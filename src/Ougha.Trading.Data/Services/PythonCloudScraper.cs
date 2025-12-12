using System.Globalization;
using HtmlAgilityPack;
using Python.Runtime;

namespace Ougha.Trading.Data.Services;

/// <summary>
/// Scrapes Forex Factory using Python's cloudscraper library for HTTP (bypasses Cloudflare)
/// and HtmlAgilityPack for HTML parsing.
/// </summary>
public class PythonCloudScraper : IDisposable
{
    private readonly bool _ownsGil;
    
    private const string PythonDll = @"C:\Users\Ougha\AppData\Local\Programs\Python\Python312\python312.dll";
    
    public PythonCloudScraper()
    {
        if (PythonEngine.IsInitialized) return;
        Runtime.PythonDLL = PythonDll;
        PythonEngine.Initialize();
        PythonEngine.BeginAllowThreads();
        _ownsGil = true;
    }
    
    /// <summary>
    /// Scrape a single week. Used for incremental caching.
    /// </summary>
    public Task<List<EconomicEvent>> ScrapeWeekAsync(DateTime weekStart)
    {
        // Python.NET requires all operations on a consistent thread to avoid GIL corruption
        var tcs = new TaskCompletionSource<List<EconomicEvent>>();
        var thread = new Thread(() =>
        {
            try
            {
                var result = ScrapeWeekSync(weekStart);
                tcs.SetResult(result);
            }
            catch (Exception ex)
            {
                tcs.SetException(ex);
            }
        })
        {
            IsBackground = true
        };
        thread.Start();
        return tcs.Task;
    }
    
    private List<EconomicEvent> ScrapeWeekSync(DateTime weekStart)
    {
        // Step 1: Fetch HTML using Python cloudscraper (bypasses Cloudflare)
        var html = FetchHtmlWithPython(weekStart);
        
        if (string.IsNullOrEmpty(html))
        {
            Console.WriteLine("[PythonCloudScraper] Failed to fetch HTML");
            return [];
        }
        
        // Step 2: Parse HTML using HtmlAgilityPack (C#)
        var events = ParseHtmlWithAgilityPack(html, weekStart);
        Console.WriteLine($"[PythonCloudScraper] Found {events.Count} events");
        return events;
    }
    
    private string FetchHtmlWithPython(DateTime weekStart)
    {
        using (Py.GIL())
        {
            try
            {
                dynamic cloudscraper = Py.Import("cloudscraper");
                var scraper = cloudscraper.create_scraper();
                
                var monthName = weekStart.ToString("MMM", CultureInfo.InvariantCulture).ToLower();
                var url = $"https://www.forexfactory.com/calendar?week={monthName}{weekStart.Day}.{weekStart.Year}";
                
                Console.WriteLine($"[PythonCloudScraper] Fetching week of {weekStart:yyyy-MM-dd}...");
                
                using PyObject response = scraper.get(url, timeout: 30);
                return response.GetAttr("text").ToString() ?? "";
            }
            catch (Exception ex)
            {
                Console.WriteLine($"[PythonCloudScraper] Fetch error: {ex.Message}");
                return "";
            }
        }
    }
    
    private static List<EconomicEvent> ParseHtmlWithAgilityPack(string html, DateTime weekStart)
    {
        var events = new List<EconomicEvent>();
        var doc = new HtmlDocument();
        doc.LoadHtml(html);
        
        var rows = doc.DocumentNode.SelectNodes("//tr").Where(c=>c.Attributes.Any(x=>x.Name == "data-event-id"));

        var currentDate = weekStart;
        
        foreach (var row in rows)
        {
            try
            {
                // Check for date header row
                var dateCell = row.SelectSingleNode(".//td[contains(@class, 'calendar__date')]");
                {
                    var dateText = dateCell.InnerText.Trim();
                    if (!string.IsNullOrEmpty(dateText) && TryParseDate(dateText, weekStart.Year, out var parsedDate))
                    {
                        currentDate = parsedDate;
                    }
                }

                // Extract currency
                var currencyCell = row.SelectSingleNode(".//td[contains(@class, 'calendar__currency')]");

                var currency = currencyCell.InnerText.Trim();

                
                // Extract time
                var timeCell = row.SelectSingleNode(".//td[contains(@class, 'calendar__time')]");
                var timeText = timeCell?.InnerText.Trim() ?? "";
                var eventTime = ParseEventTime(currentDate, timeText);
                
                // Extract event name
                var eventCell = row.SelectSingleNode(".//td[contains(@class, 'calendar__event')]");

                var eventSpan = eventCell.SelectSingleNode(".//span");

                var eventName = eventSpan.InnerText.Trim();

                
                // Extract impact from span class
                var impactCell = row.SelectSingleNode(".//td[contains(@class, 'calendar__impact')]");
                var impactSpan = impactCell?.SelectSingleNode(".//span");
                var impactClass = impactSpan?.GetAttributeValue("class", "") ?? "";
                var impact = ParseImpact(impactClass);
                
                // Extract actual/forecast/previous
                var actualCell = row.SelectSingleNode(".//td[contains(@class, 'calendar__actual')]");
                var forecastCell = row.SelectSingleNode(".//td[contains(@class, 'calendar__forecast')]");
                var previousCell = row.SelectSingleNode(".//td[contains(@class, 'calendar__previous')]");
                
                var actual = ParseNumericValue(actualCell?.InnerText);
                var forecast = ParseNumericValue(forecastCell?.InnerText);
                var previous = ParseNumericValue(previousCell?.InnerText);
                
                events.Add(new EconomicEvent(
                    eventTime,
                    currency,
                    eventName,
                    impact,
                    forecast,
                    previous,
                    actual
                ));
            }
            catch
            {
                // Skip malformed rows
            }
        }
        
        return events;
    }

    private static bool TryParseDate(string dateText, int year, out DateTime result)
    {
        result = DateTime.UtcNow.Date;
        
        var match = System.Text.RegularExpressions.Regex.Match(dateText, @"(\w+)\s+(\d+)");
        if (!match.Success) return false;
        
        var monthDay = $"{match.Groups[1].Value} {match.Groups[2].Value}";
        
        return DateTime.TryParseExact(
            $"{monthDay} {year}",
            "MMM d yyyy",
            CultureInfo.InvariantCulture,
            DateTimeStyles.None,
            out result);
    }
    
    private static DateTime ParseEventTime(DateTime date, string timeText)
    {
        if (string.IsNullOrEmpty(timeText) || timeText.Contains("Day"))
            return date;
        
        return DateTime.TryParseExact(
            timeText,
            ["h:mmtt", "htt"],
            CultureInfo.InvariantCulture,
            DateTimeStyles.None,
            out var time) ? date.Add(time.TimeOfDay) : date;
    }
    
    private static EventImpact ParseImpact(string impactClass)
    {
        if (impactClass.Contains("red") || impactClass.Contains("high"))
            return EventImpact.High;
        return impactClass.Contains("ora") || impactClass.Contains("medium") ? EventImpact.Medium : EventImpact.Low;
    }
    
    private static double? ParseNumericValue(string? text)
    {
        if (string.IsNullOrWhiteSpace(text)) return null;
        
        text = System.Text.RegularExpressions.Regex.Replace(text.Trim(), @"[%KMB]", "");
        
        if (double.TryParse(text, NumberStyles.Any, 
            CultureInfo.InvariantCulture, out var result))
            return result;
        
        return null;
    }
    
    public void Dispose()
    {
        if (_ownsGil && PythonEngine.IsInitialized)
        {
            PythonEngine.Shutdown();
        }
    }
}

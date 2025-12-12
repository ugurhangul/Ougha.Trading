using System.Globalization;
using System.Text.RegularExpressions;
using HtmlAgilityPack;

namespace Ougha.Trading.Data.Services;

/// <summary>
/// Scrapes Forex Factory economic calendar for real event data.
/// URL format: https://www.forexfactory.com/calendar?week=jan5.2025
/// </summary>
public class ForexFactoryScraper : IDisposable
{
    private readonly CloudflareBypassClient _httpClient;
    private const string BaseUrl = "https://www.forexfactory.com/calendar";
    
    private static readonly Dictionary<string, EventImpact> ImpactMap = new()
    {
        ["high"] = EventImpact.High,
        ["medium"] = EventImpact.Medium,
        ["low"] = EventImpact.Low,
        ["holiday"] = EventImpact.Low
    };
    
    public ForexFactoryScraper(CloudflareBypassClient? httpClient = null)
    {
        _httpClient = httpClient ?? new CloudflareBypassClient();
    }
    
    public void Dispose()
    {
        _httpClient.Dispose();
    }
    
    /// <summary>
    /// Scrape events for a specific week (Monday date)
    /// </summary>
    public async Task<List<EconomicEvent>> ScrapeWeekAsync(DateTime weekStart)
    {
        var events = new List<EconomicEvent>();
        
        try
        {
            var url = BuildWeekUrl(weekStart);
            var html = await _httpClient.GetStringAsync(url);
            events = ParseCalendarHtml(html);
        }
        catch (Exception ex)
        {
            Console.WriteLine($"[ForexFactoryScraper] C# scraper failed: {ex.Message}");
            // Will fall back to Python in ScrapeRangeAsync
            throw;
        }
        
        return events;
    }
    
    /// <summary>
    /// Scrape events for a date range (fetches multiple weeks).
    /// Falls back to Python cloudscraper if Cloudflare blocks us.
    /// </summary>
    public async Task<List<EconomicEvent>> ScrapeRangeAsync(DateTime from, DateTime to)
    {
        var allEvents = new List<EconomicEvent>();
        
        // Get Monday of start week
        var weekStart = from.AddDays(-(int)from.DayOfWeek + (int)DayOfWeek.Monday);
        if (weekStart > from) weekStart = weekStart.AddDays(-7);
        
        bool usePython = false;
        
        while (weekStart <= to)
        {
            if (usePython)
            {
                // Use Python cloudscraper for remaining weeks
                using var pythonScraper = new PythonCloudScraper();
                var pythonEvents = await pythonScraper.ScrapeForexFactoryAsync(weekStart, to);
                allEvents.AddRange(pythonEvents.Where(e => e.Time >= from && e.Time <= to));
                break; // Python scraper handles full range
            }
            
            try
            {
                var weekEvents = await ScrapeWeekAsync(weekStart);
                allEvents.AddRange(weekEvents.Where(e => e.Time >= from && e.Time <= to));
                weekStart = weekStart.AddDays(7);
                
                // Rate limiting
                await Task.Delay(500);
            }
            catch
            {
                // Cloudflare blocked us - fall back to Python
                Console.WriteLine("[ForexFactoryScraper] Falling back to Python cloudscraper...");
                usePython = true;
            }
        }
        
        return allEvents.OrderBy(e => e.Time).ToList();
    }
    
    private static string BuildWeekUrl(DateTime weekStart)
    {
        var monthName = weekStart.ToString("MMM", CultureInfo.InvariantCulture).ToLower();
        return $"{BaseUrl}?week={monthName}{weekStart.Day}.{weekStart.Year}";
    }
    
    private List<EconomicEvent> ParseCalendarHtml(string html)
    {
        var events = new List<EconomicEvent>();
        var doc = new HtmlDocument();
        doc.LoadHtml(html);
        
        // Forex Factory uses a table with class "calendar__table"
        var rows = doc.DocumentNode.SelectNodes("//tr[contains(@class, 'calendar__row')]");
        if (rows == null) return events;
        
        DateTime currentDate = DateTime.UtcNow.Date;
        
        foreach (var row in rows)
        {
            try
            {
                // Skip header rows
                if (row.HasClass("calendar__row--day-breaker"))
                {
                    var dateCell = row.SelectSingleNode(".//td[contains(@class, 'calendar__date')]");
                    if (dateCell != null)
                    {
                        var dateText = dateCell.InnerText.Trim();
                        if (TryParseDate(dateText, out var parsedDate))
                            currentDate = parsedDate;
                    }
                    continue;
                }
                
                // Extract currency
                var currencyNode = row.SelectSingleNode(".//td[contains(@class, 'calendar__currency')]");
                var currency = currencyNode?.InnerText.Trim() ?? "";
                if (string.IsNullOrEmpty(currency)) continue;
                
                // Extract time
                var timeNode = row.SelectSingleNode(".//td[contains(@class, 'calendar__time')]");
                var timeText = timeNode?.InnerText.Trim() ?? "";
                var eventTime = ParseEventTime(currentDate, timeText);
                
                // Extract event name
                var eventNode = row.SelectSingleNode(".//td[contains(@class, 'calendar__event')]//span");
                var eventName = eventNode?.InnerText.Trim() ?? "";
                if (string.IsNullOrEmpty(eventName)) continue;
                
                // Extract impact
                var impactNode = row.SelectSingleNode(".//td[contains(@class, 'calendar__impact')]//span");
                var impactClass = impactNode?.GetAttributeValue("class", "") ?? "";
                var impact = ParseImpact(impactClass);
                
                // Extract actual/forecast/previous
                var actualNode = row.SelectSingleNode(".//td[contains(@class, 'calendar__actual')]");
                var forecastNode = row.SelectSingleNode(".//td[contains(@class, 'calendar__forecast')]");
                var previousNode = row.SelectSingleNode(".//td[contains(@class, 'calendar__previous')]");
                
                var actual = ParseNumericValue(actualNode?.InnerText);
                var forecast = ParseNumericValue(forecastNode?.InnerText);
                var previous = ParseNumericValue(previousNode?.InnerText);
                
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
    
    private static bool TryParseDate(string dateText, out DateTime result)
    {
        result = DateTime.UtcNow.Date;
        
        // Format: "Mon Jan 6"
        var match = Regex.Match(dateText, @"(\w+)\s+(\d+)");
        if (!match.Success) return false;
        
        var monthDay = $"{match.Groups[1].Value} {match.Groups[2].Value}";
        
        if (DateTime.TryParseExact(
            $"{monthDay} {DateTime.UtcNow.Year}",
            "MMM d yyyy",
            CultureInfo.InvariantCulture,
            DateTimeStyles.None,
            out result))
        {
            return true;
        }
        
        return false;
    }
    
    private static DateTime ParseEventTime(DateTime date, string timeText)
    {
        if (string.IsNullOrEmpty(timeText) || timeText.Contains("Day"))
            return date;
        
        // Format: "8:30am" or "Tentative"
        if (DateTime.TryParseExact(
            timeText,
            new[] { "h:mmtt", "htt" },
            CultureInfo.InvariantCulture,
            DateTimeStyles.None,
            out var time))
        {
            return date.Add(time.TimeOfDay);
        }
        
        return date;
    }
    
    private static EventImpact ParseImpact(string impactClass)
    {
        impactClass = impactClass.ToLower();
        
        if (impactClass.Contains("high") || impactClass.Contains("red"))
            return EventImpact.High;
        if (impactClass.Contains("medium") || impactClass.Contains("orange") || impactClass.Contains("yellow"))
            return EventImpact.Medium;
        
        return EventImpact.Low;
    }
    
    private static double? ParseNumericValue(string? text)
    {
        if (string.IsNullOrWhiteSpace(text)) return null;
        
        // Remove % and other symbols
        text = Regex.Replace(text.Trim(), @"[%KMB]", "");
        
        if (double.TryParse(text, NumberStyles.Any, CultureInfo.InvariantCulture, out var result))
            return result;
        
        return null;
    }
}

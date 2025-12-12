using Python.Runtime;

namespace Ougha.Trading.Data.Services;

/// <summary>
/// Scrapes Forex Factory using Python's cloudscraper library via Python.NET.
/// This reliably bypasses Cloudflare protection.
/// </summary>
public class PythonCloudScraper : IDisposable
{
    private readonly bool _ownsGil;
    
    private const string PythonDll = @"C:\Users\Ougha\AppData\Local\Programs\Python\Python312\python312.dll";
    
    public PythonCloudScraper()
    {
        if (!PythonEngine.IsInitialized)
        {
            Runtime.PythonDLL = PythonDll;
            PythonEngine.Initialize();
            PythonEngine.BeginAllowThreads();
            _ownsGil = true;
        }
    }
    
    /// <summary>
    /// Scrape a single week. Used for incremental caching.
    /// </summary>
    public Task<List<EconomicEvent>> ScrapeWeekAsync(DateTime weekStart)
    {
        return Task.Run(() => ScrapeWeekSync(weekStart));
    }
    
    private List<EconomicEvent> ScrapeWeekSync(DateTime weekStart)
    {
        using (Py.GIL())
        {
            try
            {
                dynamic cloudscraper = Py.Import("cloudscraper");
                dynamic bs4 = Py.Import("bs4");
                dynamic scraper = cloudscraper.create_scraper();
                
                Console.WriteLine($"[PythonCloudScraper] Fetching week of {weekStart:yyyy-MM-dd}...");
                var events = ScrapeWeekSync(scraper, bs4, weekStart);
                Console.WriteLine($"[PythonCloudScraper] Found {events.Count} events");
                return events;
            }
            catch (Exception ex)
            {
                Console.WriteLine($"[PythonCloudScraper] Error: {ex.Message}");
                return [];
            }
        }
    }
    
    public Task<List<EconomicEvent>> ScrapeForexFactoryAsync(DateTime from, DateTime to)
    {
        // Run all Python operations synchronously in a background task
        return Task.Run(() => ScrapeForexFactorySync(from, to));
    }
    
    private List<EconomicEvent> ScrapeForexFactorySync(DateTime from, DateTime to)
    {
        var events = new List<EconomicEvent>();
        
        using (Py.GIL())
        {
            try
            {
                // Import cloudscraper and BeautifulSoup
                dynamic cloudscraper = Py.Import("cloudscraper");
                dynamic bs4 = Py.Import("bs4");
                
                dynamic scraper = cloudscraper.create_scraper();
                
                // Calculate week range
                var weekStart = from.AddDays(-(int)from.DayOfWeek + (int)DayOfWeek.Monday);
                if (weekStart > from) weekStart = weekStart.AddDays(-7);
                
                var totalWeeks = (int)Math.Ceiling((to - weekStart).TotalDays / 7) + 1;
                var currentWeek = 0;
                
                while (weekStart <= to)
                {
                    currentWeek++;
                    Console.WriteLine($"[PythonCloudScraper] ({currentWeek}/{totalWeeks}) Fetching week of {weekStart:yyyy-MM-dd}...");
                    
                    try
                    {
                        var weekEvents = ScrapeWeekSync(scraper, bs4, weekStart);
                        foreach (var evt in weekEvents)
                        {
                            if (evt.Time >= from && evt.Time <= to)
                                events.Add(evt);
                        }
                        Console.WriteLine($"[PythonCloudScraper] Found {weekEvents.Count} events");
                    }
                    catch (Exception ex)
                    {
                        Console.WriteLine($"[PythonCloudScraper] Error on week {weekStart:yyyy-MM-dd}: {ex.Message}");
                    }
                    
                    weekStart = weekStart.AddDays(7);
                    
                    // Rate limiting - brief sleep
                    Thread.Sleep(300);
                }
            }
            catch (Exception ex)
            {
                Console.WriteLine($"[PythonCloudScraper] Error: {ex.Message}");
                Console.WriteLine($"[PythonCloudScraper] Make sure cloudscraper and beautifulsoup4 are installed:");
                Console.WriteLine($"  pip install cloudscraper beautifulsoup4");
            }
        }
        
        return events.OrderBy(e => e.Time).ToList();
    }
    
    private static List<EconomicEvent> ScrapeWeekSync(dynamic scraper, dynamic bs4, DateTime weekStart)
    {
        var events = new List<EconomicEvent>();
        
        var monthName = weekStart.ToString("MMM", System.Globalization.CultureInfo.InvariantCulture).ToLower();
        var url = $"https://www.forexfactory.com/calendar?week={monthName}{weekStart.Day}.{weekStart.Year}";
        
        dynamic response = scraper.get(url, timeout: 30);
        var html = (string)response.text;
        
        dynamic soup = bs4.BeautifulSoup(html, "html.parser");
        dynamic rows = soup.find_all("tr", class_: "calendar__row");
        
        DateTime currentDate = weekStart;
        
        foreach (dynamic row in rows)
        {
            try
            {
                // Check for date header row
                dynamic? dateCell = row.find("td", class_: "calendar__date");
                if (dateCell != null)
                {
                    var dateText = ((string?)dateCell.text)?.Trim();
                    if (!string.IsNullOrEmpty(dateText) && TryParseDate(dateText, weekStart.Year, out var parsedDate))
                    {
                        currentDate = parsedDate;
                        continue;
                    }
                }
                
                // Extract currency
                dynamic? currencyCell = row.find("td", class_: "calendar__currency");
                if (currencyCell == null) continue;
                var currency = ((string?)currencyCell.text)?.Trim() ?? "";
                if (string.IsNullOrEmpty(currency)) continue;
                
                // Extract time
                dynamic? timeCell = row.find("td", class_: "calendar__time");
                var timeText = timeCell != null ? ((string?)timeCell.text)?.Trim() ?? "" : "";
                var eventTime = ParseEventTime(currentDate, timeText);
                
                // Extract event name
                dynamic? eventCell = row.find("td", class_: "calendar__event");
                dynamic? eventSpan = eventCell?.find("span");
                if (eventSpan == null) continue;
                var eventName = ((string?)eventSpan.text)?.Trim() ?? "";
                if (string.IsNullOrEmpty(eventName)) continue;
                
                // Extract impact from title attribute using raw HTML (Python.NET attr access is unreliable)
                dynamic? impactCell = row.find("td", class_: "calendar__impact").find("span");
                string impactTitle = "";
                if (impactCell != null)
                {
                    try
                    {
                        // Get raw HTML and extract title with regex
                        // Use Python's str() to convert PyObject to string
                        dynamic builtins = Py.Import("builtins");
                        string cellHtml = (string)builtins.str(impactCell);
                        var titleMatch = System.Text.RegularExpressions.Regex.Match(
                            cellHtml, @"class=""([^""]+)""",
                            System.Text.RegularExpressions.RegexOptions.IgnoreCase);
                        if (titleMatch.Success)
                        {
                            impactTitle = titleMatch.Groups[1].Value.ToLower();
                        }
                    }
                    catch
                    {
                        
                    }
                }
                var impact = ParseImpact(impactTitle);
                
   // Extract actual/forecast/previous
                dynamic? actualCell = row.find("td", class_: "calendar__actual");
                dynamic? forecastCell = row.find("td", class_: "calendar__forecast");
                dynamic? previousCell = row.find("td", class_: "calendar__previous");
                
                var actual = ParseNumericValue(actualCell?.text?.ToString());
                var forecast = ParseNumericValue(forecastCell?.text?.ToString());
                var previous = ParseNumericValue(previousCell?.text?.ToString());
                
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
            System.Globalization.CultureInfo.InvariantCulture,
            System.Globalization.DateTimeStyles.None,
            out result);
    }
    
    private static DateTime ParseEventTime(DateTime date, string timeText)
    {
        if (string.IsNullOrEmpty(timeText) || timeText.Contains("Day"))
            return date;
        
        if (DateTime.TryParseExact(
            timeText,
            new[] { "h:mmtt", "htt" },
            System.Globalization.CultureInfo.InvariantCulture,
            System.Globalization.DateTimeStyles.None,
            out var time))
        {
            return date.Add(time.TimeOfDay);
        }
        
        return date;
    }
    
    private static EventImpact ParseImpact(string impactTitle)
    {
        if (impactTitle.Contains("red"))
            return EventImpact.High;
        return impactTitle.Contains("ora") ? EventImpact.Medium : EventImpact.Low;
    }
    
    private static double? ParseNumericValue(string? text)
    {
        if (string.IsNullOrWhiteSpace(text)) return null;
        
        text = System.Text.RegularExpressions.Regex.Replace(text.Trim(), @"[%KMB]", "");
        
        if (double.TryParse(text, System.Globalization.NumberStyles.Any, 
            System.Globalization.CultureInfo.InvariantCulture, out var result))
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

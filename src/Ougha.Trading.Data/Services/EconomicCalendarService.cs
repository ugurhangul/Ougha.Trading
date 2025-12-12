using System.Text.Json;

namespace Ougha.Trading.Data.Services;

/// <summary>
/// Service for providing economic calendar events and news features.
/// Uses ForexFactoryScraper to fetch real economic event data.
/// Caches scraped events per-week to allow incremental scraping.
/// </summary>
public class EconomicCalendarService
{
    private readonly ForexFactoryScraper _scraper;
    private List<EconomicEvent> _events = [];
    private DateTime _loadedFrom = DateTime.MinValue;
    private DateTime _loadedTo = DateTime.MinValue;
    private readonly Lock _cacheLock = new();
    
    private static readonly string CacheDir = Path.Combine(
        Environment.GetFolderPath(Environment.SpecialFolder.LocalApplicationData),
        "Ougha.Trading", "economic_events_cache");
    
    private static readonly JsonSerializerOptions JsonOptions = new()
    {
        WriteIndented = true,
        PropertyNamingPolicy = JsonNamingPolicy.CamelCase
    };
    
    public EconomicCalendarService()
    {
        _scraper = new ForexFactoryScraper();
        Directory.CreateDirectory(CacheDir);
    }
    
    /// <summary>
    /// Load events for a date range. Uses per-week caching - only scrapes missing weeks.
    /// </summary>
    public async Task LoadEventsAsync(DateTime from, DateTime to)
    {
        Console.WriteLine($"[EconomicCalendarService] Loading events: {from:yyyy-MM-dd} to {to:yyyy-MM-dd}");
        
        var allEvents = new List<EconomicEvent>();
        
        // Get Monday of start week
        var weekStart = from.AddDays(-(int)from.DayOfWeek + (int)DayOfWeek.Monday);
        if (weekStart > from) weekStart = weekStart.AddDays(-7);
        
        var weeksToScrape = new List<DateTime>();
        var totalWeeks = (int)Math.Ceiling((to - weekStart).TotalDays / 7) + 1;
        var currentWeek = 0;
        
        // First pass: load from cache or mark for scraping
        var tempWeekStart = weekStart;
        while (tempWeekStart <= to)
        {
            currentWeek++;
            var cachedEvents = await LoadWeekFromCacheAsync(tempWeekStart);
            
            if (cachedEvents.Count > 0)
            {
                allEvents.AddRange(cachedEvents);
                Console.WriteLine($"[EconomicCalendarService] ({currentWeek}/{totalWeeks}) Loaded week {tempWeekStart:yyyy-MM-dd} from cache ({cachedEvents.Count} events)");
            }
            else
            {
                weeksToScrape.Add(tempWeekStart);
            }
            
            tempWeekStart = tempWeekStart.AddDays(7);
        }
        
        // Second pass: scrape missing weeks
        if (weeksToScrape.Count > 0)
        {
            Console.WriteLine($"[EconomicCalendarService] Need to scrape {weeksToScrape.Count} weeks...");
            
            using var pythonScraper = new PythonCloudScraper();
            
            foreach (var week in weeksToScrape)
            {
                var scrapedEvents = await pythonScraper.ScrapeWeekAsync(week);
                
                if (scrapedEvents.Count > 0)
                {
                    await SaveWeekToCacheAsync(week, scrapedEvents);
                    allEvents.AddRange(scrapedEvents);
                }
            }
        }
        
        // Filter to requested range and store
        lock (_cacheLock)
        {
            _events = allEvents
                .Where(e => e.Time >= from && e.Time <= to)
                .OrderBy(e => e.Time)
                .ToList();
            _loadedFrom = from;
            _loadedTo = to;
        }
        
        Console.WriteLine($"[EconomicCalendarService] Total: {_events.Count} events loaded for date range");
    }
    
    private string GetWeekCacheFileName(DateTime weekStart)
    {
        return Path.Combine(CacheDir, $"week_{weekStart:yyyyMMdd}.json");
    }
    
    private async Task<List<EconomicEvent>> LoadWeekFromCacheAsync(DateTime weekStart)
    {
        var cacheFile = GetWeekCacheFileName(weekStart);
        
        if (!File.Exists(cacheFile))
            return [];
        
        try
        {
            var json = await File.ReadAllTextAsync(cacheFile);
            var events = JsonSerializer.Deserialize<List<EconomicEventDto>>(json, JsonOptions);
            return events?.Select(e => new EconomicEvent(
                e.Time,
                e.Currency,
                e.EventName,
                (EventImpact)e.Impact,
                e.Forecast,
                e.Previous,
                e.Actual
            )).ToList() ?? [];
        }
        catch
        {
            return [];
        }
    }
    
    private async Task SaveWeekToCacheAsync(DateTime weekStart, List<EconomicEvent> events)
    {
        var cacheFile = GetWeekCacheFileName(weekStart);
        
        try
        {
            var dtos = events.Select(e => new EconomicEventDto(
                e.Time,
                e.Currency,
                e.EventName,
                (int)e.Impact,
                e.Forecast,
                e.Previous,
                e.Actual
            )).ToList();
            
            var json = JsonSerializer.Serialize(dtos, JsonOptions);
            await File.WriteAllTextAsync(cacheFile, json);
            Console.WriteLine($"[EconomicCalendarService] Cached week {weekStart:yyyy-MM-dd} ({events.Count} events)");
        }
        catch (Exception ex)
        {
            Console.WriteLine($"[EconomicCalendarService] Failed to save cache: {ex.Message}");
        }
    }
    
    /// <summary>
    /// Get all loaded events
    /// </summary>
    public IReadOnlyList<EconomicEvent> Events
    {
        get
        {
            lock (_cacheLock)
            {
                return _events.ToList();
            }
        }
    }
    
    /// <summary>
    /// Build NewsFeatures (16 floats) for the given timestamp.
    /// </summary>
    public float[] BuildNewsFeatures(DateTime timestamp, string[] activeCurrencies)
    {
        var features = new float[16];
        
        List<EconomicEvent> events;
        lock (_cacheLock)
        {
            events = _events.ToList();
        }
        
        if (events.Count == 0)
            return features;
        
        var relevantEvents = events
            .Where(e => e.Time >= timestamp && e.Time <= timestamp.AddDays(2))
            .OrderBy(e => e.Time)
            .ToList();
        
        // Features 0-2: Time to next high-impact event for USD, EUR, GBP
        var currencies = new[] { "USD", "EUR", "GBP" };
        for (var i = 0; i < 3; i++)
        {
            var nextHigh = relevantEvents
                .FirstOrDefault(e => e.Currency == currencies[i] && e.Impact == EventImpact.High);
            
            if (nextHigh != null)
            {
                var hoursUntil = (nextHigh.Time - timestamp).TotalHours;
                features[i] = (float)Math.Max(0, Math.Min(48, hoursUntil)) / 48f;
            }
            else
            {
                features[i] = 1f;
            }
        }
        
        // Features 3-5: Time to next medium+ impact event
        for (var i = 0; i < 3; i++)
        {
            var nextMedium = relevantEvents
                .FirstOrDefault(e => e.Currency == currencies[i] && e.Impact >= EventImpact.Medium);
            
            if (nextMedium != null)
            {
                var hoursUntil = (nextMedium.Time - timestamp).TotalHours;
                features[3 + i] = (float)Math.Max(0, Math.Min(48, hoursUntil)) / 48f;
            }
            else
            {
                features[3 + i] = 1f;
            }
        }
        
        // Features 6-8: Event impact score for active positions
        for (var i = 0; i < 3 && i < activeCurrencies.Length; i++)
        {
            var currency = activeCurrencies[i].Length >= 3 
                ? activeCurrencies[i][..3]
                : activeCurrencies[i];
                
            var impactEvents = relevantEvents
                .Where(e => e.Currency == currency && e.Time <= timestamp.AddHours(6))
                .ToList();
            
            var totalImpact = impactEvents.Sum(e => (int)e.Impact) / 9f;
            features[6 + i] = Math.Min(1f, totalImpact);
        }
        
        // Features 9-11: Expected volatility multiplier
        for (var i = 0; i < 3; i++)
        {
            var nearEvents = relevantEvents
                .Where(e => e.Currency == currencies[i] && 
                           Math.Abs((e.Time - timestamp).TotalMinutes) <= 60)
                .ToList();
            
            features[9 + i] = nearEvents.Any(e => e.Impact == EventImpact.High) ? 2f :
                              nearEvents.Any(e => e.Impact == EventImpact.Medium) ? 1.5f : 1f;
        }
        
        // Features 12-14: Forecast vs previous delta (normalized)
        var recentEvents = relevantEvents.Take(3).ToList();
        for (var i = 0; i < 3 && i < recentEvents.Count; i++)
        {
            var evt = recentEvents[i];
            if (evt.Forecast.HasValue && evt.Previous.HasValue && evt.Previous.Value != 0)
            {
                var delta = (evt.Forecast.Value - evt.Previous.Value) / Math.Abs(evt.Previous.Value);
                features[12 + i] = (float)Math.Clamp(delta, -1, 1);
            }
        }
        
        // Feature 15: Currently in high-impact event window flag
        var inWindow = relevantEvents.Any(e => 
            e.Impact == EventImpact.High &&
            Math.Abs((e.Time - timestamp).TotalMinutes) <= 30);
        features[15] = inWindow ? 1f : 0f;
        
        return features;
    }
    
    /// <summary>
    /// Reset service state
    /// </summary>
    public void Reset()
    {
        lock (_cacheLock)
        {
            _events.Clear();
            _loadedFrom = DateTime.MinValue;
            _loadedTo = DateTime.MinValue;
        }
    }
    
    /// <summary>
    /// Clear all cached files
    /// </summary>
    public void ClearCache()
    {
        try
        {
            foreach (var file in Directory.GetFiles(CacheDir, "*.json"))
            {
                File.Delete(file);
            }
        }
        catch { }
    }
}

// DTO for JSON serialization
internal record EconomicEventDto(
    DateTime Time,
    string Currency,
    string EventName,
    int Impact,
    double? Forecast,
    double? Previous,
    double? Actual
);

/// <summary>
/// Represents a single economic calendar event
/// </summary>
public record EconomicEvent(
    DateTime Time,
    string Currency,
    string EventName,
    EventImpact Impact,
    double? Forecast,
    double? Previous,
    double? Actual
);

/// <summary>
/// Impact level of economic event
/// </summary>
public enum EventImpact
{
    Low = 1,
    Medium = 2,
    High = 3
}

using System.Text.Json;

namespace Ougha.Trading.Data.Services;

/// <summary>
/// Service for providing economic calendar events and news features.
/// Uses ForexFactoryScraper to fetch real economic event data.
/// Caches scraped events per-currency to allow incremental updates.
/// </summary>
public class EconomicCalendarService
{
    private List<EconomicEvent> _events = [];
    private readonly Lock _cacheLock = new();
    
    private static readonly string CacheDir = Path.Combine(
        Environment.GetFolderPath(Environment.SpecialFolder.LocalApplicationData),
        "Ougha.Trading", "economic_events_cache");
    
    private static readonly JsonSerializerOptions JsonOptions = new()
    {
        WriteIndented = true,
        PropertyNamingPolicy = JsonNamingPolicy.CamelCase,
        PropertyNameCaseInsensitive = true
    };

    public EconomicCalendarService()
    {
        Directory.CreateDirectory(CacheDir);
    }
    
    /// <summary>
    /// Load events for a date range. Uses per-currency caching where each currency
    /// file tracks which weeks have been scraped for that currency.
    /// </summary>
    public async Task LoadEventsAsync(DateTime from, DateTime to)
    {
        
        // Get Monday of start week
        var weekStart = from.AddDays(-(int)from.DayOfWeek + (int)DayOfWeek.Monday).Date;
        if (weekStart > from) weekStart = weekStart.AddDays(-7);
        
        // Build list of all weeks in range
        var allWeeksInRange = new List<DateTime>();
        var tempWeekStart = weekStart;
        while (tempWeekStart <= to)
        {
            allWeeksInRange.Add(tempWeekStart);
            tempWeekStart = tempWeekStart.AddDays(7);
        }
        
        // Load all currency caches and find which weeks are missing globally
        // A week needs scraping if ANY currency hasn't been scraped for that week yet
        var scrapedWeeksPerCurrency = await LoadAllCurrencyScrapedWeeksAsync();
        
        // Find weeks that need scraping (weeks not in ANY currency cache)
        // We scrape globally, then distribute events to currency files
        var globallyScrapedWeeks = scrapedWeeksPerCurrency.Values
            .SelectMany(w => w)
            .Distinct()
            .ToHashSet();
        
        var weeksToScrape = allWeeksInRange
            .Where(w => !globallyScrapedWeeks.Contains(w))
            .ToList();
        
        // Scrape missing weeks
        if (weeksToScrape.Count > 0)
        {
            Console.WriteLine($"[EconomicCalendarService] Need to scrape {weeksToScrape.Count} weeks...");
            
            using var pythonScraper = new PythonCloudScraper();
            
            foreach (var week in weeksToScrape)
            {
                var scrapedEvents = await pythonScraper.ScrapeWeekAsync(week);
                
                // Append events to their respective currency files with week tracking
                await AppendEventsToCurrencyFilesAsync(scrapedEvents, week);
            }
        }
        
        // Load all events from currency files for the requested date range
        var allEvents = await LoadEventsFromCurrencyFilesAsync(from, to);
        
        lock (_cacheLock)
        {
            _events = allEvents.OrderBy(e => e.Time).ToList();
        }
    }
    
    private static string GetCurrencyFileName(string currency)
    {
        return Path.Combine(CacheDir, $"{currency}.json");
    }
    
    /// <summary>
    /// Load scraped weeks for all currencies from their cache files.
    /// Returns a dictionary mapping currency -> set of scraped week start dates.
    /// </summary>
    private static async Task<Dictionary<string, HashSet<DateTime>>> LoadAllCurrencyScrapedWeeksAsync()
    {
        var result = new Dictionary<string, HashSet<DateTime>>();
        
        if (!Directory.Exists(CacheDir))
            return result;
        
        var currencyFiles = Directory.GetFiles(CacheDir, "*.json")
            .Where(f => !f.EndsWith("scraped_weeks.json"))
            .ToList();
        
        foreach (var file in currencyFiles)
        {
            var currency = Path.GetFileNameWithoutExtension(file);
            
            try
            {
                var json = await File.ReadAllTextAsync(file);
                
                // Try to deserialize as new format (CurrencyCacheFile with ScrapedWeeks)
                var cache = JsonSerializer.Deserialize<CurrencyCacheFile>(json, JsonOptions);
                if (cache?.ScrapedWeeks != null)
                {
                    result[currency] = cache.ScrapedWeeks.Select(w => w.Date).ToHashSet();
                    continue;
                }
            }
            catch
            {
                // Might be old format - try to read as plain event list and infer weeks
            }
            
            try
            {
                var json = await File.ReadAllTextAsync(file);
                var events = JsonSerializer.Deserialize<List<EconomicEventDto>>(json, JsonOptions);
                
                if (events != null)
                {
                    var weeks = new HashSet<DateTime>();
                    foreach (var ev in events)
                    {
                        var weekStart = ev.Time.AddDays(-(int)ev.Time.DayOfWeek + (int)DayOfWeek.Monday).Date;
                        weeks.Add(weekStart);
                    }
                    result[currency] = weeks;
                }
            }
            catch
            {
                // Skip corrupted files
            }
        }
        
        return result;
    }
    
    /// <summary>
    /// Append events to currency-specific cache files with week tracking.
    /// </summary>
    private static async Task AppendEventsToCurrencyFilesAsync(List<EconomicEvent> events, DateTime weekStart)
    {
        // Group events by currency
        var eventsByCurrency = events.GroupBy(e => e.Currency).ToList();
        
        foreach (var group in eventsByCurrency)
        {
            var currency = group.Key;
            var currencyFile = GetCurrencyFileName(currency);
            
            // Load existing cache file
            var existingEvents = new List<EconomicEventDto>();
            var existingWeeks = new List<DateTime>();
            
            if (File.Exists(currencyFile))
            {
                try
                {
                    var json = await File.ReadAllTextAsync(currencyFile);
                    
                    // Try new format first
                    var existingCache = JsonSerializer.Deserialize<CurrencyCacheFile>(json, JsonOptions);
                    if (existingCache != null)
                    {
                        existingEvents = existingCache.Events;
                        existingWeeks = existingCache.ScrapedWeeks;
                    }
                    else
                    {
                        // Fall back to old format (plain list)
                        existingEvents = JsonSerializer.Deserialize<List<EconomicEventDto>>(json, JsonOptions) ?? [];
                    }
                }
                catch
                {
                    // Start fresh
                }
            }
            
            // Add week to scraped weeks if not present
            if (!existingWeeks.Any(w => w.Date == weekStart.Date))
            {
                existingWeeks.Add(weekStart.Date);
            }
            
            // Create a set of existing event keys for deduplication
            var existingKeys = existingEvents
                .Select(e => $"{e.Time:O}|{e.EventName}")
                .ToHashSet();
            
            // Append new events (avoiding duplicates)
            var newEvents = group
                .Where(e => !existingKeys.Contains($"{e.Time:O}|{e.EventName}"))
                .Select(e => new EconomicEventDto(
                    e.Time,
                    e.Currency,
                    e.EventName,
                    (int)e.Impact,
                    e.Forecast,
                    e.Previous,
                    e.Actual
                ))
                .ToList();
            
            existingEvents.AddRange(newEvents);
            existingEvents = existingEvents.OrderBy(e => e.Time).ToList();
            
            // Save as new format with scraped weeks
            var newCache = new CurrencyCacheFile(
                existingWeeks.OrderBy(w => w).ToList(),
                existingEvents
            );
            
            var outputJson = JsonSerializer.Serialize(newCache, JsonOptions);
            await File.WriteAllTextAsync(currencyFile, outputJson);
            
            if (newEvents.Count > 0)
            {
                Console.WriteLine($"[EconomicCalendarService] {currency}: +{newEvents.Count} events (week {weekStart:yyyy-MM-dd})");
            }
        }
    }
    
    private static async Task<List<EconomicEvent>> LoadEventsFromCurrencyFilesAsync(DateTime from, DateTime to)
    {
        var allEvents = new List<EconomicEvent>();
        
        if (!Directory.Exists(CacheDir))
            return allEvents;
        
        // Get all currency files in cache directory
        var currencyFiles = Directory.GetFiles(CacheDir, "*.json")
            .Where(f => !f.EndsWith("scraped_weeks.json"));
        
        foreach (var file in currencyFiles)
        {
            try
            {
                var json = await File.ReadAllTextAsync(file);
                
                List<EconomicEventDto>? events = null;
                
                // Try new format first (CurrencyCacheFile)
                try
                {
                    var cache = JsonSerializer.Deserialize<CurrencyCacheFile>(json, JsonOptions);
                    if (cache?.Events != null)
                    {
                        events = cache.Events;
                    }
                }
                catch
                {
                    // Fall back to old format
                }
                
                // Try old format (plain list)
                if (events == null)
                {
                    events = JsonSerializer.Deserialize<List<EconomicEventDto>>(json, JsonOptions);
                }
                
                if (events != null)
                {
                    var filtered = events
                        .Where(e => e.Time >= from && e.Time <= to)
                        .Select(e => new EconomicEvent(
                            e.Time,
                            e.Currency,
                            e.EventName,
                            (EventImpact)e.Impact,
                            e.Forecast,
                            e.Previous,
                            e.Actual
                        ));
                    
                    allEvents.AddRange(filtered);
                }
            }
            catch (Exception ex)
            {
                Console.WriteLine($"[EconomicCalendarService] Failed to load {Path.GetFileName(file)}: {ex.Message}");
            }
        }
        
        return allEvents;
    }


    /// <summary>
    /// Calculate the expected news feature size for the given number of symbols.
    /// Feature layout: 4 features per symbol + 1 global flag.
    /// </summary>
    public static int GetNewsFeatureSize(int symbolCount)
    {
        // Per symbol: high-impact time, medium-impact time, impact score, volatility multiplier
        // Global: high-impact window flag
        return (symbolCount * 4) + 1;
    }

    /// <summary>
    /// Build NewsFeatures for the given timestamp.
    /// symbols should contain the trading symbols (e.g., ["EURUSD", "GBPUSD", "USDTRY"]).
    /// For each symbol, extracts base (first 3 chars) and quote (chars 3-6) currencies.
    /// Feature size = (N symbols * 4) + 1
    /// </summary>
    public float[] BuildNewsFeatures(DateTime timestamp, string[] symbols)
    {
        var n = symbols.Length;
        var featureSize = GetNewsFeatureSize(n);
        var features = new float[featureSize];
        
        List<EconomicEvent> events;
        lock (_cacheLock)
        {
            events = _events.ToList();
        }
        
        if (events.Count == 0 || n == 0)
            return features;
        
        var relevantEvents = events
            .Where(e => e.Time >= timestamp && e.Time <= timestamp.AddDays(2))
            .OrderBy(e => e.Time)
            .ToList();
        
        // Collect all unique currencies for the global flag
        var allCurrencies = new HashSet<string>();
        
        for (var i = 0; i < n; i++)
        {
            var symbol = symbols[i];
            // Extract base and quote currencies from symbol (e.g., EURUSD -> EUR, USD)
            var baseCurrency = symbol.Length >= 3 ? symbol[..3] : symbol;
            var quoteCurrency = symbol.Length >= 6 ? symbol.Substring(3, 3) : "";
            var symbolCurrencies = new[] { baseCurrency, quoteCurrency }.Where(c => !string.IsNullOrEmpty(c)).ToArray();
            
            allCurrencies.UnionWith(symbolCurrencies);
            
            // Section 1: Time to next high-impact event for this symbol (from either currency)
            var nextHigh = relevantEvents
                .FirstOrDefault(e => symbolCurrencies.Contains(e.Currency) && e.Impact == EventImpact.High);
            
            if (nextHigh != null)
            {
                var hoursUntil = (nextHigh.Time - timestamp).TotalHours;
                features[i] = (float)Math.Max(0, Math.Min(48, hoursUntil)) / 48f;
            }
            else
            {
                features[i] = 1f;
            }
            
            // Section 2: Time to next medium+ impact event for this symbol
            var section2Start = n;
            var nextMedium = relevantEvents
                .FirstOrDefault(e => symbolCurrencies.Contains(e.Currency) && e.Impact >= EventImpact.Medium);
            
            if (nextMedium != null)
            {
                var hoursUntil = (nextMedium.Time - timestamp).TotalHours;
                features[section2Start + i] = (float)Math.Max(0, Math.Min(48, hoursUntil)) / 48f;
            }
            else
            {
                features[section2Start + i] = 1f;
            }
            
            // Section 3: Combined event impact score for this symbol (both currencies)
            var section3Start = n * 2;
            var impactEvents = relevantEvents
                .Where(e => symbolCurrencies.Contains(e.Currency) && e.Time <= timestamp.AddHours(6))
                .ToList();
            
            var totalImpact = impactEvents.Sum(e => (int)e.Impact) / 9f;
            features[section3Start + i] = Math.Min(1f, totalImpact);
            
            // Section 4: Expected volatility multiplier for this symbol (max of both currencies)
            var section4Start = n * 3;
            var nearEvents = relevantEvents
                .Where(e => symbolCurrencies.Contains(e.Currency) && 
                           Math.Abs((e.Time - timestamp).TotalMinutes) <= 60)
                .ToList();
            
            features[section4Start + i] = nearEvents.Any(e => e.Impact == EventImpact.High) ? 2f :
                              nearEvents.Any(e => e.Impact == EventImpact.Medium) ? 1.5f : 1f;
        }
        
        // Section 5: Global high-impact window flag (any symbol's currencies)
        var inWindow = relevantEvents.Any(e => 
            allCurrencies.Contains(e.Currency) &&
            e.Impact == EventImpact.High &&
            Math.Abs((e.Time - timestamp).TotalMinutes) <= 30);
        features[n * 4] = inWindow ? 1f : 0f;
        
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
        }
    }
}

// DTO for JSON serialization - includes scraped weeks metadata
internal record CurrencyCacheFile(
    List<DateTime> ScrapedWeeks,
    List<EconomicEventDto> Events
);

// DTO for individual event
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

using Ougha.Trading.Data.Downloaders;

namespace Ougha.Trading.Data.Services;

public class DataService(
    QuestDbDataLoader repo,
    Ex2ArchiveDownloader downloader)
{
    public async Task EnsureDataReadyAsync(string symbol, DateTime start, DateTime end)
    {
        Console.WriteLine($"Checking data readiness for {symbol} from {start:yyyy-MM-dd} to {end:yyyy-MM-dd}...");

        var existingDays = await repo.GetExistingDaysAsync(symbol, start, end);
        var missingDays = new List<DateTime>();

        for (var date = start.Date; date <= end.Date; date = date.AddDays(1))
        {
            if (!existingDays.Contains(date))
            {
                var isNoData = await repo.IsDayMarkedNoDataAsync(symbol, date);
                if (!isNoData)
                {
                    missingDays.Add(date);
                }
                else
                {
                }
            }
        }

        if (missingDays.Count == 0)
        {
            Console.WriteLine("All data is present (or marked as missing).");
            return;
        }

        Console.WriteLine($"Found {missingDays.Count} missing days. Starting download...");

        foreach (var day in missingDays)
        {
            Console.WriteLine($"Downloading {symbol} for {day:yyyy-MM-dd}...");
            try
            {
                var ticks = await downloader.DownloadTicksAsync(symbol, day);
                if (ticks.Count > 0)
                {
                    Console.WriteLine($"  - Downloaded {ticks.Count} ticks. Inserting...");
                    await repo.InsertTicksAsync(symbol, ticks);
                    Console.WriteLine($"  - Insert complete.");
                }
                else
                {
                    Console.WriteLine($"  - No data found for {day:yyyy-MM-dd}. Marking as no-data.");
                    await repo.MarkDayNoDataAsync(symbol, day);
                }
            }
            catch (Exception ex)
            {
                 Console.WriteLine($"  - Error: {ex.Message}");
            }
        }
        
        Console.WriteLine("Data sync complete.");
    }

    /// <summary>
    /// Ensure S1 tick data is ready for all symbols in the given date range.
    /// Downloads missing data from Ex2 archive and inserts into QuestDB.
    /// </summary>
    /// <param name="symbols">List of symbols to check</param>
    /// <param name="start">Start date</param>
    /// <param name="end">End date</param>
    /// <returns>Report of which symbols are ready and which have issues</returns>
    public async Task<TickReadinessReport> EnsureTicksReadyAsync(
        IEnumerable<string> symbols, 
        DateTime start, 
        DateTime end)
    {
        var report = new TickReadinessReport();
        var symbolList = symbols.ToList();
        
        Console.WriteLine($"Checking tick data readiness for {symbolList.Count} symbols from {start:yyyy-MM-dd} to {end:yyyy-MM-dd}...");
        
        foreach (var symbol in symbolList)
        {
            var symbolStatus = new SymbolTickStatus { Symbol = symbol };
            
            try
            {
                // Count existing candles for this symbol
                var existingCount = await repo.CountCandlesAsync(symbol, "s1", start, end);
                symbolStatus.ExistingCandleCount = existingCount;
                
                if (existingCount > 0)
                {
                    symbolStatus.IsReady = true;
                    report.ReadySymbols.Add(symbol);
                    Console.WriteLine($"  ✓ {symbol}: {existingCount:N0} candles available");
                }
                else
                {
                    // Check for missing days and download
                    Console.WriteLine($"  ⚠ {symbol}: No data found, attempting download...");
                    await EnsureDataReadyAsync(symbol, start, end);
                    
                    // Re-check after download
                    existingCount = await repo.CountCandlesAsync(symbol, "s1", start, end);
                    symbolStatus.ExistingCandleCount = existingCount;
                    
                    if (existingCount > 0)
                    {
                        symbolStatus.IsReady = true;
                        report.ReadySymbols.Add(symbol);
                        Console.WriteLine($"  ✓ {symbol}: Downloaded {existingCount:N0} candles");
                    }
                    else
                    {
                        symbolStatus.IsReady = false;
                        symbolStatus.Error = "No data available after download attempt";
                        report.MissingSymbols.Add(symbol);
                        Console.WriteLine($"  ✗ {symbol}: No data available");
                    }
                }
            }
            catch (Exception ex)
            {
                symbolStatus.IsReady = false;
                symbolStatus.Error = ex.Message;
                report.FailedSymbols.Add(symbol);
                Console.WriteLine($"  ✗ {symbol}: Error - {ex.Message}");
            }
            
            report.SymbolStatuses.Add(symbolStatus);
        }
        
        Console.WriteLine($"\nTick Readiness Summary:");
        Console.WriteLine($"  Ready: {report.ReadySymbols.Count}/{symbolList.Count}");
        Console.WriteLine($"  Missing: {report.MissingSymbols.Count}");
        Console.WriteLine($"  Failed: {report.FailedSymbols.Count}");
        
        return report;
    }
}

/// <summary>
/// Report of tick data readiness for training
/// </summary>
public class TickReadinessReport
{
    public List<string> ReadySymbols { get; } = [];
    public List<string> MissingSymbols { get; } = [];
    public List<string> FailedSymbols { get; } = [];
    public List<SymbolTickStatus> SymbolStatuses { get; } = [];
    
    public bool AllReady => MissingSymbols.Count == 0 && FailedSymbols.Count == 0;
    public long TotalCandleCount => SymbolStatuses.Sum(s => s.ExistingCandleCount);
}

public class SymbolTickStatus
{
    public required string Symbol { get; init; }
    public bool IsReady { get; set; }
    public long ExistingCandleCount { get; set; }
    public string? Error { get; set; }
}

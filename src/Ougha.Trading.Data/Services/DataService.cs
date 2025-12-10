using Ougha.Trading.Core.Models;
using Ougha.Trading.Data.Downloaders;

namespace Ougha.Trading.Data.Services;

public class DataService
{
    private readonly QuestDBDataLoader _repo;
    private readonly Ex2ArchiveDownloader _downloader;
    private readonly SymbolInfoService _symbolService;

    public DataService(
        QuestDBDataLoader repo,
        Ex2ArchiveDownloader downloader,
        SymbolInfoService symbolService)
    {
        _repo = repo;
        _downloader = downloader;
        _symbolService = symbolService;
    }

    public async Task EnsureDataReadyAsync(string symbol, DateTime start, DateTime end)
    {
        Console.WriteLine($"Checking data readiness for {symbol} from {start:yyyy-MM-dd} to {end:yyyy-MM-dd}...");

        var existingDays = await _repo.GetExistingDaysAsync(symbol, start, end);
        var missingDays = new List<DateTime>();

        for (var date = start.Date; date <= end.Date; date = date.AddDays(1))
        {
            if (!existingDays.Contains(date))
            {
                // Check if explicitly marked as no-data
                bool isNoData = await _repo.IsDayMarkedNoDataAsync(symbol, date);
                if (!isNoData)
                {
                    missingDays.Add(date);
                }
                else
                {
                    // Console.WriteLine($"Skipping {date:yyyy-MM-dd} (Marked as No-Data).");
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
                var ticks = await _downloader.DownloadTicksAsync(symbol, day);
                if (ticks != null && ticks.Count > 0)
                {
                    Console.WriteLine($"  - Downloaded {ticks.Count} ticks. Inserting...");
                    await _repo.InsertTicksAsync(symbol, ticks);
                    Console.WriteLine($"  - Insert complete.");
                }
                else
                {
                    // Mark as no data
                    Console.WriteLine($"  - No data found for {day:yyyy-MM-dd}. Marking as no-data.");
                    await _repo.MarkDayNoDataAsync(symbol, day);
                }
            }
            catch (Exception ex)
            {
                 Console.WriteLine($"  - Error: {ex.Message}");
            }
            
            // GC explicitly? In .NET usually not needed for per-loop scope if strictly scoped.
        }
        
        Console.WriteLine("Data sync complete.");
    }
    
    public SymbolInfo GetSymbolInfo(string symbol)
    {
        return _symbolService.GetSymbolInfo(symbol);
    }
}

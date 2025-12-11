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

}

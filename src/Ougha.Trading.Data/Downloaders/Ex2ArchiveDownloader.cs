using System.IO.Compression;
using System.Text;
using Ougha.Trading.Core.Models;
using System.Globalization;

namespace Ougha.Trading.Data.Downloaders;

public class Ex2ArchiveDownloader
{
    private const string BaseUrl = "https://ticks.ex2archive.com";
    private readonly HttpClient _client;
    private readonly string _cacheDir;

    public Ex2ArchiveDownloader(HttpClient client, string cacheDir = "data/archives")
    {
        _client = client;
        _cacheDir = Path.IsPathRooted(cacheDir) ? cacheDir : Path.Combine(Environment.CurrentDirectory, cacheDir);
        
        if (!Directory.Exists(_cacheDir))
            Directory.CreateDirectory(_cacheDir);
    }

    public async Task<List<Tick>> DownloadTicksAsync(string symbol, DateTime date, string broker = "Exness")
    {
        // Try Day
        var dayTicks = await TryDownloadDayAsync(symbol, date, broker);
        if (dayTicks != null) return dayTicks;

        // Try Month (and filter)
        var monthTicks = await TryDownloadMonthAsync(symbol, date, broker);
        if (monthTicks != null) 
            return FilterToDay(monthTicks, date);

        // Try Year (and filter)
        var yearTicks = await TryDownloadYearAsync(symbol, date, broker);
        if (yearTicks != null)
             return FilterToDay(yearTicks, date);

        return new List<Tick>();
    }

    private List<Tick> FilterToDay(List<Tick> ticks, DateTime date)
    {
        var start = date.Date;
        var end = start.AddDays(1);
        return ticks.Where(t => t.Time >= start && t.Time < end).ToList();
    }

    private async Task<List<Tick>?> TryDownloadDayAsync(string symbol, DateTime date, string broker)
    {
        // URL: /ticks/{symbol}/{year}/{month:02}/{day:02}/{Broker}_{Symbol}_{Year}_{Month:02}_{Day:02}.zip
        int year = date.Year;
        int month = date.Month;
        int day = date.Day;
        string url = $"{BaseUrl}/ticks/{symbol}/{year}/{month:D2}/{day:D2}/{broker}_{symbol}_{year}_{month:D2}_{day:D2}.zip";
        string cacheFile = Path.Combine(_cacheDir, $"{symbol}_{year}_{month:D2}_{day:D2}.zip");

        return await DownloadAndParseAsync(url, cacheFile);
    }

    private async Task<List<Tick>?> TryDownloadMonthAsync(string symbol, DateTime date, string broker)
    {
        int year = date.Year;
        int month = date.Month;
        string url = $"{BaseUrl}/ticks/{symbol}/{year}/{month:D2}/{broker}_{symbol}_{year}_{month:D2}.zip";
        string cacheFile = Path.Combine(_cacheDir, $"{symbol}_{year}_{month:D2}.zip");
        return await DownloadAndParseAsync(url, cacheFile);
    }
    
    private async Task<List<Tick>?> TryDownloadYearAsync(string symbol, DateTime date, string broker)
    {
        int year = date.Year;
        string url = $"{BaseUrl}/ticks/{symbol}/{year}/{broker}_{symbol}_{year}.zip";
        string cacheFile = Path.Combine(_cacheDir, $"{symbol}_{year}.zip");
        return await DownloadAndParseAsync(url, cacheFile);
    }

    private async Task<List<Tick>?> DownloadAndParseAsync(string url, string cachePath)
    {
        try 
        {
            byte[] data;
            if (File.Exists(cachePath) && new FileInfo(cachePath).Length > 0)
            {
                data = await File.ReadAllBytesAsync(cachePath);
            }
            else
            {
                var response = await _client.GetAsync(url);
                if (!response.IsSuccessStatusCode) return null;
                data = await response.Content.ReadAsByteArrayAsync();
                await File.WriteAllBytesAsync(cachePath, data);
            }

            return ParseZip(data);
        }
        catch (Exception ex)
        {
            Console.WriteLine($"Error downloading {url}: {ex.Message}");
            return null;
        }
    }

    private List<Tick> ParseZip(byte[] zipData)
    {
        var ticks = new List<Tick>();
        using var ms = new MemoryStream(zipData);
        using var archive = new ZipArchive(ms);
        
        foreach (var entry in archive.Entries)
        {
            if (!entry.FullName.EndsWith(".csv")) continue;
            
            using var stream = entry.Open();
            using var reader = new StreamReader(stream);
            
            // Assuming header exists or implied? Python code had detection.
            // Python code detected: timestamp, bid, ask, volume
            // Let's assume standard format: timestamp,bid,ask,volume
            // Or try to parse header?
            
            string? header = reader.ReadLine();
            // Simple parsing for now - assume standard format
            
            while (!reader.EndOfStream)
            {
                var line = reader.ReadLine();
                if (string.IsNullOrWhiteSpace(line)) continue;
                
                var parts = line.Split(new[] { ',', ';' }); 
                if (parts.Length < 3) continue;

                if (DateTime.TryParse(parts[0], CultureInfo.InvariantCulture, DateTimeStyles.AdjustToUniversal, out var time) &&
                    double.TryParse(parts[1], NumberStyles.Any, CultureInfo.InvariantCulture, out var bid) &&
                    double.TryParse(parts[2], NumberStyles.Any, CultureInfo.InvariantCulture, out var ask))
                {
                     double vol = 0;
                     if (parts.Length > 3) double.TryParse(parts[3], NumberStyles.Any, CultureInfo.InvariantCulture, out vol);
                     
                     // Last = Bid (approx)
                     ticks.Add(new Tick(time, bid, ask, 1, false, vol)); // TickType 1=Info/Quote?
                }
            }
        }
        return ticks;
    }
}

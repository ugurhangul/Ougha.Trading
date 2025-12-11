using System.IO.Compression;
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
        var dayTicks = await TryDownloadDayAsync(symbol, date, broker);
        if (dayTicks != null) return dayTicks;

        var monthTicks = await TryDownloadMonthAsync(symbol, date, broker);
        if (monthTicks != null) 
            return FilterToDay(monthTicks, date);

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
        var year = date.Year;
        var month = date.Month;
        var day = date.Day;
        var url = $"{BaseUrl}/ticks/{symbol}/{year}/{month:D2}/{day:D2}/{broker}_{symbol}_{year}_{month:D2}_{day:D2}.zip";
        var cacheFile = Path.Combine(_cacheDir, $"{symbol}_{year}_{month:D2}_{day:D2}.zip");

        return await DownloadAndParseAsync(url, cacheFile);
    }

    private async Task<List<Tick>?> TryDownloadMonthAsync(string symbol, DateTime date, string broker)
    {
        var year = date.Year;
        var month = date.Month;
        var url = $"{BaseUrl}/ticks/{symbol}/{year}/{month:D2}/{broker}_{symbol}_{year}_{month:D2}.zip";
        var cacheFile = Path.Combine(_cacheDir, $"{symbol}_{year}_{month:D2}.zip");
        return await DownloadAndParseAsync(url, cacheFile);
    }
    
    private async Task<List<Tick>?> TryDownloadYearAsync(string symbol, DateTime date, string broker)
    {
        var year = date.Year;
        var url = $"{BaseUrl}/ticks/{symbol}/{year}/{broker}_{symbol}_{year}.zip";
        var cacheFile = Path.Combine(_cacheDir, $"{symbol}_{year}.zip");
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

                     ticks.Add(new Tick(time, bid, ask, 1, false, vol));
                }
            }
        }
        return ticks;
    }
}

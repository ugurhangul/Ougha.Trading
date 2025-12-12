using System.Threading.Channels;
using Ougha.Trading.Core.Models;
using Ougha.Trading.Data;

namespace Ougha.Trading.RL.Training;

public record ChunkConfig(
    int ChunkDays = 7,
    int PrefetchChunks = 10,
    int HistoryBufferDays = 1,
    int EpisodeDays = 5
);

public record DataChunk(
    DateTime StartDate,
    DateTime EndDate,
    List<(DateTime Time, string Symbol, Candle Candle)> Candles,
    Dictionary<string, List<Candle>> HistoryBySymbol
);

public class ChunkBasedDataProvider : IDisposable
{
    private readonly QuestDbDataLoader _dataLoader;
    private readonly List<string> _symbols;
    private readonly DateTime _globalStartDate;
    private readonly DateTime _globalEndDate;
    private readonly ChunkConfig _config;
    private readonly string _timeframe;
    
    private readonly Channel<DataChunk> _chunkChannel;
    private readonly CancellationTokenSource _cts;
    private Task? _producerTask;
    
    private readonly List<(DateTime Start, DateTime End)> _chunkBoundaries;
    private int _currentChunkIndex;
    private DataChunk? _currentChunk;
    private readonly Random _random;
    private readonly Lock _lock = new();

    public int TotalChunks => _chunkBoundaries.Count;
    public int CurrentChunkIndex => _currentChunkIndex;

    public ChunkBasedDataProvider(
        QuestDbDataLoader dataLoader,
        IEnumerable<string> symbols,
        DateTime startDate,
        DateTime endDate,
        ChunkConfig? config = null,
        string timeframe = "s1")
    {
        _dataLoader = dataLoader;
        _symbols = symbols.ToList();
        _globalStartDate = startDate;
        _globalEndDate = endDate;
        _config = config ?? new ChunkConfig();
        _timeframe = timeframe;
        _random = new Random();
        
        _chunkBoundaries = CalculateChunkBoundaries();
        _currentChunkIndex = 0;
        
        _chunkChannel = Channel.CreateBounded<DataChunk>(new BoundedChannelOptions(_config.PrefetchChunks)
        {
            FullMode = BoundedChannelFullMode.Wait,
            SingleReader = true,
            SingleWriter = true
        });
        
        _cts = new CancellationTokenSource();
    }

    public void StartPrefetching()
    {
        var channel = _chunkChannel;
        var token = _cts.Token;
        _producerTask = Task.Run(() => ProduceChunksAsync(channel, token), token);
    }

    public async Task<DataChunk?> GetNextChunkAsync()
    {
        if (await _chunkChannel.Reader.WaitToReadAsync())
        {
            if (_chunkChannel.Reader.TryRead(out var chunk))
            {
                lock (_lock)
                {
                    _currentChunk = chunk;
                    _currentChunkIndex++;
                }
                return chunk;
            }
        }
        return null;
    }

    public (List<(DateTime Time, string Symbol, Candle Candle)> EpisodeData, 
            Dictionary<string, List<Candle>> History, 
            DateTime EpisodeStart,
            DateTime EpisodeEnd) 
        CreateEpisodeFromCurrentChunk()
    {
        lock (_lock)
        {
            if (_currentChunk == null)
                throw new InvalidOperationException("No chunk loaded. Call GetNextChunkAsync first.");

            var chunk = _currentChunk;
            var chunkDuration = (chunk.EndDate - chunk.StartDate).TotalDays;
            var maxOffset = Math.Max(0, chunkDuration - _config.EpisodeDays);
            
            var offsetDays = maxOffset > 0 ? _random.NextDouble() * maxOffset : 0;
            var episodeStart = chunk.StartDate.AddDays(offsetDays);
            var episodeEnd = episodeStart.AddDays(_config.EpisodeDays);
            
            if (episodeEnd > chunk.EndDate)
                episodeEnd = chunk.EndDate;

            var startIdx = BinarySearchTime(chunk.Candles, episodeStart);
            var endIdx = BinarySearchTime(chunk.Candles, episodeEnd);
            
            if (startIdx < 0) startIdx = ~startIdx;
            if (endIdx < 0) endIdx = ~endIdx;
            
            startIdx = Math.Max(0, Math.Min(startIdx, chunk.Candles.Count - 1));
            endIdx = Math.Min(endIdx, chunk.Candles.Count);
            
            var count = Math.Max(0, endIdx - startIdx);
            var episodeData = chunk.Candles.GetRange(startIdx, count);
            
            return (episodeData, chunk.HistoryBySymbol, episodeStart, episodeEnd);
        }
    }

    private List<(DateTime Start, DateTime End)> CalculateChunkBoundaries()
    {
        var boundaries = new List<(DateTime, DateTime)>();
        var current = _globalStartDate;
        
        while (current < _globalEndDate)
        {
            var chunkEnd = current.AddDays(_config.ChunkDays);
            if (chunkEnd > _globalEndDate)
                chunkEnd = _globalEndDate;
            
            boundaries.Add((current, chunkEnd));
            current = chunkEnd;
        }
        
        return boundaries;
    }

    private async Task ProduceChunksAsync(Channel<DataChunk> channel, CancellationToken ct)
    {
        try
        {
            foreach (var (chunkStart, chunkEnd) in _chunkBoundaries)
            {
                if (ct.IsCancellationRequested)
                    break;

                var historyStart = chunkStart.AddDays(-_config.HistoryBufferDays);
                var chunk = await LoadChunkAsync(historyStart, chunkStart, chunkEnd);

                await channel.Writer.WriteAsync(chunk, ct);
            }
        }
        finally
        {
            channel.Writer.TryComplete();
        }
    }



    private async Task<DataChunk> LoadChunkAsync(DateTime historyStart, DateTime chunkStart, DateTime chunkEnd)
    {
        var allCandles = new List<(DateTime Time, string Symbol, Candle Candle)>();
        var historyBySymbol = new Dictionary<string, List<Candle>>();
        foreach (var symbol in _symbols)
            historyBySymbol[symbol] = [];

        // Batch load all symbols in single query
        var allSymbolCandles = await _dataLoader.LoadCandlesBatchAsync(_symbols, _timeframe, historyStart, chunkEnd);
        
        foreach (var (symbol, candles) in allSymbolCandles)
        {
            foreach (var candle in candles)
            {
                if (candle.Time >= chunkStart)
                    allCandles.Add((candle.Time, symbol, candle));
                else
                    historyBySymbol[symbol].Add(candle);
            }
        }



        allCandles = allCandles.OrderBy(c => c.Time).ToList();

        foreach (var symbol in _symbols)
            historyBySymbol[symbol] = historyBySymbol[symbol].OrderBy(c => c.Time).ToList();

        return new DataChunk(chunkStart, chunkEnd, allCandles, historyBySymbol);
    }

    private static int BinarySearchTime(
        List<(DateTime Time, string Symbol, Candle Candle)> list, 
        DateTime target)
    {
        int lo = 0, hi = list.Count - 1;
        while (lo <= hi)
        {
            var mid = lo + (hi - lo) / 2;
            var cmp = list[mid].Time.CompareTo(target);
            if (cmp == 0) return mid;
            if (cmp < 0) lo = mid + 1;
            else hi = mid - 1;
        }
        return ~lo;
    }

    public void Dispose()
    {
        _cts.Cancel();
        try
        {
            _producerTask?.Wait(TimeSpan.FromSeconds(5));
        }
        catch (AggregateException ex) when (ex.InnerExceptions.All(e =>
            e is TaskCanceledException or OperationCanceledException or ChannelClosedException))
        {
        }
        _cts.Dispose();
    }
}


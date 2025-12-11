using System;
using System.Collections.Generic;
using System.Linq;
using System.Threading;
using System.Threading.Channels;
using System.Threading.Tasks;
using Ougha.Trading.Core.Models;
using Ougha.Trading.Data;

namespace Ougha.Trading.RL.Training;

public record ChunkConfig(
    int ChunkDays = 7,
    int PrefetchChunks = 2,
    int HistoryBufferDays = 1,
    int EpisodeDays = 5
);

public record DataChunk(
    DateTime StartDate,
    DateTime EndDate,
    List<(DateTime Time, string Symbol, Candle Candle)> Candles,
    Dictionary<string, List<Candle>> HistoryBySymbol,
    Dictionary<string, Dictionary<string, List<Candle>>>? MultiTimeframeCandles = null
);

public class ChunkBasedDataProvider : IDisposable
{
    private readonly QuestDbDataLoader _dataLoader;
    private readonly List<string> _symbols;
    private readonly DateTime _globalStartDate;
    private readonly DateTime _globalEndDate;
    private readonly ChunkConfig _config;
    private readonly string _timeframe;
    
    private Channel<DataChunk> _chunkChannel;
    private CancellationTokenSource _cts;
    private Task? _producerTask;
    
    private readonly List<(DateTime Start, DateTime End)> _chunkBoundaries;
    private int _currentChunkIndex;
    private DataChunk? _currentChunk;
    private readonly Random _random;
    private readonly object _lock = new();

    public int TotalChunks => _chunkBoundaries.Count;
    public int CurrentChunkIndex => _currentChunkIndex;
    public bool IsExhausted => _currentChunkIndex >= _chunkBoundaries.Count && _currentChunk == null;

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
        _producerTask = Task.Run(() => ProduceChunksAsync(channel, token));
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

    private static readonly string[] MultiTimeframes = { "m1", "m5", "m15", "h1", "h4" };

    private async Task<DataChunk> LoadChunkAsync(DateTime historyStart, DateTime chunkStart, DateTime chunkEnd)
    {
        var allCandles = new List<(DateTime Time, string Symbol, Candle Candle)>();
        var historyBySymbol = new Dictionary<string, List<Candle>>();
        var mtfCandles = new Dictionary<string, Dictionary<string, List<Candle>>>();

        foreach (var symbol in _symbols)
        {
            historyBySymbol[symbol] = new List<Candle>();
            mtfCandles[symbol] = new Dictionary<string, List<Candle>>();
        }

        foreach (var symbol in _symbols)
        {
            var candles = await _dataLoader.LoadCandlesAsync(symbol, _timeframe, historyStart, chunkEnd);
            var candleList = candles.ToList();

            foreach (var candle in candleList)
            {
                if (candle.Time >= chunkStart)
                    allCandles.Add((candle.Time, symbol, candle));
                else
                    historyBySymbol[symbol].Add(candle);
            }
        }

        var mtfTasks = new List<Task>();
        foreach (var symbol in _symbols)
        {
            foreach (var tf in MultiTimeframes)
            {
                var s = symbol;
                var t = tf;
                mtfTasks.Add(Task.Run(async () =>
                {
                    var candles = await _dataLoader.LoadCandlesAsync(s, t, historyStart, chunkEnd);
                    var list = candles.OrderBy(c => c.Time).ToList();
                    lock (mtfCandles)
                    {
                        mtfCandles[s][t.ToUpper()] = list;
                    }
                }));
            }
        }
        await Task.WhenAll(mtfTasks);

        allCandles = allCandles.OrderBy(c => c.Time).ToList();

        foreach (var symbol in _symbols)
            historyBySymbol[symbol] = historyBySymbol[symbol].OrderBy(c => c.Time).ToList();

        return new DataChunk(chunkStart, chunkEnd, allCandles, historyBySymbol, mtfCandles);
    }

    public void Reset()
    {
        lock (_lock)
        {
            _currentChunkIndex = 0;
            _currentChunk = null;
        }

        _cts.Cancel();
        try
        {
            _producerTask?.Wait(TimeSpan.FromSeconds(2));
        }
        catch (AggregateException ex) when (ex.InnerExceptions.All(e => e is TaskCanceledException or OperationCanceledException))
        {
        }

        _chunkChannel = Channel.CreateBounded<DataChunk>(new BoundedChannelOptions(_config.PrefetchChunks)
        {
            FullMode = BoundedChannelFullMode.Wait,
            SingleReader = true,
            SingleWriter = true
        });

        _cts = new CancellationTokenSource();
        StartPrefetching();
    }

    public async Task<bool> HasMoreChunksAsync()
    {
        return await _chunkChannel.Reader.WaitToReadAsync() || _currentChunkIndex < _chunkBoundaries.Count;
    }

    private static int BinarySearchTime(
        List<(DateTime Time, string Symbol, Candle Candle)> list, 
        DateTime target)
    {
        // Binary search implementation
        int lo = 0, hi = list.Count - 1;
        while (lo <= hi)
        {
            int mid = lo + (hi - lo) / 2;
            int cmp = list[mid].Time.CompareTo(target);
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


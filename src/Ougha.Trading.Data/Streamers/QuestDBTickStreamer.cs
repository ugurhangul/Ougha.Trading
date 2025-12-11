using System.Buffers;
using System.Threading.Channels;
using Ougha.Trading.Core.Models;

namespace Ougha.Trading.Data.Streamers;

public class QuestDBTickStreamer
{
    private readonly QuestDbDataLoader _dataLoader;
    private readonly int _batchSize;
    private readonly int _chunkDays;
    private readonly int _prefetchChunks;

    public QuestDBTickStreamer(
        QuestDbDataLoader dataLoader,
        int batchSize = 100000,
        int chunkDays = 7,
        int prefetchChunks = 4)
    {
        _dataLoader = dataLoader;
        _batchSize = batchSize;
        _chunkDays = chunkDays;
        _prefetchChunks = prefetchChunks;
    }

    public async IAsyncEnumerable<(string Symbol, Tick Tick)> StreamTicksAsync(
        IEnumerable<string> symbols,
        DateTime startDate,
        DateTime endDate)
    {
        var symbolList = symbols.ToList();

        if (symbolList.Count == 0)
            yield break;

        if (symbolList.Count == 1)
        {
            await foreach (var tick in StreamSingleSymbolOptimizedAsync(symbolList[0], startDate, endDate))
                yield return tick;
        }
        else
        {
            await foreach (var tick in StreamMultiSymbolOptimizedAsync(symbolList, startDate, endDate))
                yield return tick;
        }
    }

    private async IAsyncEnumerable<(string Symbol, Tick Tick)> StreamSingleSymbolOptimizedAsync(
        string symbol,
        DateTime startDate,
        DateTime endDate)
    {
        var channel = Channel.CreateBounded<List<Tick>>(new BoundedChannelOptions(_prefetchChunks)
        {
            FullMode = BoundedChannelFullMode.Wait,
            SingleReader = true,
            SingleWriter = true
        });

        var producerTask = Task.Run(async () =>
        {
            try
            {
                var current = startDate.Date;
                var end = endDate.Date;

                while (current <= end)
                {
                    var chunkEnd = current.AddDays(_chunkDays);
                    if (chunkEnd > end.AddDays(1)) chunkEnd = end.AddDays(1);

                    var ticks = await _dataLoader.LoadTicksAsync(symbol, current, chunkEnd);
                    if (ticks.Count > 0)
                        await channel.Writer.WriteAsync(ticks);
                    current = chunkEnd;
                }
            }
            finally
            {
                channel.Writer.Complete();
            }
        });

        await foreach (var chunkTicks in channel.Reader.ReadAllAsync())
        {
            foreach (var tick in chunkTicks)
            {
                yield return (symbol, tick);
            }
        }

        await producerTask;
    }

    private async IAsyncEnumerable<(string Symbol, Tick Tick)> StreamMultiSymbolOptimizedAsync(
        List<string> symbols,
        DateTime startDate,
        DateTime endDate)
    {
        var buffers = new Dictionary<string, SymbolTickBuffer>();
        var heap = new PriorityQueue<SymbolTickEntry, DateTime>();

        foreach (var symbol in symbols)
        {
            var buffer = new SymbolTickBuffer(symbol, _dataLoader, startDate, endDate, _chunkDays, _prefetchChunks);
            buffers[symbol] = buffer;

            var entry = await buffer.GetNextEntryAsync();
            if (entry.HasValue)
                heap.Enqueue(entry.Value, entry.Value.Tick.Time);
        }

        while (heap.Count > 0)
        {
            var current = heap.Dequeue();
            yield return (current.Symbol, current.Tick);

            var buffer = buffers[current.Symbol];
            var next = await buffer.GetNextEntryAsync();
            if (next.HasValue)
                heap.Enqueue(next.Value, next.Value.Tick.Time);
        }

        foreach (var buffer in buffers.Values)
            buffer.Dispose();
    }

    public async IAsyncEnumerable<TickBatch> StreamTickBatchesAsync(
        IEnumerable<string> symbols,
        DateTime startDate,
        DateTime endDate,
        int batchSize = 0)
    {
        if (batchSize <= 0) batchSize = _batchSize;

        var tickArray = ArrayPool<SymbolTick>.Shared.Rent(batchSize);
        int count = 0;

        try
        {
            await foreach (var (symbol, tick) in StreamTicksAsync(symbols, startDate, endDate))
            {
                tickArray[count++] = new SymbolTick(symbol, tick);

                if (count >= batchSize)
                {
                    yield return new TickBatch(tickArray, count);
                    count = 0;
                }
            }

            if (count > 0)
                yield return new TickBatch(tickArray, count);
        }
        finally
        {
            ArrayPool<SymbolTick>.Shared.Return(tickArray);
        }
    }

    public async Task<long> CountTicksAsync(IEnumerable<string> symbols, DateTime startDate, DateTime endDate)
    {
        var tasks = symbols.Select(s => _dataLoader.CountTicksAsync(s, startDate, endDate));
        var counts = await Task.WhenAll(tasks);
        return counts.Sum();
    }

    public async Task<List<string>> GetSymbolsWithDataAsync(IEnumerable<string> symbols, DateTime startDate, DateTime endDate)
    {
        var tasks = symbols.Select(async s => (Symbol: s, HasData: await _dataLoader.HasDataForDayAsync(s, startDate)));
        var results = await Task.WhenAll(tasks);
        return results.Where(r => r.HasData).Select(r => r.Symbol).ToList();
    }
}

internal readonly record struct SymbolTickEntry(string Symbol, Tick Tick);

internal sealed class SymbolTickBuffer : IDisposable
{
    private readonly string _symbol;
    private readonly QuestDbDataLoader _dataLoader;
    private readonly DateTime _endDate;
    private readonly int _chunkDays;
    private readonly Channel<List<Tick>> _channel;
    private readonly Task _producerTask;

    private List<Tick>? _currentBatch;
    private int _currentIndex;
    private bool _disposed;

    public SymbolTickBuffer(string symbol, QuestDbDataLoader dataLoader, DateTime startDate, DateTime endDate, int chunkDays, int prefetchChunks)
    {
        _symbol = symbol;
        _dataLoader = dataLoader;
        _endDate = endDate;
        _chunkDays = chunkDays;

        _channel = Channel.CreateBounded<List<Tick>>(new BoundedChannelOptions(prefetchChunks)
        {
            FullMode = BoundedChannelFullMode.Wait,
            SingleReader = true,
            SingleWriter = true
        });

        _producerTask = Task.Run(() => ProduceAsync(startDate));
    }

    private async Task ProduceAsync(DateTime startDate)
    {
        try
        {
            var current = startDate.Date;
            while (current <= _endDate.Date)
            {
                var chunkEnd = current.AddDays(_chunkDays);
                if (chunkEnd > _endDate.AddDays(1)) chunkEnd = _endDate.AddDays(1);

                var ticks = await _dataLoader.LoadTicksAsync(_symbol, current, chunkEnd);
                if (ticks.Count > 0)
                    await _channel.Writer.WriteAsync(ticks);
                current = chunkEnd;
            }
        }
        finally
        {
            _channel.Writer.Complete();
        }
    }

    public async ValueTask<SymbolTickEntry?> GetNextEntryAsync()
    {
        while (true)
        {
            if (_currentBatch != null && _currentIndex < _currentBatch.Count)
            {
                var tick = _currentBatch[_currentIndex++];
                return new SymbolTickEntry(_symbol, tick);
            }

            if (!await _channel.Reader.WaitToReadAsync())
                return null;

            if (_channel.Reader.TryRead(out var batch))
            {
                _currentBatch = batch;
                _currentIndex = 0;
            }
        }
    }

    public void Dispose()
    {
        if (_disposed) return;
        _disposed = true;
        _channel.Writer.TryComplete();
    }
}

public readonly record struct SymbolTick(string Symbol, Tick Tick);

public readonly struct TickBatch
{
    private readonly SymbolTick[] _ticks;
    public int Count { get; }

    public TickBatch(SymbolTick[] ticks, int count)
    {
        _ticks = ticks;
        Count = count;
    }

    public ReadOnlySpan<SymbolTick> Ticks => _ticks.AsSpan(0, Count);

    public SymbolTick this[int index] => _ticks[index];
}

public class QuestDBTickTimeline
{
    private readonly QuestDBTickStreamer _streamer;
    private readonly List<string> _symbols;
    private readonly DateTime _startDate;
    private readonly DateTime _endDate;
    
    private long? _totalTicks;
    private List<(DateTime Time, string Symbol, Tick Tick)>? _cachedTicks;
    private int _currentIndex;

    public QuestDBTickTimeline(
        QuestDbDataLoader dataLoader,
        IEnumerable<string> symbols,
        DateTime startDate,
        DateTime endDate)
    {
        _streamer = new QuestDBTickStreamer(dataLoader);
        _symbols = symbols.ToList();
        _startDate = startDate;
        _endDate = endDate;
        _currentIndex = 0;
    }

    public List<string> Symbols => _symbols;
    public DateTime StartDate => _startDate;
    public DateTime EndDate => _endDate;

    /// <summary>
    /// Get total tick count (lazy loaded).
    /// </summary>
    public async Task<long> GetCountAsync()
    {
        _totalTicks ??= await _streamer.CountTicksAsync(_symbols, _startDate, _endDate);
        return _totalTicks.Value;
    }

    /// <summary>
    /// Load all ticks into memory (for smaller datasets).
    /// Use StreamingEnumerate for large datasets.
    /// </summary>
    public async Task LoadAllAsync()
    {
        if (_cachedTicks != null) return;

        _cachedTicks = new List<(DateTime Time, string Symbol, Tick Tick)>();
        
        await foreach (var (symbol, tick) in _streamer.StreamTicksAsync(_symbols, _startDate, _endDate))
        {
            _cachedTicks.Add((tick.Time, symbol, tick));
        }
        
        _totalTicks = _cachedTicks.Count;
    }

    /// <summary>
    /// Get tick count (cached).
    /// </summary>
    public int Count => _cachedTicks?.Count ?? 0;

    /// <summary>
    /// Check if there are more ticks.
    /// </summary>
    public bool HasNext => _cachedTicks != null && _currentIndex < _cachedTicks.Count - 1;

    /// <summary>
    /// Get tick at specific index (requires LoadAllAsync first).
    /// </summary>
    public (DateTime Time, string Symbol, Tick Tick) GetAtIndex(int index)
    {
        if (_cachedTicks == null)
            throw new InvalidOperationException("Call LoadAllAsync first");
        
        if (index < 0 || index >= _cachedTicks.Count)
            throw new IndexOutOfRangeException($"Index {index} out of range (0-{_cachedTicks.Count - 1})");
        
        return _cachedTicks[index];
    }

    /// <summary>
    /// Reset timeline to beginning.
    /// </summary>
    public void Reset()
    {
        _currentIndex = 0;
    }

    /// <summary>
    /// Stream ticks without loading all into memory.
    /// </summary>
    public IAsyncEnumerable<(string Symbol, Tick Tick)> StreamAsync()
    {
        return _streamer.StreamTicksAsync(_symbols, _startDate, _endDate);
    }

    /// <summary>
    /// Get cached ticks as a list (use for conversion to TickTimeline externally).
    /// </summary>
    public List<(DateTime Time, string Symbol, Tick Tick)>? GetCachedTicks() => _cachedTicks;
}


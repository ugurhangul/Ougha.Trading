using Ougha.Trading.Core.Models;

namespace Ougha.Trading.Data.Streamers;

/// <summary>
/// Timeline that streams S1 (1-second) candles from QuestDB materialized view
/// and converts them to ticks for BacktestExecutor compatibility.
/// </summary>
public class QuestDBS1Timeline
{
    private readonly QuestDbDataLoader _dataLoader;
    private readonly List<string> _symbols;
    private readonly DateTime _startDate;
    private readonly DateTime _endDate;
    private readonly string _timeframe;

    private long? _totalCandles;

    public QuestDBS1Timeline(
        QuestDbDataLoader dataLoader,
        IEnumerable<string> symbols,
        DateTime startDate,
        DateTime endDate,
        string timeframe = "s1")
    {
        _dataLoader = dataLoader;
        _symbols = symbols.ToList();
        _startDate = startDate;
        _endDate = endDate;
        _timeframe = timeframe;
    }

    public List<string> Symbols => _symbols;
    public DateTime StartDate => _startDate;
    public DateTime EndDate => _endDate;

    /// <summary>
    /// Get total candle count (lazy loaded).
    /// </summary>
    public async Task<long> GetCountAsync()
    {
        if (_totalCandles.HasValue)
            return _totalCandles.Value;

        long total = 0;
        foreach (var symbol in _symbols)
        {
            total += await _dataLoader.CountCandlesAsync(symbol, _timeframe, _startDate, _endDate);
        }

        _totalCandles = total;
        return total;
    }

    /// <summary>
    /// Stream candles directly (for BacktestExecutor compatibility).
    /// </summary>
    public async IAsyncEnumerable<(string Symbol, Candle Candle)> StreamAsync()
    {
        if (_symbols.Count == 0)
            yield break;

        if (_symbols.Count == 1)
        {
            await foreach (var item in StreamSingleSymbolAsync(_symbols[0]))
                yield return item;
        }
        else
        {
            await foreach (var item in StreamMultiSymbolAsync())
                yield return item;
        }
    }

    private async IAsyncEnumerable<(string Symbol, Candle Candle)> StreamSingleSymbolAsync(string symbol)
    {
        await foreach (var candle in _dataLoader.StreamCandlesAsync(symbol, _timeframe, _startDate, _endDate))
        {
            yield return (symbol, candle);
        }
    }

    private async IAsyncEnumerable<(string Symbol, Candle Candle)> StreamMultiSymbolAsync()
    {
        // Use heap-based merge for multi-symbol chronological ordering
        var heap = new PriorityQueue<(string Symbol, Candle Candle, IAsyncEnumerator<Candle> Enumerator), DateTime>();
        var enumerators = new List<IAsyncEnumerator<Candle>>();

        // Initialize enumerators for each symbol
        foreach (var symbol in _symbols)
        {
            var enumerator = _dataLoader.StreamCandlesAsync(symbol, _timeframe, _startDate, _endDate).GetAsyncEnumerator();
            enumerators.Add(enumerator);

            if (await enumerator.MoveNextAsync())
            {
                heap.Enqueue((symbol, enumerator.Current, enumerator), enumerator.Current.Time);
            }
        }

        // Merge streams chronologically
        while (heap.Count > 0)
        {
            var (symbol, candle, enumerator) = heap.Dequeue();
            yield return (symbol, candle);

            if (await enumerator.MoveNextAsync())
            {
                heap.Enqueue((symbol, enumerator.Current, enumerator), enumerator.Current.Time);
            }
        }

        // Dispose enumerators
        foreach (var enumerator in enumerators)
        {
            await enumerator.DisposeAsync();
        }
    }

    /// <summary>
    /// Loads ALL candles for the configured range and symbols into memory,
    /// returning a sorted list suitable for creating a CandleTimeline.
    /// Uses parallel chunk loading for performance.
    /// </summary>
    public async Task<List<(DateTime Time, string Symbol, Candle Candle)>> LoadAllTimelineAsync()
    {
        var allCandles = new List<(DateTime Time, string Symbol, Candle Candle)>();

        foreach (var symbol in _symbols)
        {
            Console.WriteLine($"Loading {symbol}...");
            var lel = await _dataLoader.LoadCandlesAsync(symbol, _timeframe, _startDate, _endDate);
            allCandles.AddRange(lel.Select(c => (c.Time, symbol, c)));
        }

        return allCandles.OrderBy(c => c.Time).ToList();
    }
}
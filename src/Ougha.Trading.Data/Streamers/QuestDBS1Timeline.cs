using Ougha.Trading.Core.Models;

namespace Ougha.Trading.Data.Streamers;

/// <summary>
/// Timeline that streams S1 (1-second) candles from QuestDB materialized view
/// and converts them to ticks for BacktestExecutor compatibility.
/// </summary>
public class QuestDbs1Timeline(
    QuestDbDataLoader dataLoader,
    IEnumerable<string> symbols,
    DateTime startDate,
    DateTime endDate,
    string timeframe = "s1")
{
    private readonly List<string> _symbols = symbols.ToList();
    private long? _totalCandles;

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
            total += await dataLoader.CountCandlesAsync(symbol, timeframe, startDate, endDate);
        }

        _totalCandles = total;
        return total;
    }

    /// <summary>
    /// Stream candles directly (for BacktestExecutor compatibility).
    /// </summary>
    public async IAsyncEnumerable<(string Symbol, Candle Candle)> StreamAsync()
    {
        switch (_symbols.Count)
        {
            case 0:
                yield break;
            case 1:
            {
                await foreach (var item in StreamSingleSymbolAsync(_symbols[0]))
                    yield return item;
                break;
            }
            default:
            {
                await foreach (var item in StreamMultiSymbolAsync())
                    yield return item;
                break;
            }
        }
    }

    private async IAsyncEnumerable<(string Symbol, Candle Candle)> StreamSingleSymbolAsync(string symbol)
    {
        await foreach (var candle in dataLoader.StreamCandlesAsync(symbol, timeframe, startDate, endDate))
        {
            yield return (symbol, candle);
        }
    }

    private async IAsyncEnumerable<(string Symbol, Candle Candle)> StreamMultiSymbolAsync()
    {
        var heap = new PriorityQueue<(string Symbol, Candle Candle, IAsyncEnumerator<Candle> Enumerator), DateTime>();
        var enumerators = new List<IAsyncEnumerator<Candle>>();

        foreach (var symbol in _symbols)
        {
            var enumerator = dataLoader.StreamCandlesAsync(symbol, timeframe, startDate, endDate).GetAsyncEnumerator();
            enumerators.Add(enumerator);

            if (await enumerator.MoveNextAsync())
            {
                heap.Enqueue((symbol, enumerator.Current, enumerator), enumerator.Current.Time);
            }
        }

        while (heap.Count > 0)
        {
            var (symbol, candle, enumerator) = heap.Dequeue();
            yield return (symbol, candle);

            if (await enumerator.MoveNextAsync())
            {
                heap.Enqueue((symbol, enumerator.Current, enumerator), enumerator.Current.Time);
            }
        }

        foreach (var enumerator in enumerators)
        {
            await enumerator.DisposeAsync();
        }
    }
}
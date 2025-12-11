using Ougha.Trading.Core.Models;

namespace Ougha.Trading.Backtesting;

public class CandleTimeline
{
    // Flattened structure: (Time, Symbol, Candle)
    // We could optimize this, but this matches TickTimeline API for now
    private readonly IReadOnlyList<(DateTime Time, string Symbol, Candle Candle)> _allCandles;

    public int Count => _allCandles.Count;

    public CandleTimeline(IEnumerable<(DateTime Time, string Symbol, Candle Candle)> candles)
    {
        // Simple sort by time
        _allCandles = candles.OrderBy(x => x.Time).ToList();
    }

    public (DateTime Time, string Symbol, Candle Candle) GetAtIndex(int index)
    {
        return _allCandles[index];
    }
}

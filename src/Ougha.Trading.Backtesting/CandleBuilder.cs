using Ougha.Trading.Core.Models;

namespace Ougha.Trading.Backtesting;

/// <summary>
/// Aggregates ticks into candles for a specific timeframe.
/// </summary>
public class CandleBuilder
{
    private readonly TimeSpan _period;
    private Candle? _currentCandle;
    private readonly List<Candle> _completedCandles = new();

    public CandleBuilder(TimeSpan period)
    {
        _period = period;
    }

    public IReadOnlyList<Candle> Candles => _completedCandles;
    
    public Candle? CurrentCandle => _currentCandle;

    public void AddTick(Tick tick)
    {
        var periodStart = tick.Time.Ticks - (tick.Time.Ticks % _period.Ticks);
        var candleTime = new DateTime(periodStart, tick.Time.Kind);

        if (_currentCandle == null)
        {
            _currentCandle = new Candle(
                candleTime, 
                tick.Bid, tick.Bid, tick.Bid, tick.Bid, 
                (long)tick.Volume);
        }
        else if (_currentCandle.Time != candleTime)
        {
            _completedCandles.Add(_currentCandle);

            _currentCandle = new Candle(
                candleTime, 
                tick.Bid, tick.Bid, tick.Bid, tick.Bid, 
                (long)tick.Volume);
        }
        else
        {
            _currentCandle = new Candle(
                _currentCandle.Time,
                _currentCandle.Open,
                Math.Max(_currentCandle.High, tick.Bid),
                Math.Min(_currentCandle.Low, tick.Bid),
                tick.Bid,
                _currentCandle.Volume + (long)tick.Volume);
        }
    }
    
    public void Reset()
    {
        _completedCandles.Clear();
        _currentCandle = null;
    }

    /// <summary>
    /// Returns the recent N completed candles + current (snapshot).
    /// </summary>
    public List<Candle> GetRecent(int count)
    {
        var result = new List<Candle>();
        var start = Math.Max(0, _completedCandles.Count - count);

        for(var i=start; i<_completedCandles.Count; i++)
            result.Add(_completedCandles[i]);

        if (_currentCandle != null)
            result.Add(_currentCandle);

        if (result.Count > count)
             result.RemoveRange(0, result.Count - count);
             
        return result;
    }
}

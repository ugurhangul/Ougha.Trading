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
        // Align time to period start
        var periodStart = tick.Time.Ticks - (tick.Time.Ticks % _period.Ticks);
        var candleTime = new DateTime(periodStart, tick.Time.Kind);

        if (_currentCandle == null)
        {
            _currentCandle = new Candle(
                candleTime, 
                tick.Bid, tick.Bid, tick.Bid, tick.Bid, 
                tick.Volume);
        }
        else if (_currentCandle.Time != candleTime)
        {
            // Close previous
            _completedCandles.Add(_currentCandle);
            
            // Start new
            _currentCandle = new Candle(
                candleTime, 
                tick.Bid, tick.Bid, tick.Bid, tick.Bid, 
                tick.Volume);
        }
        else
        {
            // Update current
            _currentCandle = new Candle(
                _currentCandle.Time,
                _currentCandle.Open,
                Math.Max(_currentCandle.High, tick.Bid),
                Math.Min(_currentCandle.Low, tick.Bid),
                tick.Bid,
                _currentCandle.Volume + tick.Volume);
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
        int start = Math.Max(0, _completedCandles.Count - count);
        
        // Add completed
        for(int i=start; i<_completedCandles.Count; i++)
            result.Add(_completedCandles[i]);
            
        // Add current snapshot
        if (_currentCandle != null)
            result.Add(_currentCandle);
            
        // Trim if we added current and went over (though usually we want >= count or exact?)
        // FeatureBuilder usually expects 'WindowSize' completed candles, or includes current?
        // Python code usually includes current developing candle.
        
        if (result.Count > count)
             result.RemoveRange(0, result.Count - count);
             
        return result;
    }
}

using Ougha.Trading.Core.Models;

namespace Ougha.Trading.Backtesting;

/// <summary>
/// Aggregates M1 candles into higher timeframes (M5, M15, H1, H4).
/// Maintains rolling buffers for each timeframe.
/// </summary>
public class MultiTimeframeCandleAggregator
{
    private readonly Dictionary<string, TimeSpan> _timeframePeriods = new()
    {
        ["M1"] = TimeSpan.FromMinutes(1),
        ["M5"] = TimeSpan.FromMinutes(5),
        ["M15"] = TimeSpan.FromMinutes(15),
        ["H1"] = TimeSpan.FromHours(1),
        ["H4"] = TimeSpan.FromHours(4)
    };

    private readonly int _maxCandlesPerTimeframe;

    // Completed candles per timeframe
    private readonly Dictionary<string, List<Candle>> _completedCandles = new();

    // Current building candle per timeframe
    private readonly Dictionary<string, Candle?> _currentCandles = new();

    public static readonly string[] Timeframes = { "M1", "M5", "M15", "H1", "H4" };

    public MultiTimeframeCandleAggregator(int maxCandlesPerTimeframe = 100)
    {
        _maxCandlesPerTimeframe = maxCandlesPerTimeframe;

        foreach (var tf in Timeframes)
        {
            _completedCandles[tf] = new List<Candle>();
            _currentCandles[tf] = null;
        }
    }

    /// <summary>
    /// Add a tick and update all timeframe candles.
    /// </summary>
    /// <returns>List of timeframes that just closed a candle</returns>
    public List<string> AddTick(Tick tick)
    {
        var closedTimeframes = new List<string>();

        foreach (var tf in Timeframes)
        {
            var period = _timeframePeriods[tf];
            var candleTime = AlignToTimeframe(tick.Time, period);

            if (_currentCandles[tf] == null)
            {
                // Start first candle
                _currentCandles[tf] = CreateCandle(candleTime, tick);
            }
            else if (_currentCandles[tf]!.Time != candleTime)
            {
                // Close current candle and start new one
                _completedCandles[tf].Add(_currentCandles[tf]!);
                closedTimeframes.Add(tf);

                // Trim to max size
                if (_completedCandles[tf].Count > _maxCandlesPerTimeframe)
                {
                    _completedCandles[tf].RemoveAt(0);
                }

                _currentCandles[tf] = CreateCandle(candleTime, tick);
            }
            else
            {
                // Update current candle
                _currentCandles[tf] = UpdateCandle(_currentCandles[tf]!, tick);
            }
        }

        return closedTimeframes;
    }

    /// <summary>
    /// Add an M1 candle and aggregate into higher timeframes.
    /// </summary>
    /// <returns>List of timeframes that just closed a candle</returns>
    public List<string> AddM1Candle(Candle m1Candle)
    {
        var closedTimeframes = new List<string>();

        foreach (var tf in Timeframes)
        {
            var period = _timeframePeriods[tf];
            var candleTime = AlignToTimeframe(m1Candle.Time, period);

            if (_currentCandles[tf] == null)
            {
                // Start first candle
                _currentCandles[tf] = new Candle(
                    candleTime,
                    m1Candle.Open,
                    m1Candle.High,
                    m1Candle.Low,
                    m1Candle.Close,
                    m1Candle.Volume);
            }
            else if (_currentCandles[tf]!.Time != candleTime)
            {
                // Close current candle and start new one
                _completedCandles[tf].Add(_currentCandles[tf]!);
                closedTimeframes.Add(tf);

                // Trim to max size
                if (_completedCandles[tf].Count > _maxCandlesPerTimeframe)
                {
                    _completedCandles[tf].RemoveAt(0);
                }

                _currentCandles[tf] = new Candle(
                    candleTime,
                    m1Candle.Open,
                    m1Candle.High,
                    m1Candle.Low,
                    m1Candle.Close,
                    m1Candle.Volume);
            }
            else
            {
                // Merge M1 candle into current higher-timeframe candle
                var current = _currentCandles[tf]!;
                _currentCandles[tf] = new Candle(
                    current.Time,
                    current.Open,                                     // Keep original open
                    Math.Max(current.High, m1Candle.High),           // Highest high
                    Math.Min(current.Low, m1Candle.Low),             // Lowest low
                    m1Candle.Close,                                   // Latest close
                    current.Volume + m1Candle.Volume);               // Sum volume
            }
        }

        return closedTimeframes;
    }

    /// <summary>
    /// Get recent candles for a timeframe (completed + current).
    /// </summary>
    public List<Candle> GetCandles(string timeframe, int count)
    {
        if (!_completedCandles.ContainsKey(timeframe))
            return new List<Candle>();

        var completed = _completedCandles[timeframe];
        var current = _currentCandles[timeframe];

        var result = new List<Candle>();
        int start = Math.Max(0, completed.Count - count);

        for (int i = start; i < completed.Count; i++)
            result.Add(completed[i]);

        // Add current candle snapshot if exists
        if (current != null && result.Count < count)
            result.Add(current);

        return result;
    }

    /// <summary>
    /// Get all timeframe candle dictionaries (for use with MultiTimeframeStateBuilder).
    /// </summary>
    public Dictionary<string, List<Candle>> GetAllTimeframeCandles(int count)
    {
        var result = new Dictionary<string, List<Candle>>();
        foreach (var tf in Timeframes)
        {
            result[tf] = GetCandles(tf, count);
        }
        return result;
    }

    /// <summary>
    /// Check if we have sufficient candles in all timeframes.
    /// </summary>
    public bool HasSufficientData(int minCandles)
    {
        return Timeframes.All(tf => GetCandles(tf, minCandles).Count >= minCandles);
    }

    /// <summary>
    /// Get count of completed candles per timeframe.
    /// </summary>
    public Dictionary<string, int> GetCandleCounts()
    {
        return Timeframes.ToDictionary(tf => tf, tf => _completedCandles[tf].Count);
    }

    public void Reset()
    {
        foreach (var tf in Timeframes)
        {
            _completedCandles[tf].Clear();
            _currentCandles[tf] = null;
        }
    }

    private static DateTime AlignToTimeframe(DateTime time, TimeSpan period)
    {
        var periodTicks = period.Ticks;
        var alignedTicks = time.Ticks - (time.Ticks % periodTicks);
        return new DateTime(alignedTicks, time.Kind);
    }

    private static Candle CreateCandle(DateTime time, Tick tick)
    {
        return new Candle(time, tick.Bid, tick.Bid, tick.Bid, tick.Bid, tick.Volume);
    }

    private static Candle UpdateCandle(Candle current, Tick tick)
    {
        return new Candle(
            current.Time,
            current.Open,
            Math.Max(current.High, tick.Bid),
            Math.Min(current.Low, tick.Bid),
            tick.Bid,
            current.Volume + tick.Volume);
    }
}

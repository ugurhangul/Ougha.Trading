using Ougha.Trading.Core.Abstractions;
using Ougha.Trading.Core.Models;
using Ougha.Trading.RL.Agents;

namespace Ougha.Trading.RL;

/// <summary>
/// Builds multi-timeframe state for RL agent matching Python's MultiTimeframeStateBuilder.
/// Maintains feature buffers for M1, M5, M15, H1, H4 timeframes and computes confluence features.
/// </summary>
public class MultiTimeframeStateBuilder
{
    /// <summary>
    /// Supported timeframes in order (matching Python's SUPPORTED_TIMEFRAMES).
    /// </summary>
    public static readonly string[] Timeframes = ["M1", "M5", "M15", "H1", "H4", "D1"];

    private static readonly Dictionary<string, int> TimeframeIndex = Timeframes
        .Select((tf, i) => (tf, i))
        .ToDictionary(x => x.tf, x => x.i);

    private readonly int _windowSize;
    private readonly int _nFeatures;
    private readonly IFeatureBuilder _featureBuilder;

    private readonly Dictionary<string, float[,]> _featureBuffers = new();
    private readonly Dictionary<string, DateTime> _lastCandleTimes = new();

    private float[] _cachedConfluence = new float[10];
    private bool _confluenceDirty = true;

    public int WindowSize => _windowSize;
    public int NFeatures => _nFeatures;

    public MultiTimeframeStateBuilder(IFeatureBuilder featureBuilder, int windowSize = 20)
    {
        _featureBuilder = featureBuilder;
        _windowSize = windowSize;
        _nFeatures = featureBuilder.FeatureCount;

        foreach (var tf in Timeframes)
        {
            _featureBuffers[tf] = new float[windowSize, _nFeatures];
            _lastCandleTimes[tf] = DateTime.MinValue;
        }
    }

    /// <summary>
    /// Update candle buffer for a timeframe and compute features.
    /// </summary>
    /// <param name="timeframe">Timeframe name (M1, M5, etc.)</param>
    /// <param name="candles">Candle data (needs at least windowSize candles)</param>
    /// <param name="symbol">Symbol name for feature engineering</param>
    /// <returns>True if buffer was updated with new data</returns>
    public bool UpdateCandles(string timeframe, IReadOnlyList<Candle>? candles, string symbol)
    {
        if (!TimeframeIndex.ContainsKey(timeframe))
            return false;

        if (candles == null || candles.Count < _windowSize)
            return false;

        var lastCandleTime = candles[^1].Time;
        if (lastCandleTime <= _lastCandleTimes[timeframe])
            return false;

        var features = _featureBuilder.BuildFeatures(candles, symbol);

        var startRow = Math.Max(0, features.GetLength(0) - _windowSize);
        var rows = Math.Min(_windowSize, features.GetLength(0));

        var buffer = _featureBuffers[timeframe];
        for (var i = 0; i < rows; i++)
        {
            for (var j = 0; j < _nFeatures; j++)
            {
                buffer[i, j] = features[startRow + i, j];
            }
        }

        _lastCandleTimes[timeframe] = lastCandleTime;
        _confluenceDirty = true;
        return true;
    }

    /// <summary>
    /// Initialize higher timeframes with zeros if they have no data.
    /// This is preferred over copying M1 data which creates false patterns.
    /// The model learns that missing timeframes have zero features.
    /// </summary>
    public void ZeroPadMissingTimeframes()
    {
        if (_lastCandleTimes["M1"] == DateTime.MinValue)
            return;

        var m1Time = _lastCandleTimes["M1"];

        foreach (var tf in Timeframes.Skip(1))
        {
            if (_lastCandleTimes[tf] == DateTime.MinValue)
            {
                // Keep the buffer as zeros (already initialized)
                // Just mark as "initialized" at M1 time so confluence features work
                _lastCandleTimes[tf] = m1Time;
            }
        }
        _confluenceDirty = true;
    }
    
    /// <summary>
    /// DEPRECATED: Use ZeroPadMissingTimeframes instead.
    /// Copy M1 features to higher timeframes as fallback when they're unavailable.
    /// This is kept for backward compatibility but creates false patterns.
    /// </summary>
    [Obsolete("Use ZeroPadMissingTimeframes instead - copying M1 creates false patterns")]
    public void CopyM1ToMissingTimeframes()
    {
        if (_lastCandleTimes["M1"] == DateTime.MinValue)
            return;

        var m1Buffer = _featureBuffers["M1"];
        var m1Time = _lastCandleTimes["M1"];

        foreach (var tf in Timeframes.Skip(1))
        {
            if (_lastCandleTimes[tf] == DateTime.MinValue)
            {
                var buffer = _featureBuffers[tf];
                Array.Copy(m1Buffer, buffer, m1Buffer.Length);
                _lastCandleTimes[tf] = m1Time;
            }
        }
        _confluenceDirty = true;
    }

    /// <summary>
    /// Get the current timeframe feature buffers.
    /// Returns Direct reference for performance - avoid modification.
    /// </summary>
    public Dictionary<string, float[,]> GetTimeframeFeatures() => _featureBuffers;

    /// <summary>
    /// Check if we have sufficient data for all required timeframes.
    /// </summary>
    /// <param name="minTimeframes">Minimum number of timeframes with data (default 3)</param>
    public bool HasSufficientData(int minTimeframes = 3)
    {
        var count = _lastCandleTimes.Count(x => x.Value > DateTime.MinValue);
        return count >= minTimeframes;
    }

    /// <summary>
    /// Build trigger context indicating which timeframes just closed.
    /// </summary>
    /// <param name="closedTimeframes">List of timeframe names that just closed a candle</param>
    /// <returns>One-hot encoded trigger vector (6D for M1, M5, M15, H1, H4, D1)</returns>
    public float[] BuildTriggerContext(IEnumerable<string>? closedTimeframes = null)
    {
        var trigger = new float[6];
        if (closedTimeframes == null)
            return trigger;

        foreach (var tf in closedTimeframes)
        {
            if (TimeframeIndex.TryGetValue(tf, out var idx) && idx < trigger.Length)
                trigger[idx] = 1.0f;
        }
        return trigger;
    }

    /// <summary>
    /// Compute confluence features across timeframes.
    /// Features include trend alignment, volatility regime, RSI confluence, etc.
    /// </summary>
    public float[] ComputeConfluenceFeatures()
    {
        if (!_confluenceDirty)
            return _cachedConfluence;

        var confluence = new float[10];

        var availableTfs = Timeframes
            .Where(tf => _lastCandleTimes[tf] > DateTime.MinValue)
            .ToList();

        if (availableTfs.Count < 2)
        {
            _cachedConfluence = confluence;
            _confluenceDirty = false;
            return confluence;
        }

        var sma7Idx = 4;
        var sma21Idx = 6;
        var rsi14Idx = 29;
        var macdHistIdx = 33;
        var volRegimeIdx = 39;

        var trends = new List<int>();
        var volRegimes = new List<float>();
        var rsis = new List<float>();
        var macdHists = new List<float>();

        foreach (var tf in availableTfs)
        {
            var buffer = _featureBuffers[tf];
            var lastRow = _windowSize - 1;

            if (lastRow >= 0 && lastRow < buffer.GetLength(0))
            {
                var sma7 = buffer[lastRow, Math.Min(sma7Idx, _nFeatures - 1)];
                var sma21 = buffer[lastRow, Math.Min(sma21Idx, _nFeatures - 1)];
                trends.Add(sma7 > sma21 ? 1 : (sma7 < sma21 ? -1 : 0));

                if (volRegimeIdx < _nFeatures)
                    volRegimes.Add(buffer[lastRow, volRegimeIdx]);

                if (rsi14Idx < _nFeatures)
                    rsis.Add(buffer[lastRow, rsi14Idx]);

                if (macdHistIdx < _nFeatures)
                    macdHists.Add(buffer[lastRow, macdHistIdx]);
            }
        }

        if (trends.Count > 0)
            confluence[0] = trends.Sum() / (float)trends.Count;

        if (trends.Count > 1)
            confluence[1] = trends.All(t => t > 0) || trends.All(t => t < 0) ? 1f : 0f;

        if (volRegimes.Count > 0)
            confluence[2] = volRegimes.Average();

        if (volRegimes.Count > 1)
            confluence[3] = volRegimes.All(v => v > 0.5f) || volRegimes.All(v => v <= 0.5f) ? 1f : 0f;

        if (rsis.Count > 0)
            confluence[4] = rsis.Average() / 100f;

        if (rsis.Count > 1)
            confluence[5] = (rsis.Max() - rsis.Min()) / 100f;

        if (macdHists.Count > 0)
            confluence[6] = macdHists.Average();

        if (macdHists.Count > 1)
            confluence[7] = macdHists.All(m => m > 0) || macdHists.All(m => m < 0) ? 1f : 0f;

        confluence[8] = availableTfs.Count / (float)Timeframes.Length;

        if (trends.Count > 0)
            confluence[9] = trends.Count(t => t > 0) / (float)trends.Count;

        _cachedConfluence = confluence;
        _confluenceDirty = false;
        return confluence;
    }

    /// <summary>
    /// Build a complete AgentInput from current state.
    /// </summary>
    public AgentInput BuildAgentInput(
        string symbol,
        float[] portfolioFeatures,
        float[] riskState,
        IEnumerable<string>? closedTimeframes = null,
        float[]? newsFeatures = null,
        float[]? correlationFeatures = null,
        float[]? portfolioExposure = null,
        float[]? dxyFeatures = null,
        float[]? timeFeatures = null)
    {
        return new AgentInput
        {
            TimeframeFeatures = GetTimeframeFeatures(),
            SymbolId = SymbolIdMapper.GetSymbolId(symbol),
            TriggerContext = BuildTriggerContext(closedTimeframes),
            ConfluenceFeatures = ComputeConfluenceFeatures(),
            PortfolioFeatures = portfolioFeatures,
            RiskState = riskState,
            NewsFeatures = newsFeatures,
            CorrelationFeatures = correlationFeatures,
            PortfolioExposure = portfolioExposure,
            DxyFeatures = dxyFeatures,
            TimeFeatures = timeFeatures
        };
    }

    /// <summary>
    /// Build time-of-day features using cyclical encoding.
    /// Captures hour of day and day of week patterns for session-aware trading.
    /// </summary>
    /// <param name="time">Current UTC time</param>
    /// <returns>4D array: [sinHour, cosHour, sinDayOfWeek, cosDayOfWeek]</returns>
    public static float[] BuildTimeFeatures(DateTime time)
    {
        var hour = time.Hour + time.Minute / 60.0;
        var dayOfWeek = (int)time.DayOfWeek;
        
        return new float[]
        {
            (float)Math.Sin(2 * Math.PI * hour / 24.0),
            (float)Math.Cos(2 * Math.PI * hour / 24.0),
            (float)Math.Sin(2 * Math.PI * dayOfWeek / 7.0),
            (float)Math.Cos(2 * Math.PI * dayOfWeek / 7.0)
        };
    }

    /// <summary>
    /// Reset all buffers to empty state.
    /// </summary>
    public void Reset()
    {
        foreach (var tf in Timeframes)
        {
            _featureBuffers[tf] = new float[_windowSize, _nFeatures];
            _lastCandleTimes[tf] = DateTime.MinValue;
        }
        _cachedConfluence = new float[10];
        _confluenceDirty = true;
    }
}

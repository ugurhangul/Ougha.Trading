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
    public static readonly string[] Timeframes = { "M1", "M5", "M15", "H1", "H4" };

    private static readonly Dictionary<string, int> TimeframeIndex = Timeframes
        .Select((tf, i) => (tf, i))
        .ToDictionary(x => x.tf, x => x.i);

    private readonly int _windowSize;
    private readonly int _nFeatures;
    private readonly IFeatureBuilder _featureBuilder;

    // Feature buffers per timeframe [windowSize, nFeatures]
    private readonly Dictionary<string, float[,]> _featureBuffers = new();
    private readonly Dictionary<string, DateTime> _lastCandleTimes = new();

    // Confluence cache
    private float[] _cachedConfluence = new float[10];
    private bool _confluenceDirty = true;

    public int WindowSize => _windowSize;
    public int NFeatures => _nFeatures;

    public MultiTimeframeStateBuilder(IFeatureBuilder featureBuilder, int windowSize = 20)
    {
        _featureBuilder = featureBuilder;
        _windowSize = windowSize;
        _nFeatures = featureBuilder.FeatureCount;

        // Initialize empty buffers
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
    public bool UpdateCandles(string timeframe, IReadOnlyList<Candle> candles, string symbol)
    {
        if (!TimeframeIndex.ContainsKey(timeframe))
            return false;

        if (candles == null || candles.Count < _windowSize)
            return false;

        var lastCandleTime = candles[^1].Time;
        if (lastCandleTime <= _lastCandleTimes[timeframe])
            return false;

        // Build features for all candles
        var features = _featureBuilder.BuildFeatures(candles, symbol);

        // Extract last windowSize rows
        int startRow = Math.Max(0, features.GetLength(0) - _windowSize);
        int rows = Math.Min(_windowSize, features.GetLength(0));

        var buffer = _featureBuffers[timeframe];
        for (int i = 0; i < rows; i++)
        {
            for (int j = 0; j < _nFeatures; j++)
            {
                buffer[i, j] = features[startRow + i, j];
            }
        }

        _lastCandleTimes[timeframe] = lastCandleTime;
        _confluenceDirty = true;
        return true;
    }

    /// <summary>
    /// Copy M1 features to higher timeframes as fallback when they're unavailable.
    /// This ensures the model gets non-zero features even without proper MTF data.
    /// </summary>
    public void CopyM1ToMissingTimeframes()
    {
        if (_lastCandleTimes["M1"] == DateTime.MinValue)
            return; // No M1 data to copy

        var m1Buffer = _featureBuffers["M1"];
        var m1Time = _lastCandleTimes["M1"];

        foreach (var tf in Timeframes.Skip(1)) // Skip M1
        {
            if (_lastCandleTimes[tf] == DateTime.MinValue)
            {
                // Copy M1 buffer to this timeframe
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
        int count = _lastCandleTimes.Count(x => x.Value > DateTime.MinValue);
        return count >= minTimeframes;
    }

    /// <summary>
    /// Build trigger context indicating which timeframes just closed.
    /// </summary>
    /// <param name="closedTimeframes">List of timeframe names that just closed a candle</param>
    /// <returns>One-hot encoded trigger vector (5D)</returns>
    public float[] BuildTriggerContext(IEnumerable<string>? closedTimeframes = null)
    {
        var trigger = new float[5];
        if (closedTimeframes == null)
            return trigger;

        foreach (var tf in closedTimeframes)
        {
            if (TimeframeIndex.TryGetValue(tf, out var idx))
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

        // Get available timeframes with data
        var availableTfs = Timeframes
            .Where(tf => _lastCandleTimes[tf] > DateTime.MinValue)
            .ToList();

        if (availableTfs.Count < 2)
        {
            _cachedConfluence = confluence;
            _confluenceDirty = false;
            return confluence;
        }

        // Feature indices (matching Python)
        int sma7Idx = 4;  // sma_7_pct
        int sma21Idx = 6; // sma_21_pct
        int rsi14Idx = 29; // rsi_14
        int macdHistIdx = 33; // macd_hist_pct
        int volRegimeIdx = 39; // volatility_regime

        var trends = new List<int>();
        var volRegimes = new List<float>();
        var rsis = new List<float>();
        var macdHists = new List<float>();

        foreach (var tf in availableTfs)
        {
            var buffer = _featureBuffers[tf];
            int lastRow = _windowSize - 1;

            // Get feature values from last row
            if (lastRow >= 0 && lastRow < buffer.GetLength(0))
            {
                float sma7 = buffer[lastRow, Math.Min(sma7Idx, _nFeatures - 1)];
                float sma21 = buffer[lastRow, Math.Min(sma21Idx, _nFeatures - 1)];
                trends.Add(sma7 > sma21 ? 1 : (sma7 < sma21 ? -1 : 0));

                if (volRegimeIdx < _nFeatures)
                    volRegimes.Add(buffer[lastRow, volRegimeIdx]);

                if (rsi14Idx < _nFeatures)
                    rsis.Add(buffer[lastRow, rsi14Idx]);

                if (macdHistIdx < _nFeatures)
                    macdHists.Add(buffer[lastRow, macdHistIdx]);
            }
        }

        // 0: Trend alignment (-1 to 1)
        if (trends.Count > 0)
            confluence[0] = trends.Sum() / (float)trends.Count;

        // 1: Trend agreement (0 or 1)
        if (trends.Count > 1)
            confluence[1] = trends.All(t => t > 0) || trends.All(t => t < 0) ? 1f : 0f;

        // 2: Volatility regime average
        if (volRegimes.Count > 0)
            confluence[2] = volRegimes.Average();

        // 3: Volatility agreement
        if (volRegimes.Count > 1)
            confluence[3] = volRegimes.All(v => v > 0.5f) || volRegimes.All(v => v <= 0.5f) ? 1f : 0f;

        // 4: Average RSI (normalized to 0-1)
        if (rsis.Count > 0)
            confluence[4] = rsis.Average() / 100f;

        // 5: RSI spread
        if (rsis.Count > 1)
            confluence[5] = (rsis.Max() - rsis.Min()) / 100f;

        // 6: Average MACD histogram
        if (macdHists.Count > 0)
            confluence[6] = macdHists.Average();

        // 7: MACD agreement
        if (macdHists.Count > 1)
            confluence[7] = macdHists.All(m => m > 0) || macdHists.All(m => m < 0) ? 1f : 0f;

        // 8: Data coverage (0-1)
        confluence[8] = availableTfs.Count / (float)Timeframes.Length;

        // 9: Bullish timeframe ratio
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
        float[]? portfolioExposure = null)
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
            PortfolioExposure = portfolioExposure
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

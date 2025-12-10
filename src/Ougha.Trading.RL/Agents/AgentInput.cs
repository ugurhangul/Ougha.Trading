namespace Ougha.Trading.RL.Agents;

/// <summary>
/// Structured agent input matching Python's multi-input tensor architecture.
/// Each property maps to a specific ONNX model input tensor.
/// </summary>
public record AgentInput
{
    /// <summary>
    /// Feature tensors for each timeframe (M1, M5, M15, H1, H4).
    /// Each tensor has shape [windowSize, nFeatures].
    /// Keys should be timeframe names: "M1", "M5", "M15", "H1", "H4"
    /// </summary>
    public required Dictionary<string, float[,]> TimeframeFeatures { get; init; }

    /// <summary>
    /// Symbol ID (0-127 range, matching Python's symbol_to_id using CRC32).
    /// Maps to ONNX input "symbol_id" as int32 tensor.
    /// </summary>
    public int SymbolId { get; init; }

    /// <summary>
    /// Trigger context indicating which timeframes just closed (5D one-hot).
    /// Maps to ONNX input "trigger".
    /// </summary>
    public float[] TriggerContext { get; init; } = new float[5];

    /// <summary>
    /// Confluence features across timeframes (10D).
    /// Includes trend alignment, volatility agreement, RSI/MACD confluence.
    /// Maps to ONNX input "confluence".
    /// </summary>
    public float[] ConfluenceFeatures { get; init; } = new float[10];

    /// <summary>
    /// Portfolio state features (4D).
    /// [hasPosition, unrealizedPnl, holdingTimeHours, riskLevel]
    /// Maps to ONNX input "portfolio".
    /// </summary>
    public float[] PortfolioFeatures { get; init; } = new float[4];

    /// <summary>
    /// Risk state features (9D).
    /// [inPosition, direction, slDistance, tpDistance, atr, drawdown, pnl, equity, riskLevel]
    /// Maps to ONNX input "risk".
    /// </summary>
    public float[] RiskState { get; init; } = new float[9];

    /// <summary>
    /// News/calendar event features (32D). Optional.
    /// Maps to ONNX input "news".
    /// </summary>
    public float[]? NewsFeatures { get; init; }

    /// <summary>
    /// Cross-symbol correlation features (20D). Optional.
    /// Maps to ONNX input "correlation".
    /// </summary>
    public float[]? CorrelationFeatures { get; init; }

    /// <summary>
    /// Portfolio exposure per asset class (5D). Optional.
    /// Maps to ONNX input "exposure".
    /// </summary>
    public float[]? PortfolioExposure { get; init; }

    /// <summary>
    /// Creates an empty/default agent input for initialization.
    /// </summary>
    public static AgentInput Empty(int windowSize, int nFeatures) => new()
    {
        TimeframeFeatures = new Dictionary<string, float[,]>
        {
            ["M1"] = new float[windowSize, nFeatures],
            ["M5"] = new float[windowSize, nFeatures],
            ["M15"] = new float[windowSize, nFeatures],
            ["H1"] = new float[windowSize, nFeatures],
            ["H4"] = new float[windowSize, nFeatures]
        },
        SymbolId = 0,
        TriggerContext = new float[5],
        ConfluenceFeatures = new float[10],
        PortfolioFeatures = new float[4],
        RiskState = new float[9]
    };
}

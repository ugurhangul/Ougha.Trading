namespace Ougha.Trading.RL.Agents;

/// <summary>
/// Structured agent input matching Python's multi-input tensor architecture.
/// Each property maps to a specific ONNX model input tensor.
/// </summary>
public record AgentInput
{
    public required Dictionary<string, float[,]> TimeframeFeatures { get; init; }

    public int SymbolId { get; init; }

    public float[] TriggerContext { get; init; } = new float[5];

    public float[] ConfluenceFeatures { get; init; } = new float[10];

    /// <summary>
    /// Portfolio state features (5D).
    /// [positionType, unrealizedPnl, holdingTimeNorm, drawdownPct, normalizedBalance]
    /// </summary>
    public float[] PortfolioFeatures { get; init; } = new float[5];

    public float[] RiskState { get; init; } = new float[9];

    public float[]? NewsFeatures { get; init; }

    public float[]? CorrelationFeatures { get; init; }

    public float[]? PortfolioExposure { get; init; }
    
    /// <summary>
    /// DXY (US Dollar Index) features (8D).
    /// [dxyValue, change1H, change4H, change1D, rsi14, smaDeviation, volatility, trendDirection]
    /// </summary>
    public float[]? DxyFeatures { get; init; }

    /// <summary>
    /// Time-of-day features (4D) using cyclical encoding.
    /// [sinHour, cosHour, sinDayOfWeek, cosDayOfWeek]
    /// </summary>
    public float[]? TimeFeatures { get; init; }

}

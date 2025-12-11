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
        PortfolioFeatures = new float[5],
        RiskState = new float[9]
    };
}

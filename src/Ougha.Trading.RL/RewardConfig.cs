namespace Ougha.Trading.RL;

/// <summary>
/// Simplified reward configuration for price prediction strategy.
/// Only 4 reward components: R-multiple, MDD, prediction accuracy, close bonus.
/// </summary>
public class RewardConfig
{
    // ========================
    // R-MULTIPLE (PRIMARY SIGNAL)
    // ========================
    /// <summary>
    /// Scale for R-multiple reward. R-multiple = profit / risk.
    /// This is the main learning signal - higher = stronger feedback.
    /// </summary>
    public float RMultipleScale { get; set; } = 5f;

    // ========================
    // PREDICTION ACCURACY
    // ========================
    /// <summary>
    /// Enable prediction accuracy reward component.
    /// </summary>
    public bool UsePredictionAccuracyReward { get; set; } = true;
    
    /// <summary>
    /// Bonus when predicted price direction matched actual move.
    /// </summary>
    public float PredictionAccuracyBonus { get; set; } = 1.0f;
    
    /// <summary>
    /// Penalty when predicted price direction was wrong.
    /// </summary>
    public float PredictionWrongPenalty { get; set; } = 0.5f;

    // ========================
    // CLOSE ACTION
    // ========================
    /// <summary>
    /// Bonus for profitable manual close (agent CLOSE action, not TP/SL).
    /// </summary>
    public float ManualCloseBonus { get; set; } = 0.5f;

    // ========================
    // POSITION HOLDING
    // ========================
    /// <summary>
    /// Minimal per-tick signal when holding position.
    /// Just direction: +0.001 if profitable, -0.001 if losing.
    /// </summary>
    public float PositionHoldingSignal { get; set; } = 0.001f;

    // ========================
    // MAX DRAWDOWN PENALTY
    // ========================
    /// <summary>
    /// Threshold before MDD penalty kicks in (percentage).
    /// </summary>
    public float MddThreshold { get; set; } = 5f;
    
    /// <summary>
    /// Weight for max drawdown penalty.
    /// </summary>
    public float MddPenaltyWeight { get; set; } = 5f;

    // ========================
    // NORMALIZATION
    // ========================
    /// <summary>
    /// Scale for reward normalization (tanh).
    /// </summary>
    public float RewardNormalizationScale { get; set; } = 10f;
    
    /// <summary>
    /// Enable/disable reward normalization.
    /// </summary>
    public bool NormalizeRewards { get; set; } = true;
}

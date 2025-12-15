namespace Ougha.Trading.RL;

/// <summary>
/// Reward configuration for price prediction strategy.
/// Positions only close via TP/SL - no manual close option.
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
    public float PredictionWrongPenalty { get; set; } = 1.0f;  // Increased to match bonus

    // ========================
    // TP/SL REWARDS
    // ========================
    /// <summary>
    /// Bonus when TP is hit (in addition to R-multiple).
    /// Encourages letting winners run to target.
    /// </summary>
    public float TpHitBonus { get; set; } = 1.5f;
    
    /// <summary>
    /// Penalty when SL is hit (in addition to R-multiple).
    /// Discourages entries that lead to SL hits.
    /// </summary>
    public float SlHitPenalty { get; set; } = 0.5f;

    // ========================
    // POSITION HOLDING
    // ========================
    /// <summary>
    /// Minimal per-tick signal when holding position.
    /// Just direction: +0.001 if profitable, -0.001 if losing.
    /// </summary>
    public float PositionHoldingSignal { get; set; } = 0.001f;
    
    /// <summary>
    /// Progressive holding bonus per minute (only when profitable).
    /// Teaches agent that holding winners is good.
    /// </summary>
    public float HoldingBonusPerMinute { get; set; } = 0.03f;
    
    /// <summary>
    /// Maximum holding bonus (cap to prevent infinite holding).
    /// </summary>
    public float MaxHoldingBonus { get; set; } = 1.0f;

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



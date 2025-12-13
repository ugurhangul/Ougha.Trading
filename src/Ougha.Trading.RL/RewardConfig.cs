namespace Ougha.Trading.RL;

/// <summary>
/// Configuration for the improved reward function.
/// Designed for multi-symbol trading with volatility normalization.
/// </summary>
public class RewardConfig
{
    // ========================
    // VOLATILITY NORMALIZATION
    // ========================
    /// <summary>
    /// If true, profits are normalized by ATR for fair multi-symbol comparison.
    /// A 1% move in BTCUSD and 0.1% in EURUSD can have equal reward if ATR-normalized.
    /// </summary>
    public bool UseVolatilityNormalization { get; set; } = true;
    
    /// <summary>
    /// Default ATR for fallback when ATR data is unavailable.
    /// </summary>
    public double DefaultAtr { get; set; } = 0.001;

    // ========================
    // REALIZED PROFIT REWARDS
    // ========================
    /// <summary>
    /// Scale for realized profit rewards. Lower values = more stable learning.
    /// </summary>
    public float RealizedProfitScale { get; set; } = 10f;  // Normalized from 50
    
    /// <summary>
    /// Bonus for winning trades (profit > 0).
    /// </summary>
    public float WinBonus { get; set; } = 3f;  // Symmetric with LossPenalty
    
    /// <summary>
    /// Penalty for losing trades. Symmetric with WinBonus for balanced risk.
    /// </summary>
    public float LossPenalty { get; set; } = 3f;  // Symmetric with WinBonus for balanced exploration
    
    /// <summary>
    /// Risk-adjusted reward: profit / SL distance. Rewards good R:R trades.
    /// </summary>
    public float RiskRewardScale { get; set; } = 5f;  // Normalized: R-multiple capped at ±5, so max ±25

    // ========================
    // UNREALIZED PNL (SHAPING)
    // ========================
    /// <summary>
    /// Scale for unrealized PnL shaping. Provides dense signal during trades.
    /// </summary>
    public float UnrealizedPnlScale { get; set; } = 10f;  // Normalized from 50
    
    /// <summary>
    /// Scale for PnL delta (improvement since last step). Encourages progress.
    /// </summary>
    public float PnlDeltaScale { get; set; } = 5f;  // Normalized from 20
    
    /// <summary>
    /// Penalty for drawdown from peak unrealized PnL.
    /// </summary>
    public float DrawdownPenalty { get; set; } = 0.0001f;  // Reduced further - don't punish normal volatility
    
    /// <summary>
    /// Threshold before drawdown penalty kicks in.
    /// </summary>
    public float DrawdownThreshold { get; set; } = 0.15f;  // Increased to 5% - allow more room to breathe

    // ========================
    // POSITION MANAGEMENT
    // ========================
    /// <summary>
    /// Per-tick penalty for holding a position. Discourages overly long trades.
    /// </summary>
    public float HoldingTimePenalty { get; set; } = 0f;  // Disabled - let agent explore freely
    
    /// <summary>
    /// Per-tick bonus for holding a profitable position. Encourages exploring longer holds.
    /// </summary>
    public float HoldingBonus { get; set; } = 0.01f;  // Increased 5x - stronger incentive for holding profitable positions
    
    /// <summary>
    /// Minimum ticks threshold for early close penalty (soft guidance, not enforcement).
    /// </summary>
    public int MinHoldingTicks { get; set; } = 300;  // 5 minutes - soft threshold only
    
    /// <summary>
    /// Penalty for closing too early. Set to 0 to let agent explore freely.
    /// </summary>
    public float EarlyClosePenalty { get; set; } = 0f;  // Disabled - let agent discover holding value naturally
    
    /// <summary>
    /// Maximum holding ticks before increasing penalty.
    /// </summary>
    public int MaxHoldingTicks { get; set; } = 43200;  // 12 hours - very long to allow exploration
    
    /// <summary>
    /// Ticks for quick profit bonus eligibility.
    /// </summary>
    public int QuickProfitTicks { get; set; } = 600;  // 10 minutes at S1 granularity
    
    /// <summary>
    /// Bonus for quick profitable trades.
    /// </summary>
    public float QuickProfitBonus { get; set; } = 0.0001f;  // Disabled - was encouraging early closes

    // ========================
    // OPPORTUNITY COST MODEL
    // ========================
    /// <summary>
    /// Enable opportunity cost model instead of flat penalty.
    /// Only penalizes when agent misses significant price moves.
    /// </summary>
    public bool UseOpportunityCostModel { get; set; } = true;
    
    /// <summary>
    /// Penalty when agent is flat but market moved significantly.
    /// </summary>
    public float MissedOpportunityPenalty { get; set; } = 0.9f;
    
    /// <summary>
    /// ATR multiplier threshold: price move > this * ATR = missed opportunity.
    /// </summary>
    public float OpportunityThresholdAtr { get; set; } = 0.5f;
    
    /// <summary>
    /// Legacy flat penalty (used when UseOpportunityCostModel is false).
    /// </summary>
    public float FlatPenalty { get; set; } = 0.002f;

    // ========================
    // DIRECTION QUALITY
    // ========================
    /// <summary>
    /// Scale for rewarding correct direction (price moves in position direction).
    /// </summary>
    public float PositionQualityScale { get; set; } = 5f;  // Normalized from 100
    
    /// <summary>
    /// Scale for equity momentum (equity increasing).
    /// </summary>
    public float EquityMomentumScale { get; set; } = 2f;  // Normalized from 15
    
    /// <summary>
    /// Scale for equity change percentage.
    /// </summary>
    public float EquityChangeScale { get; set; } = 5f;  // Normalized from 50

    // ========================
    // EPISODE-LEVEL METRICS
    // ========================
    /// <summary>
    /// Weight for profit factor bonus/penalty.
    /// </summary>
    public float ProfitFactorWeight { get; set; } = 2f;  // Normalized from 10
    
    /// <summary>
    /// Weight for Sharpe ratio bonus/penalty.
    /// </summary>
    public float SharpeRatioWeight { get; set; } = 2f;  // Normalized from 10
    
    /// <summary>
    /// Minimum trades before PF is calculated.
    /// </summary>
    public int MinTradesForPf { get; set; } = 3;  // Increased from 2
    
    /// <summary>
    /// Minimum trades before Sharpe is calculated.
    /// </summary>
    public int MinTradesForSharpe { get; set; } = 5;  // Increased from 3
    
    /// <summary>
    /// Cap for profit factor (prevents outlier rewards).
    /// </summary>
    public float PfCap { get; set; } = 50f;  // Reduced from 10
    
    /// <summary>
    /// Cap for Sharpe ratio.
    /// </summary>
    public float SharpeCap { get; set; } = 30f;  // Reduced from 5
    
    /// <summary>
    /// Minimum PF threshold for positive reward.
    /// </summary>
    public float MinPfThreshold { get; set; } = 1.5f;  // Reduced from 2
    
    /// <summary>
    /// Annualization factor for Sharpe (252 trading days).
    /// </summary>
    public float AnnualizationFactor { get; set; } = 252f;

    // ========================
    // MAX DRAWDOWN PENALTY
    // ========================
    /// <summary>
    /// Weight for max drawdown penalty.
    /// </summary>
    public float MddPenaltyWeight { get; set; } = 5f;  // Normalized from 25
    
    /// <summary>
    /// Threshold before MDD penalty kicks in (percentage).
    /// </summary>
    public float MddThreshold { get; set; } = 5f;  // Reduced from 10
    
    /// <summary>
    /// Scale multiplier for MDD penalty.
    /// </summary>
    public float MddPenaltyScale { get; set; } = 1.5f;  // Reduced from 2

    // ========================
    // NORMALIZATION
    // ========================
    /// <summary>
    /// Scale for reward normalization. Higher = less clipping.
    /// </summary>
    public float RewardNormalizationScale { get; set; } = 10f;  // Reduced to match normalized reward magnitudes
    
    /// <summary>
    /// If true, normalize rewards to bounded range using tanh (smooth).
    /// If false, use clamp (hard cutoff).
    /// </summary>
    public bool UseTanhNormalization { get; set; } = false;  // Use clamp for clearer signal
    
    /// <summary>
    /// Enable/disable reward normalization.
    /// </summary>
    public bool NormalizeRewards { get; set; } = true;
}

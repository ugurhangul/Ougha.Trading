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
    public float RealizedProfitScale { get; set; } = 50f;  // Reduced from 200
    
    /// <summary>
    /// Bonus for winning trades (profit > 0).
    /// </summary>
    public float WinBonus { get; set; } = 3f;  // Reduced from 10
    
    /// <summary>
    /// Penalty for losing trades. Symmetric with WinBonus for balanced risk.
    /// </summary>
    public float LossPenalty { get; set; } = 4.5f;  // 1.5x asymmetry for capital preservation
    
    /// <summary>
    /// Risk-adjusted reward: profit / SL distance. Rewards good R:R trades.
    /// </summary>
    public float RiskRewardScale { get; set; } = 10f;

    // ========================
    // UNREALIZED PNL (SHAPING)
    // ========================
    /// <summary>
    /// Scale for unrealized PnL shaping. Provides dense signal during trades.
    /// </summary>
    public float UnrealizedPnlScale { get; set; } = 50f;  // Increased for denser learning signal
    
    /// <summary>
    /// Scale for PnL delta (improvement since last step). Encourages progress.
    /// </summary>
    public float PnlDeltaScale { get; set; } = 20f;
    
    /// <summary>
    /// Penalty for drawdown from peak unrealized PnL.
    /// </summary>
    public float DrawdownPenalty { get; set; } = 0.003f;  // Reduced from 0.005
    
    /// <summary>
    /// Threshold before drawdown penalty kicks in.
    /// </summary>
    public float DrawdownThreshold { get; set; } = 0.02f;  // Increased from 0.01

    // ========================
    // POSITION MANAGEMENT
    // ========================
    /// <summary>
    /// Per-tick penalty for holding a position. Discourages overly long trades.
    /// </summary>
    public float HoldingTimePenalty { get; set; } = 0.00005f;  // Reduced from 0.0001
    
    /// <summary>
    /// Minimum ticks before a position can be closed without penalty.
    /// </summary>
    public int MinHoldingTicks { get; set; } = 120;  // 2 minutes at S1 granularity
    
    /// <summary>
    /// Penalty for closing too early (before MinHoldingTicks).
    /// </summary>
    public float EarlyClosePenalty { get; set; } = 5f;  // Reduced from 10
    
    /// <summary>
    /// Maximum holding ticks before increasing penalty.
    /// </summary>
    public int MaxHoldingTicks { get; set; } = 14400;  // 4 hours at S1 granularity
    
    /// <summary>
    /// Ticks for quick profit bonus eligibility.
    /// </summary>
    public int QuickProfitTicks { get; set; } = 600;  // 10 minutes at S1 granularity
    
    /// <summary>
    /// Bonus for quick profitable trades.
    /// </summary>
    public float QuickProfitBonus { get; set; } = 5f;  // Reduced from 20

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
    public float MissedOpportunityPenalty { get; set; } = 0.5f;
    
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
    public float PositionQualityScale { get; set; } = 10f;  // Reduced from 20
    
    /// <summary>
    /// Scale for equity momentum (equity increasing).
    /// </summary>
    public float EquityMomentumScale { get; set; } = 15f;  // Reduced from 30
    
    /// <summary>
    /// Scale for equity change percentage.
    /// </summary>
    public float EquityChangeScale { get; set; } = 50f;  // Reduced from 100

    // ========================
    // EPISODE-LEVEL METRICS
    // ========================
    /// <summary>
    /// Weight for profit factor bonus/penalty.
    /// </summary>
    public float ProfitFactorWeight { get; set; } = 20f;  // Reduced from 50
    
    /// <summary>
    /// Weight for Sharpe ratio bonus/penalty.
    /// </summary>
    public float SharpeRatioWeight { get; set; } = 20f;  // Reduced from 50
    
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
    public float PfCap { get; set; } = 5f;  // Reduced from 10
    
    /// <summary>
    /// Cap for Sharpe ratio.
    /// </summary>
    public float SharpeCap { get; set; } = 3f;  // Reduced from 5
    
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
    public float MddPenaltyWeight { get; set; } = 25f;  // Reduced from 50
    
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
    public float RewardNormalizationScale { get; set; } = 20f;  // Reduced from 50
    
    /// <summary>
    /// If true, normalize rewards to bounded range using tanh (smooth).
    /// If false, use clamp (hard cutoff).
    /// </summary>
    public bool UseTanhNormalization { get; set; } = true;  // NEW: smooth normalization
    
    /// <summary>
    /// Enable/disable reward normalization.
    /// </summary>
    public bool NormalizeRewards { get; set; } = true;
}

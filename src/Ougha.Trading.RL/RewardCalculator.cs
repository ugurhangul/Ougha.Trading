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
    public float LossPenalty { get; set; } = 3f;  // Now symmetric
    
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
    public float UnrealizedPnlScale { get; set; } = 15f;  // Reduced from 50
    
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
    public int MinHoldingTicks { get; set; } = 30;  // Reduced from 50
    
    /// <summary>
    /// Penalty for closing too early (before MinHoldingTicks).
    /// </summary>
    public float EarlyClosePenalty { get; set; } = 5f;  // Reduced from 10
    
    /// <summary>
    /// Maximum holding ticks before increasing penalty.
    /// </summary>
    public int MaxHoldingTicks { get; set; } = 1000;
    
    /// <summary>
    /// Ticks for quick profit bonus eligibility.
    /// </summary>
    public int QuickProfitTicks { get; set; } = 150;  // Reduced from 200
    
    /// <summary>
    /// Bonus for quick profitable trades.
    /// </summary>
    public float QuickProfitBonus { get; set; } = 5f;  // Reduced from 20

    // ========================
    // FLAT POSITION INCENTIVE
    // ========================
    /// <summary>
    /// Penalty for being flat (no position). Encourages trading activity.
    /// Set to 0 to disable.
    /// </summary>
    public float FlatPenalty { get; set; } = 0.0005f;  // NEW: tiny incentive to trade

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

/// <summary>
/// Tracks episode-level metrics for PF and Sharpe calculation.
/// </summary>
public class EpisodeMetrics
{
    private double _grossProfit;
    private double _grossLoss;
    private int _tradeCount;
    private readonly List<double> _returns = [];

    public int TradeCount => _tradeCount;

    public void UpdateTrade(double tradeProfit)
    {
        _tradeCount++;
        if (tradeProfit > 0)
            _grossProfit += tradeProfit;
        else
            _grossLoss += Math.Abs(tradeProfit);
        
        _returns.Add(tradeProfit);
    }

    public double CalculateProfitFactor()
    {
        if (_grossLoss <= 0) return _grossProfit > 0 ? 10.0 : 1.0;
        return _grossProfit / _grossLoss;
    }

    public double CalculateSharpeRatio(float annualizationFactor = 252f)
    {
        if (_returns.Count < 2) return 0;
        
        var mean = _returns.Average();
        var variance = _returns.Sum(r => Math.Pow(r - mean, 2)) / (_returns.Count - 1);
        var stdDev = Math.Sqrt(variance);
        
        if (stdDev <= 0) return mean > 0 ? 5.0 : 0;
        
        return (mean / stdDev) * Math.Sqrt(annualizationFactor);
    }

    public void Reset()
    {
        _grossProfit = 0;
        _grossLoss = 0;
        _tradeCount = 0;
        _returns.Clear();
    }
}

/// <summary>
/// Improved reward calculator with volatility normalization and balanced scaling.
/// Designed for multi-symbol RL trading with stable learning signals.
/// </summary>
public class RewardCalculator
{
    private readonly RewardConfig _config;
    private readonly EpisodeMetrics _episodeMetrics;

    public RewardCalculator(RewardConfig? config = null)
    {
        _config = config ?? new RewardConfig();
        _episodeMetrics = new EpisodeMetrics();
    }

    public EpisodeMetrics EpisodeMetrics => _episodeMetrics;

    /// <summary>
    /// Reset episode metrics for a new episode.
    /// </summary>
    public void ResetEpisode()
    {
        _episodeMetrics.Reset();
        _previousUnrealizedPnl = 0;
        _previousEquity = 0;
    }

    /// <summary>
    /// Calculate PF and Sharpe based reward bonus.
    /// </summary>
    private float CalculatePfSharpeReward()
    {
        var reward = 0f;

        if (_episodeMetrics.TradeCount >= _config.MinTradesForPf)
        {
            var pf = _episodeMetrics.CalculateProfitFactor();
            var pfCapped = Math.Min(pf, _config.PfCap);

            if (pfCapped >= _config.MinPfThreshold)
            {
                // Progressive reward: more PF = more reward
                reward += (float)(pfCapped - 1.0) * _config.ProfitFactorWeight;
            }
            else if (pfCapped < 1.0)
            {
                // Below breakeven gets penalty
                reward -= (float)(1.0 - pfCapped) * _config.ProfitFactorWeight;
            }
        }

        if (_episodeMetrics.TradeCount >= _config.MinTradesForSharpe)
        {
            var sharpe = _episodeMetrics.CalculateSharpeRatio(_config.AnnualizationFactor);
            var sharpeCapped = Math.Clamp(sharpe, -_config.SharpeCap, _config.SharpeCap);
            
            // Linear reward/penalty based on Sharpe
            reward += (float)sharpeCapped * _config.SharpeRatioWeight;
        }

        return reward;
    }

    /// <summary>
    /// Calculate MDD penalty with progressive scaling.
    /// </summary>
    private float CalculateMddPenalty(double mddPercent)
    {
        if (mddPercent <= _config.MddThreshold)
            return 0f;

        // Progressive penalty: exponential beyond threshold
        var excessMdd = mddPercent - _config.MddThreshold;
        var penalty = (float)(excessMdd * excessMdd * _config.MddPenaltyWeight / 100.0 * _config.MddPenaltyScale);
        return penalty;
    }

    private double _previousUnrealizedPnl;
    private double _previousEquity;
    
    /// <summary>
    /// Calculate reward with optional ATR-based volatility normalization.
    /// </summary>
    /// <param name="tradeClosed">Whether a trade was closed this tick</param>
    /// <param name="tradeProfit">Profit from closed trade (absolute)</param>
    /// <param name="holdingTicks">How long the position was held</param>
    /// <param name="hasPosition">Whether currently holding a position</param>
    /// <param name="unrealizedPnl">Current unrealized PnL percentage</param>
    /// <param name="peakUnrealizedPnl">Peak unrealized PnL this trade</param>
    /// <param name="initialBalance">Starting balance for episode</param>
    /// <param name="maxDrawdownPct">Current max drawdown percentage</param>
    /// <param name="currentEquity">Current equity</param>
    /// <param name="positionDirection">1=long, -1=short, 0=flat</param>
    /// <param name="priceChange">Price change since last tick</param>
    /// <param name="symbolAtr">ATR for volatility normalization (optional)</param>
    /// <param name="slDistance">Stop loss distance for R:R calculation (optional)</param>
    public float Calculate(
        bool tradeClosed,
        double tradeProfit,
        int holdingTicks,
        bool hasPosition,
        double unrealizedPnl,
        double peakUnrealizedPnl,
        double initialBalance,
        double maxDrawdownPct,
        double currentEquity = 0, 
        int positionDirection = 0, 
        double priceChange = 0,
        double symbolAtr = 0,
        double slDistance = 0)
    {
        var reward = 0f;

        if (tradeClosed)
        {
            _episodeMetrics.UpdateTrade(tradeProfit);
        }

        var mddPenalty = CalculateMddPenalty(maxDrawdownPct);

        if (tradeClosed)
        {
            reward += CalculateClosedTradeReward(tradeProfit, holdingTicks, initialBalance, symbolAtr, slDistance);
            reward += CalculatePfSharpeReward();
            reward -= mddPenalty;
        }
        else if (hasPosition)
        {
            reward += CalculateOpenPositionReward(
                unrealizedPnl, peakUnrealizedPnl, initialBalance, 
                currentEquity, positionDirection, priceChange, holdingTicks);
            reward -= mddPenalty;
        }
        else
        {
            // Flat (no position) - small penalty to encourage trading
            reward -= _config.FlatPenalty;
            reward -= mddPenalty * 0.5f;
            _previousUnrealizedPnl = 0;
        }

        // Normalize reward for stable learning
        if (_config.NormalizeRewards)
        {
            reward = NormalizeReward(reward);
        }

        return reward;
    }

    /// <summary>
    /// Calculate reward component for a closed trade.
    /// </summary>
    private float CalculateClosedTradeReward(
        double tradeProfit, 
        int holdingTicks, 
        double initialBalance,
        double symbolAtr,
        double slDistance)
    {
        var reward = 0f;
        
        // Early close penalty
        if (holdingTicks < _config.MinHoldingTicks)
        {
            var earlyFactor = 1.0f - ((float)holdingTicks / _config.MinHoldingTicks);
            reward -= _config.EarlyClosePenalty * earlyFactor;
        }

        // Profit-based reward
        var profitPct = (float)(tradeProfit / initialBalance) * 100f;
        
        if (_config.UseVolatilityNormalization && symbolAtr > 0)
        {
            // Normalize by ATR: a 1-ATR move is "1 unit" regardless of symbol
            var atrNormalizedProfit = (float)(tradeProfit / (symbolAtr * initialBalance));
            reward += atrNormalizedProfit * _config.RealizedProfitScale;
        }
        else
        {
            // Fallback to percentage-based
            reward += profitPct * _config.RealizedProfitScale;
        }

        // Risk-adjusted reward (R-multiple)
        if (slDistance > 0 && tradeProfit != 0)
        {
            var rMultiple = (float)(tradeProfit / (slDistance * initialBalance));
            reward += Math.Clamp(rMultiple, -5f, 5f) * _config.RiskRewardScale;
        }

        // Win/loss bonus (symmetric)
        if (tradeProfit > 0)
        {
            reward += _config.WinBonus;
            
            // Quick profit bonus
            if (holdingTicks >= _config.MinHoldingTicks && holdingTicks < _config.QuickProfitTicks)
            {
                var quickFactor = 1.0f - ((float)(holdingTicks - _config.MinHoldingTicks) / 
                    (_config.QuickProfitTicks - _config.MinHoldingTicks));
                reward += _config.QuickProfitBonus * quickFactor;
            }
        }
        else
        {
            reward -= _config.LossPenalty;
        }

        return reward;
    }

    /// <summary>
    /// Calculate reward component for an open position (shaping signal).
    /// </summary>
    private float CalculateOpenPositionReward(
        double unrealizedPnl,
        double peakUnrealizedPnl,
        double initialBalance,
        double currentEquity,
        int positionDirection,
        double priceChange,
        int holdingTicks)
    {
        var reward = 0f;
        
        // Small holding time penalty (prevents infinitely holding)
        reward -= _config.HoldingTimePenalty;
        
        // Progressive holding penalty for very long trades
        if (holdingTicks > _config.MaxHoldingTicks)
        {
            var excessTicks = holdingTicks - _config.MaxHoldingTicks;
            reward -= _config.HoldingTimePenalty * (excessTicks / 100f);
        }

        // Unrealized PnL shaping (scaled down to avoid overshadowing closed trades)
        var unrealizedReward = (float)(unrealizedPnl / initialBalance) * 100f * _config.UnrealizedPnlScale;
        reward += unrealizedReward * 0.01f;

        // Direction quality: reward when price moves in position direction
        if (positionDirection != 0 && priceChange != 0)
        {
            var directionReward = (float)(priceChange * positionDirection) * _config.PositionQualityScale;
            reward += Math.Clamp(directionReward, -0.1f, 0.1f);  // Bounded
        }

        // Equity momentum: reward when equity is increasing
        if (currentEquity > 0 && _previousEquity > 0)
        {
            var equityChange = (currentEquity - _previousEquity) / initialBalance * 100;
            reward += (float)Math.Clamp(equityChange * _config.EquityMomentumScale, -0.5f, 0.5f);
        }

        // PnL delta: reward for improvement since last step
        var pnlDelta = unrealizedPnl - _previousUnrealizedPnl;
        if (pnlDelta > 0)
        {
            reward += (float)(pnlDelta / initialBalance) * _config.PnlDeltaScale;
        }
        else if (pnlDelta < 0)
        {
            reward += (float)(pnlDelta / initialBalance) * _config.PnlDeltaScale * 0.5f;  // Asymmetric
        }

        // Drawdown from peak penalty
        var dd = Math.Max(0, peakUnrealizedPnl - unrealizedPnl);
        if (dd > _config.DrawdownThreshold)
        {
            reward -= _config.DrawdownPenalty * (float)(dd / _config.DrawdownThreshold);
        }

        _previousUnrealizedPnl = unrealizedPnl;
        _previousEquity = currentEquity > 0 ? currentEquity : _previousEquity;
        
        return reward;
    }

    /// <summary>
    /// Normalize reward to bounded range for stable learning.
    /// </summary>
    private float NormalizeReward(float reward)
    {
        var scaled = reward / _config.RewardNormalizationScale;
        
        if (_config.UseTanhNormalization)
        {
            // Smooth bounded output using tanh: (-1, 1)
            return (float)Math.Tanh(scaled);
        }
        else
        {
            // Hard clamp: [-1, 1]
            return Math.Clamp(scaled, -1f, 1f);
        }
    }
}

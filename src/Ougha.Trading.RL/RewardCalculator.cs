namespace Ougha.Trading.RL;

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

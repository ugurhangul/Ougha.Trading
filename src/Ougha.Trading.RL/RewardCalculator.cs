using Ougha.Trading.Core.Models;

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
        _flatStartPrice = 0;
        _flatTicks = 0;
        _flatPreviousPrice = 0;
        _flatDirectionTicks = 0;
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
    private double _flatStartPrice;
    private int _flatTicks;
    
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
    /// <param name="symbolInfo">Symbol info for point-based normalization (optional)</param>
    /// <param name="currentPrice">Current symbol price for opportunity cost calculation</param>
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
        double slDistance = 0,
        SymbolInfo? symbolInfo = null,
        double currentPrice = 0)
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
                currentEquity, positionDirection, priceChange, holdingTicks, symbolAtr);
            reward -= mddPenalty;
        }
        else
        {
            // Opportunity cost model: only penalize if good setup was missed
            if (_config.UseOpportunityCostModel && symbolAtr > 0)
            {
                var symbolPoint = symbolInfo?.Point ?? 0.00001;
                // Use actual price, not equity, for opportunity cost tracking
                reward -= CalculateOpportunityCostPenalty(currentPrice > 0 ? currentPrice : 1.0, symbolAtr, symbolPoint);
            }
            else
            {
                reward -= _config.FlatPenalty;
            }
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
    /// Designed to let agent EXPLORE holding - longer profitable holds = more cumulative reward.
    /// Price changes are normalized by ATR for fair multi-symbol comparison.
    /// </summary>
    private float CalculateOpenPositionReward(
        double unrealizedPnl,
        double peakUnrealizedPnl,
        double initialBalance,
        double currentEquity,
        int positionDirection,
        double priceChange,
        int holdingTicks,
        double symbolAtr)
    {
        var reward = 0f;
        
        // No holding penalties - let the agent explore freely
        // Only very extreme holding (>MaxHoldingTicks) gets tiny penalty
        if (holdingTicks > _config.MaxHoldingTicks)
        {
            var excessTicks = holdingTicks - _config.MaxHoldingTicks;
            reward -= 0.00001f * (excessTicks / 1000f);  // Negligible penalty
        }
        
        // EXPLORATION INCENTIVE: Decaying bonus for holding profitable positions
        // Decays over time to encourage eventual profit-taking rather than indefinite holding
        // NOTE: unrealizedPnl is already a decimal percentage (0.01 = 1%)
        if (unrealizedPnl > 0)
        {
            // Exponential decay: bonus diminishes over ~60 minutes (slowed for day trading)
            var holdingMinutes = holdingTicks / 60f;
            var decayFactor = (float)Math.Exp(-holdingMinutes / 60f);  // Slower decay for day trading
            reward += _config.HoldingBonus * decayFactor;
            
            // Removed additional profit accumulation - already covered by unrealized PnL shaping
        }

        // Unrealized PnL shaping - strong signal that being in profit is good
        // NOTE: unrealizedPnl is already a decimal percentage (0.01 = 1%)
        var unrealizedReward = (float)unrealizedPnl * 100f * _config.UnrealizedPnlScale;
        reward += unrealizedReward * 0.25f;  // Increased to 25% for stronger hold incentive

        // Direction quality: reward when price moves in position direction
        // Normalize by ATR so EURUSD micro-moves are equivalent to BTCUSD larger moves
        if (positionDirection != 0 && priceChange != 0)
        {
            var effectiveAtr = symbolAtr > 0 ? symbolAtr : _config.DefaultAtr;
            var normalizedPriceChange = priceChange / effectiveAtr;  // Now in ATR units
            var directionReward = (float)(normalizedPriceChange * positionDirection) * _config.PositionQualityScale;
            reward += Math.Clamp(directionReward, -0.1f, 0.1f);  // Bounded
        }

        // Equity momentum: reward when equity is increasing
        if (currentEquity > 0 && _previousEquity > 0)
        {
            var equityChange = (currentEquity - _previousEquity) / initialBalance * 100;
            reward += (float)Math.Clamp(equityChange * _config.EquityMomentumScale, -0.5f, 0.5f);
        }

        // PnL delta: reward for improvement since last step
        // pnlDelta is already a percentage from UnrealizedPnlPercent
        var pnlDelta = unrealizedPnl - _previousUnrealizedPnl;
        if (pnlDelta > 0)
        {
            reward += (float)(pnlDelta * 100f) * _config.PnlDeltaScale;
        }
        else if (pnlDelta < 0)
        {
            reward += (float)(pnlDelta * 100f) * _config.PnlDeltaScale * 0.5f;  // Asymmetric
        }

        // Drawdown from peak penalty
        // NOTE: peakUnrealizedPnl and unrealizedPnl are decimal percentages
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
    private double _flatPreviousPrice;  // Track direction consistency
    private int _flatDirectionTicks;     // Ticks moving in same direction
    
    /// <summary>
    /// Calculate opportunity cost penalty based on missed price movement.
    /// Only penalizes when the agent missed a SUSTAINED move (momentum), not just spikes.
    /// Uses ATR for volatility normalization and requires directional consistency.
    /// </summary>
    private float CalculateOpportunityCostPenalty(double currentPrice, double atr, double symbolPoint)
    {
        if (_flatStartPrice <= 0)
        {
            _flatStartPrice = currentPrice;
            _flatPreviousPrice = currentPrice;
            _flatTicks = 0;
            _flatDirectionTicks = 0;
            return 0;
        }
        
        _flatTicks++;
        var priceMove = Math.Abs(currentPrice - _flatStartPrice);
        
        // Track directional consistency (momentum requirement)
        var tickDirection = Math.Sign(currentPrice - _flatPreviousPrice);
        var overallDirection = Math.Sign(currentPrice - _flatStartPrice);
        
        if (tickDirection == overallDirection && tickDirection != 0)
        {
            _flatDirectionTicks++;
        }
        else if (tickDirection == -overallDirection)
        {
            _flatDirectionTicks = Math.Max(0, _flatDirectionTicks - 2);  // Faster reset on reversal
        }
        
        _flatPreviousPrice = currentPrice;
        
        // Normalize price move by ATR (volatility-relative)
        var atrThreshold = atr * _config.OpportunityThresholdAtr;
        
        // Only penalize if move was SUSTAINED (not just a spike)
        // Require at least 10 ticks of consistent direction movement
        var hasmomentum = _flatDirectionTicks >= 10;
        
        if (priceMove > atrThreshold && hasmomentum)
        {
            // Use ATR-normalized missed opportunity (independent of symbol price scale)
            var missedAtrMultiple = (float)(priceMove / atr);
            var penalty = _config.MissedOpportunityPenalty * Math.Min(missedAtrMultiple, 3f);
            
            // Reset tracking after penalty applied
            _flatStartPrice = currentPrice;
            _flatDirectionTicks = 0;
            return penalty;
        }
        
        // Very small time decay after 120 ticks of inactivity (increased from 60)
        if (_flatTicks > 120)
        {
            return 0.00005f * (_flatTicks - 120);  // Even smaller penalty
        }
        
        return 0;
    }
}

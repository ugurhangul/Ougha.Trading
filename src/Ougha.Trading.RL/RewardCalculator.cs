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
    
    // Per-symbol state tracking (CRITICAL: must not share state across symbols)
    private readonly Dictionary<string, double> _previousUnrealizedPnlBySymbol = new();
    private readonly Dictionary<string, double> _previousEquityBySymbol = new();
    private readonly Dictionary<string, double> _flatStartPriceBySymbol = new();
    private readonly Dictionary<string, int> _flatTicksBySymbol = new();
    private readonly Dictionary<string, double> _flatPreviousPriceBySymbol = new();
    private readonly Dictionary<string, int> _flatDirectionTicksBySymbol = new();

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
        _previousUnrealizedPnlBySymbol.Clear();
        _previousEquityBySymbol.Clear();
        _flatStartPriceBySymbol.Clear();
        _flatTicksBySymbol.Clear();
        _flatPreviousPriceBySymbol.Clear();
        _flatDirectionTicksBySymbol.Clear();
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

    // NOTE: Per-symbol state is now tracked in dictionaries above
    
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
        string symbol,  // CRITICAL: Must be passed for per-symbol state tracking
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
        double currentPrice = 0,
        double volume = 0)  // Position volume for R-multiple calculation
    {
        var reward = 0f;

        if (tradeClosed)
        {
            _episodeMetrics.UpdateTrade(tradeProfit);
        }

        var mddPenalty = CalculateMddPenalty(maxDrawdownPct);

        if (tradeClosed)
        {
            reward += CalculateClosedTradeReward(tradeProfit, holdingTicks, slDistance, symbolInfo, volume);
            reward += CalculatePfSharpeReward();
            reward -= mddPenalty;
        }
        else if (hasPosition)
        {
            reward += CalculateOpenPositionReward(
                symbol, unrealizedPnl, peakUnrealizedPnl, initialBalance, 
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
                reward -= CalculateOpportunityCostPenalty(symbol, currentPrice > 0 ? currentPrice : 1.0, symbolAtr, symbolPoint);
            }
            else
            {
                reward -= _config.FlatPenalty;
            }
            // Removed MDD penalty for flat periods - don't penalize agent for past losses when not trading
            _previousUnrealizedPnlBySymbol[symbol] = 0;
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
        double slDistance,
        SymbolInfo? symbolInfo,
        double volume)
    {
        var reward = 0f;
        
        // Early close penalty
        if (holdingTicks < _config.MinHoldingTicks)
        {
            var earlyFactor = 1.0f - ((float)holdingTicks / _config.MinHoldingTicks);
            reward -= _config.EarlyClosePenalty * earlyFactor;
        }

        // Profit-based reward using R-multiple (symbol-agnostic)
        // R-multiple = profit / risk taken - same 2R win means same reward for any symbol
        if (slDistance > 0 && symbolInfo != null && volume > 0)
        {
            // Calculate ACTUAL risk using symbol-specific values
            // Risk = (SL distance in points) * TickValue * Volume
            var slPoints = slDistance / symbolInfo.Point;
            var riskInDollars = slPoints * symbolInfo.TickValue * volume;
            
            if (riskInDollars > 0)
            {
                var rMultiple = (float)(tradeProfit / riskInDollars);
                // Primary reward signal - clamped to prevent extreme values
                reward += Math.Clamp(rMultiple, -3f, 5f) * _config.RiskRewardScale;
            }
        }
        else
        {
            // Fallback: use profit directly with modest scaling
            // This ensures we still get signal even without SL data
            reward += Math.Clamp((float)tradeProfit * 0.1f, -2f, 3f) * _config.RiskRewardScale;
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
        string symbol,
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
            // Exponential decay: bonus diminishes over ~240 minutes (extended for day trading)
            var holdingMinutes = holdingTicks / 60f;
            var decayFactor = (float)Math.Exp(-holdingMinutes / 240f);  // Extended decay for day trading
            reward += _config.HoldingBonus * decayFactor;
            
            // Removed additional profit accumulation - already covered by unrealized PnL shaping
        }

        // Unrealized PnL shaping - strong signal that being in profit is good
        // NOTE: unrealizedPnl is already a decimal percentage (0.01 = 1%)
        var unrealizedReward = (float)unrealizedPnl * 100f * _config.UnrealizedPnlScale;
        reward += unrealizedReward * 0.1f;  // Reduced from 0.25 to prevent reward hacking

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
        var previousEquity = _previousEquityBySymbol.GetValueOrDefault(symbol, 0);
        if (currentEquity > 0 && previousEquity > 0)
        {
            var equityChange = (currentEquity - previousEquity) / initialBalance * 100;
            reward += (float)Math.Clamp(equityChange * _config.EquityMomentumScale, -0.5f, 0.5f);
        }

        // PnL delta: reward for improvement since last step
        var previousUnrealizedPnl = _previousUnrealizedPnlBySymbol.GetValueOrDefault(symbol, 0);
        var pnlDelta = unrealizedPnl - previousUnrealizedPnl;
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

        _previousUnrealizedPnlBySymbol[symbol] = unrealizedPnl;
        _previousEquityBySymbol[symbol] = currentEquity > 0 ? currentEquity : previousEquity;
        
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
    // NOTE: Per-symbol flat tracking is now in dictionaries above
    
    /// <summary>
    /// Calculate opportunity cost penalty based on missed price movement.
    /// Only penalizes when the agent missed a SUSTAINED move (momentum), not just spikes.
    /// Uses ATR for volatility normalization and requires directional consistency.
    /// </summary>
    private float CalculateOpportunityCostPenalty(string symbol, double currentPrice, double atr, double symbolPoint)
    {
        var flatStartPrice = _flatStartPriceBySymbol.GetValueOrDefault(symbol, 0);
        var flatPreviousPrice = _flatPreviousPriceBySymbol.GetValueOrDefault(symbol, 0);
        var flatTicks = _flatTicksBySymbol.GetValueOrDefault(symbol, 0);
        var flatDirectionTicks = _flatDirectionTicksBySymbol.GetValueOrDefault(symbol, 0);
        
        if (flatStartPrice <= 0)
        {
            _flatStartPriceBySymbol[symbol] = currentPrice;
            _flatPreviousPriceBySymbol[symbol] = currentPrice;
            _flatTicksBySymbol[symbol] = 0;
            _flatDirectionTicksBySymbol[symbol] = 0;
            return 0;
        }
        
        flatTicks++;
        _flatTicksBySymbol[symbol] = flatTicks;
        var priceMove = Math.Abs(currentPrice - flatStartPrice);
        
        // Track directional consistency (momentum requirement)
        var tickDirection = Math.Sign(currentPrice - flatPreviousPrice);
        var overallDirection = Math.Sign(currentPrice - flatStartPrice);
        
        if (tickDirection == overallDirection && tickDirection != 0)
        {
            flatDirectionTicks++;
        }
        else if (tickDirection == -overallDirection)
        {
            flatDirectionTicks = Math.Max(0, flatDirectionTicks - 2);  // Faster reset on reversal
        }
        _flatDirectionTicksBySymbol[symbol] = flatDirectionTicks;
        
        _flatPreviousPriceBySymbol[symbol] = currentPrice;
        
        // Normalize price move by ATR (volatility-relative)
        var atrThreshold = atr * _config.OpportunityThresholdAtr;
        
        // Only penalize if move was SUSTAINED (not just a spike)
        // Require at least 10 ticks of consistent direction movement
        var hasmomentum = flatDirectionTicks >= 10;
        
        if (priceMove > atrThreshold && hasmomentum)
        {
            // Use ATR-normalized missed opportunity (independent of symbol price scale)
            var missedAtrMultiple = (float)(priceMove / atr);
            var penalty = _config.MissedOpportunityPenalty * Math.Min(missedAtrMultiple, 3f);
            
            // Reset tracking after penalty applied
            _flatStartPriceBySymbol[symbol] = currentPrice;
            _flatDirectionTicksBySymbol[symbol] = 0;
            return penalty;
        }
        
        // Very small time decay after 120 ticks of inactivity (increased from 60)
        if (flatTicks > 120)
        {
            return 0.00005f * (flatTicks - 120);  // Even smaller penalty
        }
        
        return 0;
    }
}

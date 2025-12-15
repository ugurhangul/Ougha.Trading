using Ougha.Trading.Core.Models;

namespace Ougha.Trading.RL;

/// <summary>
/// Simplified reward calculator for price prediction strategy.
/// Components: R-multiple (primary), MDD penalty (risk), prediction accuracy (optional).
/// </summary>
public class RewardCalculator
{
    private readonly RewardConfig _config;
    private readonly EpisodeMetrics _episodeMetrics;
    
    // Prediction tracking for accuracy reward
    private readonly Dictionary<string, float> _entryPredictionBySymbol = new();
    private readonly Dictionary<string, double> _entryPriceBySymbol = new();

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
        _entryPredictionBySymbol.Clear();
        _entryPriceBySymbol.Clear();
    }

    /// <summary>
    /// Store prediction at trade entry for accuracy calculation on close.
    /// </summary>
    public void RecordEntryPrediction(string symbol, float prediction, double entryPrice)
    {
        _entryPredictionBySymbol[symbol] = prediction;
        _entryPriceBySymbol[symbol] = entryPrice;
    }

    /// <summary>
    /// Reward calculation for price prediction strategy.
    /// Positions only close via TP/SL - no manual close option.
    /// </summary>
    public float Calculate(
        string symbol,
        bool tradeClosed,
        double tradeProfit,
        bool hasPosition,
        double unrealizedPnl,
        double maxDrawdownPct,
        double slDistance = 0,
        SymbolInfo? symbolInfo = null,
        double volume = 0,
        double closePrice = 0,
        CloseReason closeReason = CloseReason.Unknown,
        int holdingTicks = 0)
    {
        var reward = 0f;

        if (tradeClosed)
        {
            _episodeMetrics.UpdateTrade(tradeProfit);
            
            // 1. R-MULTIPLE: Primary reward signal (symbol-agnostic)
            reward += CalculateRMultipleReward(tradeProfit, slDistance, symbolInfo, volume);
            
            // 2. PREDICTION ACCURACY: Reward for correct price direction prediction
            reward += CalculatePredictionAccuracyReward(symbol, closePrice, tradeProfit);
            
            // 3. TP/SL BONUS/PENALTY: Encourage TP hits, discourage SL hits
            if (closeReason == CloseReason.TakeProfit)
            {
                reward += _config.TpHitBonus;  // Bonus for letting winners run to target
            }
            else if (closeReason == CloseReason.StopLoss)
            {
                reward -= _config.SlHitPenalty;  // Small penalty for bad entries
            }
            
            // Clear entry tracking
            _entryPredictionBySymbol.Remove(symbol);
            _entryPriceBySymbol.Remove(symbol);
        }
        else if (hasPosition)
        {
            // Minimal shaping: sign of unrealized PnL
            reward = (float)(Math.Sign(unrealizedPnl) * _config.PositionHoldingSignal);
            
            // CONDITIONAL HOLDING BONUS: Only reward holding PROFITABLE positions
            // This prevents rewarding holding losers
            if (holdingTicks > 0 && unrealizedPnl > 0)
            {
                var holdingMinutes = holdingTicks / 60f;
                var holdingBonus = Math.Min(
                    holdingMinutes * _config.HoldingBonusPerMinute,
                    _config.MaxHoldingBonus
                );
                reward += holdingBonus;
            }
        }
        
        // 4. MDD PENALTY: Risk constraint (applied to all states)
        if (maxDrawdownPct > _config.MddThreshold)
        {
            var excessMdd = maxDrawdownPct - _config.MddThreshold;
            reward -= (float)(excessMdd * excessMdd * _config.MddPenaltyWeight / 100.0);
        }

        // Optional normalization
        if (_config.NormalizeRewards)
        {
            reward = (float)Math.Tanh(reward / _config.RewardNormalizationScale);
        }

        return reward;
    }


    /// <summary>
    /// Calculate R-multiple based reward (primary signal).
    /// R-multiple = profit / risk taken - same 2R win means same reward for any symbol.
    /// </summary>
    private float CalculateRMultipleReward(
        double tradeProfit, 
        double slDistance,
        SymbolInfo? symbolInfo,
        double volume)
    {
        if (slDistance > 0 && symbolInfo != null && volume > 0)
        {
            // Calculate ACTUAL risk using symbol-specific values
            var slPoints = slDistance / symbolInfo.Point;
            var riskInDollars = slPoints * symbolInfo.TickValue * volume;
            
            if (riskInDollars > 0)
            {
                var rMultiple = (float)(tradeProfit / riskInDollars);
                // Clamp to prevent extreme values
                return Math.Clamp(rMultiple, -3f, 5f) * _config.RMultipleScale;
            }
        }
        
        // Fallback: use raw profit with modest scaling
        return Math.Clamp((float)tradeProfit * 0.1f, -2f, 3f) * _config.RMultipleScale;
    }

    /// <summary>
    /// Calculate prediction accuracy reward.
    /// Compares predicted price direction with actual price movement.
    /// </summary>
    private float CalculatePredictionAccuracyReward(string symbol, double closePrice, double tradeProfit)
    {
        if (!_config.UsePredictionAccuracyReward)
            return 0f;
            
        var entryPrice = _entryPriceBySymbol.GetValueOrDefault(symbol, 0);
        var prediction = _entryPredictionBySymbol.GetValueOrDefault(symbol, 0);
        
        if (entryPrice <= 0 || Math.Abs(prediction) < 0.0001f)
            return 0f;
        
        // Actual price move as percentage
        var actualMove = (closePrice - entryPrice) / entryPrice;
        
        // Check if prediction direction matched actual move
        var predictionDirection = Math.Sign(prediction);
        var actualDirection = Math.Sign(actualMove);
        
        if (predictionDirection == actualDirection)
        {
            // Correct direction - reward scales with prediction confidence that matched
            var confidence = Math.Min(Math.Abs(prediction) / 0.01f, 1f);  // Normalize to 0-1
            return _config.PredictionAccuracyBonus * confidence;
        }
        else
        {
            // Wrong direction - small penalty
            return -_config.PredictionWrongPenalty;
        }
    }
}

/// <summary>
/// How the trade was closed.
/// </summary>
public enum CloseReason
{
    Unknown = 0,
    Manual = 1,   // Agent closed via CLOSE action
    TakeProfit = 2,
    StopLoss = 3,
    EndOfEpisode = 4
}

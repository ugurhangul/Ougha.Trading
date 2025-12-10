namespace Ougha.Trading.RL;

public class RewardConfig
{
    // Values from Python code
    public float RealizedProfitScale { get; set; } = 100f;
    public float UnrealizedPnlScale { get; set; } = 10f;
    public float EquityChangeScale { get; set; } = 100f;
    public float HoldingTimePenalty { get; set; } = 0.0001f;
    public int MaxHoldingTicks { get; set; } = 1000;
    public float DrawdownPenalty { get; set; } = 0.01f;
    public float DrawdownThreshold { get; set; } = 0.005f;
    public float FlatPenalty { get; set; } = 0.0001f;
    public float WinBonus { get; set; } = 50f;
    public float LossPenalty { get; set; } = 25f;
    public float QuickProfitBonus { get; set; } = 30f;
    public int QuickProfitTicks { get; set; } = 200;
    
    public int MinHoldingTicks { get; set; } = 50;
    public float EarlyClosePenalty { get; set; } = 100f;
    
    // PF/Sharpe reward shaping (from Python)
    public float ProfitFactorWeight { get; set; } = 50f;
    public float SharpeRatioWeight { get; set; } = 50f;
    public int MinTradesForPf { get; set; } = 2;
    public int MinTradesForSharpe { get; set; } = 3;
    public float PfCap { get; set; } = 10f;
    public float SharpeCap { get; set; } = 5f;
    public float MinPfThreshold { get; set; } = 2f;
    public float AnnualizationFactor { get; set; } = 252f;
    
    // MDD penalty (from Python)
    public float MddPenaltyWeight { get; set; } = 50f;
    public float MddThreshold { get; set; } = 10f;
    public float MddPenaltyScale { get; set; } = 2f;
}

/// <summary>
/// Tracks episode-level metrics for PF and Sharpe calculation.
/// Matches Python _update_episode_metrics, _calculate_profit_factor, _calculate_sharpe_ratio.
/// </summary>
public class EpisodeMetrics
{
    private double _grossProfit = 0;
    private double _grossLoss = 0;
    private int _tradeCount = 0;
    private readonly List<double> _returns = new();

    public double GrossProfit => _grossProfit;
    public double GrossLoss => _grossLoss;
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
    }

    /// <summary>
    /// Calculate PF and Sharpe based reward bonus.
    /// Matches Python _calculate_pf_sharpe_reward.
    /// </summary>
    private float CalculatePfSharpeReward()
    {
        float reward = 0f;

        // Profit Factor reward/penalty
        if (_episodeMetrics.TradeCount >= _config.MinTradesForPf)
        {
            var pf = _episodeMetrics.CalculateProfitFactor();
            var pfCapped = Math.Min(pf, _config.PfCap);

            if (pfCapped >= _config.MinPfThreshold)
            {
                reward += (float)(pfCapped - 1.0) * _config.ProfitFactorWeight;
            }
            else
            {
                // Penalty for low PF (matching Python)
                reward -= (float)(_config.MinPfThreshold - pfCapped) * _config.ProfitFactorWeight * 0.5f;
            }
        }

        // Sharpe Ratio reward/penalty
        if (_episodeMetrics.TradeCount >= _config.MinTradesForSharpe)
        {
            var sharpe = _episodeMetrics.CalculateSharpeRatio(_config.AnnualizationFactor);
            var sharpeCapped = Math.Min(Math.Max(sharpe, -_config.SharpeCap), _config.SharpeCap);

            if (sharpeCapped > 0)
            {
                reward += (float)sharpeCapped * _config.SharpeRatioWeight;
            }
            else
            {
                // Penalty for negative Sharpe (matching Python)
                reward += (float)sharpeCapped * _config.SharpeRatioWeight * 0.5f;
            }
        }

        return reward;
    }

    /// <summary>
    /// Calculate MDD penalty.
    /// Matches Python _calculate_mdd_penalty.
    /// </summary>
    private float CalculateMddPenalty(double mddPercent)
    {
        if (mddPercent <= _config.MddThreshold)
            return 0f;

        var excessMdd = mddPercent - _config.MddThreshold;
        return (float)(excessMdd * _config.MddPenaltyWeight / 100.0 * _config.MddPenaltyScale);
    }

    public float Calculate(
        bool tradeClosed,
        double tradeProfit,
        int holdingTicks,
        bool hasPosition,
        double unrealizedPnl,
        double peakUnrealizedPnl,
        double initialBalance,
        double maxDrawdownPct)
    {
        float reward = 0f;

        // Update episode metrics if trade closed
        if (tradeClosed)
        {
            _episodeMetrics.UpdateTrade(tradeProfit);
        }

        // MDD Penalty (Global)
        float mddPenalty = CalculateMddPenalty(maxDrawdownPct);

        if (tradeClosed)
        {
             // Early close penalty
             if (holdingTicks < _config.MinHoldingTicks)
             {
                 float earlyFactor = 1.0f - ((float)holdingTicks / _config.MinHoldingTicks);
                 reward -= _config.EarlyClosePenalty * earlyFactor;
             }

             // Realized profit
             float profitPct = (float)(tradeProfit / initialBalance) * 100f;
             reward += profitPct * _config.RealizedProfitScale;

             // Win/Loss Bonus
             if (tradeProfit > 0)
             {
                 reward += _config.WinBonus;
                 // Quick profit
                 if (holdingTicks >= _config.MinHoldingTicks && holdingTicks < _config.QuickProfitTicks)
                 {
                      float quickFactor = 1.0f - ((float)(holdingTicks - _config.MinHoldingTicks) / (_config.QuickProfitTicks - _config.MinHoldingTicks));
                      reward += _config.QuickProfitBonus * quickFactor;
                 }
             }
             else
             {
                 reward -= _config.LossPenalty;
             }

             // PF/Sharpe reward component
             reward += CalculatePfSharpeReward();
             
             reward -= mddPenalty;
             return reward;
        }

        if (hasPosition)
        {
            reward -= _config.HoldingTimePenalty;
            
            // Unrealized PnL reward (scaled down)
            if (unrealizedPnl > 0)
            {
                reward += (float)unrealizedPnl * _config.UnrealizedPnlScale * 0.1f;
            }

            // Drawdown from peak unrealized
            double dd = Math.Max(0, peakUnrealizedPnl - unrealizedPnl);
            if (dd > _config.DrawdownThreshold)
            {
                reward -= _config.DrawdownPenalty;
            }
        }
        else
        {
            reward -= _config.FlatPenalty;
        }

        reward -= mddPenalty;

        return reward;
    }
}

namespace Ougha.Trading.RL;

public class RewardConfig
{
    public float RealizedProfitScale { get; set; } = 200f;
    public float UnrealizedPnlScale { get; set; } = 50f;
    public float EquityChangeScale { get; set; } = 100f;
    public float HoldingTimePenalty { get; set; } = 0.0001f;
    public int MaxHoldingTicks { get; set; } = 1000;
    public float DrawdownPenalty { get; set; } = 0.005f;
    public float DrawdownThreshold { get; set; } = 0.01f;
    public float FlatPenalty { get; set; } = 0f;
    public float WinBonus { get; set; } = 10f;
    public float LossPenalty { get; set; } = 5f;
    public float QuickProfitBonus { get; set; } = 20f;
    public int QuickProfitTicks { get; set; } = 200;
    
    public int MinHoldingTicks { get; set; } = 50;
    public float EarlyClosePenalty { get; set; } = 10f;

    public float PositionQualityScale { get; set; } = 20f;
    public float EquityMomentumScale { get; set; } = 30f;

    public float ProfitFactorWeight { get; set; } = 50f;
    public float SharpeRatioWeight { get; set; } = 50f;
    public int MinTradesForPf { get; set; } = 2;
    public int MinTradesForSharpe { get; set; } = 3;
    public float PfCap { get; set; } = 10f;
    public float SharpeCap { get; set; } = 5f;
    public float MinPfThreshold { get; set; } = 2f;
    public float AnnualizationFactor { get; set; } = 252f;

    public float MddPenaltyWeight { get; set; } = 50f;
    public float MddThreshold { get; set; } = 10f;
    public float MddPenaltyScale { get; set; } = 2f;

    public float RewardNormalizationScale { get; set; } = 50f;
    public bool NormalizeRewards { get; set; } = true;
}

/// <summary>
/// Tracks episode-level metrics for PF and Sharpe calculation.
/// Matches Python _update_episode_metrics, _calculate_profit_factor, _calculate_sharpe_ratio.
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
        var reward = 0f;

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
                reward -= (float)(_config.MinPfThreshold - pfCapped) * _config.ProfitFactorWeight * 0.5f;
            }
        }

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

    private double _previousUnrealizedPnl;
    private double _previousEquity;
    
    public float Calculate(
        bool tradeClosed,
        double tradeProfit,
        int holdingTicks,
        bool hasPosition,
        double unrealizedPnl,
        double peakUnrealizedPnl,
        double initialBalance,
        double maxDrawdownPct,
        double currentEquity = 0, int positionDirection = 0, double priceChange = 0)
    {
        var reward = 0f;

        if (tradeClosed)
        {
            _episodeMetrics.UpdateTrade(tradeProfit);
        }

        var mddPenalty = CalculateMddPenalty(maxDrawdownPct);

        if (tradeClosed)
        {
            if (holdingTicks < _config.MinHoldingTicks)
             {
                 var earlyFactor = 1.0f - ((float)holdingTicks / _config.MinHoldingTicks);
                 reward -= _config.EarlyClosePenalty * earlyFactor;
             }

            var profitPct = (float)(tradeProfit / initialBalance) * 100f;
             reward += profitPct * _config.RealizedProfitScale;

             if (tradeProfit > 0)
             {
                 reward += _config.WinBonus;
                 if (holdingTicks >= _config.MinHoldingTicks && holdingTicks < _config.QuickProfitTicks)
                 {
                      var quickFactor = 1.0f - ((float)(holdingTicks - _config.MinHoldingTicks) / (_config.QuickProfitTicks - _config.MinHoldingTicks));
                      reward += _config.QuickProfitBonus * quickFactor;
                 }
             }
             else
             {
                 reward -= _config.LossPenalty;
             }

             reward += CalculatePfSharpeReward();
             
             reward -= mddPenalty;
        }
        else if (hasPosition)
        {
            reward -= _config.HoldingTimePenalty;

            var unrealizedReward = (float)(unrealizedPnl / initialBalance) * 100f * _config.UnrealizedPnlScale;
            reward += unrealizedReward * 0.01f;

            if (positionDirection != 0 && priceChange != 0)
            {
                var directionReward = (float)(priceChange * positionDirection) * _config.PositionQualityScale;
                reward += directionReward;
            }

            if (currentEquity > 0 && _previousEquity > 0)
            {
                var equityChange = (currentEquity - _previousEquity) / initialBalance * 100;
                reward += (float)equityChange * _config.EquityMomentumScale;
            }

            var pnlDelta = unrealizedPnl - _previousUnrealizedPnl;
            if (pnlDelta > 0)
            {
                reward += (float)(pnlDelta / initialBalance) * 50f;
            }

            var dd = Math.Max(0, peakUnrealizedPnl - unrealizedPnl);
            if (dd > _config.DrawdownThreshold)
            {
                reward -= _config.DrawdownPenalty;
            }
            
            reward -= mddPenalty;

            _previousUnrealizedPnl = unrealizedPnl;
            _previousEquity = currentEquity > 0 ? currentEquity : _previousEquity;
        }
        else
        {
            reward -= mddPenalty * 0.5f;

            _previousUnrealizedPnl = 0;
        }

        if (_config.NormalizeRewards)
        {
            reward = Math.Clamp(reward / _config.RewardNormalizationScale, -1f, 1f);
        }

        return reward;
    }
}

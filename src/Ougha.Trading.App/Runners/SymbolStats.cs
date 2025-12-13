namespace Ougha.Trading.App.Runners;

/// <summary>
/// Represents a single trade result for rolling window tracking.
/// </summary>
public record TradeResult(bool IsWin, double Profit, double Loss, double HoldingSeconds);

/// <summary>
/// Statistics for a single symbol during training.
/// </summary>
public class SymbolStats
{
    private const int RollingWindowSize = 100;
    
    public int Episodes { get; set; }
    public double TotalReward { get; set; }
    public double BestReward { get; set; } = double.MinValue;
    public double BestProfitFactor { get; set; }
    public double BestSharpe { get; set; }
    public double BestMaxDrawdown { get; set; }
    public double BestWinRate { get; set; }
    public int BestTrades { get; set; }
    public int NoImprovementCount { get; set; }
    public bool EarlyStopped { get; set; }
    
    // Rolling window for last N trades (excludes early exploration)
    private readonly Queue<TradeResult> _rollingTrades = new();
    
    public void AddRollingTrade(bool isWin, double profit, double loss, double holdingSeconds)
    {
        if (_rollingTrades.Count >= RollingWindowSize)
            _rollingTrades.Dequeue();
        _rollingTrades.Enqueue(new TradeResult(isWin, profit, loss, holdingSeconds));
    }
    
    public int RollingTradeCount => _rollingTrades.Count;
    
    public double RollingWinRate
    {
        get
        {
            if (_rollingTrades.Count == 0) return 0;
            var wins = _rollingTrades.Count(t => t.IsWin);
            return (double)wins / _rollingTrades.Count * 100;
        }
    }
    
    public double RollingProfitFactor
    {
        get
        {
            if (_rollingTrades.Count == 0) return 0;
            var totalProfit = _rollingTrades.Sum(t => t.Profit);
            var totalLoss = _rollingTrades.Sum(t => t.Loss);
            return totalLoss > 0 ? totalProfit / totalLoss : (totalProfit > 0 ? 999.0 : 0);
        }
    }
    
    public double RollingNetPnL
    {
        get
        {
            if (_rollingTrades.Count == 0) return 0;
            return _rollingTrades.Sum(t => t.Profit) - _rollingTrades.Sum(t => t.Loss);
        }
    }
    
    public double RollingAvgProfitPerTrade
    {
        get
        {
            if (_rollingTrades.Count == 0) return 0;
            var netPnL = _rollingTrades.Sum(t => t.Profit) - _rollingTrades.Sum(t => t.Loss);
            return netPnL / _rollingTrades.Count;
        }
    }
    
    public TimeSpan RollingAvgHoldingTime
    {
        get
        {
            if (_rollingTrades.Count == 0) return TimeSpan.Zero;
            var avgSeconds = _rollingTrades.Average(t => t.HoldingSeconds);
            return TimeSpan.FromSeconds(avgSeconds);
        }
    }
    // Cumulative stats
    public int CumulativeTrades { get; set; }
    public int CumulativeWins { get; set; }
    public int CumulativeLosses { get; set; }
    public double CumulativeProfit { get; set; }
    public double CumulativeLoss { get; set; }
    public double CumulativeMaxDrawdown { get; set; }
    public int CumulativeBuys { get; set; }
    public int CumulativeSells { get; set; }
    public double TotalHoldingTimeSeconds { get; set; }
    public double MinHoldingTimeSeconds { get; set; } = double.MaxValue;
    public double MaxHoldingTimeSeconds { get; set; }

    public TimeSpan AverageHoldingTime => CumulativeTrades > 0
        ? TimeSpan.FromSeconds(TotalHoldingTimeSeconds / CumulativeTrades)
        : TimeSpan.Zero;
    
    public TimeSpan MinHoldingTime => MinHoldingTimeSeconds < double.MaxValue
        ? TimeSpan.FromSeconds(MinHoldingTimeSeconds)
        : TimeSpan.Zero;
    
    public TimeSpan MaxHoldingTime => TimeSpan.FromSeconds(MaxHoldingTimeSeconds);
    
    // Recent performance for trend tracking
    public double LastEpisodeReward { get; set; }
    public double PrevEpisodeReward { get; set; }
    
    public double AverageReward => Episodes > 0 ? TotalReward / Episodes : 0;
    public double CumulativeWinRate => CumulativeTrades > 0 ? (double)CumulativeWins / CumulativeTrades * 100 : 0;
    public double CumulativeProfitFactor => CumulativeLoss > 0 ? CumulativeProfit / CumulativeLoss : (CumulativeProfit > 0 ? 999.0 : 0);
    public double NetProfit => CumulativeProfit - CumulativeLoss;
    public double AvgProfitPerTrade => CumulativeTrades > 0 ? NetProfit / CumulativeTrades : 0;
    public double ExpectedValue => CumulativeTrades > 0 
        ? (CumulativeWinRate / 100.0 * (CumulativeProfit / Math.Max(1, CumulativeWins))) 
          - ((100 - CumulativeWinRate) / 100.0 * (CumulativeLoss / Math.Max(1, CumulativeLosses)))
        : 0;
    public string RewardTrend => LastEpisodeReward > PrevEpisodeReward ? "↑" : LastEpisodeReward < PrevEpisodeReward ? "↓" : "→";
}

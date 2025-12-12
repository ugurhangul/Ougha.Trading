namespace Ougha.Trading.App.Runners;

/// <summary>
/// Statistics for a single symbol during training.
/// </summary>
public class SymbolStats
{
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
    
    // Cumulative stats
    public int CumulativeTrades { get; set; }
    public int CumulativeWins { get; set; }
    public int CumulativeLosses { get; set; }
    public double CumulativeProfit { get; set; }
    public double CumulativeLoss { get; set; }
    public double CumulativeMaxDrawdown { get; set; }
    public int CumulativeBuys { get; set; }
    public int CumulativeSells { get; set; }
    
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

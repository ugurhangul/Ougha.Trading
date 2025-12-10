namespace Ougha.Trading.Analysis;

public record PerformanceMetrics
{
    public double TotalReturn { get; init; }
    public double TotalProfit { get; init; }
    public int TotalTrades { get; init; }
    public double WinRate { get; init; }
    public double ProfitFactor { get; init; }
    public double SharpeRatio { get; init; }
    public double MaxDrawdown { get; init; }
    public double AverageWin { get; init; }
    public double AverageLoss { get; init; }
    public double LargestWin { get; init; }
    public double LargestLoss { get; init; }
}

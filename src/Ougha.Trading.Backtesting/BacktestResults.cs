namespace Ougha.Trading.Backtesting;

public class BacktestResults
{
    public double InitialBalance { get; set; }
    public double FinalBalance { get; set; }
    public double FinalEquity { get; set; }
    public int TotalTrades { get; set; }
    public List<TradeRecord> TradeLog { get; set; } = new();
    public List<double> EquityCurve { get; set; } = new();
}

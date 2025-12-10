using Ougha.Trading.Core.Models;

namespace Ougha.Trading.Backtesting;

public enum ExitReason
{
    Manual,
    StopLoss,
    TakeProfit,
    Signal,
    EndOfBacktest
}

public record TradeRecord
{
    public string Symbol { get; init; } = string.Empty;
    public TradeType Type { get; init; }
    public double Volume { get; init; }
    public double OpenPrice { get; init; }
    public double ClosePrice { get; init; }
    public DateTime OpenTime { get; init; }
    public DateTime CloseTime { get; init; }
    public double Profit { get; init; }
    public ExitReason ExitReason { get; init; }
}

public class BacktestResults
{
    public double InitialBalance { get; set; }
    public double FinalBalance { get; set; }
    public double FinalEquity { get; set; }
    public int TotalTrades { get; set; }
    public List<TradeRecord> TradeLog { get; set; } = new();
    public List<double> EquityCurve { get; set; } = new();
}

using Ougha.Trading.Core.Models;

namespace Ougha.Trading.Backtesting;

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

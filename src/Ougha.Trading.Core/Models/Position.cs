namespace Ougha.Trading.Core.Models;

public class Position
{
    public string Symbol { get; set; } = string.Empty;
    public TradeType Type { get; set; }
    public double Volume { get; set; }
    public double OpenPrice { get; set; }
    public double CurrentPrice { get; set; }
    public double StopLoss { get; set; }
    public double TakeProfit { get; set; }
    public DateTime OpenTime { get; set; }
    public long Ticket { get; set; }
    public double Profit { get; set; }
    public double Swap { get; set; }
    public double Commission { get; set; }
    public double BestPrice { get; set; }

    public RiskLevel RiskLevel { get; set; }
    public double InitialStopLoss { get; set; }

    public double UnrealizedPnlPercent 
    {
        get 
        {
            if (OpenPrice == 0) return 0;
            if (Type == TradeType.Buy)
                return (CurrentPrice - OpenPrice) / OpenPrice * 100;
            return (OpenPrice - CurrentPrice) / OpenPrice * 100;
        }
    }
}

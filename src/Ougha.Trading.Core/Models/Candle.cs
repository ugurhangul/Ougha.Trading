namespace Ougha.Trading.Core.Models;

public record Candle(
    DateTime Time,
    double Open,
    double High,
    double Low,
    double Close,
    long Volume,
    double Spread = 0)  // Real spread from tick data (ask - bid)
{
    public Candle() : this(default, 0, 0, 0, 0, 0, 0) { }
}

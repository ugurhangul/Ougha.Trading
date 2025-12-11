namespace Ougha.Trading.Core.Models;

public record Candle(
    DateTime Time,
    double Open,
    double High,
    double Low,
    double Close,
    long Volume)
{
    public Candle() : this(default, default, default, default, default, default) { }
}

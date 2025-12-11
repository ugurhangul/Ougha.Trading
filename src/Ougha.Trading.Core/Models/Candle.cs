namespace Ougha.Trading.Core.Models;

public record Candle(
    DateTime Time,
    double Open,
    double High,
    double Low,
    double Close,
    long Volume)
{
    public Candle() : this(default, 0, 0, 0, 0, 0) { }
}

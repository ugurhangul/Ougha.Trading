namespace Ougha.Trading.Core.Models;

public record Tick(
    DateTime Time,
    double Bid,
    double Ask,
    long TickType,
    bool HasNoData,
    double Volume);

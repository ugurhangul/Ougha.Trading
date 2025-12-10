namespace Ougha.Trading.Risk;



public record RiskProfile(
    double SlAtrMultiplier,
    double TpRrRatio,
    double TrailingTriggerRr,
    double MaxPositionPct);

public record PositionSizing(double Volume, double StopLoss, double TakeProfit);

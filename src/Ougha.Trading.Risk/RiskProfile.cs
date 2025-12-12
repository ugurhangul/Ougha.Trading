namespace Ougha.Trading.Risk;

public record RiskProfile(
    double SlAtrMultiplier,
    double TpRrRatio,
    double TrailingTriggerRr,
    double MaxPositionPct);

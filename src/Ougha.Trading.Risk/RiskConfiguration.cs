namespace Ougha.Trading.Risk;

public record RiskConfiguration(
    double MaxRiskPerTradePct = 1.0,
    double MaxPortfolioRiskPct = 20.0,
    int MaxOpenPositions = 10,
    double MaxLotSize = 100.0,
    double MinLotSize = 0.01,
    bool UseTrailingStop = true,
    double TrailingStopDistance = 50.0);

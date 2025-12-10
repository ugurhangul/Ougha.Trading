namespace Ougha.Trading.RL;

public record EnvironmentConfig(
    string Symbol,
    int WindowSize = 20,
    int MaxSteps = 50000,
    int MaxHoldingSteps = 1000,
    double MaxLossPercent = 50.0);

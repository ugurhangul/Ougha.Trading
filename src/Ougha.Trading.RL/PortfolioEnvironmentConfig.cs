using Ougha.Trading.Data.Services;

namespace Ougha.Trading.RL;

/// <summary>
/// Configuration for portfolio-based trading environment with multiple symbols.
/// </summary>
public record PortfolioEnvironmentConfig(
    string[] Symbols,
    int WindowSize = 20,
    int MaxSteps = 50000,
    int MaxHoldingSteps = 1000,
    double MaxLossPercent = 50.0,
    int ActionMemoryWindow = 1050)
{
    /// <summary>
    /// Dynamic news feature size based on symbol count.
    /// Formula: (symbols * 4) + 1
    /// </summary>
    public int NewsFeatureSize => EconomicCalendarService.GetNewsFeatureSize(Symbols.Length);
    
    /// <summary>
    /// Creates a portfolio config from a comma-separated symbol string.
    /// </summary>
    public static PortfolioEnvironmentConfig FromSymbolString(string symbolArg, int windowSize = 20)
    {
        var symbols = symbolArg.Split(',', StringSplitOptions.RemoveEmptyEntries | StringSplitOptions.TrimEntries);
        return new PortfolioEnvironmentConfig(symbols, windowSize);
    }
}

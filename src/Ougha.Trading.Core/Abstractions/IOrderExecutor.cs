using Ougha.Trading.Core.Models;

namespace Ougha.Trading.Core.Abstractions;

public interface IOrderExecutor
{
    /// <summary>
    /// Execute a market order.
    /// </summary>
    Task<OrderResult> ExecuteAsync(
        string symbol,
        TradeType type,
        double volume,
        double sl = 0,
        double tp = 0,
        string comment = "",
        RiskLevel riskLevel = RiskLevel.Conservative);

    /// <summary>
    /// Close an existing position.
    /// </summary>
    Task<CloseResult> ClosePositionAsync(string symbol);

    /// <summary>
    /// Close all open positions.
    /// </summary>
    Task CloseAllPositionsAsync();

    /// <summary>
    /// Modify an existing position (SL/TP).
    /// </summary>
    Task<bool> ModifyPositionAsync(string symbol, double sl, double tp);

    // Data Access
    /// <summary>
    /// Get all open positions.
    /// </summary>
    IEnumerable<Position> GetPositions();

    /// <summary>
    /// Get position for a specific symbol.
    /// </summary>
    Position? GetPosition(string symbol);

    /// <summary>
    /// Get symbol information.
    /// </summary>
    SymbolInfo? GetSymbolInfo(string symbol);

    /// <summary>
    /// Get current account equity.
    /// </summary>
    double GetEquity();

    /// <summary>
    /// Get current account balance.
    /// </summary>
    double GetBalance();

    /// <summary>
    /// Get available free margin.
    /// </summary>
    double GetFreeMargin();

    /// <summary>
    /// Get current bid price for a symbol.
    /// </summary>
    double GetBid(string symbol);

    /// <summary>
    /// Get current ask price for a symbol.
    /// </summary>
    double GetAsk(string symbol);

    /// <summary>
    /// Check if market is open for trading.
    /// </summary>
    bool IsMarketOpen(string symbol);
}

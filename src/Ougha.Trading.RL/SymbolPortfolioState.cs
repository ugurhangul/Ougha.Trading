using Ougha.Trading.Core.Models;

namespace Ougha.Trading.RL;

/// <summary>
/// Portfolio state for a single symbol.
/// </summary>
public record struct SymbolPortfolioState(
    bool HasPosition,
    TradeType PositionType,
    double UnrealizedPnlPct,
    double HoldingTimeNorm,
    double DrawdownPct)
{
    public static readonly SymbolPortfolioState Flat = new(false, TradeType.Buy, 0, 0, 0);
}

using Ougha.Trading.Core.Models;

namespace Ougha.Trading.RL;

/// <summary>
/// Decodes agent actions to trading operations.
/// Actions: 0=HOLD, 1=BUY, 2=SELL
/// CLOSE action removed - positions only close via TP/SL.
/// </summary>
public static class ActionDecoder
{
    public const int NumActions = 3;  // HOLD, BUY, SELL (CLOSE removed)

    public static TradeType? Decode(int action) => action switch
    {
        0 => null,        // HOLD
        1 => TradeType.Buy,
        2 => TradeType.Sell,
        _ => null
    };

    public static bool IsHold(int action) => action == 0;
    public static bool IsBuy(int action) => action == 1;
    public static bool IsSell(int action) => action == 2;
}

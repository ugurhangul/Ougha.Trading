using Ougha.Trading.Core.Models;

namespace Ougha.Trading.RL;

public static class ActionDecoder
{
    public const int NumActions = 4;  // HOLD, BUY, SELL, CLOSE

    public static TradeType? Decode(int action) => action switch
    {
        0 => null,        // HOLD
        1 => TradeType.Buy,
        2 => TradeType.Sell,
        3 => null,        // CLOSE (handled separately)
        _ => null
    };

    public static bool IsHold(int action) => action == 0;
    public static bool IsBuy(int action) => action == 1;
    public static bool IsSell(int action) => action == 2;
    public static bool IsClose(int action) => action == 3;
}

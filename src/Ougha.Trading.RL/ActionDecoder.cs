using Ougha.Trading.Core.Models;
using Ougha.Trading.Risk;

namespace Ougha.Trading.RL;

public static class ActionDecoder
{
    // Actions:
    // 0: HOLD
    // 1-3: BUY (Cons, Mod, Agg)
    // 4-6: SELL (Cons, Mod, Agg)
    // 7: CLOSE

    public static (TradeType? type, RiskLevel? level) Decode(int action) => action switch
    {
        0 => (null, null),                           // HOLD
        1 => (TradeType.Buy, RiskLevel.Conservative),
        2 => (TradeType.Buy, RiskLevel.Moderate),
        3 => (TradeType.Buy, RiskLevel.Aggressive),
        4 => (TradeType.Sell, RiskLevel.Conservative),
        5 => (TradeType.Sell, RiskLevel.Moderate),
        6 => (TradeType.Sell, RiskLevel.Aggressive),
        7 => (null, null),                           // CLOSE
        _ => (null, null)
    };

    public static bool IsClose(int action) => action == 7;
}

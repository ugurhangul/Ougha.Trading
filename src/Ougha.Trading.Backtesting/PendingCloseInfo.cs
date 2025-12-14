using Ougha.Trading.Core.Models;

namespace Ougha.Trading.Backtesting;

public record PendingCloseInfo(
    string Symbol, 
    double Profit, 
    int HoldingTicks, 
    double Volume = 0,           // Position volume for R-multiple calculation
    double SlDistance = 0,        // SL distance in price units for risk calculation
    ExitReason ExitReason = ExitReason.Manual);

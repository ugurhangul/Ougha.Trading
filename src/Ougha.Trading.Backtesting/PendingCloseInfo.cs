using Ougha.Trading.Core.Models;

namespace Ougha.Trading.Backtesting;

public record PendingCloseInfo(string Symbol, double Profit, int HoldingTicks, ExitReason ExitReason = ExitReason.Manual);

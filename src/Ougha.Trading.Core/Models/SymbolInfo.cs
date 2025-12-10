namespace Ougha.Trading.Core.Models;

public record SymbolInfo(
    string Name,
    double Point,      // e.g. 0.00001
    double ContractSize, // e.g. 100000
    double TickValue,  // e.g. 1.0 (Value of 1 point for 1 lot usually? Or profit currency?)
                       // MT5: SymbolInfoDouble(SYMBOL_TRADE_TICK_VALUE)
    double TickSize,   // e.g. 0.00001
    string BaseCurrency,
    string ProfitCurrency,
    int Digits);

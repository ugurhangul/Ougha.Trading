namespace Ougha.Trading.Core.Models;

public record SymbolInfo(
    string Name,
    double Point,
    double ContractSize,
    double TickValue,
    double TickSize,
    string BaseCurrency,
    string ProfitCurrency,
    int Digits);

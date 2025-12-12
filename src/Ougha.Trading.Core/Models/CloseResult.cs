namespace Ougha.Trading.Core.Models;

public record CloseResult(
    bool Success,
    double Profit,
    double ClosePrice,
    string ErrorMessage = "");

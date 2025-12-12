namespace Ougha.Trading.Core.Models;

public record OrderResult(
    bool Success,
    long Ticket,
    double ExecutedPrice,
    double ExecutedVolume,
    string ErrorMessage = "");

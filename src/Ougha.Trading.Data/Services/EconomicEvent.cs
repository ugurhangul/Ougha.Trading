namespace Ougha.Trading.Data.Services;

/// <summary>
/// Represents a single economic calendar event
/// </summary>
public record EconomicEvent(
    DateTime Time,
    string Currency,
    string EventName,
    EventImpact Impact,
    double? Forecast,
    double? Previous,
    double? Actual
);

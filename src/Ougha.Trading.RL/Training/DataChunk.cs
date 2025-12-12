using Ougha.Trading.Core.Models;

namespace Ougha.Trading.RL.Training;

public record DataChunk(
    DateTime StartDate,
    DateTime EndDate,
    List<(DateTime Time, string Symbol, Candle Candle)> Candles,
    Dictionary<string, List<Candle>> HistoryBySymbol
);

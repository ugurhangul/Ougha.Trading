using Ougha.Trading.Core.Models;

namespace Ougha.Trading.RL.Training;

public record ChunkConfig(
    int ChunkDays = 7,
    int PrefetchChunks = 10,
    int HistoryBufferDays = 1,
    int EpisodeDays = 5
);

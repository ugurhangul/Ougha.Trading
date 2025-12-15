using Ougha.Trading.Core.Models;

namespace Ougha.Trading.RL.Training;

/// <summary>
/// Configuration for chunk-based data loading.
/// ChunkDays automatically equals EpisodeDays for non-overlapping episodes.
/// </summary>
public record ChunkConfig(
    int PrefetchChunks = 10,
    int HistoryBufferDays = 1,
    int EpisodeDays = 3      // 3 days = ~600 episodes from 5 years of data
)
{
    /// <summary>
    /// ChunkDays is always equal to EpisodeDays for 1:1 chunk-to-episode mapping.
    /// This ensures each episode has unique price data with no overlap.
    /// </summary>
    public int ChunkDays => EpisodeDays;
}

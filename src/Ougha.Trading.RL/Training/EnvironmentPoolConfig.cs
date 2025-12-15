namespace Ougha.Trading.RL.Training;

/// <summary>
/// Configuration for environment pool prefetching.
/// Note: ChunkConfig now guarantees ChunkDays = EpisodeDays, so each chunk = 1 episode.
/// </summary>
public record EnvironmentPoolConfig(
    int PoolSize = 5
);

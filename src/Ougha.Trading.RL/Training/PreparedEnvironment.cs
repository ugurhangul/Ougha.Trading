namespace Ougha.Trading.RL.Training;

public record PreparedEnvironment(
    PortfolioTradingEnvironment Env,
    DateTime EpisodeStart,
    DateTime EpisodeEnd,
    int ChunkIndex,
    DateTime ChunkStartDate,
    DateTime ChunkEndDate,
    int EpisodeInChunk,
    int EpisodesPerChunk
);

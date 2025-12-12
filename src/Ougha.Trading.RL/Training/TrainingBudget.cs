namespace Ougha.Trading.RL.Training;

/// <summary>
/// Complete auto-calculated training budget.
/// Mirrors Python's TrainingBudget dataclass.
/// </summary>
public record TrainingBudget(
  int Episodes,
    int MaxSteps,
    int BatchSize,
    int TrainBatches,
    int TrainFreq,
    int MemorySize,
    int SaveFrequency,
    double LearningRate,
    double EpsilonDecay,
    int EarlyStopPatience,
    int EarlyStopMinEpisodes,
    double DaysPerEpisode,
    HardwareInfo Hardware,
    int ChunkDays = 7,
    int ChunkPrefetchCount = 2,
    int ChunkHistoryBufferDays = 1
)
{
}

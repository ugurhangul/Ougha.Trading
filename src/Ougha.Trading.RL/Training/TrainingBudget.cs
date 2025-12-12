namespace Ougha.Trading.RL.Training;

/// <summary>
/// Detected hardware information for budget calculation.
/// </summary>
public record HardwareInfo(
    int CpuCores,
    double RamGb,
    bool GpuAvailable,
    double GpuMemoryGb,
    string? GpuName
);

/// <summary>
/// Estimated data density information from tick sampling.
/// </summary>
public record DataDensityInfo(
    double AvgTicksPerDay,
    double MinTicksPerDay,
    double MaxTicksPerDay,
    int SampleDays,
    int SymbolsSampled,
    string Source
);

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

    public int Episodes { get; init; } = Episodes;
    public int MaxSteps { get; init; } = MaxSteps;
    public int BatchSize { get; init; } = BatchSize;
    public int TrainBatches { get; init; } = TrainBatches;
    public int TrainFreq { get; init; } = TrainFreq;
    public int MemorySize { get; init; } = MemorySize;
    public int SaveFrequency { get; init; } = SaveFrequency;
    public double LearningRate { get; init; } = LearningRate;
    public double EpsilonDecay { get; init; } = EpsilonDecay;
    public int EarlyStopPatience { get; init; } = EarlyStopPatience;
    public int EarlyStopMinEpisodes { get; init; } = EarlyStopMinEpisodes;
    public double DaysPerEpisode { get; init; } = DaysPerEpisode;
    public HardwareInfo Hardware { get; init; } = Hardware;
    public int ChunkDays { get; init; } = ChunkDays;
    public int ChunkPrefetchCount { get; init; } = ChunkPrefetchCount;
    public int ChunkHistoryBufferDays { get; init; } = ChunkHistoryBufferDays;
}

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
    // Episode parameters
    int Episodes,
    int MaxSteps,
    int TickSkipMin,
    int TickSkipMax,

    // Parallelism
    int EnvsPerSymbol,
    int Workers,

    // Training loop
    int BatchSize,
    int TrainBatches,
    int TrainFreq,
    int MemorySize,
    int SaveFrequency,

    // Model architecture
    int LstmUnits,
    int AttentionHeads,
    int[] HiddenLayers,
    double LearningRate,

    // Schedule
    double EpsilonDecay,
    int EarlyStopPatience,
    int EarlyStopMinEpisodes,

    // Metrics
    int NumSymbols,
    long TotalSamples,
    long SamplesPerSymbol,
    double StepsPerTradingDay,
    double DaysPerEpisode,
    double EstimatedTrainingHours,
    HardwareInfo Hardware,

    // Chunk-based training configuration
    int ChunkDays = 7,
    int ChunkPrefetchCount = 2,
    int ChunkHistoryBufferDays = 1,
    bool UseChunkedLoading = true
)
{
    /// <summary>
    /// Return formatted summary string matching Python output.
    /// </summary>
    public string GetSummary()
    {
        var gpuInfo = Hardware.GpuAvailable 
            ? $"{Hardware.GpuName} ({Hardware.GpuMemoryGb:F0} GB)" 
            : "Not available";
        
        return $"""
════════════════════════════════════════════════════════════════════
                    AUTO-CALCULATED TRAINING BUDGET
════════════════════════════════════════════════════════════════════

Hardware Detected:
  GPU:              {gpuInfo}
  CPU:              {Hardware.CpuCores} cores
  RAM:              {Hardware.RamGb:F0} GB

────────────────────────────────────────────────────────────────────
                         CALCULATED PARAMETERS
────────────────────────────────────────────────────────────────────

Episode Budget:
  Episodes:         {Episodes:N0}
  Max Steps:        {MaxSteps:N0}
  Tick Skip:        {TickSkipMin} - {TickSkipMax}

Parallelism:
  Envs/Symbol:      {EnvsPerSymbol}
  Workers:          {Workers}

Training Loop:
  Batch Size:       {BatchSize:N0}
  Train Batches:    {TrainBatches}
  Train Frequency:  {TrainFreq}
  Replay Buffer:    {MemorySize:N0}

Model Architecture:
  LSTM Units:       {LstmUnits}
  Attention Heads:  {AttentionHeads}
  Hidden Layers:    [{string.Join(", ", HiddenLayers)}]
  Learning Rate:    {LearningRate:F6}

Schedule:
  Epsilon Decay:    {EpsilonDecay:F6}
  Save Frequency:   {SaveFrequency} episodes
  Early Stop:       patience={EarlyStopPatience}, min_episodes={EarlyStopMinEpisodes}

Data Loading:
  Chunked Loading:  {(UseChunkedLoading ? "Enabled" : "Disabled")}
  Chunk Size:       {ChunkDays} days
  Prefetch Chunks:  {ChunkPrefetchCount}
  History Buffer:   {ChunkHistoryBufferDays} days

────────────────────────────────────────────────────────────────────
                           COVERAGE METRICS
────────────────────────────────────────────────────────────────────

  Total Samples:              {TotalSamples:N0}
  Samples/Symbol:             {SamplesPerSymbol:N0}
  Episodes/Symbol:            {(double)Episodes / NumSymbols:F1}
  Steps/Trading Day:          {StepsPerTradingDay:F0}
  Days/Episode:               {DaysPerEpisode:F1}

────────────────────────────────────────────────────────────────────
                            ESTIMATES
────────────────────────────────────────────────────────────────────

  Estimated Training Time:    ~{EstimatedTrainingHours:F1} hours

════════════════════════════════════════════════════════════════════
""";
    }
}

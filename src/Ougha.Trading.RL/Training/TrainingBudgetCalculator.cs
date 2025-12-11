using System.Diagnostics;
using System.Runtime.InteropServices;
using Ougha.Trading.Data;
using TorchSharp;

namespace Ougha.Trading.RL.Training;

/// <summary>
/// Auto-calculates optimal training parameters based on hardware, data, and date range.
/// Ported from Python budget_calculator.py.
/// </summary>
public static class TrainingBudgetCalculator
{
    private const int DefaultTicksPerDay = 50000;
    
    #region Hardware Detection
    
    /// <summary>
    /// Detect available hardware (CPU, RAM, GPU).
    /// </summary>
    public static HardwareInfo DetectHardware()
    {
        int cpuCores = Environment.ProcessorCount;
        double ramGb = GetTotalRamGb();
        
        bool gpuAvailable = false;
        double gpuMemoryGb = 0;
        string? gpuName = null;
        
        // Try TorchSharp CUDA detection
        try
        {
            if (torch.cuda.is_available())
            {
                gpuAvailable = true;
                
                // Get GPU info from nvidia-smi (TorchSharp doesn't expose device name)
                var gpuInfo = GetNvidiaGpuInfo();
                gpuName = gpuInfo.Name ?? "NVIDIA GPU";
                gpuMemoryGb = gpuInfo.MemoryGb ?? 24.0; // Default to 24GB if detection fails
            }
        }
        catch
        {
            // CUDA not available
        }
        
        return new HardwareInfo(
            CpuCores: cpuCores,
            RamGb: ramGb,
            GpuAvailable: gpuAvailable,
            GpuMemoryGb: gpuMemoryGb,
            GpuName: gpuName
        );
    }
    
    private static double GetTotalRamGb()
    {
        try
        {
            if (RuntimeInformation.IsOSPlatform(OSPlatform.Windows))
            {
                // Windows: Use WMI or GC for approximation
                return GC.GetGCMemoryInfo().TotalAvailableMemoryBytes / (1024.0 * 1024.0 * 1024.0);
            }
            // Linux/Mac: Read /proc/meminfo or similar
            return 32.0; // Default fallback
        }
        catch
        {
            return 16.0;
        }
    }
    
    private static (string? Name, double? MemoryGb) GetNvidiaGpuInfo()
    {
        try
        {
            var psi = new ProcessStartInfo
            {
                FileName = "nvidia-smi",
                Arguments = "--query-gpu=name,memory.total --format=csv,noheader,nounits",
                RedirectStandardOutput = true,
                UseShellExecute = false,
                CreateNoWindow = true
            };
            
            using var process = Process.Start(psi);
            if (process == null) return (null, null);
            
            var output = process.StandardOutput.ReadToEnd();
            process.WaitForExit();
            
            var parts = output.Trim().Split(',');
            if (parts.Length >= 2)
            {
                var name = parts[0].Trim();
                if (int.TryParse(parts[1].Trim(), out int memoryMb))
                {
                    return (name, memoryMb / 1024.0);
                }
                return (name, null);
            }
        }
        catch
        {
            // nvidia-smi not available
        }
        return (null, null);
    }
    
    #endregion
    
    #region Data Density Estimation
    
    /// <summary>
    /// Estimate tick data density by sampling from QuestDB.
    /// </summary>
    public static async Task<DataDensityInfo> EstimateDataDensityAsync(
        QuestDbDataLoader dataLoader,
        List<string> symbols,
        DateTime startDate,
        DateTime endDate,
        int sampleDays = 5)
    {
        try
        {
            int totalDays = (int)(endDate - startDate).TotalDays;
            if (totalDays <= 0)
            {
                return new DataDensityInfo(DefaultTicksPerDay, DefaultTicksPerDay, DefaultTicksPerDay, 0, 0, "default");
            }
            
            var random = new Random();
            var sampleOffsets = Enumerable.Range(0, totalDays)
                .OrderBy(_ => random.Next())
                .Take(Math.Min(sampleDays, totalDays))
                .ToList();
            
            var symbolsToSample = symbols
                .OrderBy(_ => random.Next())
                .Take(Math.Min(3, symbols.Count))
                .ToList();
            
            var tickCounts = new List<long>();
            
            foreach (var symbol in symbolsToSample)
            {
                foreach (var offset in sampleOffsets)
                {
                    var sampleDate = startDate.AddDays(offset);
                    var dayStart = sampleDate.Date;
                    var dayEnd = dayStart.AddDays(1).AddSeconds(-1);
                    
                    try
                    {
                        var count = await dataLoader.CountTicksAsync(symbol, dayStart, dayEnd);
                        if (count > 0)
                        {
                            tickCounts.Add(count);
                        }
                    }
                    catch
                    {
                        // Skip failed samples
                    }
                }
            }
            
            if (tickCounts.Count > 0)
            {
                return new DataDensityInfo(
                    AvgTicksPerDay: tickCounts.Average(),
                    MinTicksPerDay: tickCounts.Min(),
                    MaxTicksPerDay: tickCounts.Max(),
                    SampleDays: tickCounts.Count,
                    SymbolsSampled: symbolsToSample.Count,
                    Source: "questdb"
                );
            }
        }
        catch
        {
            // Fall through to default
        }
        
        return new DataDensityInfo(
            AvgTicksPerDay: DefaultTicksPerDay,
            MinTicksPerDay: DefaultTicksPerDay * 0.5,
            MaxTicksPerDay: DefaultTicksPerDay * 1.5,
            SampleDays: 0,
            SymbolsSampled: 0,
            Source: "default"
        );
    }
    
    #endregion
    
    #region Budget Calculation
    
    /// <summary>
    /// Calculate complete training budget with all parameters.
    /// </summary>
    public static TrainingBudget CalculateTrainingBudget(
        DateTime startDate,
        DateTime endDate,
        int numSymbols,
        int? episodes = null,
        int? maxSteps = null,
        int? tickSkipMin = null,
        int? tickSkipMax = null,
        int? batchSize = null,
        int? envsPerSymbol = null,
        int? workers = null,
        int? lstmUnits = null,
        int? attentionHeads = null,
        int[]? hiddenLayers = null,
        double? learningRate = null,
        double targetDaysPerEpisode = 5.0,
        double? gpuMemoryGbOverride = null,
        int? cpuCoresOverride = null,
        double? maxTrainingHours = null,
        int minEpisodes = 500,
        int maxEpisodes = 10000,
        int? ticksPerDay = null,
        int? chunkDays = null,
        int? chunkPrefetchCount = null,
        int? chunkHistoryBufferDays = null,
        bool? useChunkedLoading = null)
    {
        // Hardware detection
        var hardware = DetectHardware();
        if (gpuMemoryGbOverride.HasValue)
        {
            hardware = hardware with { GpuMemoryGb = gpuMemoryGbOverride.Value, GpuAvailable = gpuMemoryGbOverride.Value > 0 };
        }
        if (cpuCoresOverride.HasValue)
        {
            hardware = hardware with { CpuCores = cpuCoresOverride.Value };
        }
        
        int trainingDays = (int)(endDate - startDate).TotalDays;
        if (trainingDays <= 0)
        {
            throw new ArgumentException("end_date must be after start_date");
        }
        
        int effectiveTicksPerDay = ticksPerDay ?? DefaultTicksPerDay;
        
        // Parallelism
        var (calcEnvs, calcWorkers) = CalculateParallelism(hardware);
        int effectiveEnvsPerSymbol = envsPerSymbol ?? calcEnvs;
        int effectiveWorkers = workers ?? calcWorkers;
        
        // Episodes
        int effectiveEpisodes = episodes ?? CalculateEpisodes(
            trainingDays, numSymbols, effectiveEnvsPerSymbol, minEpisodes, maxEpisodes);
        
        // Tick skip
        var (calcTickSkipMin, calcTickSkipMax) = CalculateTickSkip(targetDaysPerEpisode, effectiveTicksPerDay);
        int effectiveTickSkipMin = tickSkipMin ?? calcTickSkipMin;
        int effectiveTickSkipMax = tickSkipMax ?? calcTickSkipMax;
        double tickSkipAvg = (effectiveTickSkipMin + effectiveTickSkipMax) / 2.0;
        
        // Max steps
        int effectiveMaxSteps =CalculateMaxSteps(tickSkipAvg, targetDaysPerEpisode, effectiveTicksPerDay);
        
        // Model architecture
        var (calcLstm, calcHeads, calcLayers) = CalculateModelArchitecture(hardware.GpuMemoryGb);
        int effectiveLstmUnits = lstmUnits ?? calcLstm;
        int effectiveAttentionHeads = attentionHeads ?? calcHeads;
        int[] effectiveHiddenLayers = hiddenLayers ?? calcLayers;
        
        // Batch size
        int effectiveBatchSize = batchSize ?? CalculateBatchSize(hardware.GpuMemoryGb);
        
        // Learning rate
        double effectiveLearningRate = learningRate ?? CalculateLearningRate(effectiveBatchSize);
        
        // Training loop params based on GPU
        // Keep trainBatches low for responsive training (< 1 second per call)
        int trainFreq, trainBatches;
        if (hardware.GpuMemoryGb >= 20)
        {
            trainFreq = 4;      // Train every 4 steps
            trainBatches = 4;   // 4 batches per call (~200ms per call target)
        }
        else if (hardware.GpuMemoryGb >= 10)
        {
            trainFreq = 4;
            trainBatches = 2;   // 2 batches per call
        }
        else
        {
            trainFreq = 4;
            trainBatches = 1;   // 1 batch per call
        }
        
        int memorySize = Math.Max(100000, Math.Min(5000000, effectiveEpisodes * effectiveMaxSteps / 10));
        int saveFrequency = Math.Max(10, Math.Min(100, effectiveEpisodes / 50));
        
        // Epsilon decay
        double epsilonDecay = CalculateEpsilonDecay(effectiveEpisodes);
        
        // Early stopping
        var (earlyStopPatience, earlyStopMinEpisodes) = CalculateEarlyStopping(effectiveEpisodes);
        
        // Metrics
        long totalSamples = (long)effectiveEpisodes * effectiveMaxSteps * numSymbols * effectiveEnvsPerSymbol;
        long samplesPerSymbol = totalSamples / numSymbols;
        double stepsPerTradingDay = effectiveTicksPerDay / tickSkipAvg;
        double daysPerEpisode = effectiveMaxSteps / stepsPerTradingDay;
        
        double estimatedHours = EstimateTrainingTime(
            effectiveEpisodes, effectiveMaxSteps, numSymbols, effectiveEnvsPerSymbol,
            hardware.GpuAvailable, hardware.GpuMemoryGb, hardware.CpuCores);
        
        // Constrain by max training hours if specified
        if (maxTrainingHours.HasValue && estimatedHours > maxTrainingHours.Value)
        {
            double scaleFactor = maxTrainingHours.Value / estimatedHours;
            effectiveEpisodes = Math.Max(minEpisodes, (int)(effectiveEpisodes * scaleFactor));
            estimatedHours = maxTrainingHours.Value;
            totalSamples = (long)effectiveEpisodes * effectiveMaxSteps * numSymbols * effectiveEnvsPerSymbol;
            samplesPerSymbol = totalSamples / numSymbols;
        }
        
        var effectiveChunkDays = chunkDays ?? 7;
        var effectiveChunkPrefetch = chunkPrefetchCount ?? 2;
        var effectiveChunkHistoryBuffer = chunkHistoryBufferDays ?? 1;
        var effectiveUseChunkedLoading = useChunkedLoading ?? true;

        return new TrainingBudget(
            Episodes: effectiveEpisodes,
            MaxSteps: effectiveMaxSteps,
            TickSkipMin: effectiveTickSkipMin,
            TickSkipMax: effectiveTickSkipMax,
            EnvsPerSymbol: effectiveEnvsPerSymbol,
            Workers: effectiveWorkers,
            BatchSize: effectiveBatchSize,
            TrainBatches: trainBatches,
            TrainFreq: trainFreq,
            MemorySize: memorySize,
            SaveFrequency: saveFrequency,
            LstmUnits: effectiveLstmUnits,
            AttentionHeads: effectiveAttentionHeads,
            HiddenLayers: effectiveHiddenLayers,
            LearningRate: effectiveLearningRate,
            EpsilonDecay: epsilonDecay,
            EarlyStopPatience: earlyStopPatience,
            EarlyStopMinEpisodes: earlyStopMinEpisodes,
            NumSymbols: numSymbols,
            TotalSamples: totalSamples,
            SamplesPerSymbol: samplesPerSymbol,
            StepsPerTradingDay: stepsPerTradingDay,
            DaysPerEpisode: daysPerEpisode,
            EstimatedTrainingHours: estimatedHours,
            Hardware: hardware,
            ChunkDays: effectiveChunkDays,
            ChunkPrefetchCount: effectiveChunkPrefetch,
            ChunkHistoryBufferDays: effectiveChunkHistoryBuffer,
            UseChunkedLoading: effectiveUseChunkedLoading
        );
    }
    
    #endregion
    
    #region Private Helpers
    
    private static int CalculateEpisodes(
        int trainingDays,
        int numSymbols,
        int envsPerSymbol,
        int minEpisodes,
        int maxEpisodes)
    {
        double trainingYears = trainingDays / 365.0;
        
        int durationEpisodes = (int)(1500 * (1 + Math.Log(1 + trainingYears * 3)));
        int symbolCoverage = numSymbols * 180;
        
        int baseEpisodes = Math.Max(durationEpisodes, symbolCoverage);
        
        // double parallelFactor = Math.Sqrt(envsPerSymbol);
        // int adjusted = (int)(baseEpisodes * 2.0 / parallelFactor);
        //
        return baseEpisodes;
    }
    
    private static (int Min, int Max) CalculateTickSkip(double targetDaysPerEpisode, int ticksPerDay, int targetSteps = 1500)
    {
        double tickSkipAvg = (ticksPerDay * targetDaysPerEpisode) / targetSteps;
        
        int tickSkipMin = (int)(tickSkipAvg * 0.6);
        int tickSkipMax = (int)(tickSkipAvg * 1.4);
        
        
        if (tickSkipMin >= tickSkipMax)
        {
            tickSkipMax = tickSkipMin + 100;
        }
        
        return (tickSkipMin, tickSkipMax);
    }
    
    private static int CalculateMaxSteps(double tickSkipAvg, double targetDaysPerEpisode, int ticksPerDay)
    {
        double stepsPerDay = ticksPerDay / tickSkipAvg;
        int durationSteps = (int)(stepsPerDay * targetDaysPerEpisode);
        return Math.Max(500, Math.Min(5000, durationSteps));
    }
    
    private static int CalculateBatchSize(double gpuMemoryGb)
    {
        // PPO benefits from larger batch sizes for stable gradients
        // With 3-tensor packing, we can handle larger batches efficiently
        if (gpuMemoryGb >= 20)
            return 512;   // 24GB GPUs - large batches for faster training
        if (gpuMemoryGb >= 10)
            return 256;   // 12GB GPUs
        if (gpuMemoryGb >= 6)
            return 128;   // 8GB GPUs
        return 64;        // CPU or low VRAM
    }
    
    private static (int LstmUnits, int AttentionHeads, int[] HiddenLayers) CalculateModelArchitecture(double gpuMemoryGb)
    {
        // Match Python's current hardcoded values
        int lstmUnits = 1024;
        int attentionHeads = 16;
        int[] hiddenLayers = [2048,1024,512, 256];
        
        return (lstmUnits, attentionHeads, hiddenLayers);
    }
    
    private static double CalculateLearningRate(int batchSize, double baseLr = 0.0003)
    {
        const int referenceBatch = 2048;
        double lr = baseLr * ((double)batchSize / referenceBatch);
        return Math.Max(0.0001, Math.Min(0.001, lr));
    }
    
    private static double CalculateEpsilonDecay(int episodes, double targetExploitationPct = 0.7)
    {
        int targetEpisode = (int)(episodes * targetExploitationPct);
        const double epsilonStart = 1.0;
        const double epsilonEnd = 0.01;
        
        if (targetEpisode <= 0)
        {
            return 0.999;
        }
        
        double decay = Math.Pow(epsilonEnd / epsilonStart, 1.0 / targetEpisode);
        return Math.Max(0.99, Math.Min(0.9999, decay));
    }
    
    private static (int Patience, int MinEpisodes) CalculateEarlyStopping(int episodes)
    {
        int patience = Math.Max(50, (int)(episodes * 0.15));
        int minEpisodes = Math.Max(100, (int)(episodes * 0.5));
        return (patience, minEpisodes);
    }
    
    private static (int EnvsPerSymbol, int Workers) CalculateParallelism(HardwareInfo hardware)
    {
        // Match Python's hardcoded high-performance values
        int envsPerSymbol = 512;
        int workers = 64;
        return (envsPerSymbol, workers);
    }
    
    private static double EstimateTrainingTime(
        int episodes,
        int maxSteps,
        int numSymbols,
        int envsPerSymbol,
        bool gpuAvailable,
        double gpuMemoryGb,
        int cpuCores)
    {
        long totalSteps = (long)episodes * maxSteps * numSymbols * envsPerSymbol;
        
        int stepsPerSecond;
        if (gpuAvailable)
        {
            stepsPerSecond = gpuMemoryGb >= 20 ? 200000 :
                             gpuMemoryGb >= 10 ? 120000 : 80000;
        }
        else
        {
            stepsPerSecond = cpuCores >= 24 ? 20000 :
                             cpuCores >= 16 ? 15000 :
                             cpuCores >= 8 ? 10000 : 5000;
        }
        
        double trainingSeconds = (double)totalSteps / stepsPerSecond;
        const double overheadFactor = 1.2;
        
        return (trainingSeconds * overheadFactor) / 3600;
    }
    
    #endregion
}

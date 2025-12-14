using System.Diagnostics;
using System.Runtime.InteropServices;
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
    private static HardwareInfo DetectHardware()
    {
        var cpuCores = Environment.ProcessorCount;
        var ramGb = GetTotalRamGb();
        
        var gpuAvailable = false;
        double gpuMemoryGb = 0;
        string? gpuName = null;

        try
        {
            if (torch.cuda.is_available())
            {
                gpuAvailable = true;

                var gpuInfo = GetNvidiaGpuInfo();
                gpuName = gpuInfo.Name ?? "NVIDIA GPU";
                gpuMemoryGb = gpuInfo.MemoryGb ?? 24.0;
            }
        }
        catch
        {
            // ignored
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
                return GC.GetGCMemoryInfo().TotalAvailableMemoryBytes / (1024.0 * 1024.0 * 1024.0);
            }

            return 32.0;
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
                if (int.TryParse(parts[1].Trim(), out var memoryMb))
                {
                    return (name, memoryMb / 1024.0);
                }
                return (name, null);
            }
        }
        catch
        {
            // ignored
        }

        return (null, null);
    }
    
    #endregion
    
    #region Budget Calculation
    
    /// <summary>
    /// Calculate the complete training budget with all parameters.
    /// </summary>
    public static TrainingBudget CalculateTrainingBudget(
        DateTime startDate,
        DateTime endDate,
        int numSymbols,
        int? episodes = null,
        int? tickSkipMin = null,
        int? tickSkipMax = null,
        int? batchSize = null,
        int? envsPerSymbol = null,
        double? learningRate = null,
        double targetDaysPerEpisode = 5.0,
        double? gpuMemoryGbOverride = null,
        int? cpuCoresOverride = null,
        double? maxTrainingHours = null,
        int minEpisodes = 500,
        int? ticksPerDay = null,
        int? chunkDays = null,
        int? chunkPrefetchCount = null,
        int? chunkHistoryBufferDays = null
        )
    {
        var hardware = DetectHardware();
        if (gpuMemoryGbOverride.HasValue)
        {
            hardware = hardware with { GpuMemoryGb = gpuMemoryGbOverride.Value, GpuAvailable = gpuMemoryGbOverride.Value > 0 };
        }
        if (cpuCoresOverride.HasValue)
        {
            hardware = hardware with { CpuCores = cpuCoresOverride.Value };
        }
        
        var trainingDays = (int)(endDate - startDate).TotalDays;
        if (trainingDays <= 0)
        {
            throw new ArgumentException("end_date must be after start_date");
        }
        
        var effectiveTicksPerDay = ticksPerDay ?? DefaultTicksPerDay;

        var calcEnvs = CalculateParallelism();
        var effectiveEnvsPerSymbol = envsPerSymbol ?? calcEnvs;

        var effectiveEpisodes = episodes ?? CalculateEpisodes(
            trainingDays, numSymbols);

        // Fixed tick skip of 60 = one decision per M1 candle (~4000 decisions per 5-day episode)
        var effectiveTickSkipMin = tickSkipMin ?? 60;
        var effectiveTickSkipMax = tickSkipMax ?? 60;
        var tickSkipAvg = (effectiveTickSkipMin + effectiveTickSkipMax) / 2.0;

        var effectiveMaxSteps =CalculateMaxSteps(tickSkipAvg, targetDaysPerEpisode, effectiveTicksPerDay);
        var effectiveBatchSize = batchSize ?? CalculateBatchSize(hardware.GpuMemoryGb);
        var effectiveLearningRate = learningRate ?? CalculateLearningRate(effectiveBatchSize);

        int trainFreq, trainBatches;
        switch (hardware.GpuMemoryGb)
        {
            case >= 20:
                trainFreq = 4;
                trainBatches = 4;
                break;
            case >= 10:
                trainFreq = 4;
                trainBatches = 2;
                break;
            default:
                trainFreq = 4;
                trainBatches = 1;
                break;
        }
        
        var memorySize = Math.Max(100000, Math.Min(5000000, effectiveEpisodes * effectiveMaxSteps / 10));
        var saveFrequency = Math.Max(10, Math.Min(100, effectiveEpisodes / 50));

        var epsilonDecay = CalculateEpsilonDecay(effectiveEpisodes);

        var (earlyStopPatience, earlyStopMinEpisodes) = CalculateEarlyStopping(effectiveEpisodes);

        var stepsPerTradingDay = effectiveTicksPerDay / tickSkipAvg;
        var daysPerEpisode = effectiveMaxSteps / stepsPerTradingDay;
        
        var estimatedHours = EstimateTrainingTime(
            effectiveEpisodes, effectiveMaxSteps, numSymbols, effectiveEnvsPerSymbol,
            hardware.GpuAvailable, hardware.GpuMemoryGb, hardware.CpuCores);

        if (maxTrainingHours.HasValue && estimatedHours > maxTrainingHours.Value)
        {
            var scaleFactor = maxTrainingHours.Value / estimatedHours;
            effectiveEpisodes = Math.Max(minEpisodes, (int)(effectiveEpisodes * scaleFactor));
        }
        
        var effectiveChunkDays = chunkDays ?? 7;
        var effectiveChunkPrefetch = chunkPrefetchCount ?? 2;
        var effectiveChunkHistoryBuffer = chunkHistoryBufferDays ?? 1;

        return new TrainingBudget(
            Episodes: effectiveEpisodes,
            MaxSteps: effectiveMaxSteps,
            BatchSize: effectiveBatchSize,
            TrainBatches: trainBatches,
            TrainFreq: trainFreq,
            MemorySize: memorySize,
            SaveFrequency: saveFrequency,
            LearningRate: effectiveLearningRate,
            EpsilonDecay: epsilonDecay,
            EarlyStopPatience: earlyStopPatience,
            EarlyStopMinEpisodes: earlyStopMinEpisodes,
            DaysPerEpisode: daysPerEpisode,
            Hardware: hardware,
            ChunkDays: effectiveChunkDays,
            ChunkPrefetchCount: effectiveChunkPrefetch,
            ChunkHistoryBufferDays: effectiveChunkHistoryBuffer
        );
    }
    
    #endregion
    
    #region Private Helpers
    
    private static int CalculateEpisodes(
        int trainingDays,
        int numSymbols)
    {
        var trainingYears = trainingDays / 365.0;
        
        var durationEpisodes = (int)(1500 * (1 + Math.Log(1 + trainingYears * 3)));
        var symbolCoverage = numSymbols * 180;
        
        var baseEpisodes = Math.Max(durationEpisodes, symbolCoverage);

        return baseEpisodes;
    }
    
    private static (int Min, int Max) CalculateTickSkip(double targetDaysPerEpisode, int ticksPerDay, int targetSteps = 1500)
    {
        var tickSkipAvg = (ticksPerDay * targetDaysPerEpisode) / targetSteps;
        
        var tickSkipMin = (int)(tickSkipAvg * 0.6);
        var tickSkipMax = (int)(tickSkipAvg * 1.4);
        
        
        if (tickSkipMin >= tickSkipMax)
        {
            tickSkipMax = tickSkipMin + 100;
        }
        
        return (tickSkipMin, tickSkipMax);
    }
    
    private static int CalculateMaxSteps(double tickSkipAvg, double targetDaysPerEpisode, int ticksPerDay)
    {
        var stepsPerDay = ticksPerDay / tickSkipAvg;
        var durationSteps = (int)(stepsPerDay * targetDaysPerEpisode);
        return Math.Max(500, Math.Min(5000, durationSteps));
    }
    
    private static int CalculateBatchSize(double gpuMemoryGb)
    {
        return gpuMemoryGb switch
        {
            >= 20 => 512,
            >= 10 => 256,
            _ => gpuMemoryGb >= 6 ? 128 : 64
        };
    }

    private static double CalculateLearningRate(int batchSize, double baseLr = 0.0003)
    {
        const int referenceBatch = 2048;
        var lr = baseLr * ((double)batchSize / referenceBatch);
        return Math.Max(0.0001, Math.Min(0.001, lr));
    }
    
    private static double CalculateEpsilonDecay(int episodes, double targetExploitationPct = 0.7)
    {
        var targetEpisode = (int)(episodes * targetExploitationPct);
        const double epsilonStart = 1.0;
        const double epsilonEnd = 0.01;
        
        if (targetEpisode <= 0)
        {
            return 0.999;
        }
        
        var decay = Math.Pow(epsilonEnd / epsilonStart, 1.0 / targetEpisode);
        return Math.Max(0.99, Math.Min(0.9999, decay));
    }
    
    private static (int Patience, int MinEpisodes) CalculateEarlyStopping(int episodes)
    {
        var patience = Math.Max(50, (int)(episodes * 0.15));
        var minEpisodes = Math.Max(100, (int)(episodes * 0.5));
        return (patience, 0);
    }
    
    private static int  CalculateParallelism()
    {
        var envsPerSymbol = 512;
        return envsPerSymbol;
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
        var totalSteps = (long)episodes * maxSteps * numSymbols * envsPerSymbol;
        
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
        
        var trainingSeconds = (double)totalSteps / stepsPerSecond;
        const double overheadFactor = 1.2;
        
        return (trainingSeconds * overheadFactor) / 3600;
    }
    
    #endregion
}

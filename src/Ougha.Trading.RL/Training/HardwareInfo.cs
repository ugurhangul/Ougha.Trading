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

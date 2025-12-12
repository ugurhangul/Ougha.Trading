namespace Ougha.Trading.RL.Agents;

/// <summary>
/// Training metrics from a PPO update
/// </summary>
public struct PpoMetrics
{
    public float PolicyLoss { get; init; }
    public float ValueLoss { get; init; }
    public float Entropy { get; init; }
    public float KlDivergence { get; init; }
    public float ClipFraction { get; init; }
    public float ClipEpsilon { get; init; }
}

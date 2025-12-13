using Ougha.Trading.RL.Agents;

namespace Ougha.Trading.RL.Training;

public class Experience
{
    public required AgentInput State { get; set; }
    public int Action { get; set; }
    public float Reward { get; set; }
    public AgentInput? NextState { get; set; }
    public bool Done { get; set; }

    /// <summary>
    /// Log probability of the action (logπ(a|s)) for PPO importance sampling.
    /// Used in the PPO ratio calculation: exp(newLogProb - oldLogProb).
    /// </summary>
    public float LogProb { get; set; } = 0f;
    
    /// <summary>
    /// Priority weight for DQN Prioritized Experience Replay (PER).
    /// Higher priority = more likely to be sampled.
    /// </summary>
    public float Priority { get; set; } = 1.0f;
    
    /// <summary>
    /// Take Profit multiplier [0,1] predicted by the model.
    /// Used to train the TP/SL head.
    /// </summary>
    public float TpMultiplier { get; set; }
    
    /// <summary>
    /// Stop Loss multiplier [0,1] predicted by the model.
    /// Used to train the TP/SL head.
    /// </summary>
    public float SlMultiplier { get; set; }
    
    /// <summary>
    /// Episode identifier for sequence-based batching.
    /// Experiences from the same episode should be processed together.
    /// </summary>
    public int EpisodeId { get; set; }
    
    /// <summary>
    /// Position within the episode (0, 1, 2, ...).
    /// Used to maintain temporal order within sequences.
    /// </summary>
    public int SequenceIndex { get; set; }
    
    /// <summary>
    /// Symbol index for multi-symbol environments.
    /// </summary>
    public int SymbolIdx { get; set; }
}

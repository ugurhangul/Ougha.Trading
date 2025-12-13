namespace Ougha.Trading.RL.Agents;

/// <summary>
/// Unified interface for RL agents.
/// Supports both inference-only (ONNX) and training (Torch) agents.
/// </summary>
public interface IAgent : IDisposable
{
    /// <summary>
    /// Select an action based on the current state.
    /// </summary>
    /// <param name="state">The structured input state.</param>
    /// <param name="training">Whether to use exploration (epsilon-greedy) or exploitation.</param>
    /// <returns>Selected action index.</returns>
    int Act(AgentInput state, bool training = false);

    /// <summary>
    /// Select actions for a batch of states, including TP/SL multipliers and log probabilities.
    /// Used by PPO for on-policy training where log probs must be stored with experiences.
    /// </summary>
    (int[] Actions, float[,] TpSlMultipliers, float[] LogProbs) ActBatchWithTpSlAndLogProbs(AgentInput[] inputs, bool training = true);

    /// <summary>
    /// Add an experience tuple to the replay buffer.
    /// </summary>
    void AddExperience(AgentInput state, int action, float reward, AgentInput? nextState, bool done);

    /// <summary>
    /// Add a batch of experiences with log probabilities and TP/SL multipliers for PPO training.
    /// </summary>
    void AddExperienceBatchWithLogProbs(
        AgentInput[] states,
        int[] actions,
        float[] rewards,
        AgentInput?[] nextStates,
        bool[] dones,
        float[] logProbs,
        float[] tpMultipliers,
        float[] slMultipliers);

    /// <summary>
    /// Perform a training step (optimization).
    /// </summary>
    /// <returns>Loss value (or 0 if not trained).</returns>
    float Train();
    
    float TrainMultipleBatches(int batches);

    void SyncInferenceNetwork();
    
    void DecayEpsilon();

    /// <summary>
    /// Save the model to the specified path.
    /// </summary>
    void Save(string path);

    /// <summary>
    /// Load the model from the specified path.
    /// </summary>
    void Load(string path);

    /// <summary>
    /// Reset the online learning state for a new backtest run.
    /// Clears replay buffer, resets epsilon, and step count.
    /// </summary>
    void ResetOnlineLearning();
}

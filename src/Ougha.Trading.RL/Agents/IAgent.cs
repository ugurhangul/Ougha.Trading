using System;

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
    /// Add an experience tuple to the replay buffer (if applicable).
    /// </summary>
    /// <param name="state">Current state.</param>
    /// <param name="action">Action taken.</param>
    /// <param name="reward">Reward received.</param>
    /// <param name="nextState">Next state.</param>
    /// <param name="done">Whether the episode ended.</param>
    void Observe(AgentInput state, int action, float reward, AgentInput nextState, bool done);

    /// <summary>
    /// Perform a training step (optimization).
    /// </summary>
    /// <returns>Loss value (or 0 if not trained).</returns>
    float Train();

    /// <summary>
    /// Save the model to the specified path.
    /// </summary>
    void Save(string path);

    /// <summary>
    /// Load the model from the specified path.
    /// </summary>
    void Load(string path);

    /// <summary>
    /// Reset online learning state for a new backtest run.
    /// Clears replay buffer, resets epsilon, and step count.
    /// </summary>
    void ResetOnlineLearning();
}

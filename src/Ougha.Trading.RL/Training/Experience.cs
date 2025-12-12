using Ougha.Trading.RL.Agents;

namespace Ougha.Trading.RL.Training;

public class Experience
{
    public AgentInput State { get; set; }
    public int Action { get; set; }
    public float Reward { get; set; }
    public AgentInput? NextState { get; set; }
    public bool Done { get; set; }

    public float Priority { get; set; } = 1.0f;
}

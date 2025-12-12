using Ougha.Trading.RL;

namespace Ougha.Trading.RL.Agents;

internal struct PpoBatch {
    public int[] Indices;
    public AgentInput[] States;
    public int[] Actions;
    public float[] LogProbs;
    public float[] Advantages;
    public float[] Returns;
    public float[] OldValues;
}

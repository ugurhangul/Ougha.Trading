using Ougha.Trading.RL;
using Ougha.Trading.RL.Training;

namespace Ougha.Trading.RL.Agents;

internal class PpoDataset
{
    public readonly AgentInput[] States;
    public readonly int[] Actions;
    public readonly float[] LogProbs;
    public readonly float[] Advantages;
    public readonly float[] Returns;
    
    public PpoDataset(Experience[] rollouts, float[] advantages, float[] returns)
    {
        var n = rollouts.Length;
        States = new AgentInput[n];
        Actions = new int[n];
        LogProbs = new float[n];
        Advantages = advantages;
        Returns = returns;
        
        for(var i=0; i<n; i++)
        {
            States[i] = rollouts[i].State;
            Actions[i] = rollouts[i].Action;
            LogProbs[i] = rollouts[i].Priority;
        }
    }
}

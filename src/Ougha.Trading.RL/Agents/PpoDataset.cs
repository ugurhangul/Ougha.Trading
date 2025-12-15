using Ougha.Trading.RL.Training;

namespace Ougha.Trading.RL.Agents;

internal class PpoDataset
{
    public readonly AgentInput[] States;
    public readonly int[] Actions;
    public readonly float[] LogProbs;
    public readonly float[] Advantages;
    public readonly float[] Returns;
    public readonly float[] TpMultipliers;
    public readonly float[] SlMultipliers;
    public readonly float[] HindsightSlMultipliers;  // Optimal SL from MAE tracking
    public readonly bool[] HadPositions;  // Whether agent had position (for close signal)
    public readonly float[] ActualPriceChanges;  // Actual price change for supervised prediction
    public readonly float[] OldValues;  // For value function clipping
    
    public PpoDataset(Experience[] rollouts, float[] advantages, float[] returns, float[]? oldValues = null)
    {
        var n = rollouts.Length;
        States = new AgentInput[n];
        Actions = new int[n];
        LogProbs = new float[n];
        TpMultipliers = new float[n];
        SlMultipliers = new float[n];
        HindsightSlMultipliers = new float[n];
        HadPositions = new bool[n];
        ActualPriceChanges = new float[n];
        Advantages = advantages;
        Returns = returns;
        OldValues = oldValues ?? new float[n];  // Zero if not provided
        
        for (var i = 0; i < n; i++)
        {
            States[i] = rollouts[i].State;
            Actions[i] = rollouts[i].Action;
            LogProbs[i] = rollouts[i].LogProb;
            TpMultipliers[i] = rollouts[i].TpMultiplier;
            SlMultipliers[i] = rollouts[i].SlMultiplier;
            HindsightSlMultipliers[i] = rollouts[i].HindsightSlMultiplier;
            HadPositions[i] = rollouts[i].HadPosition;
            ActualPriceChanges[i] = rollouts[i].ActualPriceChange;
        }
    }
}

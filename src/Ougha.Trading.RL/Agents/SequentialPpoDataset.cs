using Ougha.Trading.RL.Training;

namespace Ougha.Trading.RL.Agents;

/// <summary>
/// A contiguous sequence of experiences for LSTM training.
/// Maintains temporal order within the sequence.
/// </summary>
public class PpoSequence
{
    public required AgentInput[] States { get; init; }
    public required int[] Actions { get; init; }
    public required float[] LogProbs { get; init; }
    public required float[] Advantages { get; init; }
    public required float[] Returns { get; init; }
    public required float[] OldValues { get; init; }
    public required float[] TpMultipliers { get; init; }
    public required float[] SlMultipliers { get; init; }
    /// <summary>
    /// Hindsight SL multipliers computed after trade closes using MAE.
    /// -1 = no hindsight data (trade didn't close). Use SlMultipliers as fallback.
    /// </summary>
    public required float[] HindsightSlMultipliers { get; init; }
    /// <summary>
    /// Whether agent had position at each timestep (for close signal training).
    /// </summary>
    public required bool[] HadPositions { get; init; }
    /// <summary>
    /// Actual price changes when trades closed (for supervised prediction).
    /// -999 = no trade closed.
    /// </summary>
    public required float[] ActualPriceChanges { get; init; }
    public required int EpisodeId { get; init; }
    public int Length => States.Length;
}

/// <summary>
/// Dataset that preserves temporal order within sequences for LSTM training.
/// Sequences can be shuffled, but experiences within each sequence remain in chronological order.
/// Truncates at episode boundaries to prevent learning false patterns across gaps.
/// </summary>
internal class SequentialPpoDataset
{
    public readonly List<PpoSequence> Sequences;
    
    /// <summary>
    /// Sequence length in timesteps (256 = ~4 hours of M1 decisions)
    /// </summary>
    public const int DefaultSequenceLength = 256;

    public SequentialPpoDataset(
        Experience[] rollouts, 
        float[] advantages, 
        float[] returns, 
        float[] oldValues,
        int sequenceLength = DefaultSequenceLength)
    {
        Sequences = BuildSequences(rollouts, advantages, returns, oldValues, sequenceLength);
    }

    private static List<PpoSequence> BuildSequences(
        Experience[] rollouts,
        float[] advantages,
        float[] returns,
        float[] oldValues,
        int sequenceLength)
    {
        var sequences = new List<PpoSequence>();
        
        // Group experiences by episode
        var byEpisode = rollouts
            .Select((exp, idx) => (exp, idx))
            .GroupBy(x => x.exp.EpisodeId)
            .OrderBy(g => g.Key);
        
        foreach (var episodeGroup in byEpisode)
        {
            // Sort by sequence index within episode to ensure temporal order
            var episodeExperiences = episodeGroup
                .OrderBy(x => x.exp.SequenceIndex)
                .ToList();
            
            // Cut into fixed-length sequences, truncating at episode boundary
            for (var i = 0; i < episodeExperiences.Count; i += sequenceLength)
            {
                var remaining = episodeExperiences.Count - i;
                var seqLen = Math.Min(sequenceLength, remaining);
                
                // Skip very short sequences (< 25% of target length)
                if (seqLen < sequenceLength / 4)
                    continue;
                
                var seqExps = episodeExperiences.Skip(i).Take(seqLen).ToList();
                
                var states = new AgentInput[seqLen];
                var actions = new int[seqLen];
                var logProbs = new float[seqLen];
                var advs = new float[seqLen];
                var rets = new float[seqLen];
                var oldVals = new float[seqLen];
                var tpMults = new float[seqLen];
                var slMults = new float[seqLen];
                var hindsightSlMults = new float[seqLen];
                var hadPositions = new bool[seqLen];
                var actualPriceChanges = new float[seqLen];
                
                for (var j = 0; j < seqLen; j++)
                {
                    var (exp, origIdx) = seqExps[j];
                    states[j] = exp.State;
                    actions[j] = exp.Action;
                    logProbs[j] = exp.LogProb;
                    advs[j] = advantages[origIdx];
                    rets[j] = returns[origIdx];
                    oldVals[j] = oldValues[origIdx];
                    tpMults[j] = exp.TpMultiplier;
                    slMults[j] = exp.SlMultiplier;
                    hindsightSlMults[j] = exp.HindsightSlMultiplier;
                    hadPositions[j] = exp.HadPosition;
                    actualPriceChanges[j] = exp.ActualPriceChange;
                }
                
                sequences.Add(new PpoSequence
                {
                    States = states,
                    Actions = actions,
                    LogProbs = logProbs,
                    Advantages = advs,
                    Returns = rets,
                    OldValues = oldVals,
                    TpMultipliers = tpMults,
                    SlMultipliers = slMults,
                    HindsightSlMultipliers = hindsightSlMults,
                    HadPositions = hadPositions,
                    ActualPriceChanges = actualPriceChanges,
                    EpisodeId = episodeGroup.Key
                });
            }
        }
        
        return sequences;
    }
    
    /// <summary>
    /// Total number of sequences
    /// </summary>
    public int Count => Sequences.Count;
    
    /// <summary>
    /// Total experiences across all sequences
    /// </summary>
    public int TotalExperiences => Sequences.Sum(s => s.Length);
}

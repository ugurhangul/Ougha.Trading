namespace Ougha.Trading.RL.Agents;

/// <summary>
/// A batch of sequences for LSTM training.
/// Each sequence maintains temporal order.
/// </summary>
public class PpoSequenceBatch
{
    public required PpoSequence[] Sequences { get; init; }
    
    /// <summary>
    /// Maximum sequence length in this batch (for padding if needed)
    /// </summary>
    public int MaxSequenceLength => Sequences.Max(s => s.Length);
    
    /// <summary>
    /// Number of sequences in this batch
    /// </summary>
    public int BatchSize => Sequences.Length;
}

/// <summary>
/// Data loader that yields batches of sequences for LSTM training.
/// Shuffles sequences but preserves temporal order within each sequence.
/// </summary>
internal class SequenceDataLoader
{
    private readonly SequentialPpoDataset _dataset;
    private readonly int _batchSize;
    private readonly bool _shuffleSequences;
    private readonly Random _rng = new();

    public SequenceDataLoader(SequentialPpoDataset dataset, int batchSize, bool shuffleSequences = true)
    {
        _dataset = dataset;
        _batchSize = batchSize;
        _shuffleSequences = shuffleSequences;
    }

    public IEnumerable<PpoSequenceBatch> GetBatches()
    {
        var indices = Enumerable.Range(0, _dataset.Count).ToArray();
        
        if (_shuffleSequences)
        {
            // Fisher-Yates shuffle of sequence order
            for (var i = indices.Length - 1; i > 0; i--)
            {
                var k = _rng.Next(i + 1);
                (indices[i], indices[k]) = (indices[k], indices[i]);
            }
        }
        
        // Yield batches of sequences
        for (var i = 0; i < indices.Length; i += _batchSize)
        {
            var batchLen = Math.Min(_batchSize, indices.Length - i);
            var sequences = new PpoSequence[batchLen];
            
            for (var j = 0; j < batchLen; j++)
            {
                sequences[j] = _dataset.Sequences[indices[i + j]];
            }
            
            yield return new PpoSequenceBatch { Sequences = sequences };
        }
    }
}

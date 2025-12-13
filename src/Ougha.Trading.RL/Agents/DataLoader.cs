namespace Ougha.Trading.RL.Agents;

internal class DataLoader(PpoDataset ds, int batch, bool shuffle)
{
    private readonly Random _rng = new();

    public IEnumerator<PpoBatch> GetEnumerator()
    {
        var n = ds.States.Length;
        var indices = Enumerable.Range(0, n).ToArray();
        
        if (shuffle)
        {
            for (var i = n - 1; i > 0; i--)
            {
                var k = _rng.Next(i + 1);
                (indices[i], indices[k]) = (indices[k], indices[i]);
            }
        }
        
        for (var i = 0; i < n; i += batch)
        {
             var len = Math.Min(batch, n - i);
             var batchIndices = new int[len];
             Array.Copy(indices, i, batchIndices, 0, len);

             var batch1 = new PpoBatch
             {
                 Indices = batchIndices,
                 States = new AgentInput[len],
                 Actions = new int[len],
                 LogProbs = new float[len],
                 Advantages = new float[len],
                 Returns = new float[len],
                 TpMultipliers = new float[len],
                 SlMultipliers = new float[len]
             };
             
             for (var j = 0; j < len; j++)
             {
                 var idx = batchIndices[j];
                 batch1.States[j] = ds.States[idx];
                 batch1.Actions[j] = ds.Actions[idx];
                 batch1.LogProbs[j] = ds.LogProbs[idx];
                 batch1.Advantages[j] = ds.Advantages[idx];
                 batch1.Returns[j] = ds.Returns[idx];
                 batch1.TpMultipliers[j] = ds.TpMultipliers[idx];
                 batch1.SlMultipliers[j] = ds.SlMultipliers[idx];
             }
             
             yield return batch1;
        }
    }
}

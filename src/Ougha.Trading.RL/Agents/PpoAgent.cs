using TorchSharp;
using TorchSharp.Modules;
using static TorchSharp.torch;
using Ougha.Trading.RL.Models;
using Ougha.Trading.RL.Training;

namespace Ougha.Trading.RL.Agents;

public class PpoAgent : IAgent
{
    private readonly ActorCriticModel _model;
    private readonly ActorCriticModel _inferenceNet;
    private readonly Adam _optimizer;
    private readonly Device _device;

    private readonly float _gamma;
    private readonly float _gaeLambda;
    private readonly float _clipEpsilon;
    private readonly float _valueCoef;
    private float _entropyCoef;
    private readonly float _minEntropyCoef;
    private readonly int _updateEpochs;
    private readonly int _batchSize;
    
    private List<Experience> _rolloutBuffer = [];
    private readonly Lock _bufferLock = new();
    private readonly int _rolloutHorizon;
    
    private float _lastLogProb;

    private const int MAX_INFERENCE_BATCH = 64;
    private const int WINDOW_SIZE = 20;
    private const int NUM_FEATURES = 45;
    private readonly long[] _symbolBuffer = new long[MAX_INFERENCE_BATCH];

    private readonly float[]? _packedTfBuffer;
    private readonly float[]? _packedFeatBuffer;
    
    // LR Scheduling
    private readonly float _baseLr;
    private int _updateCount;
    private int _totalExpectedUpdates;
    
    // KL Adaptive Clipping
    private const float TargetKl = 0.015f;
    private const float ClipMin = 0.1f;
    private const float ClipMax = 0.3f;
    
    // NaN tracking
    private int _nanCount;
    
    // Metrics
    private PpoMetrics _lastMetrics = new();
    
    public PpoAgent(int batchSize = 256,
        int rolloutHorizon = 4096,
        float gamma = 0.99f,
        float gaeLambda = 0.95f,
        float clipEpsilon = 0.2f,
        float learningRate = 3e-4f,
        int updateEpochs = 10,
        bool useCuda = false,
        int totalExpectedUpdates = 10000)
    {
        _batchSize = batchSize;
        _rolloutHorizon = rolloutHorizon;
        _gamma = gamma;
        _gaeLambda = gaeLambda;
        _clipEpsilon = clipEpsilon;
        _valueCoef = 0.5f;
        _entropyCoef = 0.1f;
        _minEntropyCoef = 0.01f;
        _updateEpochs = updateEpochs;
        
        // LR Scheduling
        _baseLr = learningRate;
        _totalExpectedUpdates = totalExpectedUpdates;

        var cudaAvailable = cuda.is_available();
        _device = useCuda && cudaAvailable ? CUDA : CPU;
        Console.WriteLine($"[PpoAgent] Device: {(useCuda && cudaAvailable ? "CUDA" : "CPU")} (useCuda={useCuda}, cudaAvailable={cudaAvailable})");
        
        _model = new ActorCriticModel("ppo_net_train");
        _model.to(_device);
        
        _inferenceNet = new ActorCriticModel("ppo_net_infer");
        _inferenceNet.to(_device);
        SyncInferenceNetwork();
        
        _optimizer = optim.Adam(_model.parameters(), lr: learningRate);

        _packedTfBuffer = new float[MAX_INFERENCE_BATCH * 5 * WINDOW_SIZE * NUM_FEATURES];
        _packedFeatBuffer = new float[MAX_INFERENCE_BATCH * 77];
    }

    public void SyncInferenceNetwork()
    {
        using (no_grad())
        {
             var stateDict = _model.state_dict();
             _inferenceNet.load_state_dict(stateDict);
        }
    }

    public int Act(AgentInput input, bool training = true)
    {
        _inferenceNet.eval();
        using (no_grad())
        {
            var tensors = PrepareInputTensors([input]);
            var (logits, _, _) = _inferenceNet.forward(tensors);
            
            var probs = nn.functional.softmax(logits, dim: 1);
            var dist = distributions.Categorical(probs);
            var action = dist.sample();
            
            _lastLogProb = dist.log_prob(action).item<float>();
            
            return (int)action.item<long>();
        }
    }
    
    public (int[] Actions, float[,] TpSlMultipliers) ActBatchWithTpSl(AgentInput[] inputs, bool training = true)
    {
        var (actions, tpSl, _) = ActBatchWithTpSlAndLogProbs(inputs, training);
        return (actions, tpSl);
    }

    public (int[] Actions, float[,] TpSlMultipliers, float[] LogProbs) ActBatchWithTpSlAndLogProbs(AgentInput[] inputs, bool training = true)
    {
        _inferenceNet.eval();
        using (no_grad())
        {
            var tensors = PrepareInputTensors(inputs);
            var (logits, _, tpSl) = _inferenceNet.forward(tensors);

            logits = nan_to_num(logits);

            var probs = nn.functional.softmax(logits, dim: 1);

            probs = clamp(probs, 1e-8f, 1.0f);
            probs = probs / probs.sum(1, keepdim: true);

            var dist = distributions.Categorical(probs);
            var actionsTensor = dist.sample();
            var logProbsTensor = dist.log_prob(actionsTensor);

            var actionsData = actionsTensor.data<long>().ToArray();
            var logProbs = logProbsTensor.data<float>().ToArray();
            var tpSlData = tpSl.data<float>().ToArray();

            var actions = new int[inputs.Length];
            for (var i = 0; i < inputs.Length; i++)
                actions[i] = (int)actionsData[i];
            
            var tpSlMults = new float[inputs.Length, 2];
            for (var i = 0; i < inputs.Length; i++)
            {
                tpSlMults[i, 0] = tpSlData[i * 2];
                tpSlMults[i, 1] = tpSlData[i * 2 + 1];
            }
            
            return (actions, tpSlMults, logProbs);
        }
    }

    public int[] ActBatch(AgentInput[] inputs, bool training = true)
    {
        var (actions, _) = ActBatchWithTpSl(inputs, training);
        return actions;
    }

    public void AddExperience(AgentInput state, int action, float reward, AgentInput? nextState, bool done)
    {
        AddExperienceWithLogProb(state, action, reward, nextState, done, _lastLogProb);
    }

    private void AddExperienceWithLogProb(AgentInput state, int action, float reward, AgentInput? nextState, bool done, float logProb)
    {
        lock (_bufferLock)
        {
            _rolloutBuffer.Add(new Experience
            {
                State = state,
                Action = action,
                Reward = reward,
                NextState = nextState,
                Done = done,
                Priority = logProb
            });
        }
    }

    public void AddExperienceBatch(
        AgentInput[] states,
        int[] actions,
        float[] rewards,
        AgentInput?[] nextStates,
        bool[] dones)
    {
        AddExperienceBatchWithLogProbs(states, actions, rewards, nextStates, dones, new float[states.Length]);
    }

    public void AddExperienceBatchWithLogProbs(
        AgentInput[] states,
        int[] actions,
        float[] rewards,
        AgentInput?[] nextStates,
        bool[] dones,
        float[] logProbs)
    {
        lock (_bufferLock)
        {
            for (var i = 0; i < states.Length; i++)
            {
                _rolloutBuffer.Add(new Experience
                {
                    State = states[i],
                    Action = actions[i],
                    Reward = rewards[i],
                    NextState = nextStates[i],
                    Done = dones[i],
                    Priority = logProbs[i]
                });
            }
        }
    }

    public float Train()
    {
        List<Experience> bufferToTrain;

        lock(_bufferLock)
        {
            if (_rolloutBuffer.Count < _rolloutHorizon)
                return 0f;
                
            bufferToTrain = _rolloutBuffer;
            _rolloutBuffer = new List<Experience>();
        }
            
        return UpdatePpo(bufferToTrain);
    }
    
    public float TrainStep() => Train();
    public float TrainMultipleBatches(int batches) => Train();

    private float UpdatePpo(List<Experience> rollouts)
    {
        var states = rollouts.Select(e => e.State).ToArray();

        var (advantages, returns) = ComputeGae(rollouts, states);

        var dataset = new PpoDataset(rollouts, advantages, returns);
        var loader = new DataLoader(dataset, _batchSize, shuffle: true);
        
        _model.train();
        float totalLoss = 0;
        var steps = 0;

        for (var epoch = 0; epoch < _updateEpochs; epoch++)
        {
            foreach (var batch in loader)
            {
                var stateTensors = PrepareInputTensors(batch.States);
                var actions = tensor(batch.Actions, dtype: ScalarType.Int64, device: _device);
                var oldLogProbs = tensor(batch.LogProbs, dtype: ScalarType.Float32, device: _device);
                var returnsTensor = tensor(batch.Returns, dtype: ScalarType.Float32, device: _device);
                var advs = tensor(batch.Advantages, dtype: ScalarType.Float32, device: _device);

                advs = (advs - advs.mean()) / (advs.std() + 1e-8f);
                
                var (logits, values, _) = _model.forward(stateTensors);

                var probs = nn.functional.softmax(logits, dim: 1);
                var dist = distributions.Categorical(probs);
                var newLogProbs = dist.log_prob(actions);
                var entropy = dist.entropy().mean();
                
                var ratio = (newLogProbs - oldLogProbs).exp();
                var surr1 = ratio * advs;
                var surr2 = clamp(ratio, 1.0f - _clipEpsilon, 1.0f + _clipEpsilon) * advs;
                var actorLoss = -min(surr1, surr2).mean();

                var valueLoss = nn.functional.mse_loss(values.squeeze(), returnsTensor);
                
                var loss = actorLoss + _valueCoef * valueLoss - _entropyCoef * entropy;
                
                _optimizer.zero_grad();
                loss.backward();
                nn.utils.clip_grad_norm_(_model.parameters(), 0.5f);
                _optimizer.step();
                
                totalLoss += loss.item<float>();
                steps++;
            }
        }

        SyncInferenceNetwork();
        return steps > 0 ? totalLoss / steps : 0;
    }

    private (float[] Advantages, float[] Returns) ComputeGae(List<Experience> rollouts, AgentInput[] states)
    {
        var T = rollouts.Count;
        var advantages = new float[T];
        var returns = new float[T];

        var values = new float[T + 1];
        var chunkSize = 256;
        
        using (no_grad())
        {
            _model.eval();
            for (var i = 0; i < T; i += chunkSize)
            {
                var len = Math.Min(chunkSize, T - i);
                var chunk = states.Skip(i).Take(len).ToArray();
                var tensors = PrepareInputTensors(chunk);
                var (_, v, _) = _model.forward(tensors);
                var vData = v.cpu().data<float>().ToArray();
                Array.Copy(vData, 0, values, i, len);
            }

            if (!rollouts[T - 1].Done && rollouts[T - 1].NextState != null)
            {
                var tensors = PrepareInputTensors([rollouts[T - 1].NextState!]);
                var (_, v, _) = _model.forward(tensors);
                values[T] = v.item<float>();
            }
            else
            {
                values[T] = 0;
            }
        }
        
        float gae = 0;
        for (var t = T - 1; t >= 0; t--)
        {
            var exp = rollouts[t];
            var delta = exp.Reward + _gamma * values[t + 1] * (exp.Done ? 0 : 1) - values[t];
            gae = delta + _gamma * _gaeLambda * (exp.Done ? 0 : 1) * gae;
            advantages[t] = gae;

            returns[t] = advantages[t] + values[t];
        }
        
        return (advantages, returns);
    }

    public void Save(string path) => _model.save(path);
    public void Load(string path) => _model.load(path);
    public void ResetOnlineLearning()
    {
        lock (_bufferLock)
        {
            _rolloutBuffer.Clear();
        }
    }
    public void DecayEpsilon()
    {
        if (!(_entropyCoef > _minEntropyCoef)) return;
        _entropyCoef *= 0.999f;
        _entropyCoef = Math.Max(_entropyCoef, _minEntropyCoef);
    }
    public float GetEntropyCoef() => _entropyCoef;

    private Tensor[] PrepareInputTensors(AgentInput[] inputs)
    {
        var batchSize = inputs.Length;
        var usePreallocated = batchSize <= MAX_INFERENCE_BATCH;
        
        var tensors = new Tensor[3];
        string[] tfNames = ["M1", "M5", "M15", "H1", "H4"];

        var tfTotalLen = batchSize * 5 * WINDOW_SIZE * NUM_FEATURES;
        var tfPackedBuffer = usePreallocated && _packedTfBuffer != null 
            ? _packedTfBuffer 
            : new float[tfTotalLen];
        
        for (var tfIdx = 0; tfIdx < 5; tfIdx++)
        {
            var tf = tfNames[tfIdx];
            var tfOffset = tfIdx * WINDOW_SIZE * NUM_FEATURES;

            for (var b = 0; b < batchSize; b++)
            {
                var batchOffset = b * 5 * WINDOW_SIZE * NUM_FEATURES + tfOffset;
                
                if (!inputs[b].TimeframeFeatures.TryGetValue(tf, out var tfData))
                {
                    for (var i = 0; i < WINDOW_SIZE * NUM_FEATURES; i++)
                        tfPackedBuffer[batchOffset + i] = 0f;
                    continue;
                }
                
                for (var row = 0; row < WINDOW_SIZE; row++)
                {
                    var rowOffset = batchOffset + row * NUM_FEATURES;
                    for (var col = 0; col < NUM_FEATURES; col++)
                    {
                        var val = tfData[row, col];
                        tfPackedBuffer[rowOffset + col] = float.IsFinite(val) ? val : 0f;
                    }
                }
            }
        }
        
        tensors[0] = tensor(tfPackedBuffer, new long[] { batchSize, 5, WINDOW_SIZE, NUM_FEATURES }, 
            dtype: ScalarType.Float32, device: _device);

        var symBuffer = usePreallocated ? _symbolBuffer : new long[batchSize];
        for (var i = 0; i < batchSize; i++) 
            symBuffer[i] = inputs[i].SymbolId;
        tensors[1] = tensor(symBuffer, new long[] { batchSize, 1 }, 
            dtype: ScalarType.Int64, device: _device);

        const int totalFeatures = 5 + 10 + 5 + 9 + 16 + 20 + 12;
        var featTotalLen = batchSize * totalFeatures;
        var featPackedBuffer = usePreallocated && _packedFeatBuffer != null 
            ? _packedFeatBuffer 
            : new float[featTotalLen];
        
        for (var b = 0; b < batchSize; b++)
        {
            var offset = b * totalFeatures;
            var inp = inputs[b];

            CopyFeatures(featPackedBuffer, offset, inp.TriggerContext, 5); offset += 5;
            CopyFeatures(featPackedBuffer, offset, inp.ConfluenceFeatures, 10); offset += 10;
            CopyFeatures(featPackedBuffer, offset, inp.PortfolioFeatures, 5); offset += 5;
            CopyFeatures(featPackedBuffer, offset, inp.RiskState, 9); offset += 9;
            CopyFeatures(featPackedBuffer, offset, inp.NewsFeatures ?? ZeroNews, 16); offset += 16;
            CopyFeatures(featPackedBuffer, offset, inp.CorrelationFeatures ?? ZeroCorrelation, 20); offset += 20;
            CopyFeatures(featPackedBuffer, offset, inp.PortfolioExposure ?? ZeroExposure, 12);
        }
        
        tensors[2] = tensor(featPackedBuffer, new long[] { batchSize, totalFeatures }, 
            dtype: ScalarType.Float32, device: _device);
             
        return tensors;
    }

    private static void CopyFeatures(float[] dest, int destOffset, float[] source, int len)
    {
        for (var i = 0; i < len; i++)
        {
            var val = source[i];
            dest[destOffset + i] = float.IsFinite(val) ? val : 0f;
        }
    }

    private static readonly float[] ZeroNews = new float[16];
    private static readonly float[] ZeroCorrelation = new float[20];
    private static readonly float[] ZeroExposure = new float[12];


    public void Dispose()
    {
        _model.Dispose();
        _inferenceNet.Dispose();
        _optimizer.Dispose();
    }
}

internal class PpoDataset
{
    public readonly AgentInput[] States;
    public readonly int[] Actions;
    public readonly float[] LogProbs;
    public readonly float[] Advantages;
    public readonly float[] Returns;
    
    public PpoDataset(List<Experience> rollouts, float[] advantages, float[] returns)
    {
        var n = rollouts.Count;
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
        
        for(var i=0; i<n; i+=batch)
        {
             var len = Math.Min(batch, n-i);
             var batchIndices = new int[len];
             Array.Copy(indices, i, batchIndices, 0, len);

             var batch1 = new PpoBatch
             {
                 States = new AgentInput[len],
                 Actions = new int[len],
                 LogProbs = new float[len],
                 Advantages = new float[len],
                 Returns = new float[len]
             };
             
             for(var j=0; j<len; j++)
             {
                 var idx = batchIndices[j];
                 batch1.States[j] = ds.States[idx];
                 batch1.Actions[j] = ds.Actions[idx];
                 batch1.LogProbs[j] = ds.LogProbs[idx];
                 batch1.Advantages[j] = ds.Advantages[idx];
                 batch1.Returns[j] = ds.Returns[idx];
             }
             
             yield return batch1;
        }
    }
}

internal struct PpoBatch {
    public AgentInput[] States;
    public int[] Actions;
    public float[] LogProbs;
    public float[] Advantages;
    public float[] Returns;
    public float[] OldValues;
}

/// <summary>
/// Training metrics from a PPO update
/// </summary>
public struct PpoMetrics
{
    public float PolicyLoss { get; init; }
    public float ValueLoss { get; init; }
    public float Entropy { get; init; }
    public float KlDivergence { get; init; }
    public float ClipFraction { get; init; }
    public float ClipEpsilon { get; init; }
}

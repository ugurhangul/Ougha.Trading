using System;
using System.Collections.Generic;
using System.Linq;
using TorchSharp;
using TorchSharp.Modules;
using static TorchSharp.torch;
using static TorchSharp.torch.nn;
using static TorchSharp.torch.optim;
using Ougha.Trading.RL.Models;
using Ougha.Trading.RL.Training;

namespace Ougha.Trading.RL.Agents;

public class PpoAgent : IAgent, IDisposable
{
    private readonly ActorCriticModel _model; // Training network
    private readonly ActorCriticModel _inferenceNet; // Inference network
    private readonly Adam _optimizer;
    private readonly Device _device;
    
    // PPO Hyperparameters
    private readonly float _gamma;
    private readonly float _gaeLambda;
    private readonly float _clipEpsilon;
    private readonly float _valueCoef;
    private float _entropyCoef;
    private readonly float _initialEntropyCoef;
    private readonly float _minEntropyCoef;
    private readonly int _updateEpochs;
    private readonly int _batchSize;
    
    private List<Experience> _rolloutBuffer = new();
    private readonly object _bufferLock = new();
    private readonly int _rolloutHorizon;
    
    private float _lastLogProb;
    
    // Pre-allocated buffers for 3-tensor packing (performance optimization)
    private const int MAX_INFERENCE_BATCH = 64; // Max batch size for inference (increased for GPU efficiency)
    private const int WINDOW_SIZE = 20;
    private const int NUM_FEATURES = 45;
    private readonly long[] _symbolBuffer = new long[MAX_INFERENCE_BATCH];
    
    // Packed buffers for efficient GPU transfer
    private float[]? _packedTfBuffer;
    private float[]? _packedFeatBuffer;
    
    public PpoAgent(
        int batchSize = 64,
        int rolloutHorizon = 2048,
        float gamma = 0.99f,
        float gaeLambda = 0.95f,
        float clipEpsilon = 0.2f,
        float learningRate = 3e-4f,
        int updateEpochs = 10,
        bool useCuda = false)
    {
        _batchSize = batchSize;
        _rolloutHorizon = rolloutHorizon;
        _gamma = gamma;
        _gaeLambda = gaeLambda;
        _clipEpsilon = clipEpsilon;
        _valueCoef = 0.5f;
        _entropyCoef = 0.1f;           // Doubled for more exploration
        _initialEntropyCoef = 0.1f;
        _minEntropyCoef = 0.01f;       // Higher floor for continued exploration
        _updateEpochs = updateEpochs;
        
        // CUDA setup with logging
        bool cudaAvailable = torch.cuda.is_available();
        _device = useCuda && cudaAvailable ? torch.CUDA : torch.CPU;
        Console.WriteLine($"[PpoAgent] Device: {(useCuda && cudaAvailable ? "CUDA" : "CPU")} (useCuda={useCuda}, cudaAvailable={cudaAvailable})");
        
        _model = new ActorCriticModel("ppo_net_train");
        _model.to(_device);
        
        _inferenceNet = new ActorCriticModel("ppo_net_infer");
        _inferenceNet.to(_device);
        SyncInferenceNetwork();
        
        _optimizer = torch.optim.Adam(_model.parameters(), lr: learningRate);
        
        // Initialize packed buffers for 3-tensor transfer
        _packedTfBuffer = new float[MAX_INFERENCE_BATCH * 5 * WINDOW_SIZE * NUM_FEATURES];
        _packedFeatBuffer = new float[MAX_INFERENCE_BATCH * 77]; // All features concatenated
    }

    public void SyncInferenceNetwork()
    {
        using (torch.no_grad())
        {
             var stateDict = _model.state_dict();
             _inferenceNet.load_state_dict(stateDict);
        }
    }

    public int Act(AgentInput input, bool training = true)
    {
        _inferenceNet.eval();
        using (torch.no_grad())
        {
            var tensors = PrepareInputTensors(new[] { input });
            var (logits, _, _) = _inferenceNet.forward(tensors);
            
            var probs = torch.nn.functional.softmax(logits, dim: 1);
            var dist = torch.distributions.Categorical(probs);
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
        using (torch.no_grad())
        {
            var tensors = PrepareInputTensors(inputs);
            var (logits, _, tpSl) = _inferenceNet.forward(tensors);
            
            // Replace NaN with 0 unconditionally (avoid GPU sync from .item<bool>())
            logits = torch.nan_to_num(logits, 0.0f);

            var probs = torch.nn.functional.softmax(logits, dim: 1);
            
            // Clamp probabilities to avoid NaN/negative without GPU sync
            probs = torch.clamp(probs, 1e-8f, 1.0f);
            probs = probs / probs.sum(1, keepdim: true);

            var dist = torch.distributions.Categorical(probs);
            var actionsTensor = dist.sample();
            var logProbsTensor = dist.log_prob(actionsTensor);
            
            // Batch all data extractions together
            var actionsData = actionsTensor.data<long>().ToArray();
            var logProbs = logProbsTensor.data<float>().ToArray();
            var tpSlData = tpSl.data<float>().ToArray();
            
            // Convert actions to int array
            var actions = new int[inputs.Length];
            for (int i = 0; i < inputs.Length; i++)
                actions[i] = (int)actionsData[i];
            
            var tpSlMults = new float[inputs.Length, 2];
            for (int i = 0; i < inputs.Length; i++)
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
                Priority = logProb // Store log prob in Priority field
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
            for (int i = 0; i < states.Length; i++)
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
        
        // Check buffer size and SWAP if ready
        lock(_bufferLock)
        {
            if (_rolloutBuffer.Count < _rolloutHorizon)
                return 0f;
                
            bufferToTrain = _rolloutBuffer;
            _rolloutBuffer = new List<Experience>(); // New empty buffer for Collector
        }
            
        return UpdatePPO(bufferToTrain);
    }
    
    public float TrainStep() => Train(); // Interface compat
    public float TrainMultipleBatches(int batches) => Train(); // Interface compat

    private float UpdatePPO(List<Experience> rollouts)
    {
        // 1. Compute GAE
        var states = rollouts.Select(e => e.State).ToArray();
        
        // Compute Values, Advantages and Returns
        var (advantages, returns) = ComputeGAE(rollouts, states);
        
        // Prepare training data
        var dataset = new PpoDataset(rollouts, advantages, returns);
        var loader = new DataLoader(dataset, _batchSize, shuffle: true);
        
        _model.train();
        float totalLoss = 0;
        int steps = 0;
        
        // 2. Multi-epoch update
        for (int epoch = 0; epoch < _updateEpochs; epoch++)
        {
            foreach (var batch in loader)
            {
                var stateTensors = PrepareInputTensors(batch.States);
                var actions = torch.tensor(batch.Actions, dtype: ScalarType.Int64, device: _device);
                var oldLogProbs = torch.tensor(batch.LogProbs, dtype: ScalarType.Float32, device: _device);
                var returnsTensor = torch.tensor(batch.Returns, dtype: ScalarType.Float32, device: _device);
                var advs = torch.tensor(batch.Advantages, dtype: ScalarType.Float32, device: _device);
                
                // Normalize advantages
                advs = (advs - advs.mean()) / (advs.std() + 1e-8f);
                
                var (logits, values, _) = _model.forward(stateTensors);
                
                // 3. Loss Calculation
                // Policy Loss
                var probs = torch.nn.functional.softmax(logits, dim: 1);
                var dist = torch.distributions.Categorical(probs);
                var newLogProbs = dist.log_prob(actions);
                var entropy = dist.entropy().mean();
                
                var ratio = (newLogProbs - oldLogProbs).exp();
                var surr1 = ratio * advs;
                var surr2 = torch.clamp(ratio, 1.0f - _clipEpsilon, 1.0f + _clipEpsilon) * advs;
                var actorLoss = -torch.min(surr1, surr2).mean();
                
                // Value Loss
                var valueLoss = torch.nn.functional.mse_loss(values.squeeze(), returnsTensor);
                
                var loss = actorLoss + _valueCoef * valueLoss - _entropyCoef * entropy;
                
                _optimizer.zero_grad();
                loss.backward();
                torch.nn.utils.clip_grad_norm_(_model.parameters(), 0.5f);
                _optimizer.step();
                
                totalLoss += loss.item<float>();
                steps++;
            }
        }
        
        // _rolloutBuffer.Clear(); // Already swapped
        SyncInferenceNetwork(); // Sync weights to inference net
        return steps > 0 ? totalLoss / steps : 0;
    }
    
    // Pass rollouts specifically to avoid using _rolloutBuffer field
    private (float[] Advantages, float[] Returns) ComputeGAE(List<Experience> rollouts, AgentInput[] states)
    {
        int T = rollouts.Count;
        var advantages = new float[T];
        var returns = new float[T];
        
        // Get Values for all states + last next state
        // Batching in chunks to avoid OOM
        var values = new float[T + 1];
        int chunkSize = 256;
        
        using (torch.no_grad())
        {
            _model.eval();
            // Get value for every step
            for (int i = 0; i < T; i += chunkSize)
            {
                int len = Math.Min(chunkSize, T - i);
                var chunk = states.Skip(i).Take(len).ToArray();
                var tensors = PrepareInputTensors(chunk);
                // ...
                var (_, v, _) = _model.forward(tensors);
                var vData = v.cpu().data<float>().ToArray();
                Array.Copy(vData, 0, values, i, len);
            }
            
            // Last value (bootstrapping if episode not done)
            if (!rollouts[T - 1].Done && rollouts[T - 1].NextState != null)
            {
                var tensors = PrepareInputTensors(new[] { rollouts[T - 1].NextState! });
                var (_, v, _) = _model.forward(tensors);
                values[T] = v.item<float>();
            }
            else
            {
                values[T] = 0;
            }
        }
        
        float gae = 0;
        for (int t = T - 1; t >= 0; t--)
        {
            var exp = rollouts[t];
            // Delta = r + gamma * V(s') * (1-d) - V(s)
            float delta = exp.Reward + _gamma * values[t + 1] * (exp.Done ? 0 : 1) - values[t];
            gae = delta + _gamma * _gaeLambda * (exp.Done ? 0 : 1) * gae;
            advantages[t] = gae;
            
            // Return = Advantage + Value (Target for Value Head)
            returns[t] = advantages[t] + values[t];
        }
        
        return (advantages, returns);
    }

    public void Save(string path) => _model.save(path);
    public void Load(string path) => _model.load(path);
    public void ResetOnlineLearning() { _rolloutBuffer.Clear(); }
    public void DecayEpsilon()
    {
        // PPO uses adaptive entropy decay instead of epsilon
        if (_entropyCoef > _minEntropyCoef)
        {
            _entropyCoef *= 0.999f; // Slower decay for longer exploration
            _entropyCoef = Math.Max(_entropyCoef, _minEntropyCoef);
        }
    }
    public int BufferCount => _rolloutBuffer.Count;
    public float GetEntropyCoef() => _entropyCoef;
    
    // OPTIMIZED: Pack into 3 tensors instead of 13 for faster GPU transfer
    // Returns: [PackedTimeframes, SymbolId, PackedFeatures]
    // PackedTimeframes: [batch, 5, 20, 45] - all 5 timeframes stacked
    // SymbolId: [batch, 1] - int64
    // PackedFeatures: [batch, 77] - all 7 feature arrays concatenated
    private Tensor[] PrepareInputTensors(AgentInput[] inputs)
    {
        int batchSize = inputs.Length;
        bool usePreallocated = batchSize <= MAX_INFERENCE_BATCH;
        
        var tensors = new Tensor[3]; // Only 3 tensors now!
        string[] tfNames = { "M1", "M5", "M15", "H1", "H4" };
        
        // 1. Pack all 5 timeframes into one 4D tensor [batch, 5, 20, 45]
        int tfTotalLen = batchSize * 5 * WINDOW_SIZE * NUM_FEATURES;
        float[] tfPackedBuffer = usePreallocated && _packedTfBuffer != null 
            ? _packedTfBuffer 
            : new float[tfTotalLen];
        
        for (int tfIdx = 0; tfIdx < 5; tfIdx++)
        {
            string tf = tfNames[tfIdx];
            int tfOffset = tfIdx * WINDOW_SIZE * NUM_FEATURES; // Offset within each batch item
            
            for (int b = 0; b < batchSize; b++)
            {
                int batchOffset = b * 5 * WINDOW_SIZE * NUM_FEATURES + tfOffset;
                
                if (!inputs[b].TimeframeFeatures.TryGetValue(tf, out var tfData))
                {
                    // Zero-fill for missing timeframe
                    for (int i = 0; i < WINDOW_SIZE * NUM_FEATURES; i++)
                        tfPackedBuffer[batchOffset + i] = 0f;
                    continue;
                }
                
                for (int row = 0; row < WINDOW_SIZE; row++)
                {
                    int rowOffset = batchOffset + row * NUM_FEATURES;
                    for (int col = 0; col < NUM_FEATURES; col++)
                    {
                        float val = tfData[row, col];
                        tfPackedBuffer[rowOffset + col] = float.IsFinite(val) ? val : 0f;
                    }
                }
            }
        }
        
        tensors[0] = torch.tensor(tfPackedBuffer, new long[] { batchSize, 5, WINDOW_SIZE, NUM_FEATURES }, 
            dtype: ScalarType.Float32, device: _device);

        // 2. Symbol ID tensor (unchanged, it's tiny)
        long[] symBuffer = usePreallocated ? _symbolBuffer : new long[batchSize];
        for (int i = 0; i < batchSize; i++) 
            symBuffer[i] = inputs[i].SymbolId;
        tensors[1] = torch.tensor(symBuffer, new long[] { batchSize, 1 }, 
            dtype: ScalarType.Int64, device: _device);
        
        // 3. Pack all 7 feature arrays into one [batch, 77] tensor
        // Layout: Trigger(5) + Confluence(10) + Portfolio(5) + Risk(9) + News(16) + Correlation(20) + Exposure(12) = 77
        const int TOTAL_FEATURES = 5 + 10 + 5 + 9 + 16 + 20 + 12; // 77
        int featTotalLen = batchSize * TOTAL_FEATURES;
        float[] featPackedBuffer = usePreallocated && _packedFeatBuffer != null 
            ? _packedFeatBuffer 
            : new float[featTotalLen];
        
        for (int b = 0; b < batchSize; b++)
        {
            int offset = b * TOTAL_FEATURES;
            var inp = inputs[b];
            
            // Copy each feature array in order
            CopyFeatures(featPackedBuffer, offset, inp.TriggerContext, 5); offset += 5;
            CopyFeatures(featPackedBuffer, offset, inp.ConfluenceFeatures, 10); offset += 10;
            CopyFeatures(featPackedBuffer, offset, inp.PortfolioFeatures, 5); offset += 5;
            CopyFeatures(featPackedBuffer, offset, inp.RiskState, 9); offset += 9;
            CopyFeatures(featPackedBuffer, offset, inp.NewsFeatures ?? _zeroNews, 16); offset += 16;
            CopyFeatures(featPackedBuffer, offset, inp.CorrelationFeatures ?? _zeroCorrelation, 20); offset += 20;
            CopyFeatures(featPackedBuffer, offset, inp.PortfolioExposure ?? _zeroExposure, 12);
        }
        
        tensors[2] = torch.tensor(featPackedBuffer, new long[] { batchSize, TOTAL_FEATURES }, 
            dtype: ScalarType.Float32, device: _device);
             
        return tensors;
    }
    
    // Fast copy helper with NaN check
    private static void CopyFeatures(float[] dest, int destOffset, float[] source, int len)
    {
        for (int i = 0; i < len; i++)
        {
            float val = source[i];
            dest[destOffset + i] = float.IsFinite(val) ? val : 0f;
        }
    }
    
    // Zero arrays for optional features
    private static readonly float[] _zeroNews = new float[16];
    private static readonly float[] _zeroCorrelation = new float[20];
    private static readonly float[] _zeroExposure = new float[12];

    
    private Tensor BatchFloatArrayOptimized(AgentInput[] inputs, int batchSize, float[]? prealloc, 
        Func<AgentInput, float[]> selector, int dim)
    {
        int len = batchSize * dim;
        float[] flat = prealloc ?? new float[len];
        
        for (int i = 0; i < batchSize; i++)
        {
            var source = selector(inputs[i]);
            int offset = i * dim;
            
            // Use Buffer.BlockCopy for source arrays (4 bytes per float)
            Buffer.BlockCopy(source, 0, flat, offset * sizeof(float), dim * sizeof(float));
        }
        
        // NaN check pass (vectorizable by JIT)
        for (int i = 0; i < len; i++)
        {
            if (!float.IsFinite(flat[i]))
                flat[i] = 0f;
        }
        
        return torch.tensor(flat, new long[] { batchSize, dim }, dtype: ScalarType.Float32, device: _device);
    }

    public void Dispose()
    {
        _model.Dispose();
        _inferenceNet.Dispose();
        _optimizer.Dispose();
    }
}

// Simple Dataset/Loader helpers
class PpoDataset
{
    public AgentInput[] States;
    public int[] Actions;
    public float[] LogProbs;
    public float[] Advantages;
    public float[] Returns;
    
    public PpoDataset(List<Experience> rollouts, float[] advantages, float[] returns)
    {
        int n = rollouts.Count;
        States = new AgentInput[n];
        Actions = new int[n];
        LogProbs = new float[n];
        Advantages = advantages;
        Returns = returns;
        
        for(int i=0; i<n; i++)
        {
            States[i] = rollouts[i].State;
            Actions[i] = rollouts[i].Action;
            LogProbs[i] = rollouts[i].Priority; // Hijacked field
        }
    }
}

class DataLoader 
{
    private PpoDataset _ds;
    private int _batch;
    private bool _shuffle;
    private Random _rng = new Random();
    
    public DataLoader(PpoDataset ds, int batch, bool shuffle) { _ds = ds; _batch = batch; _shuffle = shuffle; }
    
    public IEnumerator<PpoBatch> GetEnumerator()
    {
        int n = _ds.States.Length;
        var indices = Enumerable.Range(0, n).ToArray();
        
        if (_shuffle)
        {
            // Fisher-Yates shuffle
            for (int i = n - 1; i > 0; i--)
            {
                int k = _rng.Next(i + 1);
                (indices[i], indices[k]) = (indices[k], indices[i]);
            }
        }
        
        for(int i=0; i<n; i+=_batch)
        {
             int len = Math.Min(_batch, n-i);
             var batchIndices = new int[len];
             Array.Copy(indices, i, batchIndices, 0, len);
             
             // Construct batch
             var batch = new PpoBatch
             {
                 States = new AgentInput[len],
                 Actions = new int[len],
                 LogProbs = new float[len],
                 Advantages = new float[len],
                 Returns = new float[len]
             };
             
             for(int j=0; j<len; j++)
             {
                 int idx = batchIndices[j];
                 batch.States[j] = _ds.States[idx];
                 batch.Actions[j] = _ds.Actions[idx];
                 batch.LogProbs[j] = _ds.LogProbs[idx];
                 batch.Advantages[j] = _ds.Advantages[idx];
                 batch.Returns[j] = _ds.Returns[idx];
             }
             
             yield return batch;
        }
    }
}

struct PpoBatch {
    public AgentInput[] States;
    public int[] Actions;
    public float[] LogProbs;
    public float[] Advantages;
    public float[] Returns;
}

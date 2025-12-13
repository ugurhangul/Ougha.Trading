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
    
    private readonly AsyncRolloutBuffer _rolloutBuffer;
    private readonly int _newsFeatureSize;
    private readonly float[] _zeroNews;
    
    private float _lastLogProb;

    private const int MAX_INFERENCE_BATCH = 64;
    private const int WINDOW_SIZE = 20;
    private const int NUM_FEATURES = 45;
    private const int GAE_CHUNK_SIZE = 1024; // 4x larger for fewer forward passes
    private readonly long[] _symbolBuffer = new long[MAX_INFERENCE_BATCH];

    private readonly float[]? _packedTfBuffer;
    private readonly float[]? _packedFeatBuffer;
    
    // Pre-allocated Experience array for buffering
    private readonly Experience[] _experienceBuffer = new Experience[MAX_INFERENCE_BATCH];
 
    public PpoAgent(int batchSize = 256,
        int rolloutHorizon = 8192,
        float gamma = 0.99f,
        float gaeLambda = 0.95f,
        float clipEpsilon = 0.2f,
        float learningRate = 1e-3f,
        bool useCuda = false,
        int newsFeatureSize = 17)
    {
        _batchSize = batchSize;
        _gamma = gamma;
        _gaeLambda = gaeLambda;
        _clipEpsilon = clipEpsilon;
        _valueCoef = 0.5f;
        _entropyCoef = 0.12f;  // Lower start for more exploitation with massive model
        _minEntropyCoef = 0.02f;  // Lower floor allows sharper final policy
        _updateEpochs = 8; // More epochs per rollout for larger model
        
        // LR Scheduling

        var cudaAvailable = cuda.is_available();
        _device = useCuda && cudaAvailable ? CUDA : CPU;
        Console.WriteLine($"[PpoAgent] Device: {(useCuda && cudaAvailable ? "CUDA" : "CPU")} (useCuda={useCuda}, cudaAvailable={cudaAvailable})");
        
        _model = new ActorCriticModel("ppo_net_train");
        _model.to(_device);
        
        _inferenceNet = new ActorCriticModel("ppo_net_infer");
        _inferenceNet.to(_device);
        SyncInferenceNetwork();
        
        _optimizer = optim.Adam(_model.parameters(), lr: learningRate);

        _rolloutBuffer = new AsyncRolloutBuffer(rolloutHorizon);

        _newsFeatureSize = newsFeatureSize;
        _zeroNews = new float[newsFeatureSize];
        
        // totalFeatures = 5 + 10 + 5 + 9 + newsFeatureSize + 20 + 12
        var totalFeatures = 5 + 10 + 5 + 9 + _newsFeatureSize + 20 + 12;
        
        _packedTfBuffer = new float[MAX_INFERENCE_BATCH * 5 * WINDOW_SIZE * NUM_FEATURES];
        _packedFeatBuffer = new float[MAX_INFERENCE_BATCH * totalFeatures];
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
        var count = inputs.Length;
        
        _inferenceNet.eval();
        using (no_grad())
        using (NewDisposeScope())
        {
            var tensors = PrepareInputTensors(inputs);
            var (logits, _, tpSl) = _inferenceNet.forward(tensors);

            // Sanitize logits to prevent NaN/Inf issues
            var sanitizedLogits = nan_to_num(logits, nan: 0.0, posinf: 10.0, neginf: -10.0);
            var clampedLogits = clamp(sanitizedLogits, -20.0f, 20.0f);

            // Use logits-based Categorical (more numerically stable than probs-based)
            var dist = distributions.Categorical(logits: clampedLogits);
            
            // Use sample() with explicit fallback for edge cases
            Tensor actionsTensor;
            try
            {
                actionsTensor = dist.sample();
            }
            catch
            {
                actionsTensor = clampedLogits.argmax(dim: 1);
            }
            var logProbsTensor = dist.log_prob(actionsTensor);

            // Copy data to managed arrays before tensors are disposed
            var actions = new int[count];
            var tpSlMults = new float[count, 2];
            var logProbs = new float[count];
            
            var actionsData = actionsTensor.cpu().data<long>().ToArray();
            var logProbsData = logProbsTensor.cpu().data<float>().ToArray();
            var tpSlData = tpSl.cpu().data<float>().ToArray();
            
            for (var i = 0; i < count; i++)
            {
                actions[i] = (int)actionsData[i];
                logProbs[i] = logProbsData[i];
                tpSlMults[i, 0] = tpSlData[i * 2];
                tpSlMults[i, 1] = tpSlData[i * 2 + 1];
            }
            
            // Explicitly dispose tensors
            foreach (var t in tensors) t.Dispose();
            
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
        _rolloutBuffer.AddExperience(new Experience
        {
            State = state,
            Action = action,
            Reward = reward,
            NextState = nextState,
            Done = done,
            Priority = logProb
        });
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
        var count = states.Length;
        var usePreallocated = count <= MAX_INFERENCE_BATCH;
        
        // Use pre-allocated buffer when batch size allows
        var experiences = usePreallocated ? _experienceBuffer : new Experience[count];
        
        for (var i = 0; i < count; i++)
        {
            experiences[i] = new Experience
            {
                State = states[i],
                Action = actions[i],
                Reward = rewards[i],
                NextState = nextStates[i],
                Done = dones[i],
                Priority = logProbs[i]
            };
        }
        
        // Pass a span/slice if using pre-allocated buffer with smaller batch
        if (usePreallocated && count < MAX_INFERENCE_BATCH)
        {
            _rolloutBuffer.AddExperienceBatch(new ArraySegment<Experience>(experiences, 0, count));
        }
        else
        {
            _rolloutBuffer.AddExperienceBatch(experiences);
        }
    }

    public float Train()
    {
        // Check if a rollout is ready (non-blocking)
        var rollout = _rolloutBuffer.TryGetRollout();
        if (rollout == null)
            return 0f;
            
        return UpdatePpo(rollout);
    }
    
    public float TrainStep() => Train();
    public float TrainMultipleBatches(int batches) => Train();

    private float UpdatePpo(Experience[] rollouts)
    {
        var T = rollouts.Length;
        var states = new AgentInput[T];
        for (var i = 0; i < T; i++)
            states[i] = rollouts[i].State;

        var (advantages, returns) = ComputeGae(rollouts, states);

        var dataset = new PpoDataset(rollouts, advantages, returns);
        var loader = new DataLoader(dataset, _batchSize, shuffle: true);
        
        _model.train();
        float totalLoss = 0;
        var steps = 0;
        
        // Use DisposeScope for all tensors to prevent heap corruption
        using (NewDisposeScope())
        {
            // Pre-compute all tensors once to avoid repeated creation during epochs
            var allStateTensors = PrepareInputTensors(dataset.States);
            var allActions = tensor(dataset.Actions, dtype: ScalarType.Int64, device: _device);
            var allLogProbs = tensor(dataset.LogProbs, dtype: ScalarType.Float32, device: _device);
            var allReturns = tensor(dataset.Returns, dtype: ScalarType.Float32, device: _device);
            var allAdvantages = tensor(dataset.Advantages, dtype: ScalarType.Float32, device: _device);

            for (var epoch = 0; epoch < _updateEpochs; epoch++)
            {
                foreach (var batch in loader)
                {
                    // Use inner scope for batch tensors to free memory after each batch
                    using (NewDisposeScope())
                    {
                        var batchIndices = tensor(batch.Indices, dtype: ScalarType.Int64, device: _device);
                        
                        var stateTensors = new Tensor[allStateTensors.Length];
                        for (var t = 0; t < allStateTensors.Length; t++)
                            stateTensors[t] = allStateTensors[t].index_select(0, batchIndices);
                        
                        var actions = allActions.index_select(0, batchIndices);
                        var oldLogProbs = allLogProbs.index_select(0, batchIndices);
                        var returnsTensor = allReturns.index_select(0, batchIndices);
                        var advs = allAdvantages.index_select(0, batchIndices);

                        var normalizedAdvs = (advs - advs.mean()) / (advs.std() + 1e-8f);
                        
                        var (logits, values, _) = _model.forward(stateTensors);

                        var probs = nn.functional.softmax(logits, dim: 1);
                        var dist = distributions.Categorical(probs);
                        var newLogProbs = dist.log_prob(actions);
                        var entropy = dist.entropy().mean();
                        
                        var ratio = (newLogProbs - oldLogProbs).exp();
                        var surr1 = ratio * normalizedAdvs;
                        var surr2 = clamp(ratio, 1.0f - _clipEpsilon, 1.0f + _clipEpsilon) * normalizedAdvs;
                        var actorLoss = -min(surr1, surr2).mean();

                        var valueLoss = nn.functional.mse_loss(values.squeeze(), returnsTensor);
                        
                        var loss = actorLoss + _valueCoef * valueLoss - _entropyCoef * entropy;
                        
                        _optimizer.zero_grad();
                        loss.backward();
                        nn.utils.clip_grad_norm_(_model.parameters(), 0.5f);
                        _optimizer.step();
                        
                        totalLoss += loss.item<float>();
                        steps++;
                    } // batchScope disposes all batch tensors
                }
            }
            
            // Explicitly dispose input tensors
            foreach (var t in allStateTensors) t.Dispose();
        } // outerScope disposes any remaining tensors

        SyncInferenceNetwork();
        return steps > 0 ? totalLoss / steps : 0;
    }

    private (float[] Advantages, float[] Returns) ComputeGae(Experience[] rollouts, AgentInput[] states)
    {
        var T = rollouts.Length;
        var advantages = new float[T];
        var returns = new float[T];

        var values = new float[T + 1];
        
        using (no_grad())
        {
            _model.eval();
            
            // Pre-allocate chunk array to avoid repeated allocations
            var chunkBuffer = new AgentInput[GAE_CHUNK_SIZE];
            
            for (var i = 0; i < T; i += GAE_CHUNK_SIZE)
            {
                // Use DisposeScope per chunk to free tensors after each iteration
                using (NewDisposeScope())
                {
                    var len = Math.Min(GAE_CHUNK_SIZE, T - i);
                    
                    // Use Array.Copy instead of LINQ Skip/Take to avoid allocations
                    Array.Copy(states, i, chunkBuffer, 0, len);
                    
                    // Create view of correct length (avoid processing garbage in buffer)
                    AgentInput[] chunk;
                    if (len == GAE_CHUNK_SIZE)
                    {
                        chunk = chunkBuffer;
                    }
                    else
                    {
                        chunk = new AgentInput[len];
                        Array.Copy(chunkBuffer, chunk, len);
                    }
                    
                    var tensors = PrepareInputTensors(chunk);
                    var (_, v, _) = _model.forward(tensors);
                    // Squeeze on GPU first, then transfer once
                    var vSqueezeData = v.squeeze().cpu().data<float>().ToArray();
                    Array.Copy(vSqueezeData, 0, values, i, len);
                    
                    // Explicitly dispose input tensors
                    foreach (var t in tensors) t.Dispose();
                }
            }

            if (!rollouts[T - 1].Done && rollouts[T - 1].NextState != null)
            {
                using (NewDisposeScope())
                {
                    var tensors = PrepareInputTensors([rollouts[T - 1].NextState!]);
                    var (_, v, _) = _model.forward(tensors);
                    values[T] = v.item<float>();
                    foreach (var t in tensors) t.Dispose();
                }
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
    
    public bool HasPendingRollout() => _rolloutBuffer.HasReadyRollout();
    public int GetPendingRolloutsCount() => _rolloutBuffer.PendingRolloutsCount;
    public int GetActiveBufferCount() => _rolloutBuffer.ActiveBufferCount;
    
    public void ResetOnlineLearning()
    {
        _rolloutBuffer.Reset();
    }
    public void DecayEpsilon()
    {
        if (!(_entropyCoef > _minEntropyCoef)) return;
        _entropyCoef *= 0.9999f;  // Very slow decay for large model stability
        _entropyCoef = Math.Max(_entropyCoef, _minEntropyCoef);
    }
    
    /// <summary>
    /// Reset entropy coefficient to boost exploration when policy is collapsing.
    /// </summary>
    public void ResetEntropy(float? newValue = null)
    {
        _entropyCoef = newValue ?? 0.15f;  // Reset to high value
        // Note: Console output removed - it corrupts the Spectre.Console Live display
    }
    
    /// <summary>
    /// Check action distribution and reset entropy if too skewed.
    /// Call this periodically (e.g., every 100 episodes) with action counts.
    /// </summary>
    /// <param name="actionCounts">Array of action counts [Hold, Buy1, Buy2, Buy3, Sell1, Sell2, Sell3, Close]</param>
    /// <param name="skewThreshold">Max allowed percentage for any single action (0.0-1.0)</param>
    /// <returns>True if entropy was reset</returns>
    public bool CheckAndResetEntropy(int[] actionCounts, float skewThreshold = 0.70f)
    {
        if (actionCounts == null || actionCounts.Length == 0) return false;
        
        var total = actionCounts.Sum();
        if (total == 0) return false;
        
        // Check if any action dominates
        var maxPct = (float)actionCounts.Max() / total;
        
        // Also check Buy vs Sell imbalance
        var buyCount = actionCounts[1];
        var sellCount = actionCounts[2];
        var tradeCount = buyCount + sellCount;
        
        // If one action > threshold OR extreme buy/sell imbalance, reset entropy
        var needsReset = maxPct > skewThreshold;
        if (tradeCount > 100)
        {
            var buyPct = (float)buyCount / tradeCount;
            var sellPct = (float)sellCount / tradeCount;
            // If buy or sell is < 15% of trades, that's too imbalanced
            if (buyPct < 0.15f || sellPct < 0.15f)
                needsReset = true;
        }
        
        if (needsReset)
        {
            // Reset entropy but not too high to avoid instability
            var newEntropy = Math.Max(_entropyCoef * 2f, 0.08f);
            newEntropy = Math.Min(newEntropy, 0.15f);
            ResetEntropy(newEntropy);
            return true;
        }
        
        return false;
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

        var totalFeatures = 5 + 10 + 5 + 9 + _newsFeatureSize + 20 + 12;
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
            CopyFeatures(featPackedBuffer, offset, inp.NewsFeatures ?? _zeroNews, _newsFeatureSize); offset += _newsFeatureSize;
            CopyFeatures(featPackedBuffer, offset, inp.CorrelationFeatures ?? ZeroCorrelation, 20); offset += 20;
            CopyFeatures(featPackedBuffer, offset, inp.PortfolioExposure ?? ZeroExposure, 12);
        }
        
        tensors[2] = tensor(featPackedBuffer, new long[] { batchSize, totalFeatures }, 
            dtype: ScalarType.Float32, device: _device);
             
        return tensors;
    }

    private static void CopyFeatures(float[] dest, int destOffset, float[] source, int len)
    {
        // Fast path: use Buffer.BlockCopy for bulk copy, then sanitize
        Buffer.BlockCopy(source, 0, dest, destOffset * sizeof(float), len * sizeof(float));
        
        // Sanitize NaN/Inf values in-place
        for (var i = 0; i < len; i++)
        {
            if (!float.IsFinite(dest[destOffset + i]))
                dest[destOffset + i] = 0f;
        }
    }

    private static readonly float[] ZeroCorrelation = new float[20];
    private static readonly float[] ZeroExposure = new float[12];


    public void Dispose()
    {
        _model.Dispose();
        _inferenceNet.Dispose();
        _optimizer.Dispose();
    }
}


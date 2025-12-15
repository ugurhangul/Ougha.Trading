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
    private readonly float _tpSlCoef;
    private float _entropyCoef;
    private readonly float _minEntropyCoef;
    private readonly int _updateEpochs;
    private readonly int _batchSize;
    
    private readonly AsyncRolloutBuffer _rolloutBuffer;
    private readonly int _newsFeatureSize;
    private readonly float[] _zeroNews;
    
    private float _lastLogProb;
    
    // LR scheduling
    private readonly float _initialLr;
    private int _totalUpdates;
    private readonly int _expectedTotalUpdates;
    
    // KL divergence early stopping threshold - relaxed from 0.015 to allow more learning
    private const float KL_TARGET = 0.02f;
    
    // Training metrics for diagnostics
    private float _lastValueLoss;
    private float _lastKlDivergence;
    private float _lastPolicyEntropy;  // Actual policy entropy from distribution (not the coefficient)

    private const int MAX_INFERENCE_BATCH = 64;
    private const int WINDOW_SIZE = 20;
    private const int NUM_FEATURES = 45;
    private const int GAE_CHUNK_SIZE = 1024; // 4x larger for fewer forward passes
    
    // Price prediction thresholds (consistent across all methods)
    private const float TRADE_THRESHOLD = 0.0005f;  // 0.05% minimum predicted move
    private const float EXPLORATION_RATE = 0.10f;   // 10% random trades to bootstrap supervised learning
    private readonly Random _random = new();
    private readonly long[] _symbolBuffer = new long[MAX_INFERENCE_BATCH];

    private readonly float[]? _packedTfBuffer;
    private readonly float[]? _packedFeatBuffer;
    
    // Pre-allocated Experience array for buffering
    private readonly Experience[] _experienceBuffer = new Experience[MAX_INFERENCE_BATCH];
    
    // Last predictions for accuracy tracking
    private float[] _lastPredictions = [];
    public float[] LastPredictions => _lastPredictions;
 
    public PpoAgent(int batchSize = 256,
        int rolloutHorizon = 8192,
        float gamma = 0.99f,
        float gaeLambda = 0.95f,
        float clipEpsilon = 0.2f,
        float learningRate = 1e-3f,
        bool useCuda = false,
        int newsFeatureSize = 17,
        bool debugMode = false)
    {
        _batchSize = batchSize;
        _gamma = gamma;
        _gaeLambda = gaeLambda;
        _clipEpsilon = clipEpsilon;
        _valueCoef = 0.5f;
        _tpSlCoef = 0.3f;  // Increased from 0.1 for better SL learning
        _entropyCoef = 0.05f;  // Reduced from 0.60 for policy convergence
        _minEntropyCoef = 0.01f;  // Reduced from 0.50
        _updateEpochs = 6;
        
        // LR Scheduling - linear decay over expected training
        _initialLr = learningRate;
        _totalUpdates = 0;
        _expectedTotalUpdates = 10000;  // Estimated total PPO updates

        var cudaAvailable = cuda.is_available();
        _device = useCuda && cudaAvailable ? CUDA : CPU;
        
        var modeStr = debugMode ? "DEBUG (small model ~600K params)" : "FULL (large model ~10M params)";
        Console.WriteLine($"[PpoAgent] Device: {(useCuda && cudaAvailable ? "CUDA" : "CPU")} | Mode: {modeStr}");
        
        _model = new ActorCriticModel("ppo_net_train", debugMode: debugMode);
        _model.to(_device);
        
        _inferenceNet = new ActorCriticModel("ppo_net_infer", debugMode: debugMode);
        _inferenceNet.to(_device);
        SyncInferenceNetwork();
        
        _optimizer = optim.Adam(_model.parameters(), lr: learningRate);

        _rolloutBuffer = new AsyncRolloutBuffer(rolloutHorizon);

        _newsFeatureSize = newsFeatureSize;
        _zeroNews = new float[newsFeatureSize];
        
        // totalFeatures = 5 + 10 + 5 + 9 + newsFeatureSize + 20 + 12 + 8 (DxyFeatures)
        var totalFeatures = 5 + 10 + 5 + 9 + _newsFeatureSize + 20 + 12 + 8;
        
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
        
        // Reset LSTM hidden states - cached states may be invalid after weight sync
        // The model's internal state dict changed, so old cached tensors could be stale
        _inferenceNet.ResetAllHiddenStates();
    }

    public int Act(AgentInput input, bool training = true)
    {
        // Epsilon-greedy exploration during training
        if (training && _random.NextDouble() < EXPLORATION_RATE)
        {
            _lastLogProb = -1.0f;  // Mark as exploration action
            return _random.Next(0, 4);  // Random action 0-3
        }
        
        _inferenceNet.eval();
        using (no_grad())
        {
            var tensors = PrepareInputTensors([input]);
            var (pricePred, _, slMult, closeSignal) = _inferenceNet.forward(tensors);
            
            // Derive action from price prediction
            var predValue = pricePred.item<float>();
            var closeValue = closeSignal.item<float>();
            
            int action;
            if (closeValue > 0.5f)
            {
                action = 3;  // CLOSE
            }
            else if (predValue > TRADE_THRESHOLD)
            {
                action = 1;  // BUY (predicted UP)
            }
            else if (predValue < -TRADE_THRESHOLD)
            {
                action = 2;  // SELL (predicted DOWN)
            }
            else
            {
                action = 0;  // HOLD
            }
            
            // Store prediction magnitude as "log prob" (higher = more confident)
            _lastLogProb = -Math.Abs(predValue);
            
            return action;
        }
    }
    
    public (int[] Actions, float[,] TpSlMultipliers) ActBatchWithTpSl(AgentInput[] inputs, bool training = true)
    {
        var (actions, tpSl, _) = ActBatchWithTpSlAndLogProbs(inputs, training);
        return (actions, tpSl);
    }

    public (int[] Actions, float[,] TpSlMultipliers, float[] LogProbs) ActBatchWithTpSlAndLogProbs(AgentInput[] inputs, bool training = true, bool[]? hasPositions = null)
    {
        var count = inputs.Length;
        
        var actions = new int[count];
        var tpSlMults = new float[count, 2];
        var logProbs = new float[count];
        
        _inferenceNet.eval();
        using (no_grad())
        using (NewDisposeScope())
        {
            var tensors = PrepareInputTensors(inputs);
            var (pricePred, _, slMult, closeSignal) = _inferenceNet.forward(tensors);

            // Extract predictions to CPU
            var predData = pricePred.cpu().data<float>().ToArray();
            var slData = slMult.cpu().data<float>().ToArray();
            var closeData = closeSignal.cpu().data<float>().ToArray();
            
            for (var i = 0; i < count; i++)
            {
                var predValue = predData[i];
                var closeValue = closeData[i];
                var hasPosition = hasPositions?[i] ?? false;
                
                int action;
                
                // Epsilon-greedy exploration during training
                if (training && _random.NextDouble() < EXPLORATION_RATE)
                {
                    action = _random.Next(0, 4);  // Random action
                    logProbs[i] = -1.0f;  // Mark as exploration
                }
                else
                {
                    // Derive action from prediction using consistent threshold
                    if (hasPosition && closeValue > 0.5f)
                    {
                        action = 3;  // CLOSE signal when holding
                    }
                    else if (predValue > TRADE_THRESHOLD)
                    {
                        action = 1;  // BUY (predicted UP)
                    }
                    else if (predValue < -TRADE_THRESHOLD)
                    {
                        action = 2;  // SELL (predicted DOWN)
                    }
                    else
                    {
                        action = 0;  // HOLD
                    }
                    
                    // Log prob based on prediction magnitude
                    logProbs[i] = -Math.Abs(predValue);
                }
                
                actions[i] = action;
                
                // TP multiplier: minimum 0.3 to ensure meaningful TP distance
                var tpMult = (float)Math.Max(0.3, Math.Min(1.0, Math.Abs(predData[i]) / 0.03));
                tpSlMults[i, 0] = tpMult;
                tpSlMults[i, 1] = Math.Max(0.3f, slData[i]);  // Minimum SL multiplier too
            }
            
            // Track prediction variance for diagnostics
            _lastPolicyEntropy = (float)predData.Select(p => Math.Abs(p)).Average();
            _lastPredictions = predData;  // Store for accuracy tracking
            
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
            LogProb = logProb
        });
    }

    public void AddExperienceBatch(
        AgentInput[] states,
        int[] actions,
        float[] rewards,
        AgentInput?[] nextStates,
        bool[] dones)
    {
        AddExperienceBatchWithLogProbs(states, actions, rewards, nextStates, dones, 
            new float[states.Length], new float[states.Length], new float[states.Length]);
    }

    public void AddExperienceBatchWithLogProbs(
        AgentInput[] states,
        int[] actions,
        float[] rewards,
        AgentInput?[] nextStates,
        bool[] dones,
        float[] logProbs,
        float[] tpMultipliers,
        float[] slMultipliers,
        float[]? hindsightSlMultipliers = null,
        bool[]? hadPositions = null,
        float[]? actualPriceChanges = null)
    {
        var count = states.Length;
        var usePreallocated = count <= MAX_INFERENCE_BATCH;
        
        // Use pre-allocated buffer when batch size allows
        var experiences = usePreallocated ? _experienceBuffer : new Experience[count];
        
        for (var i = 0; i < count; i++)
        {
            var seqIdx = _rolloutBuffer.GetNextSequenceIndex();
            experiences[i] = new Experience
            {
                State = states[i],
                Action = actions[i],
                Reward = rewards[i],
                NextState = nextStates[i],
                Done = dones[i],
                LogProb = logProbs[i],
                TpMultiplier = tpMultipliers[i],
                SlMultiplier = slMultipliers[i],
                HindsightSlMultiplier = hindsightSlMultipliers?[i] ?? -1f,
                HadPosition = hadPositions?[i] ?? false,
                ActualPriceChange = actualPriceChanges?[i] ?? -999f,
                EpisodeId = _rolloutBuffer.CurrentEpisodeId,
                SequenceIndex = seqIdx,
                SymbolIdx = states[i].SymbolId  // Use actual symbol ID, not loop index
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

    /// <summary>
    /// Signal start of a new episode for sequence tracking.
    /// Call this before collecting experiences for a new episode.
    /// </summary>
    public void StartNewEpisode()
    {
        _rolloutBuffer.StartNewEpisode();
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
        // Clear any cached LSTM hidden states from previous runs to prevent stale tensor errors
        _model.ResetAllHiddenStates();
        
        var T = rollouts.Length;
        var states = new AgentInput[T];
        for (var i = 0; i < T; i++)
            states[i] = rollouts[i].State;

        var (advantages, returns, oldValues) = ComputeGaeWithValues(rollouts, states);

        // Use sequence-based dataset for LSTM temporal coherence
        var seqDataset = new SequentialPpoDataset(rollouts, advantages, returns, oldValues, 
            sequenceLength: SequentialPpoDataset.DefaultSequenceLength);
        
        // Also keep original dataset for fallback if no valid sequences
        if (seqDataset.Count == 0)
        {
            // Fall back to original shuffled training if no sequences formed
            return UpdatePpoShuffled(rollouts, advantages, returns, oldValues);
        }
        
        // For sequences, we batch process the entire sequence at once using standard forward()
        // The key benefit is that experiences within a sequence are from the same episode,
        // maintaining temporal coherence without per-timestep overhead
        var seqLoader = new SequenceDataLoader(seqDataset, batchSize: 8, shuffleSequences: true);
        
        _model.train();
        float totalLoss = 0;
        var steps = 0;
        
        // Apply LR scheduling (linear decay)
        _totalUpdates++;
        var progress = Math.Min(1.0f, (float)_totalUpdates / _expectedTotalUpdates);
        var currentLr = _initialLr * (1.0f - 0.9f * progress);  // Decay to 10% of initial LR
        foreach (var paramGroup in _optimizer.ParamGroups)
        {
            paramGroup.LearningRate = currentLr;
        }
        
        using (NewDisposeScope())
        {
            for (var epoch = 0; epoch < _updateEpochs; epoch++)
            {
                foreach (var seqBatch in seqLoader.GetBatches())
                {
                    // Process sequences with LSTM state propagation for temporal coherence
                    foreach (var sequence in seqBatch.Sequences)
                    {
                        using (NewDisposeScope())
                        {
                            // Prepare tensors for all timesteps
                            var actions = tensor(sequence.Actions, dtype: ScalarType.Int64, device: _device);
                            var oldLogProbs = tensor(sequence.LogProbs, dtype: ScalarType.Float32, device: _device);
                            var returnsArr = tensor(sequence.Returns, dtype: ScalarType.Float32, device: _device);
                            var advantagesArr = tensor(sequence.Advantages, dtype: ScalarType.Float32, device: _device);
                            var seqOldValues = tensor(sequence.OldValues, dtype: ScalarType.Float32, device: _device);
                            // Note: TpMultipliers not currently used for learning (formula-based)
                            
                            // Normalize advantages for this sequence
                            var normalizedAdvs = (advantagesArr - advantagesArr.mean()) / (advantagesArr.std() + 1e-8f);
                            
                            // OPTIMIZED: Process entire sequence in ONE forward pass
                            var stateTensors = PrepareInputTensors(sequence.States);
                            
                            var (pricePred, valuesTensor, slPred, closePred) = _model.ForwardSequenceBatch(
                                stateTensors[0], stateTensors[1], stateTensors[2]);
                            
                            // Clean up input tensors
                            foreach (var st in stateTensors) st.Dispose();
                            
                            // ============================================
                            // SUPERVISED PREDICTION LOSS (new approach)
                            // ============================================
                            // Train prediction head on ACTUAL price changes, not advantage-based targets
                            // This decouples prediction learning from RL value estimation
                            var predSqueezed = pricePred.squeeze();
                            
                            // Get actual price changes from closed trades
                            var actualChanges = tensor(sequence.ActualPriceChanges, ScalarType.Float32, _device);
                            
                            // Mask: only train on experiences where a trade closed (sentinel = -999)
                            var hasActualData = (actualChanges > -900f).to_type(ScalarType.Float32);
                            
                            // Supervised MSE loss on actual price movements
                            var predError = (predSqueezed - actualChanges).pow(2) * hasActualData;
                            var predictionLoss = predError.sum() / (hasActualData.sum() + 1e-8f);

                            // Value loss (unchanged)
                            var valuesSqueezed = valuesTensor.squeeze();
                            var valueClipped = seqOldValues + clamp(valuesSqueezed - seqOldValues, -_clipEpsilon, _clipEpsilon);
                            var valueLoss1 = (valuesSqueezed - returnsArr).pow(2);
                            var valueLoss2 = (valueClipped - returnsArr).pow(2);
                            var valueLoss = 0.5f * max(valueLoss1, valueLoss2).mean();
                            
                            // SL loss: learn optimal SL multiplier from HINDSIGHT targets
                            // Use hindsight SL when available (>0), otherwise fallback to stored prediction
                            var hindsightSl = sequence.HindsightSlMultipliers;
                            var targetSl = new float[sequence.Length];
                            var hasHindsight = new float[sequence.Length];
                            for (var k = 0; k < sequence.Length; k++)
                            {
                                if (hindsightSl[k] > 0f)
                                {
                                    targetSl[k] = hindsightSl[k];
                                    hasHindsight[k] = 1f;
                                }
                                else
                                {
                                    targetSl[k] = sequence.SlMultipliers[k];
                                    hasHindsight[k] = 0f;
                                }
                            }
                            var targetSlTensor = tensor(targetSl, dtype: ScalarType.Float32, device: _device);
                            var hasHindsightTensor = tensor(hasHindsight, dtype: ScalarType.Float32, device: _device);
                            
                            var slSqueezed = slPred.squeeze();
                            var tradeMask = (actions != 0).to_type(ScalarType.Float32);
                            
                            // Weight SL loss: 3x weight for experiences with hindsight targets
                            var slWeights = hasHindsightTensor * 2f + 1f;  // 1.0 normal, 3.0 hindsight
                            var slError = (slSqueezed - targetSlTensor).pow(2) * tradeMask * slWeights;
                            var slLoss = slError.sum() / (tradeMask.sum() + 1e-8f);
                            
                            // Close signal loss: only train when agent HAD a position
                            var closeSqueezed = closePred.squeeze();
                            var closeTarget = (actions == 3).to_type(ScalarType.Float32);
                            var hadPosMask = tensor(sequence.HadPositions.Select(p => p ? 1f : 0f).ToArray(), ScalarType.Float32, _device);
                            var closeError = (closeSqueezed - closeTarget).pow(2) * hadPosMask;
                            var closeLoss = closeError.sum() / (hadPosMask.sum() + 1e-8f);
                            
                            // Exploration bonus: encourage diverse predictions
                            var predVariance = predSqueezed.var();
                            var explorationBonus = predVariance;  // Higher variance = more exploration
                            
                            var loss = predictionLoss + _valueCoef * valueLoss + _tpSlCoef * (slLoss + closeLoss) - _entropyCoef * explorationBonus;
                            
                            // Track metrics (predictionLoss used for training diagnostics)
                            _lastKlDivergence = predictionLoss.item<float>();  // Approximate: higher = worse
                            _lastValueLoss = valueLoss.item<float>();
                            
                            _optimizer.zero_grad();
                            loss.backward();
                            nn.utils.clip_grad_norm_(_model.parameters(), 0.5f);
                            _optimizer.step();
                            
                            totalLoss += loss.item<float>();
                            steps++;
                        }
                    }
                    
                    // Early stop epoch if KL divergence too high
                    if (_lastKlDivergence > KL_TARGET)
                        break;
                }
                
                // Early stop epochs if KL divergence too high
                if (_lastKlDivergence > KL_TARGET)
                    break;
            }
        }

        SyncInferenceNetwork();
        return steps > 0 ? totalLoss / steps : 0;
    }
    
    /// <summary>
    /// Fallback to original shuffled training when sequence-based training isn't possible.
    /// Used when experiences don't have episode tracking or sequences are too short.
    /// </summary>
    private float UpdatePpoShuffled(Experience[] rollouts, float[] advantages, float[] returns, float[] oldValues)
    {
        var dataset = new PpoDataset(rollouts, advantages, returns, oldValues);
        var loader = new DataLoader(dataset, _batchSize, shuffle: true);
        
        _model.train();
        float totalLoss = 0;
        var steps = 0;
        
        using (NewDisposeScope())
        {
            var allStateTensors = PrepareInputTensors(dataset.States);
            var allActions = tensor(dataset.Actions, dtype: ScalarType.Int64, device: _device);
            var allReturns = tensor(dataset.Returns, dtype: ScalarType.Float32, device: _device);
            var allAdvantages = tensor(dataset.Advantages, dtype: ScalarType.Float32, device: _device);
            var allSlMults = tensor(dataset.SlMultipliers, dtype: ScalarType.Float32, device: _device);
            var allHindsightSl = tensor(dataset.HindsightSlMultipliers, dtype: ScalarType.Float32, device: _device);
            var allHadPositions = tensor(dataset.HadPositions.Select(p => p ? 1f : 0f).ToArray(), ScalarType.Float32, _device);
            var allActualPriceChanges = tensor(dataset.ActualPriceChanges, dtype: ScalarType.Float32, device: _device);
            var allOldValues = tensor(dataset.OldValues, dtype: ScalarType.Float32, device: _device);

            for (var epoch = 0; epoch < _updateEpochs; epoch++)
            {
                foreach (var batch in loader)
                {
                    using (NewDisposeScope())
                    {
                        var batchIndices = tensor(batch.Indices, dtype: ScalarType.Int64, device: _device);
                        
                        var stateTensors = new Tensor[allStateTensors.Length];
                        for (var t = 0; t < allStateTensors.Length; t++)
                            stateTensors[t] = allStateTensors[t].index_select(0, batchIndices);
                        
                        var actions = allActions.index_select(0, batchIndices);
                        var returnsTensor = allReturns.index_select(0, batchIndices);
                        var advs = allAdvantages.index_select(0, batchIndices);
                        var slMultsBatch = allSlMults.index_select(0, batchIndices);
                        var hindsightSlBatch = allHindsightSl.index_select(0, batchIndices);
                        var hadPosMask = allHadPositions.index_select(0, batchIndices);
                        var oldValuesBatch = allOldValues.index_select(0, batchIndices);

                        var normalizedAdvs = (advs - advs.mean()) / (advs.std() + 1e-8f);
                        
                        var (pricePred, values, slPred, closePred) = _model.forward(stateTensors);

                        // SUPERVISED PREDICTION LOSS (same as sequence-based)
                        var predSqueezed = pricePred.squeeze();
                        var actualChanges = allActualPriceChanges.index_select(0, batchIndices);
                        var hasActualData = (actualChanges > -900f).to_type(ScalarType.Float32);
                        var predError = (predSqueezed - actualChanges).pow(2) * hasActualData;
                        var predictionLoss = predError.sum() / (hasActualData.sum() + 1e-8f);

                        var valuesSqueezed = values.squeeze();
                        var valueClipped = oldValuesBatch + clamp(valuesSqueezed - oldValuesBatch, -_clipEpsilon, _clipEpsilon);
                        var valueLoss1 = (valuesSqueezed - returnsTensor).pow(2);
                        var valueLoss2 = (valueClipped - returnsTensor).pow(2);
                        var valueLoss = 0.5f * max(valueLoss1, valueLoss2).mean();
                        
                        // SL loss with HINDSIGHT targets (same as sequence-based)
                        var slSqueezed = slPred.squeeze();
                        var tradeMask = (actions != 0).to_type(ScalarType.Float32);
                        var hasHindsight = (hindsightSlBatch > 0).to_type(ScalarType.Float32);
                        var targetSl = where(hindsightSlBatch > 0, hindsightSlBatch, slMultsBatch);
                        var slWeights = hasHindsight * 2f + 1f;
                        var slError = (slSqueezed - targetSl).pow(2) * tradeMask * slWeights;
                        var slLoss = slError.sum() / (tradeMask.sum() + 1e-8f);
                        
                        // Close signal loss: only train when agent HAD a position
                        var closeSqueezed = closePred.squeeze();
                        var closeTarget = (actions == 3).to_type(ScalarType.Float32);
                        var closeError = (closeSqueezed - closeTarget).pow(2) * hadPosMask;
                        var closeLoss = closeError.sum() / (hadPosMask.sum() + 1e-8f);
                        
                        // Exploration bonus
                        var predVariance = predSqueezed.var();
                        
                        var loss = predictionLoss + _valueCoef * valueLoss + _tpSlCoef * (slLoss + closeLoss) - _entropyCoef * predVariance;
                        
                        _lastKlDivergence = predictionLoss.item<float>();
                        _lastValueLoss = valueLoss.item<float>();
                        
                        _optimizer.zero_grad();
                        loss.backward();
                        nn.utils.clip_grad_norm_(_model.parameters(), 0.5f);
                        _optimizer.step();
                        
                        totalLoss += loss.item<float>();
                        steps++;
                    }
                }
            }
            
            foreach (var t in allStateTensors) t.Dispose();
        }

        SyncInferenceNetwork();
        return steps > 0 ? totalLoss / steps : 0;
    }

    private (float[] Advantages, float[] Returns, float[] OldValues) ComputeGaeWithValues(Experience[] rollouts, AgentInput[] states)
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
                    var (_, v, _, _) = _model.forward(tensors);
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
                    var (_, v, _, _) = _model.forward(tensors);
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
        
        // Extract old values (first T elements) for value clipping
        var oldValues = new float[T];
        Array.Copy(values, oldValues, T);
        
        return (advantages, returns, oldValues);
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
    
    /// <summary>
    /// Force flush the buffer and train on current experiences.
    /// Call this at the end of each episode to train every episode.
    /// </summary>
    /// <returns>Average loss from training, or 0 if no experiences to train on</returns>
    public async Task<float> TrainEpisodeAsync()
    {
        // Force flush any pending experiences to make them available for training
        await _rolloutBuffer.FlushAsync();
        
        // Now try to train on the flushed rollout
        var rollout = _rolloutBuffer.TryGetRollout();
        if (rollout == null || rollout.Length == 0)
            return 0f;
            
        return UpdatePpo(rollout);
    }
    
    /// <summary>
    /// Synchronous version of TrainEpisodeAsync for simpler usage.
    /// </summary>
    public float TrainEpisode()
    {
        return TrainEpisodeAsync().GetAwaiter().GetResult();
    }
    public void DecayEpsilon()
    {
        if (!(_entropyCoef > _minEntropyCoef)) return;
        _entropyCoef *= 0.999995f;  // 10x slower decay (was 0.99995)
        _entropyCoef = Math.Max(_entropyCoef, _minEntropyCoef);
    }
    
    /// <summary>
    /// Reset entropy coefficient to boost exploration when policy is collapsing.
    /// </summary>
    public void ResetEntropy(float? newValue = null)
    {
        _entropyCoef = newValue ?? 0.60f;  // Reset to high value (raised from 0.30)
        // Note: Console output removed - it corrupts the Spectre.Console Live display
    }
    
    /// <summary>
    /// Check action distribution and reset entropy if too skewed.
    /// Call this periodically (e.g., every 100 episodes) with action counts.
    /// </summary>
    /// <param name="actionCounts">Array of action counts [Hold, Buy, Sell, Close] (4 actions)</param>
    /// <param name="skewThreshold">Max allowed percentage for any single action (0.0-1.0)</param>
    /// <returns>True if entropy was reset</returns>
    public bool CheckAndResetEntropy(int[] actionCounts, float skewThreshold = 0.45f)  // Aggressive - intervene when any action > 45%
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
            // Reset entropy aggressively to break HOLD dominance
            var newEntropy = Math.Max(_entropyCoef * 2f, 0.20f);
            newEntropy = Math.Min(newEntropy, 0.30f);
            ResetEntropy(newEntropy);
            return true;
        }
        
        return false;
    }
    /// <summary>
    /// Get the entropy coefficient (hyperparameter that weights entropy in the loss).
    /// This is NOT the actual policy entropy - use GetPolicyEntropy() for that.
    /// </summary>
    public float GetEntropyCoefficient() => _entropyCoef;
    
    /// <summary>
    /// Get the actual policy entropy from the last action distribution.
    /// Higher values = more exploration (max ~1.39 for 4 actions).
    /// </summary>
    public float GetPolicyEntropy() => _lastPolicyEntropy;
    
    // Keep old name for backwards compatibility (deprecated)
    [Obsolete("Use GetEntropyCoefficient() instead")]
    public float GetEntropyCoef() => _entropyCoef;
    
    public float GetValueLoss() => _lastValueLoss;
    public float GetKlDivergence() => _lastKlDivergence;

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

        // Fixed feature sizes (model expects these dimensions)
        const int TriggerLen = 5;
        const int ConfluenceLen = 10;
        const int PortfolioLen = 5;
        const int RiskLen = 9;
        const int CorrelationLen = 20;
        const int ExposureLen = 12;
        const int DxyLen = 8;
        
        var totalFeatures = TriggerLen + ConfluenceLen + PortfolioLen + RiskLen + _newsFeatureSize + CorrelationLen + ExposureLen + DxyLen;
        var featTotalLen = batchSize * totalFeatures;
        var featPackedBuffer = usePreallocated && _packedFeatBuffer != null 
            ? _packedFeatBuffer 
            : new float[featTotalLen];
        
        for (var b = 0; b < batchSize; b++)
        {
            var offset = b * totalFeatures;
            var inp = inputs[b];

            // Use Math.Min to handle cases where actual array is smaller or larger than expected
            CopyFeaturesWithPadding(featPackedBuffer, offset, inp.TriggerContext, TriggerLen); offset += TriggerLen;
            CopyFeaturesWithPadding(featPackedBuffer, offset, inp.ConfluenceFeatures, ConfluenceLen); offset += ConfluenceLen;
            CopyFeaturesWithPadding(featPackedBuffer, offset, inp.PortfolioFeatures, PortfolioLen); offset += PortfolioLen;
            CopyFeaturesWithPadding(featPackedBuffer, offset, inp.RiskState, RiskLen); offset += RiskLen;
            CopyFeaturesWithPadding(featPackedBuffer, offset, inp.NewsFeatures ?? _zeroNews, _newsFeatureSize); offset += _newsFeatureSize;
            CopyFeaturesWithPadding(featPackedBuffer, offset, inp.CorrelationFeatures ?? ZeroCorrelation, CorrelationLen); offset += CorrelationLen;
            CopyFeaturesWithPadding(featPackedBuffer, offset, inp.PortfolioExposure ?? ZeroExposure, ExposureLen); offset += ExposureLen;
            CopyFeaturesWithPadding(featPackedBuffer, offset, inp.DxyFeatures ?? ZeroDxy, DxyLen);
        }
        
        tensors[2] = tensor(featPackedBuffer, new long[] { batchSize, totalFeatures }, 
            dtype: ScalarType.Float32, device: _device);
             
        return tensors;
    }

    /// <summary>
    /// Copy features with automatic padding/truncation to match expected length.
    /// If source is smaller than expectedLen, remaining slots are zeroed.
    /// If source is larger, only first expectedLen elements are copied.
    /// </summary>
    private static void CopyFeaturesWithPadding(float[] dest, int destOffset, float[] source, int expectedLen)
    {
        var copyLen = Math.Min(source.Length, expectedLen);
        
        // Copy available data
        if (copyLen > 0)
        {
            Buffer.BlockCopy(source, 0, dest, destOffset * sizeof(float), copyLen * sizeof(float));
        }
        
        // Zero-pad remaining slots if source is shorter than expected
        for (var i = copyLen; i < expectedLen; i++)
        {
            dest[destOffset + i] = 0f;
        }
        
        // Sanitize NaN/Inf values in-place
        for (var i = 0; i < expectedLen; i++)
        {
            if (!float.IsFinite(dest[destOffset + i]))
                dest[destOffset + i] = 0f;
        }
    }

    private static readonly float[] ZeroCorrelation = new float[20];
    private static readonly float[] ZeroExposure = new float[12];
    private static readonly float[] ZeroDxy = new float[8];
    private static readonly float[] ZeroTime = new float[4];


    public void Dispose()
    {
        _model.Dispose();
        _inferenceNet.Dispose();
        _optimizer.Dispose();
    }
}


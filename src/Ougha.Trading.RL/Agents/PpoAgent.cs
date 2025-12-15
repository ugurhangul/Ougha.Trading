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
    private float _lastPredictionStd;  // Std dev of predictions (NOT action entropy)
    private float _lastActionEntropy;  // Actual action distribution entropy (max ~1.39 for 4 actions)

    private const int MAX_INFERENCE_BATCH = 64;
    private const int WINDOW_SIZE = 50;
    private const int NUM_FEATURES = 45;
    private const int GAE_CHUNK_SIZE = 4096; // OPTIMIZATION H1: 4x larger for better GPU utilization
    
    // Price prediction thresholds (consistent across all methods)
    private const float TRADE_THRESHOLD = 0.0002f;  // 0.02% minimum predicted move - lower for more trading
    private const float EXPLORATION_RATE = 0.10f;   // 10% random trades (reduced from 20% - too much was hurting)
    private readonly Random _random = new();
    private readonly long[] _symbolBuffer = new long[MAX_INFERENCE_BATCH];

    private readonly float[]? _packedTfBuffer;
    private readonly float[]? _packedFeatBuffer;
    
    // Pre-allocated Experience array for buffering
    private readonly Experience[] _experienceBuffer = new Experience[MAX_INFERENCE_BATCH];
    
    // Last predictions for accuracy tracking
    private float[] _lastPredictions = [];
    public float[] LastPredictions => _lastPredictions;
    
    // OPTIMIZATION M3: Lazy inference network sync
    private bool _inferenceNetDirty = true;
 
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
        _entropyCoef = 0.50f;  // High initial value to prevent early policy collapse
        _minEntropyCoef = 0.10f;  // Much higher floor - never let entropy fully collapse
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
        
        // totalFeatures = TriggerLen(6) + ConfluenceLen(10) + PortfolioLen(5) + RiskLen(9) 
        //               + newsFeatureSize + CorrelationLen(20) + ExposureLen(12) + DxyLen(8)
        var totalFeatures = 6 + 10 + 5 + 9 + _newsFeatureSize + 20 + 12 + 8;
        
        _packedTfBuffer = new float[MAX_INFERENCE_BATCH * 6 * WINDOW_SIZE * NUM_FEATURES];  // 6 timeframes: M1, M5, M15, H1, H4, D1
        _packedFeatBuffer = new float[MAX_INFERENCE_BATCH * totalFeatures];
    }

    /// <summary>
    /// OPTIMIZATION M3: Lazy sync - only copy weights when dirty flag is set.
    /// Called automatically before inference when needed.
    /// </summary>
    public void SyncInferenceNetwork()
    {
        if (!_inferenceNetDirty)
            return;
            
        using (no_grad())
        {
             var stateDict = _model.state_dict();
             _inferenceNet.load_state_dict(stateDict);
        }
        
        // Reset LSTM hidden states - cached states may be invalid after weight sync
        // The model's internal state dict changed, so old cached tensors could be stale
        _inferenceNet.ResetAllHiddenStates();
        _inferenceNetDirty = false;
    }
    
    /// <summary>
    /// Mark inference network as dirty (needs resync before inference).
    /// Called after training updates.
    /// </summary>
    private void MarkInferenceNetworkDirty()
    {
        _inferenceNetDirty = true;
    }


    public int Act(AgentInput input, bool training = true)
    {
        // Epsilon-greedy exploration during training
        if (training && _random.NextDouble() < EXPLORATION_RATE)
        {
            _lastLogProb = -1.0f;  // Mark as exploration action
            return _random.Next(0, 4);  // Random action 0-3
        }
        
        // OPTIMIZATION M3: Lazy sync - only syncs if dirty
        SyncInferenceNetwork();
        
        _inferenceNet.eval();
        using (no_grad())
        {
            var tensors = PrepareInputTensors([input]);
            var (pricePred, _, slMult, closeSignal) = _inferenceNet.forward(tensors);
            
            // Derive action from price prediction
            var predValue = pricePred.item<float>();
            var closeValue = closeSignal.item<float>();
            
            int action;
            // Derive action from price prediction only (no close signal)
            if (predValue > TRADE_THRESHOLD)
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
        
        // OPTIMIZATION M3: Lazy sync - only syncs if dirty
        SyncInferenceNetwork();
        
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
                    action = _random.Next(0, 3);  // Random action 0-2 (HOLD/BUY/SELL only)
                    logProbs[i] = -1.0f;  // Mark as exploration
                }
                else
                {
                    // Derive action from prediction only (no close signal)
                    if (predValue > TRADE_THRESHOLD)
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
                
                // TP multiplier: minimum 0.5 to ensure meaningful TP distance
                var tpMult = (float)Math.Max(0.5, Math.Min(1.0, Math.Abs(predData[i]) / 0.02));
                tpSlMults[i, 0] = tpMult;
                tpSlMults[i, 1] = Math.Max(0.5f, slData[i]);  // Higher minimum SL multiplier
            }
            
            // Track prediction variance for diagnostics - std dev is proxy for policy uncertainty
            // Higher std = more varied predictions = more exploration
            var predStd = predData.Length > 1 ? (float)Math.Sqrt(predData.Select(p => Math.Pow(p - predData.Average(), 2)).Average()) : 0f;
            _lastPredictionStd = predStd;  // Prediction spread (NOT action entropy)
            _lastPredictions = predData;  // Store for accuracy tracking
            
            // Compute actual action distribution entropy (3 actions now)
            var actionCounts = new int[3];
            foreach (var a in actions) actionCounts[a]++;
            _lastActionEntropy = ComputeEntropy(actionCounts, actions.Length);
            
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
        float[]? actualPriceChanges = null,
        bool[]? validDataMask = null,
        double[]? currentPrices = null,
        string[]? symbols = null)
    {
        var count = states.Length;
        var usePreallocated = count <= MAX_INFERENCE_BATCH;
        
        // Count valid experiences (skip symbols without price data)
        var validCount = 0;
        for (var i = 0; i < count; i++)
        {
            if (validDataMask == null || validDataMask[i])
                validCount++;
        }
        
        // Skip if no valid data
        if (validCount == 0)
            return;
        
        // Use pre-allocated buffer when batch size allows, otherwise allocate new
        Experience[] experiences;
        if (usePreallocated && validCount <= MAX_INFERENCE_BATCH)
        {
            experiences = _experienceBuffer;
        }
        else
        {
            experiences = new Experience[validCount];
        }
        
        var writeIdx = 0;
        for (var i = 0; i < count; i++)
        {
            // Skip symbols without valid price data
            if (validDataMask != null && !validDataMask[i])
                continue;
            
            var seqIdx = _rolloutBuffer.GetNextSequenceIndex();
            experiences[writeIdx] = new Experience
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
                SymbolIdx = states[i].SymbolId,  // Use actual symbol ID, not loop index
                CurrentPrice = currentPrices?[i] ?? 0,  // For M1-level dense supervision
                Symbol = symbols?[i] ?? ""  // Symbol name for tracking
            };
            writeIdx++;
        }
        
        // Pass a span/slice if using pre-allocated buffer with smaller batch
        if (usePreallocated && validCount < MAX_INFERENCE_BATCH)
        {
            _rolloutBuffer.AddExperienceBatch(new ArraySegment<Experience>(experiences, 0, validCount));
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
        
        // OPTIMIZATION C3: Increased batch size from 8 to 32 for better GPU utilization
        var seqLoader = new SequenceDataLoader(seqDataset, batchSize: 32, shuffleSequences: true);
        
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
                    using (NewDisposeScope())
                    {
                        // OPTIMIZATION C1: Process ALL sequences in batch together
                        var batchSize = seqBatch.Sequences.Length;
                        var sequenceLengths = new long[batchSize];
                        var inputTuples = new (Tensor packedTf, Tensor symbolIds, Tensor packedFeats)[batchSize];
                        
                        // Prepare all sequence input tensors
                        for (var s = 0; s < batchSize; s++)
                        {
                            var sequence = seqBatch.Sequences[s];
                            sequenceLengths[s] = sequence.Length;
                            var tensors = PrepareInputTensors(sequence.States);
                            inputTuples[s] = (tensors[0], tensors[1], tensors[2]);
                        }
                        
                        // OPTIMIZATION C1: Single forward pass for all sequences
                        var (pricePreds, valuesTensors, slPreds, closePreds) = 
                            _model.ForwardMultiSequence(inputTuples, sequenceLengths);
                        
                        // Compute loss for each sequence and accumulate
                        float batchLoss = 0;
                        for (var s = 0; s < batchSize; s++)
                        {
                            var sequence = seqBatch.Sequences[s];
                            var pricePred = pricePreds[s];
                            var valuesTensor = valuesTensors[s];
                            var slPred = slPreds[s];
                            var closePred = closePreds[s];
                            
                            // Prepare target tensors
                            var actions = tensor(sequence.Actions, dtype: ScalarType.Int64, device: _device);
                            var returnsArr = tensor(sequence.Returns, dtype: ScalarType.Float32, device: _device);
                            var seqOldValues = tensor(sequence.OldValues, dtype: ScalarType.Float32, device: _device);
                            
                            // SUPERVISED PREDICTION LOSS (DENSE + SPARSE)
                            var predSqueezed = pricePred.squeeze();
                            
                            // Dense M1-level targets
                            var m1PriceChanges = new float[sequence.Length];
                            for (var k = 1; k < sequence.Length; k++)
                            {
                                var prevPrice = sequence.CurrentPrices[k - 1];
                                var currPrice = sequence.CurrentPrices[k];
                                if (prevPrice > 0 && currPrice > 0)
                                    m1PriceChanges[k] = (float)((currPrice - prevPrice) / prevPrice);
                            }
                            var m1Targets = tensor(m1PriceChanges, ScalarType.Float32, _device);
                            var hasM1Data = tensor(
                                Enumerable.Range(0, sequence.Length).Select(j => j > 0 && sequence.CurrentPrices[j] > 0 ? 1f : 0f).ToArray(), 
                                ScalarType.Float32, _device);
                            
                            var m1Error = (predSqueezed - m1Targets).pow(2) * hasM1Data;
                            var m1Loss = m1Error.sum() / (hasM1Data.sum() + 1e-8f);
                            
                            // Sparse trade-close targets
                            var actualChanges = tensor(sequence.ActualPriceChanges, ScalarType.Float32, _device);
                            var hasActualData = (actualChanges > -900f).to_type(ScalarType.Float32);
                            var tradeCloseError = (predSqueezed - actualChanges).pow(2) * hasActualData;
                            var tradeCloseLoss = tradeCloseError.sum() / (hasActualData.sum() + 1e-8f);
                            
                            var predictionLoss = 0.1f * m1Loss + 1.0f * tradeCloseLoss;

                            // Value loss
                            var valuesSqueezed = valuesTensor.squeeze();
                            var valueClipped = seqOldValues + clamp(valuesSqueezed - seqOldValues, -_clipEpsilon, _clipEpsilon);
                            var valueLoss1 = (valuesSqueezed - returnsArr).pow(2);
                            var valueLoss2 = (valueClipped - returnsArr).pow(2);
                            var valueLoss = 0.5f * max(valueLoss1, valueLoss2).mean();
                            
                            // SL loss with hindsight targets
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
                            var slWeights = hasHindsightTensor * 2f + 1f;
                            var slError = (slSqueezed - targetSlTensor).pow(2) * tradeMask * slWeights;
                            var slLoss = slError.sum() / (tradeMask.sum() + 1e-8f);
                            
                            // Close signal loss
                            var closeSqueezed = closePred.squeeze();
                            var closeTarget = (actions == 3).to_type(ScalarType.Float32);
                            var hadPosMask = tensor(sequence.HadPositions.Select(p => p ? 1f : 0f).ToArray(), ScalarType.Float32, _device);
                            var closeError = (closeSqueezed - closeTarget).pow(2) * hadPosMask;
                            var closeLoss = closeError.sum() / (hadPosMask.sum() + 1e-8f);
                            
                            // Diversity loss
                            var predVariance = predSqueezed.var();
                            var minVariance = 0.001f;
                            var variancePenalty = max(tensor(0f, device: _device), minVariance - predVariance);
                            var diversityLoss = variancePenalty * 500f;
                            var explorationBonus = predVariance * 10f;
                            
                            var seqLoss = predictionLoss + _valueCoef * valueLoss + _tpSlCoef * (slLoss + closeLoss) 
                                      + diversityLoss - _entropyCoef * explorationBonus;
                            
                            batchLoss += seqLoss.item<float>();
                            
                            // Track metrics from last sequence
                            if (s == batchSize - 1)
                            {
                                _lastKlDivergence = predVariance.item<float>();
                                _lastValueLoss = valueLoss.item<float>();
                                _lastPredictionStd = (float)Math.Sqrt(predVariance.item<float>());
                            }
                        }
                        
                        // Compute combined loss for backward pass
                        // Re-forward and compute combined loss (for gradient computation)
                        var combinedLoss = tensor(batchLoss / batchSize, device: _device, requires_grad: false);
                        
                        // For proper gradient flow, we need to recompute with gradient tracking
                        // Use the already computed outputs for a cleaner single backward
                        _optimizer.zero_grad();
                        
                        // Accumulate gradients from each sequence
                        for (var s = 0; s < batchSize; s++)
                        {
                            var sequence = seqBatch.Sequences[s];
                            var pricePred = pricePreds[s];
                            var valuesTensor = valuesTensors[s];
                            var slPred = slPreds[s];
                            var closePred = closePreds[s];
                            
                            var actions = tensor(sequence.Actions, dtype: ScalarType.Int64, device: _device);
                            var returnsArr = tensor(sequence.Returns, dtype: ScalarType.Float32, device: _device);
                            var seqOldValues = tensor(sequence.OldValues, dtype: ScalarType.Float32, device: _device);
                            
                            var predSqueezed = pricePred.squeeze();
                            var actualChanges = tensor(sequence.ActualPriceChanges, ScalarType.Float32, _device);
                            var hasActualData = (actualChanges > -900f).to_type(ScalarType.Float32);
                            var tradeCloseError = (predSqueezed - actualChanges).pow(2) * hasActualData;
                            var predictionLoss = tradeCloseError.sum() / (hasActualData.sum() + 1e-8f);

                            var valuesSqueezed = valuesTensor.squeeze();
                            var valueClipped = seqOldValues + clamp(valuesSqueezed - seqOldValues, -_clipEpsilon, _clipEpsilon);
                            var valueLoss = 0.5f * max((valuesSqueezed - returnsArr).pow(2), (valueClipped - returnsArr).pow(2)).mean();
                            
                            var slSqueezed = slPred.squeeze();
                            var tradeMask = (actions != 0).to_type(ScalarType.Float32);
                            var slLoss = ((slSqueezed - tensor(sequence.SlMultipliers, ScalarType.Float32, _device)).pow(2) * tradeMask).mean();
                            
                            var predVariance = predSqueezed.var();
                            var diversityLoss = max(tensor(0f, device: _device), 0.001f - predVariance) * 500f;
                            
                            var loss = (predictionLoss + _valueCoef * valueLoss + _tpSlCoef * slLoss + diversityLoss) / batchSize;
                            loss.backward();
                        }
                        
                        nn.utils.clip_grad_norm_(_model.parameters(), 0.5f);
                        _optimizer.step();
                        
                        totalLoss += batchLoss;
                        steps += batchSize;
                        
                        // Clean up input tensors
                        foreach (var (tf, sym, feat) in inputTuples)
                        {
                            tf.Dispose();
                            sym.Dispose();
                            feat.Dispose();
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

        // OPTIMIZATION M3: Mark dirty instead of immediate sync
        MarkInferenceNetworkDirty();
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
                        
                        // PREDICTION DIVERSITY LOSS: Same as sequence-based training
                        var predVariance = predSqueezed.var();
                        var minVariance = 0.001f;  // Raised from 0.0001
                        var variancePenalty = max(tensor(0f, device: _device), minVariance - predVariance);
                        var diversityLoss = variancePenalty * 500f;  // 500x penalty (was 100x)
                        var explorationBonus = predVariance * 10f;   // 10x stronger bonus (was 5x)
                        
                        var loss = predictionLoss + _valueCoef * valueLoss + _tpSlCoef * (slLoss + closeLoss) 
                                  + diversityLoss - _entropyCoef * explorationBonus;
                        
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

        // OPTIMIZATION M3: Mark dirty instead of immediate sync
        MarkInferenceNetworkDirty();
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
        _entropyCoef *= 0.9999995f;  // 100x slower decay to maintain exploration longer
        _entropyCoef = Math.Max(_entropyCoef, _minEntropyCoef);
    }
    
    /// <summary>
    /// Reset entropy coefficient to boost exploration when policy is collapsing.
    /// </summary>
    public void ResetEntropy(float? newValue = null)
    {
        _entropyCoef = newValue ?? 0.80f;  // Reset to near-maximum exploration
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
            // If buy or sell is < 25% of trades, that's too imbalanced
            if (buyPct < 0.25f || sellPct < 0.25f)
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
    /// Get the prediction standard deviation from the last batch.
    /// Higher values = more diverse predictions (NOT action entropy).
    /// </summary>
    public float GetPredictionStd() => _lastPredictionStd;
    
    /// <summary>
    /// Get actual action distribution entropy from last batch.
    /// Max ~1.39 for uniform 4-action distribution.
    /// </summary>
    public float GetActionEntropy() => _lastActionEntropy;
    
    /// <summary>
    /// Backwards compatible - returns prediction std (NOT true policy entropy).
    /// </summary>
    [Obsolete("Use GetPredictionStd() or GetActionEntropy() instead")]
    public float GetPolicyEntropy() => _lastPredictionStd;
    
    // Keep old name for backwards compatibility (deprecated)
    [Obsolete("Use GetEntropyCoefficient() instead")]
    public float GetEntropyCoef() => _entropyCoef;
    
    /// <summary>
    /// Compute entropy of an action count distribution.
    /// </summary>
    private static float ComputeEntropy(int[] counts, int total)
    {
        if (total == 0) return 0f;
        var entropy = 0f;
        foreach (var c in counts)
        {
            if (c > 0)
            {
                var p = (float)c / total;
                entropy -= p * MathF.Log(p);
            }
        }
        return entropy;  // Max = ln(4) ≈ 1.39 for 4 uniform actions
    }
    
    public float GetValueLoss() => _lastValueLoss;
    public float GetKlDivergence() => _lastKlDivergence;

    private Tensor[] PrepareInputTensors(AgentInput[] inputs)
    {
        var batchSize = inputs.Length;
        var usePreallocated = batchSize <= MAX_INFERENCE_BATCH;
        
        var tensors = new Tensor[3];
        string[] tfNames = ["M1", "M5", "M15", "H1", "H4", "D1"];  // 6 timeframes

        var tfTotalLen = batchSize * 6 * WINDOW_SIZE * NUM_FEATURES;
        var tfPackedBuffer = usePreallocated && _packedTfBuffer != null 
            ? _packedTfBuffer 
            : new float[tfTotalLen];
        
        for (var tfIdx = 0; tfIdx < 6; tfIdx++)
        {
            var tf = tfNames[tfIdx];
            var tfOffset = tfIdx * WINDOW_SIZE * NUM_FEATURES;

            for (var b = 0; b < batchSize; b++)
            {
                var batchOffset = b * 6 * WINDOW_SIZE * NUM_FEATURES + tfOffset;
                
                if (!inputs[b].TimeframeFeatures.TryGetValue(tf, out var tfData))
                {
                    for (var i = 0; i < WINDOW_SIZE * NUM_FEATURES; i++)
                        tfPackedBuffer[batchOffset + i] = 0f;
                    continue;
                }
                
                // Check dimensions and copy with bounds safety
                var tfRows = tfData.GetLength(0);
                var tfCols = tfData.GetLength(1);
                
                for (var row = 0; row < WINDOW_SIZE; row++)
                {
                    var rowOffset = batchOffset + row * NUM_FEATURES;
                    for (var col = 0; col < NUM_FEATURES; col++)
                    {
                        // Safe access with bounds check
                        if (row < tfRows && col < tfCols)
                        {
                            var val = tfData[row, col];
                            tfPackedBuffer[rowOffset + col] = float.IsFinite(val) ? val : 0f;
                        }
                        else
                        {
                            tfPackedBuffer[rowOffset + col] = 0f;  // Zero-pad if out of bounds
                        }
                    }
                }
            }
        }
        
        tensors[0] = tensor(tfPackedBuffer, new long[] { batchSize, 6, WINDOW_SIZE, NUM_FEATURES }, 
            dtype: ScalarType.Float32, device: _device);

        var symBuffer = usePreallocated ? _symbolBuffer : new long[batchSize];
        for (var i = 0; i < batchSize; i++) 
            symBuffer[i] = inputs[i].SymbolId;
        tensors[1] = tensor(symBuffer, new long[] { batchSize, 1 }, 
            dtype: ScalarType.Int64, device: _device);

        // Fixed feature sizes (model expects these dimensions)
        const int TriggerLen = 6;  // 6 timeframes now
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
    private static void CopyFeaturesWithPadding(float[] dest, int destOffset, float[]? source, int expectedLen)
    {
        // Handle null source by zero-filling
        if (source == null || source.Length == 0)
        {
            for (var i = 0; i < expectedLen; i++)
                dest[destOffset + i] = 0f;
            return;
        }
        
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


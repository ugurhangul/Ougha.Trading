using TorchSharp;
using TorchSharp.Modules;
using static TorchSharp.torch;
using static TorchSharp.torch.nn;
using Ougha.Trading.RL.Models;
using Ougha.Trading.RL.Training;

namespace Ougha.Trading.RL.Agents;

public class TorchAgent : IAgent
{
    private readonly DqnModel _policyNet;
    private readonly DqnModel _targetNet;
    private readonly DqnModel _inferenceNet;
    private readonly Adam _optimizer;
    private readonly PrioritizedReplayBuffer _buffer;
    private readonly Lock _bufferLock = new();

    private readonly int _batchSize;
    private readonly float _gamma;
    private float _epsilon;
    private readonly float _initialEpsilon;
    private readonly float _epsilonMin;
    private readonly float _epsilonDecay;
    private readonly int _targetUpdateFreq;
    private int _stepCount;

    private const float TAU = 0.005f;
    
    private readonly Device _device;

    private readonly long[] _trainActionBuffer;
    private readonly float[] _trainRewardBuffer;
    private readonly float[] _trainDoneBuffer;
    private readonly AgentInput[] _trainStateBuffer;
    private readonly AgentInput[] _trainNextStateBuffer;

    private const int TF_BUFFER_SIZE = 64 * 20 * 45;

    private readonly Dictionary<string, float[]> _trainTfBatchBuffers;
    private readonly Dictionary<string, float[]> _inferenceTfBatchBuffers;

    private readonly long[] _trainSymbolIdBuffer = new long[64];
    private readonly float[] _trainTriggerBuffer = new float[64 * 5];
    private readonly float[] _trainConfluenceBuffer = new float[64 * 10];
    private readonly float[] _trainPortfolioBuffer = new float[64 * 5];
    private readonly float[] _trainRiskBuffer = new float[64 * 9];
    private float[] _trainNewsBuffer;
    private readonly float[] _trainCorrelationBuffer = new float[64 * 20];
    private readonly float[] _trainExposureBuffer = new float[64 * 12];

    private readonly long[] _inferenceSymbolIdBuffer = new long[64];
    private readonly float[] _inferenceTriggerBuffer = new float[64 * 5];
    private readonly float[] _inferenceConfluenceBuffer = new float[64 * 10];
    private readonly float[] _inferencePortfolioBuffer = new float[64 * 5];
    private readonly float[] _inferenceRiskBuffer = new float[64 * 9];
    private float[] _inferenceNewsBuffer;
    private readonly float[] _inferenceCorrelationBuffer = new float[64 * 20];
    private readonly float[] _inferenceExposureBuffer = new float[64 * 12];
    
    private readonly int _newsFeatureSize;
    
    public TorchAgent(
        int batchSize = 64,
        float gamma = 0.99f,
        float epsilon = 1.0f,
        float epsilonMin = 0.01f,
        float epsilonDecay = 0.995f,
        int targetUpdateFreq = 1000,
        int bufferSize = 100000,
        bool useCuda = false,
        int newsFeatureSize = 17)
    {
        _batchSize = batchSize;
        _gamma = gamma;
        _epsilon = epsilon;
        _initialEpsilon = epsilon;
        _epsilonMin = epsilonMin;
        _epsilonDecay = epsilonDecay;
        _targetUpdateFreq = targetUpdateFreq;
        
        _device = useCuda && cuda.is_available() ? CUDA : CPU;

        _policyNet = new DqnModel("policy_net");
        _targetNet = new DqnModel("target_net");
        _inferenceNet = new DqnModel("inference_net");

        _policyNet.to(_device);
        _targetNet.to(_device);
        _inferenceNet.to(_device);

        UpdateTargetNetwork(hard: true);
        SyncInferenceNetwork();

        _optimizer = optim.Adam(_policyNet.parameters(), lr: 0.0005);

        _buffer = new PrioritizedReplayBuffer(
            capacity: bufferSize,
            alpha: 0.6f, betaStart: 0.4f, betaEnd: 1.0f, nSteps: 3, gamma: gamma
        );

        _trainActionBuffer = new long[batchSize];
        _trainRewardBuffer = new float[batchSize];
        _trainDoneBuffer = new float[batchSize];
        _trainStateBuffer = new AgentInput[batchSize];
        _trainNextStateBuffer = new AgentInput[batchSize];

        _trainTfBatchBuffers = CreateTfBuffers();
        _inferenceTfBatchBuffers = CreateTfBuffers();
        
        _newsFeatureSize = newsFeatureSize;
        _trainNewsBuffer = new float[64 * newsFeatureSize];
        _inferenceNewsBuffer = new float[64 * newsFeatureSize];
    }

    private Dictionary<string, float[]> CreateTfBuffers()
    {
        return new Dictionary<string, float[]>
        {
            ["M1"] = new float[TF_BUFFER_SIZE],
            ["M5"] = new float[TF_BUFFER_SIZE],
            ["M15"] = new float[TF_BUFFER_SIZE],
            ["H1"] = new float[TF_BUFFER_SIZE],
            ["H4"] = new float[TF_BUFFER_SIZE]
        };
    }
    
    /// <summary>
    /// Soft update of the target network using Polyak averaging.
    /// Target = τ * policy + (1 - τ) * target
    /// </summary>
    private void UpdateTargetNetwork(bool hard = false)
    {
        if (hard)
        {
            var stateDict = _policyNet.state_dict();
            _targetNet.load_state_dict(stateDict);
            return;
        }

        using (no_grad())
        {
            var targetParams = _targetNet.named_parameters().ToList();
            var policyParams = _policyNet.named_parameters().ToDictionary(p => p.name, p => p.parameter);
            
            foreach (var (name, targetParam) in targetParams)
            {
                if (policyParams.TryGetValue(name, out var policyParam))
                {
                    targetParam.mul_(1 - TAU).add_(policyParam, alpha: TAU);
                }
            }
        }
    }

    /// <summary>
    /// Syncs the Inference Network with the current Policy Network weights.
    /// Call this periodically to update the Actor behavior.
    /// </summary>
    public void SyncInferenceNetwork()
    {
        using (no_grad())
        {
             var stateDict = _policyNet.state_dict();
             _inferenceNet.load_state_dict(stateDict);
        }
    }
    
    private static readonly Random Random = new();
    private const int NUM_ACTIONS = ActionDecoder.NumActions;

    private float[,]? _lastTpSlMultipliers;
    public float[,]? LastTpSlMultipliers => _lastTpSlMultipliers;

    public int Act(AgentInput input, bool training = true)
    {
        if (training && Random.NextDouble() < _epsilon)
        {
            _lastTpSlMultipliers = new[,] { { (float)Random.NextDouble(), (float)Random.NextDouble() } };
            return Random.Next(NUM_ACTIONS);
        }

        _inferenceNet.eval();
        using (no_grad())
        {
            var tensors = PrepareInputTensors([input], useTrainingBuffers: false);
            var (qValues, tpSl) = _inferenceNet.forward(tensors);
            _lastTpSlMultipliers = new float[1, 2];
            var tpSlData = tpSl.data<float>().ToArray();
            _lastTpSlMultipliers[0, 0] = tpSlData[0];
            _lastTpSlMultipliers[0, 1] = tpSlData[1];
            return (int)qValues.argmax(1).item<long>();
        }
    }

    public (int[] Actions, float[,] TpSlMultipliers) ActBatchWithTpSl(AgentInput[] inputs, bool training = true)
    {
        var count = inputs.Length;
        var actions = new int[count];
        var tpSlMults = new float[count, 2];

        if (training)
        {
            for (var i = 0; i < count; i++)
            {
                if (Random.NextDouble() < _epsilon)
                {
                    actions[i] = Random.Next(NUM_ACTIONS);
                    tpSlMults[i, 0] = (float)Random.NextDouble();
                    tpSlMults[i, 1] = (float)Random.NextDouble();
                }
                else
                {
                    actions[i] = -1;
                }
            }

            if (Array.TrueForAll(actions, a => a >= 0))
            {
                _lastTpSlMultipliers = tpSlMults;
                return (actions, tpSlMults);
            }
        }

        _inferenceNet.eval();

        using (no_grad())
        {
            var tensors = PrepareInputTensors(inputs, useTrainingBuffers: false);
            var (qValues, tpSl) = _inferenceNet.forward(tensors);
            var argmax = qValues.argmax(1).data<long>().ToArray();
            var tpSlData = tpSl.data<float>().ToArray();

            for (var i = 0; i < count; i++)
            {
                if (actions[i] < 0)
                    actions[i] = (int)argmax[i];
                if (tpSlMults[i, 0] == 0 && tpSlMults[i, 1] == 0)
                {
                    tpSlMults[i, 0] = tpSlData[i * 2];
                    tpSlMults[i, 1] = tpSlData[i * 2 + 1];
                }
            }
        }

        _lastTpSlMultipliers = tpSlMults;
        return (actions, tpSlMults);
    }

    public int[] ActBatch(AgentInput[] inputs, bool training = true)
    {
        var (actions, _) = ActBatchWithTpSl(inputs, training);
        return actions;
    }

    private float TrainStep()
    {
        Experience[] batch;
        int[] indices;
        float[] weights;

        lock (_bufferLock)
        {
            if (_buffer.Count < _batchSize)
                return 0f;

            var result = _buffer.SampleWithPriority(_batchSize);
            batch = result.Samples;
            indices = result.Indices;
            weights = result.Weights;
        }

        for (var i = 0; i < _batchSize; i++)
        {
            _trainStateBuffer[i] = batch[i].State;
            _trainNextStateBuffer[i] = batch[i].NextState ?? batch[i].State;
            _trainActionBuffer[i] = batch[i].Action;
            _trainRewardBuffer[i] = batch[i].Reward;
            _trainDoneBuffer[i] = batch[i].Done ? 1f : 0f;
        }

        var stateTensors = PrepareInputTensors(_trainStateBuffer, useTrainingBuffers: true);

        var actions = tensor(_trainActionBuffer, dtype: ScalarType.Int64, device: _device).unsqueeze(1);
        var rewards = tensor(_trainRewardBuffer, dtype: ScalarType.Float32, device: _device);
        var dones = tensor(_trainDoneBuffer, dtype: ScalarType.Float32, device: _device);
        var isWeights = tensor(weights, dtype: ScalarType.Float32, device: _device);

        // Get Q-values AND TP/SL predictions (no longer discarding TP/SL)
        var (qValuesAll, tpSlPred) = _policyNet.forward(stateTensors);
        var qValues = qValuesAll.gather(1, actions).squeeze(1);

        Tensor targetQ;
        using (no_grad())
        {
            var nextStateTensors = PrepareInputTensors(_trainNextStateBuffer, useTrainingBuffers: true);

            var (policyNextQ, _) = _policyNet.forward(nextStateTensors);
            var bestActions = policyNextQ.argmax(1).unsqueeze(1);

            var (targetNextQ, _) = _targetNet.forward(nextStateTensors);
            var nextQPositions = targetNextQ.gather(1, bestActions).squeeze(1);

            var nextQValues = nextQPositions * (1 - dones);
            targetQ = rewards + _gamma * nextQValues;
        }

        var tdErrors = (qValues - targetQ).abs();
        var tdErrorsArray = tdErrors.detach().cpu().data<float>().ToArray();
        
        lock (_bufferLock)
        {
            _buffer.UpdatePriorities(indices, tdErrorsArray);
        }

        var elementWiseLoss = functional.smooth_l1_loss(qValues, targetQ, reduction: Reduction.None);
        var qLoss = (elementWiseLoss * isWeights).mean();

        // TP/SL loss: train on experiences with positive rewards (profitable actions)
        // Extract stored TP/SL targets from experiences
        var tpTargets = new float[_batchSize];
        var slTargets = new float[_batchSize];
        for (var i = 0; i < _batchSize; i++)
        {
            tpTargets[i] = batch[i].TpMultiplier;
            slTargets[i] = batch[i].SlMultiplier;
        }
        
        var targetTp = tensor(tpTargets, dtype: ScalarType.Float32, device: _device);
        var targetSl = tensor(slTargets, dtype: ScalarType.Float32, device: _device);
        var predTp = tpSlPred[TensorIndex.Colon, 0];
        var predSl = tpSlPred[TensorIndex.Colon, 1];
        
        // Weight by positive rewards and non-HOLD actions
        var actionsTensor = tensor(_trainActionBuffer, dtype: ScalarType.Int64, device: _device);
        var tradeMask = (actionsTensor != 0).to_type(ScalarType.Float32);
        var posRewardMask = (rewards > 0).to_type(ScalarType.Float32);
        var tpSlWeight = tradeMask * posRewardMask;
        
        var tpError = (predTp - targetTp).pow(2) * tpSlWeight;
        var slError = (predSl - targetSl).pow(2) * tpSlWeight;
        var tpSlLoss = (tpError.sum() + slError.sum()) / (tpSlWeight.sum() + 1e-8f);
        
        // Combined loss with TP/SL coefficient
        var loss = qLoss + 0.1f * tpSlLoss;

        _optimizer.zero_grad();
        loss.backward();
        nn.utils.clip_grad_norm_(_policyNet.parameters(), 1.0);
        _optimizer.step();

        _stepCount++;
        
        if (_stepCount % _targetUpdateFreq == 0) UpdateTargetNetwork();

        if (_stepCount % 100 == 0)
            SyncInferenceNetwork();

        return loss.item<float>();
    }
    
    public void Observe(AgentInput state, int action, float reward, AgentInput nextState, bool done)
    {
        AddExperience(state, action, reward, nextState, done);
    }

    public float Train()
    {
        return TrainStep();
    }
    
    /// <summary>
    /// Train multiple batches in a sequence to maximize GPU utilization.
    /// Returns average loss across all batches.
    /// </summary>
    public float TrainMultipleBatches(int numBatches)
    {
        int count;
        lock(_bufferLock) count = _buffer.Count;
        
        if (count < _batchSize)
            return 0f;
            
        var totalLoss = 0f;
        var successfulBatches = 0;
        
        for (var i = 0; i < numBatches; i++)
        {
            var loss = TrainStep();
            if (loss > 0)
            {
                totalLoss += loss;
                successfulBatches++;
            }
        }
        
        return successfulBatches > 0 ? totalLoss / successfulBatches : 0f;
    }

    public void Save(string path)
    {
        _policyNet.save(path);
    }

    public void Load(string path)
    {
        _policyNet.load(path);
        UpdateTargetNetwork(hard: true);
        SyncInferenceNetwork();
    }

    public void ResetOnlineLearning()
    {
        lock (_bufferLock)
        {
            _buffer.Clear();
        }
        _epsilon = _initialEpsilon;
        _stepCount = 0;
    }
    
    /// <summary>
    /// Decay epsilon by one step. Call once per episode for a proper exploration schedule.
    /// </summary>
    public void DecayEpsilon()
    {
        if (_epsilon > _epsilonMin)
            _epsilon *= _epsilonDecay;
    }

    public int BufferCount
    {
        get { lock (_bufferLock) return _buffer.Count; }
    }

    public void AddExperience(AgentInput state, int action, float reward, AgentInput? nextState, bool done)
    {
        lock (_bufferLock)
        {
            _buffer.Add(state, action, reward, nextState, done);
        }
    }

    /// <summary>
    /// Add a batch of experiences efficiently (vectorized add).
    /// Thread-safe batch insertion for parallel environment stepping.
    /// </summary>
    public void AddExperienceBatch(
        AgentInput[] states,
        int[] actions,
        float[] rewards,
        AgentInput?[] nextStates,
        bool[] dones)
    {
        lock (_bufferLock)
        {
            _buffer.AddBatch(states, actions, rewards, nextStates, dones);
        }
    }

    public (int[] Actions, float[,] TpSlMultipliers, float[] LogProbs) ActBatchWithTpSlAndLogProbs(AgentInput[] inputs, bool training = true)
    {
        var (actions, tpSl) = ActBatchWithTpSl(inputs, training);
        return (actions, tpSl, new float[inputs.Length]);
    }

    public void AddExperienceBatchWithLogProbs(
        AgentInput[] states,
        int[] actions,
        float[] rewards,
        AgentInput?[] nextStates,
        bool[] dones,
        float[] logProbs,
        float[] tpMultipliers,
        float[] slMultipliers)
    {
        // DQN now stores TP/SL for training the TP/SL head
        lock (_bufferLock)
        {
            _buffer.AddBatchWithTpSl(states, actions, rewards, nextStates, dones, tpMultipliers, slMultipliers);
        }
    }

    private Tensor[] PrepareInputTensors(AgentInput[] inputs, bool useTrainingBuffers)
    {
        var batchSize = inputs.Length;
        var tensors = new Tensor[13];
        var tensorIdx = 0;
        
        string[] tfNames = ["M1", "M5", "M15", "H1", "H4"];
        const int window = 20;
        const int feats = 45;

        var tfBuffers = useTrainingBuffers ? _trainTfBatchBuffers : _inferenceTfBatchBuffers;
        
        foreach (var tf in tfNames)
        {
            var buffer = batchSize <= 64 ? tfBuffers[tf] : new float[batchSize * window * feats];
            
            for (var b = 0; b < batchSize; b++)
            {
                var tfData = inputs[b].TimeframeFeatures[tf];
                Buffer.BlockCopy(tfData, 0, buffer, b * window * feats * sizeof(float), window * feats * sizeof(float));
            }
            
            tensors[tensorIdx++] = tensor(buffer, new long[] { batchSize, window, feats }, dtype: ScalarType.Float32, device: _device);
        }

        long[] symBuffer;
        float[] triggerBuffer, confluenceBuffer, portfolioBuffer, riskBuffer, newsBuffer, correlationBuffer, exposureBuffer;

        if (useTrainingBuffers)
        {
             symBuffer = _trainSymbolIdBuffer;
             triggerBuffer = _trainTriggerBuffer;
             confluenceBuffer = _trainConfluenceBuffer;
             portfolioBuffer = _trainPortfolioBuffer;
             riskBuffer = _trainRiskBuffer;
             newsBuffer = _trainNewsBuffer;
             correlationBuffer = _trainCorrelationBuffer;
             exposureBuffer = _trainExposureBuffer;
        }
        else
        {
             symBuffer = _inferenceSymbolIdBuffer;
             triggerBuffer = _inferenceTriggerBuffer;
             confluenceBuffer = _inferenceConfluenceBuffer;
             portfolioBuffer = _inferencePortfolioBuffer;
             riskBuffer = _inferenceRiskBuffer;
             newsBuffer = _inferenceNewsBuffer;
             correlationBuffer = _inferenceCorrelationBuffer;
             exposureBuffer = _inferenceExposureBuffer;
        }

        var activeSymBuffer = batchSize <= 64 ? symBuffer : new long[batchSize];
        for (var i = 0; i < batchSize; i++)
            activeSymBuffer[i] = inputs[i].SymbolId;
        tensors[tensorIdx++] = tensor(activeSymBuffer, new long[] { batchSize, 1 }, dtype: ScalarType.Int64, device: _device);

        tensors[tensorIdx++] = BatchFloatArrayOptimized(inputs, i => i.TriggerContext, 5, batchSize <= 64 ? triggerBuffer : null);

        tensors[tensorIdx++] = BatchFloatArrayOptimized(inputs, i => i.ConfluenceFeatures, 10, batchSize <= 64 ? confluenceBuffer : null);

        tensors[tensorIdx++] = BatchFloatArrayOptimized(inputs, i => i.PortfolioFeatures, 5, batchSize <= 64 ? portfolioBuffer : null);

        tensors[tensorIdx++] = BatchFloatArrayOptimized(inputs, i => i.RiskState, 9, batchSize <= 64 ? riskBuffer : null);

        if (inputs[0].NewsFeatures != null)
            tensors[tensorIdx++] = BatchFloatArrayOptimized(inputs, i => i.NewsFeatures!, inputs[0].NewsFeatures!.Length, batchSize <= 64 ? newsBuffer : null);
        else
            tensors[tensorIdx++] = zeros(new long[] { batchSize, _newsFeatureSize }, device: _device);

        if (inputs[0].CorrelationFeatures != null)
            tensors[tensorIdx++] = BatchFloatArrayOptimized(inputs, i => i.CorrelationFeatures!, 20, batchSize <= 64 ? correlationBuffer : null);
        else
            tensors[tensorIdx++] = zeros(new long[] { batchSize, 20 }, device: _device);

        if (inputs[0].PortfolioExposure != null)
            tensors[tensorIdx] = BatchFloatArrayOptimized(inputs, i => i.PortfolioExposure!, 12, batchSize <= 64 ? exposureBuffer : null);
        else
            tensors[tensorIdx] = zeros(new long[] { batchSize, 12 }, device: _device);
             
        return tensors;
    }
    
    private Tensor BatchFloatArrayOptimized(AgentInput[] inputs, Func<AgentInput, float[]> selector, int dim, float[]? buffer)
    {
        var batch = inputs.Length;
        var flat = buffer ?? new float[batch * dim];
        for (var i = 0; i < batch; i++)
        {
            var arr = selector(inputs[i]);
            Array.Copy(arr, 0, flat, i * dim, dim);
        }
        return tensor(flat, new long[] { batch, dim }, dtype: ScalarType.Float32, device: _device);
    }

    public void Dispose()
    {
        _policyNet.Dispose();
        _targetNet.Dispose();
        _inferenceNet.Dispose();
        _optimizer.Dispose();
    }
}

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

public class TorchAgent : IAgent, IDisposable
{
    private readonly DqnModel _policyNet; // Kept private fields...
    private readonly DqnModel _targetNet;
    private readonly Adam _optimizer;
    private readonly ReplayBuffer _replayBuffer;
    
    private readonly int _batchSize;
    private readonly float _gamma;
    private float _epsilon;
    private readonly float _initialEpsilon;
    private readonly float _epsilonMin;
    private readonly float _epsilonDecay;
    private readonly int _targetUpdateFreq;
    private int _stepCount;
    
    private readonly Device _device;
    
    public TorchAgent(
        int batchSize = 64,
        float gamma = 0.99f,
        float epsilon = 1.0f,
        float epsilonMin = 0.01f,
        float epsilonDecay = 0.995f,
        int targetUpdateFreq = 1000,
        int bufferSize = 100000,
        bool useCuda = false)
    {
        _batchSize = batchSize;
        _gamma = gamma;
        _epsilon = epsilon;
        _initialEpsilon = epsilon;
        _epsilonMin = epsilonMin;
        _epsilonDecay = epsilonDecay;
        _targetUpdateFreq = targetUpdateFreq;
        
        _device = useCuda && torch.cuda.is_available() ? torch.CUDA : torch.CPU;
        
        // Initialize Models
        _policyNet = new DqnModel("policy_net");
        _targetNet = new DqnModel("target_net");
        
        _policyNet.to(_device);
        _targetNet.to(_device);
        
        // Copy weights
        UpdateTargetNetwork();
        
        _optimizer = torch.optim.Adam(_policyNet.parameters(), lr: 0.0005);
        _replayBuffer = new ReplayBuffer(bufferSize);
    }
    
    public void UpdateTargetNetwork()
    {
        // Load state dict from policy to target
        // Simpler way in TorchSharp might be stricter, but load_state_dict works
        var stateDict = _policyNet.state_dict();
        _targetNet.load_state_dict(stateDict);
    }
    
    private static readonly Random _random = new();
    private const int NUM_ACTIONS = 8;

    public int Act(AgentInput input, bool training = true)
    {
        // if (training && _random.NextDouble() < _epsilon)
        // {
        //     return _random.Next(NUM_ACTIONS);
        // }

        using (torch.no_grad())
        {
            var tensors = PrepareInputTensors(new[] { input });
            var qValues = _policyNet.forward(tensors);
            return (int)qValues.argmax(1).item<long>();
        }
    }
    
    public float TrainStep()
    {
        if (_replayBuffer.Count < _batchSize)
            return 0f;
            
        var batch = _replayBuffer.Sample(_batchSize);
        
        // Prepare Batches
        var states = batch.Select(e => e.State).ToArray();
        var nextStates = batch.Select(e => e.NextState).Where(s => s != null).ToArray()!; // Handle nulls if any (shouldn't be for non-done)
        
        var stateTensors = PrepareInputTensors(states);
        
        // Actions: (Batch, 1)
        var actions = torch.tensor(batch.Select(e => (long)e.Action).ToArray(), dtype: ScalarType.Int64, device: _device).unsqueeze(1);
        
        // Rewards: (Batch)
        var rewards = torch.tensor(batch.Select(e => e.Reward).ToArray(), dtype: ScalarType.Float32, device: _device);
        
        // Dones: (Batch)
        var dones = torch.tensor(batch.Select(e => e.Done ? 1f : 0f).ToArray(), dtype: ScalarType.Float32, device: _device);
        
        // 1. Current Q values
        // Gather Q values for the selected actions
        var qValues = _policyNet.forward(stateTensors).gather(1, actions).squeeze(1);
        
        // 2. Next Q values (from Target Net)
        var nextQValues = torch.zeros(_batchSize, device: _device);
        
        if (nextStates.Length > 0)
        {
           using (torch.no_grad())
           {
               var nextStateTensors = PrepareInputTensors(nextStates); // Warning: this assumes all nextStates valid or handled. 
                                                                     // A strictly correct implementation handles masks. 
                                                                     // Simplified here: nextStates batch might be smaller if we filter nulls?
                                                                     // Actually, simplified: assuming nextState is never null unless Done.
                                                                     // If Done, next Q is 0.
               
               // We need a full batch of next states. For Done steps, next state doesn't matter, but input must be valid.
               // Let's create a full batch where Done states are just copies of current.
               var validNextStatesBatch = batch.Select(e => e.NextState ?? e.State).ToArray(); 
               var nextStateBatchTensors = PrepareInputTensors(validNextStatesBatch);
               
               var nextQPositions = _targetNet.forward(nextStateBatchTensors).max(1).values;
               // Zero out done states
               nextQValues = nextQPositions * (1 - dones);
           }
        }
        
        // 3. Target Q
        var targetQ = rewards + _gamma * nextQValues;
        
        // 4. Loss (MSE or Huber)
        var criterion = torch.nn.MSELoss();
        var loss = criterion.forward(qValues, targetQ);
        
        // 5. Optimize
        _optimizer.zero_grad();
        loss.backward();
        // torch.nn.utils.clip_grad_norm_(_policyNet.parameters(), 1.0); // Clip gradients
        _optimizer.step();
        
        // Update Epsilon
        if (_epsilon > _epsilonMin)
            _epsilon *= _epsilonDecay;
            
        // Update Target Network
        _stepCount++;
        if (_stepCount % _targetUpdateFreq == 0)
            UpdateTargetNetwork();
            
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

    public void Save(string path)
    {
        _policyNet.save(path);
    }

    public void Load(string path)
    {
        _policyNet.load(path);
        UpdateTargetNetwork();
    }

    public void ResetOnlineLearning()
    {
        _replayBuffer.Clear();
        _epsilon = _initialEpsilon;
        _stepCount = 0;
    }

    public void AddExperience(AgentInput state, int action, float reward, AgentInput? nextState, bool done)
    {
        _replayBuffer.Add(state, action, reward, nextState, done);
    }
    
    // Memory Management helper
    private Tensor[] PrepareInputTensors(AgentInput[] inputs)
    {
        int batchSize = inputs.Length;
        // Assuming all inputs have valid data.
        
        // Helper to batch arrays
        // Timeframes: [TF_INDEX] -> Tensor(Batch, Window, Features)
        var tensors = new List<Tensor>();
        
        string[] tfNames = { "M1", "M5", "M15", "H1", "H4" }; // Order must match DqnModel
        
        foreach (var tf in tfNames)
        {
            // Gather data for this timeframe across the batch
            // Flatten to 1D array first for speed? Or simpler loop.
            // input.TimeframeFeatures[tf] is [Window, Features]
            // We need [Batch, Window, Features]
            
            // Assuming fixed sizes for now
            int window = 20; 
            int feats = 45;
            float[] batchedData = new float[batchSize * window * feats];
            
            for (int b = 0; b < batchSize; b++)
            {
                var tfData = inputs[b].TimeframeFeatures[tf]; // [20, 45]
                // Memcopy or loop
                Buffer.BlockCopy(tfData, 0, batchedData, b * window * feats * sizeof(float), window * feats * sizeof(float));
            }
            
            tensors.Add(torch.tensor(batchedData, new long[] { batchSize, window, feats }, dtype: ScalarType.Float32, device: _device));
        }

        // Symbol ID
        var symData = inputs.Select(i => (long)i.SymbolId).ToArray();
        tensors.Add(torch.tensor(symData, new long[] { batchSize, 1 }, dtype: ScalarType.Int64, device: _device));
        
        // Trigger (5)
        tensors.Add(BatchFloatArray(inputs, i => i.TriggerContext, 5));
        
        // Confluence (10)
        tensors.Add(BatchFloatArray(inputs, i => i.ConfluenceFeatures, 10));
        
        // Portfolio (4)
        tensors.Add(BatchFloatArray(inputs, i => i.PortfolioFeatures, 4));
        
        // Risk (9)
        tensors.Add(BatchFloatArray(inputs, i => i.RiskState, 9));
        
        // Optionals - Always add them as DqnModel defaults to expecting them.
        // If data is missing (null), provide zeros.
        
        // News (16)
        if (inputs[0].NewsFeatures != null)
             tensors.Add(BatchFloatArray(inputs, i => i.NewsFeatures!, 16));
        else
             tensors.Add(torch.zeros(new long[] { batchSize, 16 }, device: _device));

        // Correlation (20)
        if (inputs[0].CorrelationFeatures != null)
             tensors.Add(BatchFloatArray(inputs, i => i.CorrelationFeatures!, 20));
        else
             tensors.Add(torch.zeros(new long[] { batchSize, 20 }, device: _device));

        // Portfolio Exposure (12)
        if (inputs[0].PortfolioExposure != null)
             tensors.Add(BatchFloatArray(inputs, i => i.PortfolioExposure!, 12));
        else
             tensors.Add(torch.zeros(new long[] { batchSize, 12 }, device: _device));
             
        return tensors.ToArray();
    }
    
    private Tensor BatchFloatArray(AgentInput[] inputs, Func<AgentInput, float[]> selector, int dim)
    {
        int batch = inputs.Length;
        float[] flat = new float[batch * dim];
        for(int i=0; i<batch; i++)
        {
            var arr = selector(inputs[i]);
            Array.Copy(arr, 0, flat, i * dim, dim);
        }
        return torch.tensor(flat, new long[] { batch, dim }, dtype: ScalarType.Float32, device: _device);
    }

    public void Dispose()
    {
        _policyNet.Dispose();
        _targetNet.Dispose();
        _optimizer.Dispose();
    }
}

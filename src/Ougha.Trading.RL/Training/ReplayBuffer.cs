using System;
using System.Collections.Generic;
using Ougha.Trading.RL.Agents;

namespace Ougha.Trading.RL.Training;

public class Experience
{
    public AgentInput State { get; set; }
    public int Action { get; set; }
    public float Reward { get; set; }
    public AgentInput? NextState { get; set; }
    public bool Done { get; set; }
    
    // For PER
    public float Priority { get; set; } = 1.0f;
}

/// <summary>
/// Prioritized Experience Replay buffer with N-step returns support.
/// Combines PER (samples important experiences more often) with N-step TD learning.
/// </summary>
public class PrioritizedReplayBuffer
{
    private readonly Experience[] _buffer;
    private readonly float[] _priorities;
    private readonly SumTree _sumTree;
    
    private int _position;
    private int _count;
    private readonly Random _random = new();
    
    // PER hyperparameters
    private readonly float _alpha;  // Priority exponent (0 = uniform, 1 = full prioritization)
    private readonly float _betaStart;
    private readonly float _betaEnd;
    private float _beta;
    private readonly int _betaAnnealingSteps;
    private int _stepCount;
    
    // N-step returns
    private readonly int _nSteps;
    private readonly float _gamma;
    private readonly Queue<Experience> _nStepBuffer;
    
    private const float PriorityEpsilon = 0.01f;
    private const float MaxPriority = 1.0f;

    public int Count => _count;
    public int Capacity => _buffer.Length;

    public PrioritizedReplayBuffer(
        int capacity,
        float alpha = 0.6f,
        float betaStart = 0.4f,
        float betaEnd = 1.0f,
        int betaAnnealingSteps = 100000,
        int nSteps = 3,
        float gamma = 0.99f)
    {
        _buffer = new Experience[capacity];
        _priorities = new float[capacity];
        _sumTree = new SumTree(capacity);
        
        _alpha = alpha;
        _betaStart = betaStart;
        _betaEnd = betaEnd;
        _beta = betaStart;
        _betaAnnealingSteps = betaAnnealingSteps;
        
        _nSteps = nSteps;
        _gamma = gamma;
        _nStepBuffer = new Queue<Experience>(nSteps);
        
        _position = 0;
        _count = 0;
    }

    /// <summary>
    /// Add experience with N-step return computation.
    /// </summary>
    public void Add(AgentInput state, int action, float reward, AgentInput? nextState, bool done)
    {
        var experience = new Experience
        {
            State = state,
            Action = action,
            Reward = reward,
            NextState = nextState,
            Done = done,
            Priority = MaxPriority
        };

        _nStepBuffer.Enqueue(experience);

        // Process N-step buffer when full or episode ends
        if (_nStepBuffer.Count >= _nSteps || done)
        {
            ProcessNStepBuffer(done);
        }
    }

    /// <summary>
    /// Add a batch of experiences efficiently (vectorized add).
    /// Bypasses N-step processing for raw batch insertion.
    /// </summary>
    public void AddBatch(
        AgentInput[] states,
        int[] actions,
        float[] rewards,
        AgentInput?[] nextStates,
        bool[] dones)
    {
        int batchSize = states.Length;
        for (int i = 0; i < batchSize; i++)
        {
            var experience = new Experience
            {
                State = states[i],
                Action = actions[i],
                Reward = rewards[i],
                NextState = nextStates[i],
                Done = dones[i],
                Priority = MaxPriority
            };
            StoreExperience(experience);
        }
    }

    /// <summary>
    /// Add a batch with N-step processing (slower but correct for training).
    /// </summary>
    public void AddBatchWithNStep(
        AgentInput[] states,
        int[] actions,
        float[] rewards,
        AgentInput?[] nextStates,
        bool[] dones)
    {
        int batchSize = states.Length;
        for (int i = 0; i < batchSize; i++)
        {
            Add(states[i], actions[i], rewards[i], nextStates[i], dones[i]);
        }
    }

    private void ProcessNStepBuffer(bool episodeDone)
    {
        while (_nStepBuffer.Count > 0 && (_nStepBuffer.Count >= _nSteps || episodeDone))
        {
            var nStepExps = _nStepBuffer.ToArray();
            var firstExp = _nStepBuffer.Dequeue();
            
            // Compute n-step return: R = r_0 + γ*r_1 + γ²*r_2 + ... + γⁿ*V(s_n)
            float nStepReward = 0f;
            float gammaMultiplier = 1f;
            bool reachedTerminal = false;
            AgentInput? nStepNextState = null;
            
            for (int i = 0; i < nStepExps.Length && i < _nSteps; i++)
            {
                nStepReward += gammaMultiplier * nStepExps[i].Reward;
                gammaMultiplier *= _gamma;
                
                if (nStepExps[i].Done)
                {
                    reachedTerminal = true;
                    break;
                }
                
                nStepNextState = nStepExps[i].NextState;
            }
            
            // Store the n-step experience
            var nStepExp = new Experience
            {
                State = firstExp.State,
                Action = firstExp.Action,
                Reward = nStepReward,
                NextState = reachedTerminal ? null : nStepNextState,
                Done = reachedTerminal,
                Priority = MaxPriority
            };
            
            StoreExperience(nStepExp);
            
            if (!episodeDone)
                break; // Only process one at a time unless episode ended
        }
    }

    private void StoreExperience(Experience experience)
    {
        float priority = MaxPriority;
        
        _buffer[_position] = experience;
        _priorities[_position] = priority;
        _sumTree.Update(_position, (float)Math.Pow(priority + PriorityEpsilon, _alpha));
        
        _position = (_position + 1) % _buffer.Length;
        if (_count < _buffer.Length)
            _count++;
    }

    /// <summary>
    /// Sample with prioritized weights. Returns (experiences, indices, importance weights).
    /// </summary>
    public (Experience[] Samples, int[] Indices, float[] Weights) SampleWithPriority(int batchSize)
    {
        var samples = new Experience[batchSize];
        var indices = new int[batchSize];
        var weights = new float[batchSize];
        
        // Anneal beta
        _stepCount++;
        _beta = Math.Min(_betaEnd, _betaStart + (_betaEnd - _betaStart) * _stepCount / _betaAnnealingSteps);
        
        float totalPriority = _sumTree.Total;
        float segment = totalPriority / batchSize;
        
        float minProbability = _sumTree.Min / totalPriority;
        float maxWeight = (float)Math.Pow(_count * minProbability, -_beta);
        
        for (int i = 0; i < batchSize; i++)
        {
            float a = segment * i;
            float b = segment * (i + 1);
            float value = (float)(_random.NextDouble() * (b - a) + a);
            
            int idx = _sumTree.Find(value);
            idx = Math.Clamp(idx, 0, _count - 1);
            
            indices[i] = idx;
            samples[i] = _buffer[idx];
            
            float priority = _priorities[idx];
            float probability = (float)Math.Pow(priority + PriorityEpsilon, _alpha) / totalPriority;
            float weight = (float)Math.Pow(_count * probability, -_beta);
            weights[i] = weight / maxWeight; // Normalize
        }
        
        return (samples, indices, weights);
    }

    /// <summary>
    /// Update priorities based on TD errors.
    /// </summary>
    public void UpdatePriorities(int[] indices, float[] tdErrors)
    {
        for (int i = 0; i < indices.Length; i++)
        {
            int idx = indices[i];
            float priority = Math.Abs(tdErrors[i]) + PriorityEpsilon;
            priority = Math.Min(priority, MaxPriority);
            
            _priorities[idx] = priority;
            _sumTree.Update(idx, (float)Math.Pow(priority, _alpha));
        }
    }

    /// <summary>
    /// Simple uniform sampling (fallback, for compatibility).
    /// </summary>
    public void SampleInto(Experience[] target)
    {
        int batchSize = target.Length;
        for (int i = 0; i < batchSize; i++)
        {
            int index = _random.Next(0, _count);
            target[i] = _buffer[index];
        }
    }

    public void Clear()
    {
        Array.Clear(_buffer, 0, _buffer.Length);
        Array.Clear(_priorities, 0, _priorities.Length);
        _sumTree.Clear();
        _nStepBuffer.Clear();
        _position = 0;
        _count = 0;
    }
}

/// <summary>
/// Sum tree data structure for O(log n) priority sampling.
/// </summary>
internal class SumTree
{
    private readonly float[] _tree;
    private readonly int _capacity;

    public SumTree(int capacity)
    {
        _capacity = capacity;
        _tree = new float[2 * capacity - 1];
    }

    public float Total => _tree[0];
    
    public float Min
    {
        get
        {
            float min = float.MaxValue;
            for (int i = _capacity - 1; i < _tree.Length; i++)
            {
                if (_tree[i] > 0 && _tree[i] < min)
                    min = _tree[i];
            }
            return min > 0 ? min : 1e-6f;
        }
    }

    public void Update(int dataIndex, float priority)
    {
        int treeIndex = dataIndex + _capacity - 1;
        float change = priority - _tree[treeIndex];
        _tree[treeIndex] = priority;
        
        // Propagate changes up the tree
        while (treeIndex > 0)
        {
            treeIndex = (treeIndex - 1) / 2;
            _tree[treeIndex] += change;
        }
    }

    public int Find(float value)
    {
        int idx = 0;
        
        while (idx < _capacity - 1)
        {
            int left = 2 * idx + 1;
            int right = left + 1;
            
            if (value <= _tree[left])
            {
                idx = left;
            }
            else
            {
                value -= _tree[left];
                idx = right;
            }
        }
        
        return idx - (_capacity - 1);
    }

    public void Clear()
    {
        Array.Clear(_tree, 0, _tree.Length);
    }
}

/// <summary>
/// Simple uniform replay buffer (original implementation, kept for compatibility).
/// </summary>
public class ReplayBuffer
{
    private readonly Experience[] _buffer;
    private int _position;
    private int _count;
    private readonly Random _random = new();

    public int Count => _count;
    public int Capacity => _buffer.Length;

    public ReplayBuffer(int capacity)
    {
        _buffer = new Experience[capacity];
        _position = 0;
        _count = 0;
    }

    public void Add(AgentInput state, int action, float reward, AgentInput? nextState, bool done)
    {
        var experience = new Experience
        {
            State = state,
            Action = action,
            Reward = reward,
            NextState = nextState,
            Done = done
        };

        _buffer[_position] = experience;
        _position = (_position + 1) % _buffer.Length;

        if (_count < _buffer.Length)
        {
            _count++;
        }
    }

    /// <summary>
    /// Add a batch of experiences efficiently (vectorized add).
    /// </summary>
    public void AddBatch(
        AgentInput[] states,
        int[] actions,
        float[] rewards,
        AgentInput?[] nextStates,
        bool[] dones)
    {
        int batchSize = states.Length;
        for (int i = 0; i < batchSize; i++)
        {
            var experience = new Experience
            {
                State = states[i],
                Action = actions[i],
                Reward = rewards[i],
                NextState = nextStates[i],
                Done = dones[i]
            };

            _buffer[_position] = experience;
            _position = (_position + 1) % _buffer.Length;

            if (_count < _buffer.Length)
                _count++;
        }
    }

    public List<Experience> Sample(int batchSize)
    {
        var batch = new List<Experience>(batchSize);
        for (int i = 0; i < batchSize; i++)
        {
            int index = _random.Next(0, _count);
            batch.Add(_buffer[index]);
        }
        return batch;
    }

    public void SampleInto(Experience[] target)
    {
        int batchSize = target.Length;
        for (int i = 0; i < batchSize; i++)
        {
            int index = _random.Next(0, _count);
            target[i] = _buffer[index];
        }
    }

    public void Clear()
    {
        Array.Clear(_buffer, 0, _buffer.Length);
        _position = 0;
        _count = 0;
    }
}

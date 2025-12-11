using Ougha.Trading.RL.Agents;

namespace Ougha.Trading.RL.Training;

public class Experience
{
    public AgentInput State { get; set; }
    public int Action { get; set; }
    public float Reward { get; set; }
    public AgentInput? NextState { get; set; }
    public bool Done { get; set; }

    public float Priority { get; set; } = 1.0f;
}

/// <summary>
/// Prioritized Experience Replay buffer with N-step returns support.
/// Combines PER (samples important experiences more often) with N-step TD learning.
/// </summary>
public class PrioritizedReplayBuffer(
    int capacity,
    float alpha = 0.6f,
    float betaStart = 0.4f,
    float betaEnd = 1.0f,
    int betaAnnealingSteps = 100000,
    int nSteps = 3,
    float gamma = 0.99f)
{
    private readonly Experience[] _buffer = new Experience[capacity];
    private readonly float[] _priorities = new float[capacity];
    private readonly SumTree _sumTree = new(capacity);
    
    private int _position;
    private int _count;
    private readonly Random _random = new();

    private readonly float _betaStart = betaStart;
    private float _beta = betaStart;
    private int _stepCount;

    private readonly Queue<Experience> _nStepBuffer = new(nSteps);
    
    private const float PriorityEpsilon = 0.01f;
    private const float MaxPriority = 1.0f;

    public int Count => _count;

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

        if (_nStepBuffer.Count >= nSteps || done)
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
        var batchSize = states.Length;
        for (var i = 0; i < batchSize; i++)
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

    private void ProcessNStepBuffer(bool episodeDone)
    {
        while (_nStepBuffer.Count > 0 && (_nStepBuffer.Count >= nSteps || episodeDone))
        {
            var nStepExps = _nStepBuffer.ToArray();
            var firstExp = _nStepBuffer.Dequeue();

            var nStepReward = 0f;
            var gammaMultiplier = 1f;
            var reachedTerminal = false;
            AgentInput? nStepNextState = null;
            
            for (var i = 0; i < nStepExps.Length && i < nSteps; i++)
            {
                nStepReward += gammaMultiplier * nStepExps[i].Reward;
                gammaMultiplier *= gamma;
                
                if (nStepExps[i].Done)
                {
                    reachedTerminal = true;
                    break;
                }
                
                nStepNextState = nStepExps[i].NextState;
            }

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
                break;
        }
    }

    private void StoreExperience(Experience experience)
    {
        var priority = MaxPriority;
        
        _buffer[_position] = experience;
        _priorities[_position] = priority;
        _sumTree.Update(_position, (float)Math.Pow(priority + PriorityEpsilon, alpha));
        
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

        _stepCount++;
        _beta = Math.Min(betaEnd, _betaStart + (betaEnd - _betaStart) * _stepCount / betaAnnealingSteps);
        
        var totalPriority = _sumTree.Total;
        var segment = totalPriority / batchSize;
        
        var minProbability = _sumTree.Min / totalPriority;
        var maxWeight = (float)Math.Pow(_count * minProbability, -_beta);
        
        for (var i = 0; i < batchSize; i++)
        {
            var a = segment * i;
            var b = segment * (i + 1);
            var value = (float)(_random.NextDouble() * (b - a) + a);
            
            var idx = _sumTree.Find(value);
            idx = Math.Clamp(idx, 0, _count - 1);
            
            indices[i] = idx;
            samples[i] = _buffer[idx];
            
            var priority = _priorities[idx];
            var probability = (float)Math.Pow(priority + PriorityEpsilon, alpha) / totalPriority;
            var weight = (float)Math.Pow(_count * probability, -_beta);
            weights[i] = weight / maxWeight;
        }
        
        return (samples, indices, weights);
    }

    /// <summary>
    /// Update priorities based on TD errors.
    /// </summary>
    public void UpdatePriorities(int[] indices, float[] tdErrors)
    {
        for (var i = 0; i < indices.Length; i++)
        {
            var idx = indices[i];
            var priority = Math.Abs(tdErrors[i]) + PriorityEpsilon;
            priority = Math.Min(priority, MaxPriority);
            
            _priorities[idx] = priority;
            _sumTree.Update(idx, (float)Math.Pow(priority, alpha));
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
internal class SumTree(int capacity)
{
    private readonly float[] _tree = new float[2 * capacity - 1];

    public float Total => _tree[0];
    
    public float Min
    {
        get
        {
            var min = float.MaxValue;
            for (var i = capacity - 1; i < _tree.Length; i++)
            {
                if (_tree[i] > 0 && _tree[i] < min)
                    min = _tree[i];
            }
            return min > 0 ? min : 1e-6f;
        }
    }

    public void Update(int dataIndex, float priority)
    {
        var treeIndex = dataIndex + capacity - 1;
        var change = priority - _tree[treeIndex];
        _tree[treeIndex] = priority;

        while (treeIndex > 0)
        {
            treeIndex = (treeIndex - 1) / 2;
            _tree[treeIndex] += change;
        }
    }

    public int Find(float value)
    {
        var idx = 0;
        
        while (idx < capacity - 1)
        {
            var left = 2 * idx + 1;
            var right = left + 1;
            
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
        
        return idx - (capacity - 1);
    }

    public void Clear()
    {
        Array.Clear(_tree, 0, _tree.Length);
    }
}
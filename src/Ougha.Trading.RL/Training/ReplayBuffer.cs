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
}

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

    public List<Experience> Sample(int batchSize)
    {
        var batch = new List<Experience>(batchSize);

        // Simple random sampling
        // Note: For very large buffers, this can be optimized
        for (int i = 0; i < batchSize; i++)
        {
            int index = _random.Next(0, _count);
            batch.Add(_buffer[index]);
        }

        return batch;
    }

    public void Clear()
    {
        Array.Clear(_buffer, 0, _buffer.Length);
        _position = 0;
        _count = 0;
    }
}

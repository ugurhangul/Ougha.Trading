using System.Collections.Concurrent;
using System.Threading.Channels;

namespace Ougha.Trading.RL.Training;

/// <summary>
/// Async double-buffered rollout buffer for PPO.
/// Allows collecting new experiences while the previous rollout is being trained.
/// 
/// Flow:
/// 1. Buffer A collects experiences while Buffer B trains
/// 2. When A is full, swap roles
/// 3. This overlaps rollout collection with gradient updates
/// </summary>
public class AsyncRolloutBuffer
{
    private readonly int _capacity;
    private readonly Channel<Experience[]> _readyRollouts;
    
    private List<Experience> _activeBuffer = [];
    private readonly Lock _bufferLock = new();
    
    public int ActiveBufferCount => _activeBuffer.Count;
    public int PendingRolloutsCount => _readyRollouts.Reader.Count;
    
    public AsyncRolloutBuffer(int capacity, int maxPendingRollouts = 2)
    {
        _capacity = capacity;
        _readyRollouts = Channel.CreateBounded<Experience[]>(
            new BoundedChannelOptions(maxPendingRollouts)
            {
                FullMode = BoundedChannelFullMode.Wait,
                SingleReader = true,
                SingleWriter = true
            });
    }
    
    /// <summary>
    /// Add an experience to the active buffer.
    /// When buffer reaches capacity, it's sent to the training channel.
    /// </summary>
    public async ValueTask AddExperienceAsync(Experience experience)
    {
        bool rolloutReady;
        Experience[]? rollout = null;
        
        lock (_bufferLock)
        {
            _activeBuffer.Add(experience);
            rolloutReady = _activeBuffer.Count >= _capacity;
            
            if (rolloutReady)
            {
                rollout = [.. _activeBuffer];
                _activeBuffer = [];
            }
        }
        
        if (rolloutReady && rollout != null)
        {
            await _readyRollouts.Writer.WriteAsync(rollout);
        }
    }
    
    /// <summary>
    /// Add a batch of experiences
    /// </summary>
    public async ValueTask AddExperienceBatchAsync(IEnumerable<Experience> experiences)
    {
        bool rolloutReady;
        Experience[]? rollout = null;
        
        lock (_bufferLock)
        {
            _activeBuffer.AddRange(experiences);
            rolloutReady = _activeBuffer.Count >= _capacity;
            
            if (rolloutReady)
            {
                rollout = [.. _activeBuffer];
                _activeBuffer = [];
            }
        }
        
        if (rolloutReady && rollout != null)
        {
            await _readyRollouts.Writer.WriteAsync(rollout);
        }
    }
    
    /// <summary>
    /// Try to get a ready rollout for training without waiting.
    /// Returns null if no rollout is ready.
    /// </summary>
    public Experience[]? TryGetRollout()
    {
        return _readyRollouts.Reader.TryRead(out var rollout) ? rollout : null;
    }
    
    /// <summary>
    /// Wait for a rollout to be ready for training.
    /// </summary>
    public async ValueTask<Experience[]> GetRolloutAsync(CancellationToken ct = default)
    {
        return await _readyRollouts.Reader.ReadAsync(ct);
    }
    
    /// <summary>
    /// Check if a rollout is ready without consuming it.
    /// </summary>
    public bool HasReadyRollout() => _readyRollouts.Reader.TryPeek(out _);
    
    /// <summary>
    /// Force flush the active buffer to ready (for end of training).
    /// </summary>
    public async ValueTask FlushAsync()
    {
        Experience[]? rollout = null;
        
        lock (_bufferLock)
        {
            if (_activeBuffer.Count > 0)
            {
                rollout = [.. _activeBuffer];
                _activeBuffer = [];
            }
        }
        
        if (rollout != null && rollout.Length > 0)
        {
            await _readyRollouts.Writer.WriteAsync(rollout);
        }
    }
    
    /// <summary>
    /// Signal that no more rollouts will be added.
    /// </summary>
    public void Complete()
    {
        _readyRollouts.Writer.Complete();
    }
    
    /// <summary>
    /// Reset the buffer state.
    /// </summary>
    public void Reset()
    {
        lock (_bufferLock)
        {
            _activeBuffer.Clear();
        }
        
        // Drain any pending rollouts
        while (_readyRollouts.Reader.TryRead(out _)) { }
    }
}

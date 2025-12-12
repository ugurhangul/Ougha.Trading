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
/// 3. This overlaps a rollout collection with gradient updates
/// </summary>
public class AsyncRolloutBuffer(int capacity, int maxPendingRollouts = 4)
{
    private readonly Channel<Experience[]> _readyRollouts = Channel.CreateBounded<Experience[]>(
        new BoundedChannelOptions(maxPendingRollouts)
        {
            FullMode = BoundedChannelFullMode.DropOldest,
            SingleReader = true,
            SingleWriter = false
        });
    
    private List<Experience> _activeBuffer = [];
    private readonly Lock _bufferLock = new();
    
    public int ActiveBufferCount => _activeBuffer.Count;
    public int PendingRolloutsCount => _readyRollouts.Reader.Count;

    /// <summary>
    /// Add an experience to the active buffer (synchronous, non-blocking).
    /// When buffer reaches capacity, it's sent to the training channel.
    /// </summary>
    public void AddExperience(Experience experience)
    {
        Experience[]? rollout = null;
        
        lock (_bufferLock)
        {
            _activeBuffer.Add(experience);
            
            if (_activeBuffer.Count >= capacity)
            {
                rollout = [.. _activeBuffer];
                _activeBuffer = [];
            }
        }
        
        if (rollout != null)
        {
            _readyRollouts.Writer.TryWrite(rollout);
        }
    }
    
    /// <summary>
    /// Add a batch of experiences (synchronous, non-blocking)
    /// </summary>
    public void AddExperienceBatch(IEnumerable<Experience> experiences)
    {
        Experience[]? rollout = null;
        
        lock (_bufferLock)
        {
            _activeBuffer.AddRange(experiences);
            
            if (_activeBuffer.Count >= capacity)
            {
                rollout = [.. _activeBuffer];
                _activeBuffer = [];
            }
        }
        
        if (rollout != null)
        {
            _readyRollouts.Writer.TryWrite(rollout);
        }
    }
    
    /// <summary>
    /// Add an experience to the active buffer.
    /// When buffer reaches capacity, it's sent to the training channel.
    /// </summary>
    public async ValueTask AddExperienceAsync(Experience experience)
    {
        Experience[]? rollout = null;
        
        lock (_bufferLock)
        {
            _activeBuffer.Add(experience);
            
            if (_activeBuffer.Count >= capacity)
            {
                rollout = [.. _activeBuffer];
                _activeBuffer = [];
            }
        }
        
        if (rollout != null)
        {
            await _readyRollouts.Writer.WriteAsync(rollout);
        }
    }
    
    /// <summary>
    /// Add a batch of experiences
    /// </summary>
    public async ValueTask AddExperienceBatchAsync(IEnumerable<Experience> experiences)
    {
        Experience[]? rollout = null;
        
        lock (_bufferLock)
        {
            _activeBuffer.AddRange(experiences);
            
            if (_activeBuffer.Count >= capacity)
            {
                rollout = [.. _activeBuffer];
                _activeBuffer = [];
            }
        }
        
        if (rollout != null)
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

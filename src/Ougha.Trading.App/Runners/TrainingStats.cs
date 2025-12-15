using System.Diagnostics;

namespace Ougha.Trading.App.Runners;

/// <summary>
/// Tracks all training statistics and metrics for the rich UI display.
/// Thread-safe for parallel episode training.
/// </summary>
public class TrainingStats
{
    private readonly Lock _lock = new();

    private int _episode;
    public int Episode { get => _episode; set => _episode = value; }
    public int TotalEpisodes { get; set; }
    public int CurrentStep { get; set; }
    public int MaxSteps { get; set; }
    public int CurrentChunk { get; set; }
    public int TotalChunks { get; set; }
    public int EpisodesPerChunk { get; set; }
    public int EpisodeInChunk { get; set; }
    public DateTime ChunkStartDate { get; set; }
    public DateTime ChunkEndDate { get; set; }
    public DateTime EpisodeStartDate { get; set; }
    public DateTime EpisodeEndDate { get; set; }
    public string CurrentSymbol { get; set; } = "";
    public string CurrentAction { get; set; } = "HOLD";

    public int CompletedEpisodes => _completedEpisodes;
    private int _completedEpisodes;
    public int ActiveEpisodes { get; set; }

    public DateTime StartTime { get; set; } = DateTime.Now;
    public TimeSpan Elapsed => DateTime.Now - StartTime;
    
    public double EpisodesPerHour => Elapsed.TotalHours > 0 ? _completedEpisodes / Elapsed.TotalHours : 0;
    public double SecondsPerEpisode => _completedEpisodes > 0 ? Elapsed.TotalSeconds / _completedEpisodes : 0;

    public TimeSpan EstimatedTimeRemaining
    {
        get
        {
            var completed = _completedEpisodes;
            if (completed <= 0) return TimeSpan.Zero;
            var episodesRemaining = TotalEpisodes - completed;
            var secondsPerEpisode = Elapsed.TotalSeconds / completed;
            return TimeSpan.FromSeconds(secondsPerEpisode * episodesRemaining);
        }
    }
    
    public string RewardTrend
    {
        get
        {
            lock (_lock)
            {
                if (_recentRewards.Count < 5) return "---";
                
                var recent5 = _recentRewards.TakeLast(5).Average();
                var recent20 = _recentRewards.TakeLast(Math.Min(20, _recentRewards.Count)).Average();
                
                if (recent5 > recent20 * 1.05) return "UP";
                if (recent5 < recent20 * 0.95) return "DOWN";
                return "FLAT";
            }
        }
    }

    public double EpisodeReward { get; set; }
    public double EpisodeLoss { get; set; }
    private double _bestReward = double.MinValue;
    public double BestReward { get => _bestReward; set => _bestReward = value; }
    public float Epsilon { get; set; } = 1.0f;
    
    /// <summary>
    /// Entropy coefficient (hyperparameter that weights entropy in PPO loss).
    /// </summary>
    public float EntropyCoefficient { get; set; } = 0.60f;
    
    /// <summary>
    /// Actual policy entropy from the action distribution (max ~1.39 for 4 actions).
    /// Higher = more exploration.
    /// </summary>
    public float PolicyEntropy { get; set; } = 0f;
    
    /// <summary>
    /// Backwards compatible alias for EntropyCoefficient.
    /// </summary>
    public float Entropy
    {
        get => EntropyCoefficient;
        set => EntropyCoefficient = value;
    }
    
    public bool IsPpoAgent { get; set; }

    private readonly Queue<double> _recentRewards = new();
    private int RewardHistorySize { get; set; } = 100;

    public void AddReward(double reward)
    {
        lock (_lock)
        {
            _recentRewards.Enqueue(reward);
            while (_recentRewards.Count > RewardHistorySize)
                _recentRewards.Dequeue();
        }
    }

    public double AverageReward100
    {
        get
        {
            lock (_lock)
                return _recentRewards.Count > 0 ? _recentRewards.Average() : 0;
        }
    }

    public double AverageReward10
    {
        get
        {
            lock (_lock)
                return _recentRewards.Count > 0
                    ? _recentRewards.TakeLast(Math.Min(10, _recentRewards.Count)).Average()
                    : 0;
        }
    }

    public int Positions { get; set; }
    public double Equity { get; set; } = 10000;
    public double InitialBalance { get; set; } = 10000;
    public int BufferSize { get; set; }

    public double PnLPercent => InitialBalance > 0
        ? (Equity - InitialBalance) / InitialBalance * 100
        : 0;

    private readonly Stopwatch _stepTimer = new();
    private int _stepCount;
    private double _totalStepTime;

    public double StepsPerSecond { get; private set; }
    public double ActionTimeMs { get; set; }
    public double EnvStepTimeMs { get; set; }
    public double TrainTimeMs { get; set; }
    public double BufferAddTimeMs { get; set; }
    private int _trainCalls;
    public int TrainCalls { get => _trainCalls; set => _trainCalls = value; }
    private int _totalSteps;
    public int TotalSteps { get => _totalSteps; set => _totalSteps = value; }

    public void StartStepTimer() => _stepTimer.Restart();

    public void EndStepTimer(int stepsTaken = 1)
    {
        _stepTimer.Stop();
        Interlocked.Add(ref _stepCount, stepsTaken);
        lock (_lock)
        {
            _totalStepTime += _stepTimer.Elapsed.TotalSeconds;
            if (_totalStepTime > 0)
                StepsPerSecond = _stepCount / _totalStepTime;
        }
    }

    private readonly Dictionary<int, int> _actionCounts = new()
    {
        { 0, 0 },
        { 1, 0 },
        { 2, 0 }
    };

    public Dictionary<int, int> ActionCounts
    {
        get
        {
            lock (_lock)
                return new Dictionary<int, int>(_actionCounts);
        }
    }

    public void RecordAction(int action)
    {
        lock (_lock)
        {
            if (_actionCounts.ContainsKey(action))
                _actionCounts[action]++;
            else
                _actionCounts[action] = 1;
        }
    }

    public int TotalActions
    {
        get
        {
            lock (_lock)
                return _actionCounts.Values.Sum();
        }
    }

    /// <summary>
    /// Get action counts as an array [Hold, Buy1, Buy2, Buy3, Sell1, Sell2, Sell3, Close]
    /// </summary>
    public int[] GetActionCountsAsArray()
    {
        lock (_lock)
        {
            var result = new int[8];
            for (var i = 0; i < 8; i++)
                result[i] = _actionCounts.GetValueOrDefault(i, 0);
            return result;
        }
    }

    /// <summary>
    /// Reset action counts (e.g., after checking for entropy reset)
    /// </summary>
    public void ResetActionCounts()
    {
        lock (_lock)
        {
            for (var i = 0; i < 8; i++)
                _actionCounts[i] = 0;
        }
    }

    private int _tradesOpened;
    private int _tradesClosed;
    private int _wins;
    private int _losses;
    private double _totalProfit;
    private double _totalLoss;
    private double _totalHoldingTimeSeconds;

    public int TradesOpened { get => _tradesOpened; set => _tradesOpened = value; }
    public int TradesClosed { get => _tradesClosed; set => _tradesClosed = value; }
    public int Wins { get => _wins; set => _wins = value; }
    public int Losses { get => _losses; set => _losses = value; }
    public double TotalProfit { get => _totalProfit; set => _totalProfit = value; }
    public double TotalLoss { get => _totalLoss; set => _totalLoss = value; }
    public double TotalHoldingTimeSeconds { get => _totalHoldingTimeSeconds; set => _totalHoldingTimeSeconds = value; }

    public double WinRate => _tradesClosed > 0 ? (double)_wins / _tradesClosed * 100 : 0;
    public double ProfitFactor => _totalLoss > 0 ? _totalProfit / _totalLoss : (_totalProfit > 0 ? 999.0 : 0);
    public TimeSpan AverageHoldingTime => _tradesClosed > 0
        ? TimeSpan.FromSeconds(_totalHoldingTimeSeconds / _tradesClosed)
        : TimeSpan.Zero;
    
    // Prediction accuracy tracking for supervised learning
    private int _correctPredictions;
    private int _totalPredictions;
    private double _totalPredictionError;  // Sum of |actual - predicted|
    
    public int CorrectPredictions { get => _correctPredictions; set => _correctPredictions = value; }
    public int TotalPredictions { get => _totalPredictions; set => _totalPredictions = value; }
    public double TotalPredictionError { get => _totalPredictionError; set => _totalPredictionError = value; }
    
    /// <summary>
    /// Prediction accuracy: % of trades where predicted direction matched actual price movement.
    /// </summary>
    public double PredictionAccuracy => _totalPredictions > 0 ? (double)_correctPredictions / _totalPredictions * 100 : 0;
    
    /// <summary>
    /// Average prediction error (MAE of predicted vs actual price change).
    /// </summary>
    public double AvgPredictionError => _totalPredictions > 0 ? _totalPredictionError / _totalPredictions : 0;
    
    /// <summary>
    /// Record a prediction outcome when a trade closes.
    /// </summary>
    public void RecordPrediction(float predicted, float actual)
    {
        lock (_lock)
        {
            _totalPredictions++;
            _totalPredictionError += Math.Abs(actual - predicted);
            
            // Direction accuracy: did sign match?
            if (Math.Sign(predicted) == Math.Sign(actual) && Math.Abs(predicted) > 0.0001f)
            {
                _correctPredictions++;
            }
        }
    }
    
    // Rolling window aggregates - computed from all symbol rolling windows
    public int RollingTrades
    {
        get
        {
            lock (_lock)
                return _symbolPerformance.Values.Sum(s => s.RollingTradeCount);
        }
    }
    
    public double RollingWinRate
    {
        get
        {
            lock (_lock)
            {
                var allTrades = _symbolPerformance.Values.Sum(s => s.RollingTradeCount);
                if (allTrades == 0) return 0;
                // Compute weighted average
                var totalWins = _symbolPerformance.Values.Sum(s => 
                    s.RollingTradeCount > 0 ? s.RollingWinRate * s.RollingTradeCount / 100.0 : 0);
                return totalWins / allTrades * 100;
            }
        }
    }
    
    public double RollingProfitFactor
    {
        get
        {
            lock (_lock)
            {
                // Sum up all rolling profits/losses across symbols
                double totalProfit = 0, totalLoss = 0;
                foreach (var s in _symbolPerformance.Values)
                {
                    if (s.RollingTradeCount == 0) continue;
                    // Access the underlying trades through the public PF if possible
                    // We'll estimate from PF and trade count
                    var pf = s.RollingProfitFactor;
                    // Assume normalized: if PF = profit/loss, and we sum proportionally
                    if (pf >= 999) totalProfit += s.RollingTradeCount; // All wins
                    else if (pf > 0)
                    {
                        // Approximate contribution
                        totalProfit += pf * s.RollingTradeCount;
                        totalLoss += s.RollingTradeCount;
                    }
                }
                return totalLoss > 0 ? totalProfit / totalLoss : (totalProfit > 0 ? 999.0 : 0);
            }
        }
    }
    
    public double RollingNetPnL
    {
        get
        {
            lock (_lock)
            {
                // Sum net PnL from all symbol rolling windows (need to add this to SymbolStats)
                return _symbolPerformance.Values.Sum(s => s.RollingNetPnL);
            }
        }
    }

    public double RollingAvgProfitPerTrade
    {
        get
        {
            lock (_lock)
            {
                var totalTrades = _symbolPerformance.Values.Sum(s => s.RollingTradeCount);
                if (totalTrades == 0) return 0;
                var totalNetPnL = _symbolPerformance.Values.Sum(s => s.RollingNetPnL);
                return totalNetPnL / totalTrades;
            }
        }
    }
    
    public TimeSpan RollingAvgHoldingTime
    {
        get
        {
            lock (_lock)
            {
                var totalTrades = _symbolPerformance.Values.Sum(s => s.RollingTradeCount);
                if (totalTrades == 0) return TimeSpan.Zero;
                // Weighted average of holding times
                var totalSeconds = _symbolPerformance.Values
                    .Where(s => s.RollingTradeCount > 0)
                    .Sum(s => s.RollingAvgHoldingTime.TotalSeconds * s.RollingTradeCount);
                return TimeSpan.FromSeconds(totalSeconds / totalTrades);
            }
        }
    }
    
    public void IncrementEpisode() => Interlocked.Increment(ref _completedEpisodes);

    private readonly Dictionary<string, SymbolStats> _symbolPerformance = new();

    public Dictionary<string, SymbolStats> SymbolPerformance
    {
        get
        {
            lock (_lock)
                return new Dictionary<string, SymbolStats>(_symbolPerformance);
        }
    }

    public SymbolStats GetOrCreateSymbolStats(string symbol)
    {
        lock (_lock)
        {
            if (!_symbolPerformance.TryGetValue(symbol, out var stats))
            {
                stats = new SymbolStats();
                _symbolPerformance[symbol] = stats;
            }
            return stats;
        }
    }
    public bool GpuAvailable { get; set; }
    public int EarlyStopPatience { get; set; } = 300;
    public int EarlyStopMinEpisodes { get; set; } = 500;
    
    // Error tracking without console output
    public string? LastError { get; set; }
    public int EntropyResetCount { get; set; }
}

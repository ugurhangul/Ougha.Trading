using System.Diagnostics;

namespace Ougha.Trading.App.Runners;

/// <summary>
/// Statistics for a single symbol during training.
/// </summary>
public class SymbolStats
{
    public int Episodes { get; set; }
    public double TotalReward { get; set; }
    public double BestReward { get; set; } = double.MinValue;
    public double BestProfitFactor { get; set; }
    public double BestSharpe { get; set; }
    public double BestMaxDrawdown { get; set; }
    public double BestWinRate { get; set; }
    public int BestTrades { get; set; }
    public int NoImprovementCount { get; set; }
    public bool EarlyStopped { get; set; }
    
    public double AverageReward => Episodes > 0 ? TotalReward / Episodes : 0;
}

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

    public double EpisodeReward { get; set; }
    public double EpisodeLoss { get; set; }
    private double _bestReward = double.MinValue;
    public double BestReward { get => _bestReward; set => _bestReward = value; }
    public float Epsilon { get; set; } = 1.0f;
    public float Entropy { get; set; } = 0.05f;
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
        { 2, 0 },
        { 3, 0 },
        { 4, 0 },
        { 5, 0 },
        { 6, 0 },
        { 7, 0 }
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

    public void RecordActions(int[] actions)
    {
        lock (_lock)
        {
            foreach (var action in actions)
            {
                if (_actionCounts.ContainsKey(action))
                    _actionCounts[action]++;
                else
                    _actionCounts[action] = 1;
            }
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

    private int _tradesOpened;
    private int _tradesClosed;
    private int _wins;
    private int _losses;
    private double _totalProfit;
    private double _totalLoss;

    public int TradesOpened { get => _tradesOpened; set => _tradesOpened = value; }
    public int TradesClosed { get => _tradesClosed; set => _tradesClosed = value; }
    public int Wins { get => _wins; set => _wins = value; }
    public int Losses { get => _losses; set => _losses = value; }
    public double TotalProfit { get => _totalProfit; set => _totalProfit = value; }
    public double TotalLoss { get => _totalLoss; set => _totalLoss = value; }

    public double WinRate => _tradesClosed > 0 ? (double)_wins / _tradesClosed * 100 : 0;
    public double ProfitFactor => _totalLoss > 0 ? _totalProfit / _totalLoss : (_totalProfit > 0 ? 999.0 : 0);

    public void IncrementEpisode() => Interlocked.Increment(ref _completedEpisodes);
    public void IncrementTotalSteps(int count = 1) => Interlocked.Add(ref _totalSteps, count);
    public void IncrementTrainCalls() => Interlocked.Increment(ref _trainCalls);

    public void AddTradeStats(int opened, int closed, int wins, int losses, double profit, double loss)
    {
        Interlocked.Add(ref _tradesOpened, opened);
        Interlocked.Add(ref _tradesClosed, closed);
        Interlocked.Add(ref _wins, wins);
        Interlocked.Add(ref _losses, losses);
        lock (_lock)
        {
            _totalProfit += profit;
            _totalLoss += loss;
        }
    }

    public void UpdateBestReward(double reward)
    {
        double current;
        do
        {
            current = _bestReward;
            if (reward <= current) return;
        } while (Math.Abs(Interlocked.CompareExchange(ref _bestReward, reward, current) - current) > 0.1);
    }

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
}

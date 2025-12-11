using Ougha.Trading.Backtesting;
using Ougha.Trading.Core.Abstractions;
using Ougha.Trading.Core.Models;
using Ougha.Trading.Risk;
using Ougha.Trading.Features.Indicators;
using Ougha.Trading.RL.Agents;

namespace Ougha.Trading.RL;

/// <summary>
/// Portfolio-based trading environment that processes all symbols together.
/// Builds combined state for all symbols and applies actions per symbol.
/// </summary>
public class PortfolioTradingEnvironment
{
    private readonly BacktestExecutor _executor;
    private readonly PortfolioStateBuilder _stateBuilder;
    private readonly RewardCalculator _rewardCalculator;
    private readonly PortfolioEnvironmentConfig _config;
    private readonly PortfolioManager _portfolioManager;
    private readonly IFeatureBuilder _featureBuilder;

    // Per-symbol state tracking - now uses MTF aggregator for all timeframes
    private readonly Dictionary<string, MultiTimeframeCandleAggregator> _mtfAggregators;
    private readonly Dictionary<string, MultiTimeframeStateBuilder> _mtfBuilders;
    private readonly Dictionary<string, int> _positionOpenTicks;
    private readonly Dictionary<string, double> _peakUnrealizedPnls;
    private readonly Dictionary<string, List<string>> _lastClosedTimeframes;
    
    // Action memory tracking (matches Python's action_memory_window)
    private readonly Dictionary<string, int> _lastExecutedAction;
    private readonly Dictionary<string, int> _lastExecutedTick;
    private readonly int _actionMemoryWindow; // Configurable, default 1050
    private const int MIN_HOLDING_TICKS = 50; // Minimum ticks before closing

    // Per-symbol action tracking for online learning (tracks every action, not just executed)
    private readonly Dictionary<string, int> _lastActionBySymbol;

    // Per-symbol TP/SL multipliers from model output
    private readonly Dictionary<string, (float TpMult, float SlMult)> _lastTpSlMultipliers;

    // Global state
    private int _currentTick;
    private double _peakEquity;
    private double _initialBalance;
    private bool _isDone;
    private int _lastM1Minute = -1;

    public int StateSize => _stateBuilder.GetStateSize(_config.Symbols.Length);
    public int ActionSize => ActionDecoder.NumActions; // 3 actions: HOLD, BUY, SELL
    public string[] Symbols => _config.Symbols;
    
    /// <summary>
    /// Expose RewardCalculator for episode metric resets.
    /// </summary>
    public RewardCalculator RewardCalculator => _rewardCalculator;
    
    /// <summary>
    /// Expose executor for stats access (positions, balance, results).
    /// </summary>
    public BacktestExecutor Executor => _executor;

    /// <summary>
    /// Expose MTF aggregator for internal preloading or debugging.
    /// </summary>
    public MultiTimeframeCandleAggregator GetMtFAggregator(string symbol) => _mtfAggregators[symbol];

    public PortfolioTradingEnvironment(
        BacktestExecutor executor,
        IFeatureBuilder featureBuilder,
        PortfolioManager portfolioManager,
        RewardCalculator rewardCalculator,
        PortfolioEnvironmentConfig config)
    {
        _executor = executor;
        _featureBuilder = featureBuilder;
        _portfolioManager = portfolioManager;
        _rewardCalculator = rewardCalculator;
        _config = config;
        _stateBuilder = new PortfolioStateBuilder(featureBuilder, config.WindowSize);
        _initialBalance = executor.GetBalance();
        _peakEquity = _initialBalance;

        // Initialize per-symbol state with MTF aggregators
        _mtfAggregators = new Dictionary<string, MultiTimeframeCandleAggregator>();
        _mtfBuilders = new Dictionary<string, MultiTimeframeStateBuilder>();
        _positionOpenTicks = new Dictionary<string, int>();
        _peakUnrealizedPnls = new Dictionary<string, double>();
        _lastClosedTimeframes = new Dictionary<string, List<string>>();
        _lastExecutedAction = new Dictionary<string, int>();
        _lastExecutedTick = new Dictionary<string, int>();
        _lastActionBySymbol = new Dictionary<string, int>();
        _lastTpSlMultipliers = new Dictionary<string, (float, float)>();

        _actionMemoryWindow = config.ActionMemoryWindow;

        foreach (var symbol in config.Symbols)
        {
            _mtfAggregators[symbol] = new MultiTimeframeCandleAggregator(maxCandlesPerTimeframe: config.WindowSize + 50);
            _mtfBuilders[symbol] = new MultiTimeframeStateBuilder(featureBuilder, config.WindowSize);
            _positionOpenTicks[symbol] = 0;
            _peakUnrealizedPnls[symbol] = 0;
            _lastClosedTimeframes[symbol] = new List<string>();
            _lastActionBySymbol[symbol] = 0;
            _lastTpSlMultipliers[symbol] = (0.5f, 0.5f); // Default middle values
        }

        _reusableRewards = new float[config.Symbols.Length];
        _reusableDones = new bool[config.Symbols.Length];
    }

    /// <summary>
    /// Specialized step method for RL Training.
    /// Returns per-symbol rewards and structured AgentInput[] states.
    /// Optimized: Only rebuilds features when M1 candle closes.
    /// </summary>
    public async Task<(AgentInput[] NextStates, float[] Rewards, bool[] Dones)> StepTrainingAsync(int[] actions)
    {
        var (rewards, dones, m1CandleClosed) = await StepFastAsync(actions);
        
        // OPTIMIZATION: Only rebuild features when M1 candle closes or cache is empty
        // This avoids expensive feature computation on every tick
        AgentInput[] nextStates;
        if (_isDone)
        {
            nextStates = _cachedAgentInputs ?? BuildAgentInputs();
        }
        else if (m1CandleClosed || _cachedAgentInputs == null)
        {
            nextStates = BuildAgentInputs();
            _cachedAgentInputs = nextStates;
        }
        else
        {
            nextStates = _cachedAgentInputs;
        }
        
        return (nextStates, rewards, dones);
    }


    private AgentInput[]? _cachedAgentInputs;
    private static readonly float[] _emptyRewards = new float[8];
    private static readonly bool[] _doneFlagsTrue = Enumerable.Repeat(true, 8).ToArray();
    private readonly float[] _reusableRewards;
    private readonly bool[] _reusableDones;

    /// <summary>
    /// Fast step that only advances simulation and processes actions.
    /// Returns whether any M1 candle closed (signal to rebuild features).
    /// </summary>
    public async Task<(float[] Rewards, bool[] Dones, bool M1CandleClosed)> StepFastAsync(int[] actions)
    {
        int symbolCount = _config.Symbols.Length;

        if (_isDone)
        {
            var emptyRewards = symbolCount <= 8 ? _emptyRewards : new float[symbolCount];
            var trueDones = symbolCount <= 8 ? _doneFlagsTrue : Enumerable.Repeat(true, symbolCount).ToArray();
            return (emptyRewards, trueDones, false);
        }

        Array.Clear(_reusableRewards, 0, symbolCount);
        Array.Clear(_reusableDones, 0, symbolCount);

        for (int i = 0; i < symbolCount; i++)
        {
            _lastActionBySymbol[_config.Symbols[i]] = actions[i];
        }

        bool moreData = await _executor.AdvanceAsync();
        _currentTick++;

        // Process automatic TP/SL closes that happened during AdvanceAsync
        // CRITICAL: This gives the agent reward signals for profitable closes!
        var pendingCloses = _executor.GetAndClearPendingCloses();
        foreach (var closeInfo in pendingCloses)
        {
            int symbolIndex = Array.IndexOf(_config.Symbols, closeInfo.Symbol);
            if (symbolIndex >= 0)
            {
                double maxDrawdownPct = 0;
                if (_peakEquity > 0)
                    maxDrawdownPct = (_peakEquity - _executor.GetEquity()) / _peakEquity;

                float closeReward = _rewardCalculator.Calculate(
                    tradeClosed: true,
                    tradeProfit: closeInfo.Profit,
                    holdingTicks: closeInfo.HoldingTicks,
                    hasPosition: false,
                    unrealizedPnl: 0,
                    peakUnrealizedPnl: _peakUnrealizedPnls.GetValueOrDefault(closeInfo.Symbol),
                    initialBalance: _initialBalance,
                    maxDrawdownPct: maxDrawdownPct);

                _reusableRewards[symbolIndex] += closeReward;
                _positionOpenTicks[closeInfo.Symbol] = 0;
                _peakUnrealizedPnls[closeInfo.Symbol] = 0;
            }
        }

        string? tickedSymbol = _executor.LastTickedSymbol;
        bool m1CandleClosed = false;

        if (tickedSymbol != null)
        {
            int symbolIndex = Array.IndexOf(_config.Symbols, tickedSymbol);
            if (symbolIndex >= 0)
            {
                var action = actions[symbolIndex];
                var (symbolReward, _) = await ProcessSymbolAction(tickedSymbol, action);
                _reusableRewards[symbolIndex] += symbolReward;
            }

            var candle = _executor.GetLastKnownCandle(tickedSymbol);
            if (candle != null)
            {
                var closedTimeframes = _mtfAggregators[tickedSymbol].AddCandle(candle);
                _lastClosedTimeframes[tickedSymbol] = closedTimeframes;
                if (closedTimeframes.Contains("M1"))
                    m1CandleClosed = true;
            }
        }

        // Use time-based M1 detection for determinism (independent of tick order)
        var currentTime = _executor.CurrentTime;
        if (_lastM1Minute != currentTime.Minute)
        {
            _lastM1Minute = currentTime.Minute;
            m1CandleClosed = true;
        }

        double currentEquity = _executor.GetEquity();
        if (currentEquity > _peakEquity) _peakEquity = currentEquity;

        bool globalDone = !moreData;
        if (currentEquity < _initialBalance * (1 - _config.MaxLossPercent / 100.0))
            globalDone = true;

        _isDone = globalDone;
        if (globalDone) Array.Fill(_reusableDones, true);

        return (_reusableRewards, _reusableDones, m1CandleClosed);
    }
    
    private async Task<(float Reward, bool TradeClosed)> ProcessSymbolAction(string symbol, int action)
    {
        if (ActionDecoder.IsHold(action)) return (0, false);

        var entryType = ActionDecoder.Decode(action);

        bool tradeClosed = false;
        double tradeProfit = 0;
        int holdingTicks = 0;

        var pos = _executor.GetPosition(symbol);
        bool hasPosition = pos != null;

        int lastAction = _lastExecutedAction.GetValueOrDefault(symbol, -1);
        int lastTick = _lastExecutedTick.GetValueOrDefault(symbol, -_actionMemoryWindow);
        int ticksSinceLastAction = _currentTick - lastTick;

        if (lastAction == action && ticksSinceLastAction < _actionMemoryWindow)
            return (0, false);

        int currentHoldingTicks = hasPosition ? _currentTick - _positionOpenTicks.GetValueOrDefault(symbol) : 0;

        // If we have a position and action is opposite direction, close first
        if (hasPosition && entryType.HasValue && pos!.Type != entryType.Value)
        {
            if (currentHoldingTicks >= MIN_HOLDING_TICKS)
            {
                var closeResult = await _executor.ClosePositionAsync(symbol);
                if (closeResult.Success)
                {
                    tradeClosed = true;
                    tradeProfit = closeResult.Profit;
                    holdingTicks = currentHoldingTicks;
                    _positionOpenTicks[symbol] = 0;
                    hasPosition = false;
                }
            }
            else
            {
                return (0, false);
            }
        }

        // Open new position if no position
        if (entryType.HasValue && !hasPosition)
        {
            var candle = _executor.GetLastKnownCandle(symbol);
            if (candle != null)
            {
                double atr = CalculateAtr(symbol);
                if (atr <= 0 || double.IsNaN(atr))
                    atr = candle.Close * 0.001;

                var symInfo = _executor.GetSymbolInfo(symbol)
                    ?? new SymbolInfo(symbol, 0.00001, 100000, 1, 0.00001, "USD", "USD", 5);

                // Get TP/SL multipliers from model output
                var (tpMult, slMult) = _lastTpSlMultipliers.GetValueOrDefault(symbol, (0.5f, 0.5f));

                // Scale multipliers to ATR ranges:
                // TP: 1.0 - 5.0 ATR (tpMult is 0-1 from sigmoid)
                // SL: 0.5 - 3.0 ATR (slMult is 0-1 from sigmoid)
                double tpAtrMult = 1.0 + tpMult * 4.0;  // 1.0 to 5.0
                double slAtrMult = 0.5 + slMult * 2.5;  // 0.5 to 3.0

                double slDistance = atr * slAtrMult;
                double tpDistance = atr * tpAtrMult;
                
                // Use Bid/Ask from executor (which uses Close +/- spread now)
                double bid = _executor.GetBid(symbol);
                double ask = _executor.GetAsk(symbol);

                double sl = entryType.Value == TradeType.Buy
                    ? bid - slDistance
                    : ask + slDistance;

                double tp = entryType.Value == TradeType.Buy
                    ? bid + tpDistance
                    : ask - tpDistance;

                // Calculate position size based on risk
                double riskAmount = _executor.GetEquity() * 0.01; // 1% risk per trade
                double slPoints = slDistance / symInfo.Point;
                double tickValue = symInfo.TickValue;
                // Avoid DBZ
                double volume = tickValue > 0 && slPoints > 0
                    ? riskAmount / (tickValue * slPoints)
                    : 0.01;
                volume = Math.Max(0.01, Math.Min(volume, 100.0));
                volume = Math.Round(volume, 2);

                var result = await _executor.ExecuteAsync(symbol, entryType.Value, volume,
                    sl, tp, "RL Portfolio Agent", RiskLevel.Moderate);

                if (result.Success)
                {
                    _positionOpenTicks[symbol] = _currentTick;
                    _peakUnrealizedPnls[symbol] = 0;
                    _lastExecutedAction[symbol] = action;
                    _lastExecutedTick[symbol] = _currentTick;
                }
            }
        }

        pos = _executor.GetPosition(symbol);
        double unrealized = pos?.UnrealizedPnlPercent / 100.0 ?? 0;

        if (unrealized > _peakUnrealizedPnls.GetValueOrDefault(symbol))
            _peakUnrealizedPnls[symbol] = unrealized;

        double maxDrawdownPct = 0;
        if (_peakEquity > 0)
            maxDrawdownPct = (_peakEquity - _executor.GetEquity()) / _peakEquity;

        float reward = _rewardCalculator.Calculate(
            tradeClosed, tradeProfit, holdingTicks,
            pos != null, unrealized, _peakUnrealizedPnls.GetValueOrDefault(symbol),
            _initialBalance, maxDrawdownPct);

        return (reward, tradeClosed);
    }

    public void SetTpSlMultipliers(string symbol, float tpMult, float slMult)
    {
        _lastTpSlMultipliers[symbol] = (tpMult, slMult);
    }

    public void SetTpSlMultipliersBatch(float[,] tpSlMultipliers)
    {
        for (int i = 0; i < _config.Symbols.Length && i < tpSlMultipliers.GetLength(0); i++)
        {
            _lastTpSlMultipliers[_config.Symbols[i]] = (tpSlMultipliers[i, 0], tpSlMultipliers[i, 1]);
        }
    }

    private double CalculateAtr(string symbol)
    {
        var history = _mtfAggregators[symbol].GetCandles("M1", 50);
        if (history.Count < 15)
            return 0;

        double[] h = history.Select(c => c.High).ToArray();
        double[] l = history.Select(c => c.Low).ToArray();
        double[] c = history.Select(c => c.Close).ToArray();

        var atrSeries = Technicals.Atr(h, l, c, 14);
        return atrSeries[^1];
    }

    private float[] BuildCurrentState()
    {
        var symbolCandles = new Dictionary<string, IReadOnlyList<Candle>>();
        var portfolioStates = new Dictionary<string, SymbolPortfolioState>();

        foreach (var symbol in _config.Symbols)
        {
            symbolCandles[symbol] = _mtfAggregators[symbol].GetCandles("M1", _config.WindowSize);

            var pos = _executor.GetPosition(symbol);
            if (pos != null)
            {
                double unrealized = pos.UnrealizedPnlPercent / 100.0;
                double peakPnl = _peakUnrealizedPnls.GetValueOrDefault(symbol);
                double drawdown = Math.Max(0, peakPnl - unrealized);
                
                int heldTicks = _currentTick - _positionOpenTicks.GetValueOrDefault(symbol);
                double holdingTimeNorm = Math.Min(1.0, heldTicks / (double)_config.MaxHoldingSteps);

                portfolioStates[symbol] = new SymbolPortfolioState(
                    HasPosition: true,
                    PositionType: pos.Type,
                    UnrealizedPnlPct: unrealized,
                    HoldingTimeNorm: holdingTimeNorm,
                    DrawdownPct: drawdown);
            }
            else
            {
                portfolioStates[symbol] = SymbolPortfolioState.Flat;
            }
        }

        return _stateBuilder.BuildPortfolioState(symbolCandles, portfolioStates, _config.Symbols);
    }

    /// <summary>
    /// Build structured AgentInput for each symbol.
    /// This is the recommended method for new ONNX agent interface.
    /// </summary>
    public AgentInput[] BuildAgentInputs()
    {
        var inputs = new AgentInput[_config.Symbols.Length];

        for (int i = 0; i < _config.Symbols.Length; i++)
        {
            var symbol = _config.Symbols[i];
            inputs[i] = BuildAgentInputForSymbol(symbol);
        }

        return inputs;
    }

    /// <summary>
    /// Build structured AgentInput for a single symbol.
    /// </summary>
    private AgentInput BuildAgentInputForSymbol(string symbol)
    {
        var mtfBuilder = _mtfBuilders[symbol];
        var mtfAggregator = _mtfAggregators[symbol];
        
        // Update MTF builder with all timeframe candles from aggregator
        var allTimeframeCandles = mtfAggregator.GetAllTimeframeCandles(_config.WindowSize + 50);
        foreach (var (timeframe, candles) in allTimeframeCandles)
        {
            if (candles.Count >= _config.WindowSize)
            {
                mtfBuilder.UpdateCandles(timeframe, candles, symbol);
            }
        }
        
        // Fallback: copy M1 to missing timeframes if not enough data yet
        mtfBuilder.CopyM1ToMissingTimeframes();

        var pos = _executor.GetPosition(symbol);
        bool hasPosition = pos != null;

        // Build portfolio features [5D] - includes balance
        // [0] = position_type (1.0 for BUY, -1.0 for SELL, 0.0 for no position)
        // [1] = unrealized_pnl * 100 (percent)
        // [2] = min(1.0, holding_ticks / max_holding_ticks) (normalized)
        // [3] = drawdown * 100 (percent)
        // [4] = normalized balance (current_balance / initial_balance - 1.0)
        var portfolioFeatures = new float[5];

        // Normalized balance: (current / initial) - 1.0, so 0 = break-even, positive = profit, negative = loss
        double currentBalance = _executor.GetBalance();
        portfolioFeatures[4] = (float)((currentBalance / _initialBalance) - 1.0);

        if (hasPosition)
        {
            portfolioFeatures[0] = pos!.Type == TradeType.Buy ? 1.0f : -1.0f;

            double unrealizedPnl = (pos.CurrentPrice - pos.OpenPrice) / pos.OpenPrice;
            if (pos.Type == TradeType.Sell) unrealizedPnl *= -1;
            portfolioFeatures[1] = (float)(unrealizedPnl * 100.0);

            int heldTicks = _currentTick - _positionOpenTicks.GetValueOrDefault(symbol);
            portfolioFeatures[2] = (float)Math.Min(1.0, heldTicks / (double)_config.MaxHoldingSteps);

            if (unrealizedPnl > _peakUnrealizedPnls.GetValueOrDefault(symbol))
                _peakUnrealizedPnls[symbol] = unrealizedPnl;
            double drawdown = Math.Max(0.0, _peakUnrealizedPnls.GetValueOrDefault(symbol) - unrealizedPnl);
            portfolioFeatures[3] = (float)(drawdown * 100.0);
        }

        var riskState = new float[9];
        if (hasPosition)
        {
            riskState[0] = 1.0f;
            riskState[1] = pos!.Type == TradeType.Buy ? 1.0f : -1.0f;
            riskState[6] = (float)(pos.Profit / 100.0);
            riskState[8] = 1.0f;
        }

        var closedTfs = _lastClosedTimeframes.GetValueOrDefault(symbol) ?? new List<string>();

        return mtfBuilder.BuildAgentInput(
            symbol: symbol,
            portfolioFeatures: portfolioFeatures,
            riskState: riskState,
            closedTimeframes: closedTfs,
            newsFeatures: null,
            correlationFeatures: null,
            portfolioExposure: null
        );
    }

    public Task<float[]> ResetAsync()
    {
        // Reset executor to enable fresh episode data
        _executor.Reset();
        
        // Reset reward calculator episode metrics for proper PF/Sharpe tracking
        _rewardCalculator.ResetEpisode();
        
        // No warmup skip needed - candles are preloaded from QuestDB materialized views
        _currentTick = 0;
        _isDone = false;
        _initialBalance = _executor.GetBalance();
        _peakEquity = _initialBalance;
        _lastM1Minute = -1;

        foreach (var symbol in _config.Symbols)
        {
            _positionOpenTicks[symbol] = 0;
            _peakUnrealizedPnls[symbol] = 0;
            _mtfBuilders[symbol].Reset();
            _mtfAggregators[symbol].Reset();
            _lastClosedTimeframes[symbol].Clear();
            _lastActionBySymbol[symbol] = 0;
            _lastTpSlMultipliers[symbol] = (0.5f, 0.5f);
            _lastExecutedAction.Remove(symbol);
            _lastExecutedTick.Remove(symbol);
        }

        return Task.FromResult(new float[StateSize]);
    }

    /// <summary>
    /// Preload historical candles from QuestDB materialized views for all symbols.
    /// This eliminates the need for warmup by loading pre-built candles.
    /// Call this after ResetAsync if historical candles are available.
    /// </summary>
    /// <param name="dbLoader">QuestDB data loader</param>
    /// <param name="episodeStartTime">The timestamp when the episode starts (loads candles before this)</param>
    /// <param name="candleCount">Number of historical candles to load per timeframe (default: WindowSize + 50)</param>
    public async Task PreloadHistoricalCandlesAsync(
        Data.QuestDbDataLoader dbLoader, 
        DateTime episodeStartTime,
        int? candleCount = null)
    {
        int count = candleCount ?? _config.WindowSize + 50;
        
        // Map internal timeframe keys to QuestDB materialized view names
        var timeframeMap = new Dictionary<string, string>
        {
            ["M1"] = "m1",
            ["M5"] = "m5",
            ["M15"] = "m15",
            ["H1"] = "h1",
            ["H4"] = "h4"
        };

        foreach (var symbol in _config.Symbols)
        {
            var aggregator = _mtfAggregators[symbol];
            
            foreach (var (internalTf, questdbTf) in timeframeMap)
            {
                var candles = await dbLoader.LoadHistoricalCandlesAsync(
                    symbol, questdbTf, episodeStartTime, count);
                
                if (candles.Count > 0)
                {
                    aggregator.PreloadCandles(internalTf, candles);
                }
            }
        }
        
        // Reset tick counter since we have pre-built candle history
        _currentTick = 0;
    }

    /// <summary>
    /// Get the last action taken for a specific symbol.
    /// Used for online learning experience storage.
    /// </summary>
    public int GetLastAction(string symbol) => _lastActionBySymbol.GetValueOrDefault(symbol, 0);

    /// <summary>
    /// Get the last actions for all symbols in order.
    /// Used for online learning experience storage.
    /// </summary>
    public int[] GetLastActions() => _config.Symbols.Select(s => _lastActionBySymbol.GetValueOrDefault(s, 0)).ToArray();
}

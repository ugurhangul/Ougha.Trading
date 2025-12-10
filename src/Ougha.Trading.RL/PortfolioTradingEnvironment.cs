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
    private const int ACTION_MEMORY_WINDOW = 1050; // Ticks before same action can repeat
    private const int MIN_HOLDING_TICKS = 50; // Minimum ticks before closing

    // Per-symbol action tracking for online learning (tracks every action, not just executed)
    private readonly Dictionary<string, int> _lastActionBySymbol;

    // Global state
    private int _currentTick;
    private double _peakEquity;
    private double _initialBalance;
    private bool _isDone;
    private int _lastM1Minute = -1;

    public int StateSize => _stateBuilder.GetStateSize(_config.Symbols.Length);
    public int ActionSize => 8; // Per symbol: HOLD, BUY_CONS, BUY_MOD, BUY_AGG, SELL_CONS, SELL_MOD, SELL_AGG, CLOSE
    public string[] Symbols => _config.Symbols;

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

        foreach (var symbol in config.Symbols)
        {
            _mtfAggregators[symbol] = new MultiTimeframeCandleAggregator(maxCandlesPerTimeframe: config.WindowSize + 50);
            _mtfBuilders[symbol] = new MultiTimeframeStateBuilder(featureBuilder, config.WindowSize);
            _positionOpenTicks[symbol] = 0;
            _peakUnrealizedPnls[symbol] = 0;
            _lastClosedTimeframes[symbol] = new List<string>();
            _lastActionBySymbol[symbol] = 0;
        }

        _reusableRewards = new float[config.Symbols.Length];
        _reusableDones = new bool[config.Symbols.Length];
    }

    /// <summary>
    /// Specialized step method for RL Training.
    /// Returns per-symbol rewards and structured AgentInput[] states.
    /// In tick-by-tick mode, only the symbol that receives the next tick will have its action processed.
    /// </summary>
    public async Task<(AgentInput[] NextStates, float[] Rewards, bool[] Dones, Dictionary<string, object> Infos)> StepTrainingAsync(int[] actions)
    {
        var (rewards, dones, _) = await StepFastAsync(actions);
        var nextStates = _isDone ? _cachedAgentInputs ?? BuildAgentInputs() : BuildAgentInputs();
        return (nextStates, rewards, dones, new Dictionary<string, object>());
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

        string? tickedSymbol = _executor.LastTickedSymbol;
        bool m1CandleClosed = false;

        if (tickedSymbol != null)
        {
            int symbolIndex = Array.IndexOf(_config.Symbols, tickedSymbol);
            if (symbolIndex >= 0)
            {
                var action = actions[symbolIndex];
                var (symbolReward, _) = await ProcessSymbolAction(tickedSymbol, action);
                _reusableRewards[symbolIndex] = symbolReward;
            }

            var tick = _executor.GetLastKnownTick(tickedSymbol);
            if (tick != null)
            {
                var closedTimeframes = _mtfAggregators[tickedSymbol].AddTick(tick);
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
        // Skip HOLD actions early - no need to process
        if (action == 0) return (0, false);

        bool isClose = ActionDecoder.IsClose(action);
        var (entryType, riskLevel) = ActionDecoder.Decode(action);

        bool tradeClosed = false;
        double tradeProfit = 0;
        int holdingTicks = 0;

        var pos = _executor.GetPosition(symbol);
        bool hasPosition = pos != null;

        // Action memory window check (Python: ticks_since_last_action < action_memory_window)
        int lastAction = _lastExecutedAction.GetValueOrDefault(symbol, -1);
        int lastTick = _lastExecutedTick.GetValueOrDefault(symbol, -ACTION_MEMORY_WINDOW);
        int ticksSinceLastAction = _currentTick - lastTick;

        if (lastAction == action && ticksSinceLastAction < ACTION_MEMORY_WINDOW)
        {
            // Skip - same action within memory window
            return (0, false);
        }
        
        // Min holding ticks check (Python: holding_ticks < min_holding_ticks)
        int currentHoldingTicks = hasPosition ? _currentTick - _positionOpenTicks.GetValueOrDefault(symbol) : 0;
        if (hasPosition && currentHoldingTicks < MIN_HOLDING_TICKS)
        {
            // Don't close or reverse too early
            if (isClose || (entryType.HasValue && pos!.Type != (entryType.Value == TradeType.Buy ? TradeType.Buy : TradeType.Sell)))
            {
                return (0, false);
            }
        }

        // Execute action - MATCHING PYTHON BEHAVIOR:
        // - BUY/SELL only work when NOT has_position (no flipping allowed)
        // - CLOSE only works when has_position
        if (isClose)
        {
            if (hasPosition)
            {
                var result = await _executor.ClosePositionAsync(symbol);
                if (result.Success)
                {
                    tradeClosed = true;
                    tradeProfit = result.Profit;
                    holdingTicks = _currentTick - _positionOpenTicks.GetValueOrDefault(symbol);
                    _positionOpenTicks[symbol] = 0;

                    // Update action memory
                    _lastExecutedAction[symbol] = action;
                    _lastExecutedTick[symbol] = _currentTick;
                }
            }
            // else: CLOSE with no position - action is ignored (no penalty, just no-op)
        }
        else if (entryType.HasValue && riskLevel.HasValue && !hasPosition)
        {
            // Python: BUY/SELL only execute when NOT has_position
            var tick = _executor.GetLastKnownTick(symbol);
            if (tick != null)
            {
                double atr = CalculateAtr(symbol);
                if (atr <= 0 || double.IsNaN(atr))
                    atr = tick.Bid * 0.001;

                var symInfo = _executor.GetSymbolInfo(symbol)
                    ?? new SymbolInfo(symbol, 0.00001, 100000, 1, 0.00001, "USD", "USD", 5);

                var sizing = _portfolioManager.CalculatePositionSize(
                    symbol, entryType.Value, riskLevel.Value,
                    _executor.GetEquity(), tick.Bid, atr, symInfo);

                var result = await _executor.ExecuteAsync(symbol, entryType.Value, sizing.Volume,
                    sizing.StopLoss, sizing.TakeProfit, "RL Portfolio Agent", riskLevel.Value);

                if (result.Success)
                {
                    _positionOpenTicks[symbol] = _currentTick;
                    _peakUnrealizedPnls[symbol] = 0;

                    // Update action memory
                    _lastExecutedAction[symbol] = action;
                    _lastExecutedTick[symbol] = _currentTick;
                }
            }
        }
        // If has_position and trying to BUY/SELL, do nothing (matches Python)

        // Recalculate position state for reward
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

        // Get position info
        var pos = _executor.GetPosition(symbol);
        bool hasPosition = pos != null;

        // Build portfolio features [4D] - EXACTLY matching Python's _get_portfolio_features()
        // Python: portfolio[0] = position_type (1.0 for BUY, -1.0 for SELL, 0.0 for no position)
        // Python: portfolio[1] = unrealized_pnl * 100 (percent)
        // Python: portfolio[2] = min(1.0, holding_ticks / max_holding_ticks) (normalized)
        // Python: portfolio[3] = drawdown * 100 (percent)
        var portfolioFeatures = new float[4];
        if (hasPosition)
        {
            // Python: position_type = 1 if pos.position_type == PositionType.BUY else -1
            portfolioFeatures[0] = pos!.Type == TradeType.Buy ? 1.0f : -1.0f;

            // Python: unrealized_pnl = (current_price - open_price) / open_price
            // Then multiplied by -1 if SELL, then * 100
            double unrealizedPnl = (pos.CurrentPrice - pos.OpenPrice) / pos.OpenPrice;
            if (pos.Type == TradeType.Sell) unrealizedPnl *= -1;
            portfolioFeatures[1] = (float)(unrealizedPnl * 100.0);

            int heldTicks = _currentTick - _positionOpenTicks.GetValueOrDefault(symbol);
            portfolioFeatures[2] = (float)Math.Min(1.0, heldTicks / (double)_config.MaxHoldingSteps);

            // Update peak and calculate drawdown
            if (unrealizedPnl > _peakUnrealizedPnls.GetValueOrDefault(symbol))
                _peakUnrealizedPnls[symbol] = unrealizedPnl;
            double drawdown = Math.Max(0.0, _peakUnrealizedPnls.GetValueOrDefault(symbol) - unrealizedPnl);
            portfolioFeatures[3] = (float)(drawdown * 100.0);
        }
        // else: all zeros (no position) - matches Python behavior

        // Build risk state [9D] - EXACTLY matching Python's _build_state()
        // Python only fills: [0]=in_position, [1]=direction, [6]=profit/100, [8]=risk_level
        var riskState = new float[9];
        if (hasPosition)
        {
            riskState[0] = 1.0f; // in position
            riskState[1] = pos!.Type == TradeType.Buy ? 1.0f : -1.0f; // direction
            // riskState[2-5] = 0 (Python doesn't fill these)
            riskState[6] = (float)(pos.Profit / 100.0); // profit / 100 (Python: pos.profit / 100.0)
            // riskState[7] = 0 (Python doesn't fill this)
            riskState[8] = 1.0f; // risk_level.MODERATE = 1
        }

        // Get timeframes that just closed (for trigger context)
        var closedTfs = _lastClosedTimeframes.GetValueOrDefault(symbol) ?? new List<string>();

        return mtfBuilder.BuildAgentInput(
            symbol: symbol,
            portfolioFeatures: portfolioFeatures,
            riskState: riskState,
            closedTimeframes: closedTfs,
            newsFeatures: null,     // Not implemented yet
            correlationFeatures: null, // Not implemented yet
            portfolioExposure: null    // Not implemented yet
        );
    }

    public Task<float[]> ResetAsync()
    {
        _currentTick = _config.WindowSize * 100;
        _isDone = false;
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
            _lastExecutedAction.Remove(symbol);
            _lastExecutedTick.Remove(symbol);
        }

        return Task.FromResult(new float[StateSize]);
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

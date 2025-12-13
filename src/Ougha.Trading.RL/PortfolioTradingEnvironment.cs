using Ougha.Trading.Backtesting;
using Ougha.Trading.Core.Abstractions;
using Ougha.Trading.Core.Models;
using Ougha.Trading.Data.Services;
using Ougha.Trading.Features.Indicators;
using Ougha.Trading.RL.Agents;
using Serilog;

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

    private readonly Dictionary<string, MultiTimeframeCandleAggregator> _mtfAggregators;
    private readonly Dictionary<string, MultiTimeframeStateBuilder> _mtfBuilders;
    private readonly Dictionary<string, int> _positionOpenTicks;
    private readonly Dictionary<string, double> _peakUnrealizedPnls;
    private readonly Dictionary<string, List<string>> _lastClosedTimeframes;

    private readonly Dictionary<string, int> _lastExecutedAction;
    private readonly Dictionary<string, int> _lastExecutedTick;
    private readonly int _actionMemoryWindow;
    private const int MIN_HOLDING_TICKS = 30;  // 30 seconds - minimal enforcement, let agent explore freely

    private readonly Dictionary<string, int> _lastActionBySymbol;

    private readonly Dictionary<string, (float TpMult, float SlMult)> _lastTpSlMultipliers;
    
    // DXY Index service for USD strength features
    private readonly DxyIndexService _dxyService;
    
    // Economic calendar service for news/event features
    private readonly EconomicCalendarService _newsService;
    
    // Cross-symbol correlation service
    private readonly CorrelationService _correlationService;
    
    // Portfolio exposure service
    private readonly PortfolioExposureService _exposureService;

    private int _currentTick;
    private double _peakEquity;
    private double _initialBalance;
    private bool _isDone;
    private int _lastM1Minute = -1;

    public int StateSize => _stateBuilder.GetStateSize(_config.Symbols.Length);
    
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
        RewardCalculator rewardCalculator,
        PortfolioEnvironmentConfig config)
    {
        _executor = executor;
        _rewardCalculator = rewardCalculator;
        _config = config;
        _stateBuilder = new PortfolioStateBuilder(featureBuilder, config.WindowSize);
        _initialBalance = executor.GetBalance();
        _peakEquity = _initialBalance;

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
            _lastTpSlMultipliers[symbol] = (0.5f, 0.5f);
        }

        _reusableRewards = new float[config.Symbols.Length];
        _reusableDones = new bool[config.Symbols.Length];
        
        // Initialize DXY service
        _dxyService = new DxyIndexService();
        
        // Initialize economic calendar service
        _newsService = new EconomicCalendarService();
        
        // Initialize correlation service
        _correlationService = new CorrelationService();
        
        // Initialize portfolio exposure service
        _exposureService = new PortfolioExposureService();
    }

    /// <summary>
    /// Specialized step method for RL Training.
    /// Returns per-symbol rewards and structured AgentInput[] states.
    /// Optimized: Only rebuilds features when M1 candle closes.
    /// </summary>
    public async Task<(AgentInput[] NextStates, float[] Rewards, bool[] Dones)> StepTrainingAsync(int[] actions)
    {
        var (rewards, dones, m1CandleClosed) = await StepFastAsync(actions);

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

    /// <summary>
    /// Step until M1 candle closes, accumulating rewards.
    /// This is the optimized training method that makes decisions at M1 granularity
    /// while maintaining S1 precision for SL/TP triggers.
    /// </summary>
    /// <param name="actions">Actions to hold throughout the M1 period</param>
    /// <param name="maxSteps">Maximum S1 steps to take (safety limit)</param>
    /// <returns>Accumulated rewards, done flags, next states, and number of S1 steps taken</returns>
    public async Task<(AgentInput[] NextStates, float[] AccumulatedRewards, bool[] Dones, int StepsTaken)> 
        StepUntilM1CloseAsync(int[] actions, int maxSteps = 120)
    {
        var symbolCount = _config.Symbols.Length;
        var accumulatedRewards = new float[symbolCount];
        var stepsTaken = 0;
        
        // Fast path: already done
        if (_isDone)
        {
            var emptyRewards = symbolCount <= 8 ? EmptyRewards : new float[symbolCount];
            var trueDones = symbolCount <= 8 ? DoneFlagsTrue : Enumerable.Repeat(true, symbolCount).ToArray();
            return (_cachedAgentInputs ?? BuildAgentInputs(), emptyRewards, trueDones, 0);
        }
        
        bool m1Closed;
        bool episodeDone;
        
        do
        {
            // Step environment with same action
            var (rewards, dones, m1CandleClosed) = await StepFastAsync(actions);
            stepsTaken++;
            
            // Accumulate rewards across all S1 steps
            for (var i = 0; i < symbolCount; i++)
                accumulatedRewards[i] += rewards[i];
            
            m1Closed = m1CandleClosed;
            episodeDone = dones.All(d => d) || _isDone;
            
        } while (!m1Closed && !episodeDone && stepsTaken < maxSteps);
        
        // Build next states (only at M1 boundaries)
        var nextStates = BuildAgentInputs();
        _cachedAgentInputs = nextStates;
        
        var finalDones = new bool[symbolCount];
        if (_isDone) Array.Fill(finalDones, true);
        
        return (nextStates, accumulatedRewards, finalDones, stepsTaken);
    }


    private AgentInput[]? _cachedAgentInputs;
    private static readonly float[] EmptyRewards = new float[8];
    private static readonly bool[] DoneFlagsTrue = Enumerable.Repeat(true, 8).ToArray();
    private readonly float[] _reusableRewards;
    private readonly bool[] _reusableDones;

    /// <summary>
    /// Fast step that only advances simulation and processes actions.
    /// Returns whether any M1 candle closed (signal to rebuild features).
    /// </summary>
    public async Task<(float[] Rewards, bool[] Dones, bool M1CandleClosed)> StepFastAsync(int[] actions)
    {
        var symbolCount = _config.Symbols.Length;

        if (_isDone)
        {
            var emptyRewards = symbolCount <= 8 ? EmptyRewards : new float[symbolCount];
            var trueDones = symbolCount <= 8 ? DoneFlagsTrue : Enumerable.Repeat(true, symbolCount).ToArray();
            return (emptyRewards, trueDones, false);
        }

        Array.Clear(_reusableRewards, 0, symbolCount);
        Array.Clear(_reusableDones, 0, symbolCount);

        for (var i = 0; i < symbolCount; i++)
        {
            _lastActionBySymbol[_config.Symbols[i]] = actions[i];
        }

        var moreData = await _executor.AdvanceAsync();
        _currentTick++;

        var pendingCloses = _executor.GetAndClearPendingCloses();
        foreach (var closeInfo in pendingCloses)
        {
            var symbolIndex = Array.IndexOf(_config.Symbols, closeInfo.Symbol);
            if (symbolIndex >= 0)
            {
                double maxDrawdownPct = 0;
                if (_peakEquity > 0)
                    maxDrawdownPct = (_peakEquity - _executor.GetEquity()) / _peakEquity;

                // Get ATR for volatility-normalized rewards
                var symbolAtr = CalculateAtr(closeInfo.Symbol);
                
                // Get symbol info for point-based normalization
                var symbolInfo = _executor.GetSymbolInfo(closeInfo.Symbol);

                var closeReward = _rewardCalculator.Calculate(
                    tradeClosed: true,
                    tradeProfit: closeInfo.Profit,
                    holdingTicks: closeInfo.HoldingTicks,
                    hasPosition: false,
                    unrealizedPnl: 0,
                    peakUnrealizedPnl: _peakUnrealizedPnls.GetValueOrDefault(closeInfo.Symbol),
                    initialBalance: _initialBalance,
                    maxDrawdownPct: maxDrawdownPct,
                    symbolAtr: symbolAtr,
                    symbolInfo: symbolInfo);

                _reusableRewards[symbolIndex] += closeReward;
                _positionOpenTicks[closeInfo.Symbol] = 0;
                _peakUnrealizedPnls[closeInfo.Symbol] = 0;
            }
        }

        var tickedSymbol = _executor.LastTickedSymbol;
        var m1CandleClosed = false;

        if (tickedSymbol != null)
        {
            var symbolIndex = Array.IndexOf(_config.Symbols, tickedSymbol);
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

        var currentTime = _executor.CurrentTime;
        if (_lastM1Minute != currentTime.Minute)
        {
            _lastM1Minute = currentTime.Minute;
            m1CandleClosed = true;
        }

        var currentEquity = _executor.GetEquity();
        if (currentEquity > _peakEquity) _peakEquity = currentEquity;

        // Hard SL enforcement: force close positions with excessive unrealized loss
        // Uses ATR-based threshold to account for symbol volatility
        // This protects against gaps where price skips over the SL level
        foreach (var symbol in _config.Symbols)
        {
            var pos = _executor.GetPosition(symbol);
            if (pos == null) continue;
            
            // Calculate ATR-based hard stop threshold
            // Volatile symbols (BTC) get more room, stable symbols keep tighter stops
            var symbolAtr = CalculateAtr(symbol);
            var currentBid = _executor.GetBid(symbol);
            var atrPct = symbolAtr > 0 && currentBid > 0 
                ? (symbolAtr / currentBid) * 100 * 3  // 3x ATR
                : 2.0;  // Fallback to 2%
            var hardSlThreshold = -Math.Max(atrPct, 2.0);  // At least -2%, up to ~-6% for volatile symbols
            
            if (pos.UnrealizedPnlPercent < hardSlThreshold)
            {
                await _executor.ClosePositionAsync(symbol);
                _positionOpenTicks[symbol] = 0;
                _peakUnrealizedPnls[symbol] = 0;
                Log.Debug("[HardSL] Force closed {Symbol} at {Loss:F2}% loss (threshold: {Threshold:F2}%)", 
                    symbol, pos.UnrealizedPnlPercent, hardSlThreshold);
            }
        }


        var globalDone = !moreData;
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
        var isCloseAction = ActionDecoder.IsClose(action);

        var tradeClosed = false;
        double tradeProfit = 0;
        var holdingTicks = 0;

        var pos = _executor.GetPosition(symbol);
        var hasPosition = pos != null;

        var lastAction = _lastExecutedAction.GetValueOrDefault(symbol, -1);
        var lastTick = _lastExecutedTick.GetValueOrDefault(symbol, -_actionMemoryWindow);
        var ticksSinceLastAction = _currentTick - lastTick;

        if (lastAction == action && ticksSinceLastAction < _actionMemoryWindow)
            return (0, false);

        var currentHoldingTicks = hasPosition ? _currentTick - _positionOpenTicks.GetValueOrDefault(symbol) : 0;

        // Handle explicit CLOSE action - just close position, don't open new one
        if (isCloseAction && hasPosition)
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
                    _lastExecutedAction[symbol] = action;
                    _lastExecutedTick[symbol] = _currentTick;
                }
            }
            // After close action, skip to reward calculation (don't open new position)
        }
        // Close existing position if opening opposite direction
        else if (hasPosition && entryType.HasValue && pos!.Type != entryType.Value)
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

        if (entryType.HasValue && !hasPosition)
        {
            var candle = _executor.GetLastKnownCandle(symbol);
            if (candle != null)
            {
                var atr = CalculateAtr(symbol);
                if (atr <= 0 || double.IsNaN(atr))
                    atr = candle.Close * 0.001;

                var symInfo = _executor.GetSymbolInfo(symbol)
                    ?? new SymbolInfo(symbol, 0.00001, 100000, 1, 0.00001, "USD", "USD", 5);

                var (tpMult, slMult) = _lastTpSlMultipliers.GetValueOrDefault(symbol, (0.5f, 0.5f));

                var tpAtrMult = 1.0 + tpMult * 4.0;
                var slAtrMult = 0.5 + slMult * 2.5;

                var slDistance = atr * slAtrMult;
                var tpDistance = atr * tpAtrMult;

                var bid = _executor.GetBid(symbol);
                var ask = _executor.GetAsk(symbol);

                var sl = entryType.Value == TradeType.Buy
                    ? bid - slDistance
                    : ask + slDistance;

                var tp = entryType.Value == TradeType.Buy
                    ? bid + tpDistance
                    : ask - tpDistance;

                var riskAmount = _executor.GetEquity() * 0.01;
                var slPoints = slDistance / symInfo.Point;
                var tickValue = symInfo.TickValue;
                var volume = tickValue > 0 && slPoints > 0
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
        var unrealized = pos?.UnrealizedPnlPercent / 100.0 ?? 0;

        if (unrealized > _peakUnrealizedPnls.GetValueOrDefault(symbol))
            _peakUnrealizedPnls[symbol] = unrealized;

        double maxDrawdownPct = 0;
        if (_peakEquity > 0)
            maxDrawdownPct = (_peakEquity - _executor.GetEquity()) / _peakEquity;

        // Get ATR for volatility-normalized rewards
        var symbolAtr = CalculateAtr(symbol);
        
        // Get symbol info for point-based normalization
        var symbolInfo = _executor.GetSymbolInfo(symbol);
        
        // Get position direction and price change for shaping rewards
        var posDirection = pos?.Type == TradeType.Buy ? 1 : (pos?.Type == TradeType.Sell ? -1 : 0);
        var currentPrice = _executor.GetBid(symbol);
        var lastCandle = _executor.GetLastKnownCandle(symbol);
        var priceChange = lastCandle != null ? currentPrice - lastCandle.Close : 0;

        var reward = _rewardCalculator.Calculate(
            tradeClosed, tradeProfit, holdingTicks,
            pos != null, unrealized, _peakUnrealizedPnls.GetValueOrDefault(symbol),
            _initialBalance, maxDrawdownPct,
            currentEquity: _executor.GetEquity(),
            positionDirection: posDirection,
            priceChange: priceChange,
            symbolAtr: symbolAtr,
            symbolInfo: symbolInfo,
            currentPrice: currentPrice);  // Pass actual price for opportunity cost

        return (reward, tradeClosed);
    }

    public void SetTpSlMultipliers(string symbol, float tpMult, float slMult)
    {
        _lastTpSlMultipliers[symbol] = (tpMult, slMult);
    }

    public void SetTpSlMultipliersBatch(float[,] tpSlMultipliers)
    {
        for (var i = 0; i < _config.Symbols.Length && i < tpSlMultipliers.GetLength(0); i++)
        {
            _lastTpSlMultipliers[_config.Symbols[i]] = (tpSlMultipliers[i, 0], tpSlMultipliers[i, 1]);
        }
    }

    private double CalculateAtr(string symbol)
    {
        var history = _mtfAggregators[symbol].GetCandles("M1", 50);
        if (history.Count < 15)
            return 0;

        var h = history.Select(c => c.High).ToArray();
        var l = history.Select(c => c.Low).ToArray();
        var c = history.Select(c => c.Close).ToArray();

        var atrSeries = Technicals.Atr(h, l, c, 14);
        return atrSeries[^1];
    }

    /// <summary>
    /// Build structured AgentInput for each symbol.
    /// This is the recommended method for the new ONNX agent interface.
    /// </summary>
    public AgentInput[] BuildAgentInputs()
    {
        var inputs = new AgentInput[_config.Symbols.Length];

        for (var i = 0; i < _config.Symbols.Length; i++)
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

        var allTimeframeCandles = mtfAggregator.GetAllTimeframeCandles(_config.WindowSize + 50);
        foreach (var (timeframe, candles) in allTimeframeCandles)
        {
            if (candles.Count >= _config.WindowSize)
            {
                mtfBuilder.UpdateCandles(timeframe, candles, symbol);
            }
        }

        mtfBuilder.ZeroPadMissingTimeframes();

        var pos = _executor.GetPosition(symbol);
        var hasPosition = pos != null;

        var portfolioFeatures = new float[5];

        var currentBalance = _executor.GetBalance();
        portfolioFeatures[4] = (float)((currentBalance / _initialBalance) - 1.0);

        if (hasPosition)
        {
            portfolioFeatures[0] = pos!.Type == TradeType.Buy ? 1.0f : -1.0f;

            var unrealizedPnl = (pos.CurrentPrice - pos.OpenPrice) / pos.OpenPrice;
            if (pos.Type == TradeType.Sell) unrealizedPnl *= -1;
            portfolioFeatures[1] = (float)(unrealizedPnl * 100.0);

            var heldTicks = _currentTick - _positionOpenTicks.GetValueOrDefault(symbol);
            portfolioFeatures[2] = (float)Math.Min(1.0, heldTicks / (double)_config.MaxHoldingSteps);

            if (unrealizedPnl > _peakUnrealizedPnls.GetValueOrDefault(symbol))
                _peakUnrealizedPnls[symbol] = unrealizedPnl;
            var drawdown = Math.Max(0.0, _peakUnrealizedPnls.GetValueOrDefault(symbol) - unrealizedPnl);
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
        
        // Build DXY features from current prices
        float[]? dxyFeatures = null;
        try
        {
            var currentPrices = new Dictionary<string, double>();
            foreach (var sym in _config.Symbols)
            {
                var price = _executor.GetBid(sym);
                if (price > 0)
                    currentPrices[sym] = price;
            }
            if (currentPrices.Count > 0)
                dxyFeatures = _dxyService.BuildDxyFeatures(currentPrices, _executor.CurrentTime);
        }
        catch (Exception ex) { Log.Debug(ex, "DXY calculation error for {Symbol}", symbol); }
        
        // Build news features for economic events
        float[]? newsFeatures = null;
        try
        {
            newsFeatures = _newsService.BuildNewsFeatures(_executor.CurrentTime, _config.Symbols);
        }
        catch (Exception ex) { Log.Debug(ex, "News feature calculation error"); }
        
        // Build correlation features
        float[]? correlationFeatures = null;
        try
        {
            var currentPrices = new Dictionary<string, double>();
            foreach (var sym in _config.Symbols)
            {
                var price = _executor.GetBid(sym);
                if (price > 0)
                    currentPrices[sym] = price;
            }
            if (currentPrices.Count > 0)
                correlationFeatures = _correlationService.BuildCorrelationFeatures(currentPrices, _config.Symbols);
        }
        catch (Exception ex) { Log.Debug(ex, "Correlation calculation error"); }
        
        // Build portfolio exposure features
        float[]? portfolioExposure = null;
        try
        {
            var positions = _executor.GetPositions().ToList();
            var equity = _executor.GetEquity();
            var freeMargin = _executor.GetFreeMargin();
            var usedMargin = equity - freeMargin;
            portfolioExposure = _exposureService.BuildExposureFeatures(positions, equity, freeMargin, usedMargin, _config.Symbols.Length * 2);
        }
        catch (Exception ex) { Log.Debug(ex, "Exposure calculation error"); }

        // Build time-of-day features for session awareness
        var timeFeatures = MultiTimeframeStateBuilder.BuildTimeFeatures(_executor.CurrentTime);

        return mtfBuilder.BuildAgentInput(
            symbol: symbol,
            portfolioFeatures: portfolioFeatures,
            riskState: riskState,
            closedTimeframes: closedTfs,
            newsFeatures: newsFeatures,
            correlationFeatures: correlationFeatures,
            portfolioExposure: portfolioExposure,
            dxyFeatures: dxyFeatures,
            timeFeatures: timeFeatures
        );
    }

    public Task<float[]> ResetAsync()
    {
        _executor.Reset();

        _rewardCalculator.ResetEpisode();

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
        
        // Reset DXY service
        _dxyService.Reset();
        
        // Reset news service
        _newsService.Reset();
        
        // Reset correlation service
        _correlationService.Reset();
        
        // Reset exposure service
        _exposureService.Reset();

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
        var count = candleCount ?? _config.WindowSize + 50;

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

        _currentTick = 0;
        
        // Preload economic calendar events for the training period
        // This fetches real data from Forex Factory
        try
        {
            var eventStart = episodeStartTime.AddDays(-1);  // Include day before
            var eventEnd = episodeStartTime.AddDays(7);      // Load a week ahead
            await _newsService.LoadEventsAsync(eventStart, eventEnd);
        }
        catch (Exception ex)
        {
            Log.Warning(ex, "[PortfolioTradingEnvironment] Failed to load calendar events");
        }
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

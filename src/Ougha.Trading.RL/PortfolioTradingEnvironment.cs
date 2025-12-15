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
    
    // Hindsight SL: Track price path for optimal SL calculation after trade closes
    private readonly Dictionary<string, double> _positionEntryPrice = new();
    private readonly Dictionary<string, double> _maxAdverseExcursion = new();   // MAE as % of entry
    private readonly Dictionary<string, double> _maxFavorableExcursion = new(); // MFE as % of entry
    private readonly Dictionary<string, double> _positionAtr = new();           // ATR at entry time
    private readonly Dictionary<string, float> _actualPriceChange = new();      // Price change when trade closed (for supervised learning)
    private readonly Dictionary<string, float> _entryPrediction = new();        // Model's prediction at trade entry
    
    // DXY Index service for USD strength features
    private readonly DxyIndexService _dxyService;
    
    // Economic calendar service for news/event features
    private readonly EconomicCalendarService _newsService;
    
    // Cross-symbol correlation service
    private readonly CorrelationService _correlationService;
    
    // Portfolio exposure service
    private readonly PortfolioExposureService _exposureService;
    
    // OPTIMIZATION C3: Cached ATR to avoid recomputation every call
    private readonly Dictionary<string, double> _cachedAtr = new();
    private readonly Dictionary<string, int> _atrLastM1Minute = new();
    
    // OPTIMIZATION C4: Pre-allocated arrays for ATR calculation (avoid LINQ allocations)
    private readonly double[] _atrHighBuffer = new double[50];
    private readonly double[] _atrLowBuffer = new double[50];
    private readonly double[] _atrCloseBuffer = new double[50];
    
    // OPTIMIZATION C2: Cached shared features for BuildAgentInputs (computed once per step)
    private float[]? _cachedSharedDxy;
    private float[]? _cachedSharedNews;
    private float[]? _cachedSharedCorrelation;
    private float[]? _cachedSharedExposure;
    private float[]? _cachedSharedTimeFeatures;
    private Dictionary<string, double>? _cachedPrices;
    private int _cachedFeaturesM1Minute = -1;

    private int _currentTick;
    private double _peakEquity;
    private double _initialBalance;
    private bool _isDone;
    private int _lastM1Minute = -1;
    
    // Daily loss limit tracking
    private double _dailyStartEquity;
    private DateTime _lastDailyReset = DateTime.MinValue;
    private const double DAILY_LOSS_LIMIT = 0.3;  // 3% max daily loss

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
            // Initialize hindsight tracking
            _positionEntryPrice[symbol] = 0;
            _maxAdverseExcursion[symbol] = 0;
            _maxFavorableExcursion[symbol] = 0;
            _positionAtr[symbol] = 0;
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
                    symbol: closeInfo.Symbol,
                    tradeClosed: true,
                    tradeProfit: closeInfo.Profit,
                    hasPosition: false,
                    unrealizedPnl: 0,
                    maxDrawdownPct: maxDrawdownPct,
                    slDistance: closeInfo.SlDistance,
                    symbolInfo: symbolInfo,
                    volume: closeInfo.Volume,
                    closePrice: _executor.GetBid(closeInfo.Symbol),
                    closeReason: MapExitReason(closeInfo.ExitReason));

                _reusableRewards[symbolIndex] += closeReward;
                _positionOpenTicks[closeInfo.Symbol] = 0;
                _peakUnrealizedPnls[closeInfo.Symbol] = 0;
                
                // Compute actual price change for supervised prediction training
                // Use PROFIT SIGN to determine actual direction, not current price (which may have bounced)
                // This ensures prediction accuracy aligns with trade outcome
                var entryPrice = _positionEntryPrice.GetValueOrDefault(closeInfo.Symbol, 0);
                if (entryPrice > 0)
                {
                    // Profit > 0 means price moved in predicted direction
                    // Profit < 0 means price moved against prediction
                    // Scale by approximate percentage (using SL distance as reference)
                    var approxMovePercent = closeInfo.SlDistance > 0 
                        ? closeInfo.Profit / (closeInfo.SlDistance * closeInfo.Volume) 
                        : closeInfo.Profit / entryPrice;
                    
                    // Get entry prediction sign to determine direction
                    var entryPred = _entryPrediction.GetValueOrDefault(closeInfo.Symbol, 0f);
                    
                    // For LONG trades (positive prediction), profit > 0 means price went UP
                    // For SHORT trades (negative prediction), profit > 0 means price went DOWN
                    // So actual direction = sign(prediction) * sign(profit)
                    var actualDirection = Math.Sign(entryPred) * Math.Sign(closeInfo.Profit);
                    var priceChange = (float)(actualDirection * Math.Abs(approxMovePercent) * 0.01);
                    
                    _actualPriceChange[closeInfo.Symbol] = priceChange;
                }
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
            // Reduced from 3x to 2x ATR for tighter risk control in day trading
            var symbolAtr = CalculateAtr(symbol);
            var currentBid = _executor.GetBid(symbol);
            var atrPct = symbolAtr > 0 && currentBid > 0 
                ? (symbolAtr / currentBid) * 100 * 2  // 2x ATR (was 3x)
                : 2.0;  // Fallback to 2%
            var hardSlThreshold = -Math.Max(atrPct, 2.0);  // At least -2%, capped at -3% for day trading
            hardSlThreshold = Math.Max(hardSlThreshold, -3.0);  // Cap at 3% max loss per trade
            
            if (pos.UnrealizedPnlPercent < hardSlThreshold)
            {
                await _executor.ClosePositionAsync(symbol);
                _positionOpenTicks[symbol] = 0;
                _peakUnrealizedPnls[symbol] = 0;
                Log.Debug("[HardSL] Force closed {Symbol} at {Loss:F2}% loss (threshold: {Threshold:F2}%)", 
                    symbol, pos.UnrealizedPnlPercent, hardSlThreshold);
            }
        }

        // Daily loss limit enforcement
        // Reset daily tracking at the start of each new trading day
        if (currentTime.Date != _lastDailyReset.Date)
        {
            _lastDailyReset = currentTime.Date;
            _dailyStartEquity = currentEquity;
        }
        
        // If daily loss exceeds limit, close all positions and stop opening new ones
        var dailyLoss = (_dailyStartEquity - currentEquity) / _dailyStartEquity;
        if (dailyLoss >= DAILY_LOSS_LIMIT)
        {
            foreach (var symbol in _config.Symbols)
            {
                var pos = _executor.GetPosition(symbol);
                if (pos != null)
                {
                    await _executor.ClosePositionAsync(symbol);
                    _positionOpenTicks[symbol] = 0;
                    _peakUnrealizedPnls[symbol] = 0;
                    Log.Debug("[DailyLimit] Force closed {Symbol} - daily loss limit {Loss:F2}% hit", 
                        symbol, dailyLoss * 100);
                }
            }
        }

        // Market hours awareness: force close positions before weekend to avoid gap risk
        // Friday 21:00 UTC is when most forex markets close
        // Skip for crypto symbols (trade 24/7)
        if (currentTime.DayOfWeek == DayOfWeek.Friday && currentTime.Hour >= 21)
        {
            foreach (var symbol in _config.Symbols)
            {
                // Crypto trades 24/7, no weekend gap risk
                var symbolInfo = _executor.GetSymbolInfo(symbol);
                if (symbolInfo?.Category == SymbolCategory.Crypto)
                    continue;
                    
                var pos = _executor.GetPosition(symbol);
                if (pos != null)
                {
                    await _executor.ClosePositionAsync(symbol);
                    _positionOpenTicks[symbol] = 0;
                    _peakUnrealizedPnls[symbol] = 0;
                    Log.Debug("[WeekendClose] Force closed {Symbol} at {Time} to avoid gap risk", 
                        symbol, currentTime);
                }
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
        // HOLD action - do nothing
        if (ActionDecoder.IsHold(action)) return (0, false);

        var entryType = ActionDecoder.Decode(action);
        
        // Not a valid entry action
        if (!entryType.HasValue) return (0, false);

        var pos = _executor.GetPosition(symbol);
        var hasPosition = pos != null;
        
        // If we already have a position, ignore new entry signals
        // Positions can only close via TP/SL
        if (hasPosition)
        {
            return (0, false);  // Do nothing - wait for TP/SL
        }

        var lastAction = _lastExecutedAction.GetValueOrDefault(symbol, -1);
        var lastTick = _lastExecutedTick.GetValueOrDefault(symbol, -_actionMemoryWindow);
        var ticksSinceLastAction = _currentTick - lastTick;

        if (lastAction == action && ticksSinceLastAction < _actionMemoryWindow)
            return (0, false);

        // Open new position
        var candle = _executor.GetLastKnownCandle(symbol);
        if (candle != null)
        {
            var atr = CalculateAtr(symbol);
            if (atr <= 0 || double.IsNaN(atr))
                atr = candle.Close * 0.001;

            var symInfo = _executor.GetSymbolInfo(symbol)
                ?? throw new InvalidOperationException($"SymbolInfo not found for {symbol}. Ensure MT5 is connected and symbol info is loaded.");

            var (tpMult, slMult) = _lastTpSlMultipliers.GetValueOrDefault(symbol, (0.5f, 0.5f));

            // Base multipliers: TP at 1.5 ATR, SL at 1.0 ATR
            var tpAtrMult = 1.5 + tpMult * 4.0;
            var slAtrMult = 1.0 + slMult * 2.5;

            var slDistance = atr * slAtrMult;
            var tpDistance = atr * tpAtrMult;
                
            // Universal minimum SL: 0.1% of price
            var bid = _executor.GetBid(symbol);
            var ask = _executor.GetAsk(symbol);
            var minSlDistance = bid * 0.001;
            
            if (slDistance < minSlDistance)
            {
                var ratio = slDistance > 0 ? tpDistance / slDistance : 2.0;
                slDistance = minSlDistance;
                tpDistance = slDistance * ratio;
            }
            
            // Add realistic slippage modeling
            var spread = ask - bid;
            var slippage = spread * 0.5;
            var effectiveBid = bid - slippage;
            var effectiveAsk = ask + slippage;

            var sl = entryType.Value == TradeType.Buy
                ? effectiveBid - slDistance
                : effectiveAsk + slDistance;

            var tp = entryType.Value == TradeType.Buy
                ? effectiveBid + tpDistance
                : effectiveAsk - tpDistance;

            var riskAmount = _executor.GetEquity() * 0.003;
            var slPoints = slDistance / symInfo.Point;
            var tickValue = symInfo.TickValue;
            
            if (tickValue <= 0)
            {
                throw new ArgumentException(
                    $"Invalid TickValue ({tickValue}) for symbol {symbol}. " +
                    $"Ensure MT5 is connected and SymbolInfo is properly loaded.");
            }
            
            var volume = slPoints > 0
                ? riskAmount / (tickValue * slPoints)
                : 0.01;
                
            volume = Math.Max(0.01, Math.Min(volume, 0.1));
            volume = Math.Round(volume, 2);
            
            if (volume >= 0.5)
            {
                Log.Debug("[VolumeCalc] {Symbol}: risk={RiskAmt:F2}, slPts={SlPts:F1}, tickVal={TickVal:F4}, vol={Vol:F2}",
                    symbol, riskAmount, slPoints, tickValue, volume);
            }

            var result = await _executor.ExecuteAsync(symbol, entryType.Value, volume,
                sl, tp, "RL Portfolio Agent", RiskLevel.Moderate);

            if (result.Success)
            {
                _positionOpenTicks[symbol] = _currentTick;
                _peakUnrealizedPnls[symbol] = 0;
                _lastExecutedAction[symbol] = action;
                _lastExecutedTick[symbol] = _currentTick;
                
                // Initialize hindsight SL tracking
                var entryPrice = entryType.Value == TradeType.Buy ? ask : bid;
                _positionEntryPrice[symbol] = entryPrice;
                _maxAdverseExcursion[symbol] = 0;
                _maxFavorableExcursion[symbol] = 0;
                _positionAtr[symbol] = atr;
                
                // Entry quality bonus: reward entries aligned with momentum
                var candlePriceChange = candle.Close - candle.Open;
                var momentum = candlePriceChange / (atr > 0 ? atr : candle.Close * 0.001);
                
                var entryQualityBonus = 0f;
                if (entryType.Value == TradeType.Buy && momentum > 0.2)
                {
                    entryQualityBonus = Math.Min(0.5f, (float)momentum * 0.3f);
                }
                else if (entryType.Value == TradeType.Sell && momentum < -0.2)
                {
                    entryQualityBonus = Math.Min(0.5f, (float)Math.Abs(momentum) * 0.3f);
                }
                else if ((entryType.Value == TradeType.Buy && momentum < -0.2) ||
                         (entryType.Value == TradeType.Sell && momentum > 0.2))
                {
                    entryQualityBonus = -0.1f;
                }
                
                if (Math.Abs(entryQualityBonus) > 0.01f)
                {
                    return (entryQualityBonus, false);
                }
            }
        }

        pos = _executor.GetPosition(symbol);
        var unrealized = pos?.UnrealizedPnlPercent / 100.0 ?? 0;

        if (unrealized > _peakUnrealizedPnls.GetValueOrDefault(symbol))
            _peakUnrealizedPnls[symbol] = unrealized;
        
        // Update MAE/MFE for hindsight SL calculation
        if (pos != null && _positionEntryPrice.GetValueOrDefault(symbol) > 0)
        {
            var entry = _positionEntryPrice[symbol];
            var current = pos.Type == TradeType.Buy 
                ? _executor.GetBid(symbol)  // Bid for closing longs
                : _executor.GetAsk(symbol); // Ask for closing shorts
            var direction = pos.Type == TradeType.Buy ? 1 : -1;
            
            // Excursion as percentage of entry price
            var priceMove = (current - entry) / entry * direction;
            
            if (priceMove > 0)
                _maxFavorableExcursion[symbol] = Math.Max(_maxFavorableExcursion[symbol], priceMove);
            else
                _maxAdverseExcursion[symbol] = Math.Max(_maxAdverseExcursion[symbol], Math.Abs(priceMove));
        }

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

        // Get holding ticks for progressive holding bonus (reuse existing var or compute fresh)
        var rewardHoldingTicks = pos != null ? _currentTick - _positionOpenTicks.GetValueOrDefault(symbol) : 0;

        var reward = _rewardCalculator.Calculate(
            symbol: symbol,
            tradeClosed: false,  // Positions only close via TP/SL (handled in OnPendingClose)
            tradeProfit: 0,
            hasPosition: pos != null,
            unrealizedPnl: unrealized,
            maxDrawdownPct: maxDrawdownPct,
            slDistance: 0,
            symbolInfo: symbolInfo,
            volume: pos?.Volume ?? 0,
            closePrice: currentPrice,
            closeReason: CloseReason.Unknown,
            holdingTicks: rewardHoldingTicks);

        return (reward, false);  // Never close from here - wait for TP/SL
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

    /// <summary>
    /// Compute optimal SL multiplier based on hindsight (MAE tracking).
    /// Call this when a trade closes to get the training target.
    /// Returns -1 if no tracking data available.
    /// </summary>
    public float GetHindsightSlMultiplier(string symbol)
    {
        var mae = _maxAdverseExcursion.GetValueOrDefault(symbol, 0);
        var atr = _positionAtr.GetValueOrDefault(symbol, 0);
        var entry = _positionEntryPrice.GetValueOrDefault(symbol, 0);
        
        if (mae <= 0 || atr <= 0 || entry <= 0)
            return -1f;  // No valid data
        
        // Convert MAE percentage to price distance
        var maePrice = entry * mae;
        
        // Optimal SL = MAE + 15% buffer (to not get stopped out on winning trades)
        var optimalSlDistance = maePrice * 1.15;
        
        // Convert to ATR multiplier: SL distance / ATR
        var optimalSlAtrMult = optimalSlDistance / atr;
        
        // Normalize to [0,1] range: formula is (mult - 0.5) / 2.5
        // Where 0.5 ATR -> 0, 3.0 ATR -> 1
        var hindsightSlMult = (float)((optimalSlAtrMult - 0.5) / 2.5);
        hindsightSlMult = Math.Clamp(hindsightSlMult, 0.1f, 1.0f);
        
        return hindsightSlMult;
    }
    
    /// <summary>
    /// Clear hindsight tracking for a symbol (call after trade closes).
    /// </summary>
    public void ClearHindsightTracking(string symbol)
    {
        _positionEntryPrice[symbol] = 0;
        _maxAdverseExcursion[symbol] = 0;
        _maxFavorableExcursion[symbol] = 0;
        _positionAtr[symbol] = 0;
        _actualPriceChange[symbol] = -999f;  // Reset to sentinel
        _entryPrediction[symbol] = 0f;       // Clear entry prediction
    }
    
    /// <summary>
    /// Get the actual price change from the last closed trade for this symbol.
    /// Returns -999 if no trade closed.
    /// </summary>
    public float GetActualPriceChange(string symbol)
    {
        return _actualPriceChange.GetValueOrDefault(symbol, -999f);
    }
    
    /// <summary>
    /// Store the model's prediction at trade entry.
    /// Call this when opening a new position.
    /// </summary>
    public void SetEntryPrediction(string symbol, float prediction)
    {
        _entryPrediction[symbol] = prediction;
    }
    
    /// <summary>
    /// Get the entry prediction for computing accuracy.
    /// Returns 0 if no prediction stored.
    /// </summary>
    public float GetEntryPrediction(string symbol)
    {
        return _entryPrediction.GetValueOrDefault(symbol, 0f);
    }
    
    /// <summary>
    /// Map ExitReason to CloseReason for reward calculation.
    /// </summary>
    private static CloseReason MapExitReason(Backtesting.ExitReason exitReason)
    {
        return exitReason switch
        {
            Backtesting.ExitReason.Manual => CloseReason.Manual,
            Backtesting.ExitReason.Signal => CloseReason.Manual,
            Backtesting.ExitReason.TakeProfit => CloseReason.TakeProfit,
            Backtesting.ExitReason.StopLoss => CloseReason.StopLoss,
            Backtesting.ExitReason.EndOfBacktest => CloseReason.EndOfEpisode,
            _ => CloseReason.Unknown
        };
    }

    /// <summary>
    /// Get cached ATR value, only recomputing when M1 candle closes.
    /// OPTIMIZATION C3: Reduces ATR calculations by 60x.
    /// </summary>
    private double CalculateAtr(string symbol)
    {
        // FIX: Don't use cache when _lastM1Minute is -1 (no M1 candles yet)
        // Check if cached value is still valid (same M1 minute and valid minute)
        if (_lastM1Minute >= 0
            && _atrLastM1Minute.TryGetValue(symbol, out var lastMinute) 
            && lastMinute == _lastM1Minute 
            && _cachedAtr.TryGetValue(symbol, out var cached)
            && cached > 0)  // Ensure we have a valid cached value
        {
            return cached;
        }
        
        var history = _mtfAggregators[symbol].GetCandles("M1", 50);
        if (history.Count < 15)
            return 0;

        // OPTIMIZATION C4: Use pre-allocated buffers instead of LINQ ToArray()
        var count = Math.Min(history.Count, 50);
        for (var i = 0; i < count; i++)
        {
            _atrHighBuffer[i] = history[i].High;
            _atrLowBuffer[i] = history[i].Low;
            _atrCloseBuffer[i] = history[i].Close;
        }

        var atrSeries = Technicals.Atr(
            _atrHighBuffer.AsSpan(0, count).ToArray(), 
            _atrLowBuffer.AsSpan(0, count).ToArray(), 
            _atrCloseBuffer.AsSpan(0, count).ToArray(), 14);
        var atrValue = atrSeries[^1];
        
        // Only cache if we have a valid M1 minute
        if (_lastM1Minute >= 0)
        {
            _cachedAtr[symbol] = atrValue;
            _atrLastM1Minute[symbol] = _lastM1Minute;
        }
        
        return atrValue;
    }

    /// <summary>
    /// Check if symbol has valid price data for trading decisions.
    /// Returns false if:
    /// - No current bid/ask price (market closed or no data)
    /// - Price is zero or invalid
    /// - Weekend/holiday for forex pairs
    /// </summary>
    public bool HasValidPriceData(string symbol)
    {
        var bid = _executor.GetBid(symbol);
        var ask = _executor.GetAsk(symbol);
        
        // No valid price
        if (bid <= 0 || ask <= 0 || double.IsNaN(bid) || double.IsNaN(ask))
            return false;
        
        // Check for weekend on forex pairs (crypto trades 24/7)
        var timestamp = _executor.CurrentTime;
        if (IsForexPair(symbol) && IsWeekend(timestamp))
            return false;
        
        // Note: Don't check candle count here - feature building handles missing data with zero-padding
        // This check is only for bid/ask availability
        return true;
    }
    
    /// <summary>
    /// Get valid data mask for all symbols (true = has valid data, can trade).
    /// Use this for action masking to force HOLD on symbols without data.
    /// </summary>
    public bool[] GetValidDataMask()
    {
        return _config.Symbols.Select(HasValidPriceData).ToArray();
    }
    
    /// <summary>
    /// Get current bid prices for all symbols (for M1-level price change tracking).
    /// Used to compute dense supervised signal in experience collection.
    /// </summary>
    public double[] GetCurrentPrices()
    {
        var prices = new double[_config.Symbols.Length];
        for (var i = 0; i < _config.Symbols.Length; i++)
        {
            prices[i] = Executor.GetBid(_config.Symbols[i]);
        }
        return prices;
    }
    
    /// <summary>
    /// Check if symbol is a forex pair (trades only during market hours).
    /// Crypto pairs like BTCUSD trade 24/7.
    /// </summary>
    private static bool IsForexPair(string symbol)
    {
        // Crypto pairs that trade 24/7
        var cryptoSymbols = new[] { "BTCUSD", "BTCJPY", "BTCEUR", "ETHUSD", "ETHBTC", "LTCUSD",
            "BTCXAG", "BTCXAU", "BTCZAR", "BTCAUD", "BTCCNH" };
        
        foreach (var crypto in cryptoSymbols)
        {
            if (symbol.Equals(crypto, StringComparison.OrdinalIgnoreCase))
                return false;
        }
        
        // All others are forex/stock/commodities that close on weekends
        return true;
    }
    
    /// <summary>
    /// Check if timestamp falls on weekend (Saturday after close or Sunday before open).
    /// Forex market closes Friday 22:00 UTC and opens Sunday 22:00 UTC.
    /// </summary>
    private static bool IsWeekend(DateTime timestamp)
    {
        var utc = timestamp.ToUniversalTime();
        
        // Saturday all day
        if (utc.DayOfWeek == DayOfWeek.Saturday)
            return true;
        
        // Sunday before 22:00 UTC
        if (utc.DayOfWeek == DayOfWeek.Sunday && utc.Hour < 22)
            return true;
        
        // Friday after 22:00 UTC
        if (utc.DayOfWeek == DayOfWeek.Friday && utc.Hour >= 22)
            return true;
        
        return false;
    }


    /// <summary>
    /// Build structured AgentInput for each symbol.
    /// OPTIMIZATION C2: Compute shared features ONCE, then reuse for all symbols.
    /// </summary>
    public AgentInput[] BuildAgentInputs()
    {
        var inputs = new AgentInput[_config.Symbols.Length];
        
        // OPTIMIZATION C2: Compute shared features ONCE per M1 candle
        // These features are the same for all symbols, so computing them once saves 90%
        RefreshSharedFeaturesIfNeeded();

        for (var i = 0; i < _config.Symbols.Length; i++)
        {
            var symbol = _config.Symbols[i];
            inputs[i] = BuildAgentInputForSymbolOptimized(symbol);
        }

        return inputs;
    }
    
    /// <summary>
    /// Refresh shared features cache if M1 minute changed.
    /// OPTIMIZATION C2: Avoids recomputing same data for each symbol.
    /// </summary>
    private void RefreshSharedFeaturesIfNeeded()
    {
        // FIX: Always compute on first call (_cachedFeaturesM1Minute starts at -1)
        // Also ensure we recompute when minute changes
        var needsRefresh = _cachedSharedDxy == null 
                          || _cachedFeaturesM1Minute != _lastM1Minute
                          || _cachedFeaturesM1Minute == -1;
        
        if (!needsRefresh)
            return;
            
        _cachedFeaturesM1Minute = _lastM1Minute;
        
        // Build price dictionary ONCE
        _cachedPrices ??= new Dictionary<string, double>(_config.Symbols.Length);
        _cachedPrices.Clear();
        foreach (var sym in _config.Symbols)
        {
            var price = _executor.GetBid(sym);
            if (price > 0)
                _cachedPrices[sym] = price;
        }
        
        // Compute shared features ONCE - set to null first to ensure we don't use stale data
        _cachedSharedDxy = null;
        _cachedSharedNews = null;
        _cachedSharedCorrelation = null;
        _cachedSharedExposure = null;
        
        try
        {
            if (_cachedPrices.Count > 0)
                _cachedSharedDxy = _dxyService.BuildDxyFeatures(_cachedPrices, _executor.CurrentTime);
        }
        catch (Exception ex) { Log.Debug(ex, "DXY calculation error"); }
        
        try
        {
            _cachedSharedNews = _newsService.BuildNewsFeatures(_executor.CurrentTime, _config.Symbols);
        }
        catch (Exception ex) { Log.Debug(ex, "News feature calculation error"); }
        
        try
        {
            if (_cachedPrices.Count > 0)
                _cachedSharedCorrelation = _correlationService.BuildCorrelationFeatures(_cachedPrices, _config.Symbols);
        }
        catch (Exception ex) { Log.Debug(ex, "Correlation calculation error"); }
        
        try
        {
            var positions = _executor.GetPositions().ToList();
            var equity = _executor.GetEquity();
            var freeMargin = _executor.GetFreeMargin();
            var usedMargin = equity - freeMargin;
            _cachedSharedExposure = _exposureService.BuildExposureFeatures(positions, equity, freeMargin, usedMargin, _config.Symbols.Length * 2);
        }
        catch (Exception ex) { Log.Debug(ex, "Exposure calculation error"); }
        
        _cachedSharedTimeFeatures = MultiTimeframeStateBuilder.BuildTimeFeatures(_executor.CurrentTime);
    }



    /// <summary>
    /// Build structured AgentInput for a single symbol.
    /// DEPRECATED: Use BuildAgentInputForSymbolOptimized via BuildAgentInputs() for better performance.
    /// </summary>
    private AgentInput BuildAgentInputForSymbol(string symbol)
    {
        // Fallback to ensure shared features are computed
        RefreshSharedFeaturesIfNeeded();
        return BuildAgentInputForSymbolOptimized(symbol);
    }
    
    /// <summary>
    /// OPTIMIZED: Build AgentInput using cached shared features.
    /// OPTIMIZATION C2: Shared features (DXY, correlation, etc.) are computed once per step.
    /// </summary>
    private AgentInput BuildAgentInputForSymbolOptimized(string symbol)
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
        
        // OPTIMIZATION C2: Use cached shared features instead of recomputing
        return mtfBuilder.BuildAgentInput(
            symbol: symbol,
            portfolioFeatures: portfolioFeatures,
            riskState: riskState,
            closedTimeframes: closedTfs,
            newsFeatures: _cachedSharedNews,
            correlationFeatures: _cachedSharedCorrelation,
            portfolioExposure: _cachedSharedExposure,
            dxyFeatures: _cachedSharedDxy,
            timeFeatures: _cachedSharedTimeFeatures
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
        
        // Reset daily loss tracking
        _dailyStartEquity = _initialBalance;
        _lastDailyReset = DateTime.MinValue;

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
            // Reset hindsight SL tracking
            _positionEntryPrice[symbol] = 0;
            _maxAdverseExcursion[symbol] = 0;
            _maxFavorableExcursion[symbol] = 0;
            _positionAtr[symbol] = 0;
        }
        
        // Reset DXY service
        _dxyService.Reset();
        
        // Reset news service
        _newsService.Reset();
        
        // Reset correlation service
        _correlationService.Reset();
        
        // Reset exposure service
        _exposureService.Reset();
        
        // OPTIMIZATION: Reset cached values
        _cachedAtr.Clear();
        _atrLastM1Minute.Clear();
        _cachedFeaturesM1Minute = -1;
        _cachedSharedDxy = null;
        _cachedSharedNews = null;
        _cachedSharedCorrelation = null;
        _cachedSharedExposure = null;
        _cachedSharedTimeFeatures = null;
        _cachedPrices?.Clear();

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
            ["H4"] = "h4",
            ["D1"] = "d1"
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

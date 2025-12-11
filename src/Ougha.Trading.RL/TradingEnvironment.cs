using RLMatrix;
using Ougha.Trading.Backtesting;
using Ougha.Trading.Core.Abstractions;
using Ougha.Trading.Core.Models;
using Ougha.Trading.Data;
using Ougha.Trading.Risk;
using Ougha.Trading.Features.Indicators;

namespace Ougha.Trading.RL;

// StateBuilder Helper
public class StateBuilder
{
    private readonly int _windowSize;
    private readonly IFeatureBuilder _featureBuilder;
    private readonly int _featuresPerCandle;

    public StateBuilder(IFeatureBuilder featureBuilder, int windowSize)
    {
        _featureBuilder = featureBuilder;
        _windowSize = windowSize;
        _featuresPerCandle = featureBuilder.FeatureCount;
    }

    public int GetStateSize()
    {
        // Reference: base_features = window_size * len(self.feature_columns) + 4 (portfolio)
        return _windowSize * _featuresPerCandle + 4; 
    }

    public float[] BuildState(
        IReadOnlyList<Candle> recentCandles, 
        string symbol,
        bool hasPosition, 
        TradeType positionType,
        double unrealizedPnlPct,
        double holdingTimeNorm,
        double drawdownPct)
    {
        // 1. Market Features
        var marketFeatures = _featureBuilder.BuildFlattenedFeatures(recentCandles, symbol, _windowSize);
        
        // 2. Portfolio Features (4 dims)
        float[] portfolioFeatures = new float[4];
        
        // Position Type: 1 (Long), -1 (Short), 0 (Flat)
        if (!hasPosition) portfolioFeatures[0] = 0f;
        else portfolioFeatures[0] = positionType == TradeType.Buy ? 1f : -1f;

        portfolioFeatures[1] = (float)unrealizedPnlPct * 100f; // Scale
        portfolioFeatures[2] = (float)holdingTimeNorm;
        portfolioFeatures[3] = (float)drawdownPct * 100f; // Scale

        // Concatenate
        var state = new float[marketFeatures.Length + 4];
        Array.Copy(marketFeatures, state, marketFeatures.Length);
        Array.Copy(portfolioFeatures, 0, state, marketFeatures.Length, 4);

        return state;
    }
}

public class TradingEnvironment
{
    private readonly BacktestExecutor _executor;
    private readonly StateBuilder _stateBuilder;
    private readonly RewardCalculator _rewardCalculator;
    private readonly EnvironmentConfig _config;
    private readonly PortfolioManager _portfolioManager;
    private readonly IFeatureBuilder _featureBuilder;
    
    // State
    private int _startTick;
    private int _currentTick;
    private double _peakUnrealizedPnl;
    private double _peakEquity; // For MDD tracking
    private int _positionOpenTick;
    private double _initialBalance;
    private bool _isDone;

    // Buffer for candles
    private readonly CandleBuilder _candleBuilder;
    
    public int StateSize => _stateBuilder.GetStateSize();
    public int ActionSize => 8;

    public TradingEnvironment(
        BacktestExecutor executor,
        IFeatureBuilder featureBuilder,
        PortfolioManager portfolioManager,
        RewardCalculator rewardCalculator,
        EnvironmentConfig config)
    {
        _executor = executor;
        _featureBuilder = featureBuilder;
        _portfolioManager = portfolioManager;
        _rewardCalculator = rewardCalculator;
        _config = config;
        _stateBuilder = new StateBuilder(featureBuilder, config.WindowSize);
        _initialBalance = executor.GetBalance();
        _peakEquity = _initialBalance;
        // Assume M1 for now or parse from config.Timeframe
        _candleBuilder = new CandleBuilder(TimeSpan.FromMinutes(1)); 
    }
    
    public async Task<(float[] NextState, float Reward, bool Done)> StepAsync(int action)
    {
        if (_isDone) return (new float[StateSize], 0, true);

        var entryType = ActionDecoder.Decode(action);

        bool tradeClosed = false;
        double tradeProfit = 0;
        int holdingTicks = 0;

        var pos = _executor.GetPosition(_config.Symbol);
        bool hasPosition = pos != null;

        // Close position if opposite direction requested
        if (hasPosition && entryType.HasValue && pos!.Type != entryType.Value)
        {
            var result = await _executor.ClosePositionAsync(_config.Symbol);
            if (result.Success)
            {
                tradeClosed = true;
                tradeProfit = result.Profit;
                holdingTicks = _currentTick - _positionOpenTick;
                _positionOpenTick = 0;
                hasPosition = false;
            }
        }

        if (entryType.HasValue && !hasPosition)
        {
            var candle = _executor.GetLastKnownCandle(_config.Symbol);
            if (candle != null)
            {
                var history = _candleBuilder.GetRecent(50);
                double atr = 0;

                if (history.Count >= 15)
                {
                    double[] h = history.Select(c => c.High).ToArray();
                    double[] l = history.Select(c => c.Low).ToArray();
                    double[] c = history.Select(c => c.Close).ToArray();

                    var atrSeries = Technicals.Atr(h, l, c, 14);
                    atr = atrSeries[^1];
                }

                if (atr <= 0 || double.IsNaN(atr))
                    atr = candle.Close * 0.001;

                var symInfo = _executor.GetSymbolInfo(_config.Symbol)
                    ?? new SymbolInfo(_config.Symbol, 0.00001, 100000, 1, 0.00001, "USD", "USD", 5);

                // Use Close (or calc bid/ask from close)
                var bid = _executor.GetBid(_config.Symbol);

                var sizing = _portfolioManager.CalculatePositionSize(
                    _config.Symbol, entryType.Value, RiskLevel.Moderate,
                    _executor.GetEquity(), bid, atr, symInfo);

                await _executor.ExecuteAsync(_config.Symbol, entryType.Value, sizing.Volume, sizing.StopLoss, sizing.TakeProfit, "RL Agent", RiskLevel.Moderate);
                _positionOpenTick = _currentTick;
                _peakUnrealizedPnl = 0;
            }
        }

        // Advance Simulation
        bool moreData = await _executor.AdvanceAsync();
        _currentTick++;

        // Process automatic TP/SL closes that happened during AdvanceAsync
        // CRITICAL: This gives the agent reward signals for profitable closes!
        var pendingCloses = _executor.GetAndClearPendingCloses();
        foreach (var closeInfo in pendingCloses)
        {
            if (closeInfo.Symbol == _config.Symbol)
            {
                tradeClosed = true;
                tradeProfit = closeInfo.Profit;
                holdingTicks = closeInfo.HoldingTicks;
                _positionOpenTick = 0;
                _peakUnrealizedPnl = 0;
            }
        }

        // PnL/State for Reward
        pos = _executor.GetPosition(_config.Symbol); // Re-fetch
        double unrealized = pos?.UnrealizedPnlPercent / 100.0 ?? 0;
        if (unrealized > _peakUnrealizedPnl) _peakUnrealizedPnl = unrealized;

        // Track Equity & Drawdown
        double currentEquity = _executor.GetEquity();
        if (currentEquity > _peakEquity) _peakEquity = currentEquity;

        double maxDrawdownPct = 0;
        if (_peakEquity > 0)
            maxDrawdownPct = (_peakEquity - currentEquity) / _peakEquity;

        // Holding Time
        double holdingTimeNorm = 0;
        if (pos != null)
        {
             // _positionOpenTick is updated on open.
             int heldTicks = _currentTick - _positionOpenTick;
             // Normalize: max holding steps (e.g. 500?)
             // Use value from config or default
             double maxSteps = _config.MaxHoldingSteps > 0 ? _config.MaxHoldingSteps : 1000.0;
             holdingTimeNorm = Math.Min(1.0, heldTicks / maxSteps);
        }

        // Calculate Reward
        float reward = _rewardCalculator.Calculate(
            tradeClosed, tradeProfit, holdingTicks,
            pos != null, unrealized, _peakUnrealizedPnl,
            _initialBalance, maxDrawdownPct);

        // Check Termination
        bool done = !moreData;
        if (currentEquity < _initialBalance * (1 - _config.MaxLossPercent/100.0))
            done = true;
            
        _isDone = done;

        // Build Next State
        // Feed current candle to CandleBuilder?
        // Wait, _candleBuilder builds from Ticks. 
        // If we have Candles natively, we might NOT need _candleBuilder tick aggregation!
        // We can just feed the candle directly if we update CandleBuilder to generic,
        // OR just use the candle list directly for StateBuilder.
        // Assuming CandleBuilder aggregates ticks to M1. If we are inputting S1 candles, 
        // we might still want to aggregate them to M1.
        // Let's check CandleBuilder. It probably takes AddTick.
        var currentCandle = _executor.GetLastKnownCandle(_config.Symbol);
        // We can synthesize a tick from candle close for compatibility or update CandleBuilder 
        if (currentCandle != null) 
        {
             // Hack: AddTick using Candle data to keep logic simple for now
             _candleBuilder.AddTick(new Tick(currentCandle.Time, currentCandle.Close, currentCandle.Close, 1, false, currentCandle.Volume));
        }

        var candles = _candleBuilder.GetRecent(_config.WindowSize);
        
        var nextState = _stateBuilder.BuildState(
            candles, 
            _config.Symbol, 
            pos != null, 
            pos?.Type ?? TradeType.Buy, 
            unrealized, 
            holdingTimeNorm, 
            maxDrawdownPct
        );

        return (nextState, reward, done);
    }
    
    public Task<float[]> ResetAsync()
    {
        // Reset Logic - no warmup skip needed since candles are preloaded from materialized views
        _currentTick = 0;
        _isDone = false;
        return Task.FromResult(new float[StateSize]);
    }
}

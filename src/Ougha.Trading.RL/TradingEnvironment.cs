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

        // Decode Action
        bool isClose = ActionDecoder.IsClose(action);
        var (entryType, riskLevel) = ActionDecoder.Decode(action);

        bool tradeClosed = false;
        double tradeProfit = 0;
        int holdingTicks = 0;

        // Current State Info (Before Action)
        var pos = _executor.GetPosition(_config.Symbol);
        bool hasPosition = pos != null;

        // Execute Action - MATCHING PYTHON BEHAVIOR:
        // - BUY/SELL only work when NOT has_position (no flipping allowed)
        // - CLOSE only works when has_position
        if (isClose && hasPosition)
        {
            var result = await _executor.ClosePositionAsync(_config.Symbol);
            if (result.Success)
            {
                tradeClosed = true;
                tradeProfit = result.Profit;
                holdingTicks = _currentTick - _positionOpenTick;
                _positionOpenTick = 0;
            }
        }
        else if (entryType.HasValue && riskLevel.HasValue && !hasPosition)
        {
            // Python: BUY/SELL only execute when NOT has_position
            var tick = _executor.GetLastKnownTick(_config.Symbol);
            if (tick != null)
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
                    atr = tick.Bid * 0.001;

                var symInfo = _executor.GetSymbolInfo(_config.Symbol)
                    ?? new SymbolInfo(_config.Symbol, 0.00001, 100000, 1, 0.00001, "USD", "USD", 5);

                var sizing = _portfolioManager.CalculatePositionSize(
                    _config.Symbol, entryType.Value, riskLevel.Value,
                    _executor.GetEquity(), tick.Bid, atr, symInfo);

                await _executor.ExecuteAsync(_config.Symbol, entryType.Value, sizing.Volume, sizing.StopLoss, sizing.TakeProfit, "RL Agent", riskLevel.Value);
                _positionOpenTick = _currentTick;
                _peakUnrealizedPnl = 0;
            }
        }
        // If has_position and trying to BUY/SELL, do nothing (matches Python)

        // Advance Simulation
        bool moreData = await _executor.AdvanceAsync();
        _currentTick++;
        
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
        // Feed current tick to CandleBuilder
        var currentTick = _executor.GetLastKnownTick(_config.Symbol);
        if (currentTick != null) 
        {
            _candleBuilder.AddTick(currentTick);
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
        // Reset Logic
        _currentTick = _config.WindowSize * 100; // Skip warmup
        _isDone = false;
        return Task.FromResult(new float[StateSize]);
    }
}

using Ougha.Trading.Backtesting;
using Ougha.Trading.Core.Abstractions;
using Ougha.Trading.Core.Models;
using Ougha.Trading.Risk;
using Ougha.Trading.Features.Indicators;

namespace Ougha.Trading.RL;

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
        var marketFeatures = _featureBuilder.BuildFlattenedFeatures(recentCandles, symbol, _windowSize);

        var portfolioFeatures = new float[4];

        if (!hasPosition) portfolioFeatures[0] = 0f;
        else portfolioFeatures[0] = positionType == TradeType.Buy ? 1f : -1f;

        portfolioFeatures[1] = (float)unrealizedPnlPct * 100f;
        portfolioFeatures[2] = (float)holdingTimeNorm;
        portfolioFeatures[3] = (float)drawdownPct * 100f;

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

    private int _currentTick;
    private double _peakUnrealizedPnl;
    private double _peakEquity;
    private int _positionOpenTick;
    private readonly double _initialBalance;
    private bool _isDone;

    private readonly CandleBuilder _candleBuilder;

    private int StateSize => _stateBuilder.GetStateSize();

    public TradingEnvironment(
        BacktestExecutor executor,
        IFeatureBuilder featureBuilder,
        PortfolioManager portfolioManager,
        RewardCalculator rewardCalculator,
        EnvironmentConfig config)
    {
        _executor = executor;
        _portfolioManager = portfolioManager;
        _rewardCalculator = rewardCalculator;
        _config = config;
        _stateBuilder = new StateBuilder(featureBuilder, config.WindowSize);
        _initialBalance = executor.GetBalance();
        _peakEquity = _initialBalance;
        _candleBuilder = new CandleBuilder(TimeSpan.FromMinutes(1)); 
    }
    
    public async Task<(float[] NextState, float Reward, bool Done)> StepAsync(int action)
    {
        if (_isDone) return (new float[StateSize], 0, true);

        var entryType = ActionDecoder.Decode(action);

        var tradeClosed = false;
        double tradeProfit = 0;
        var holdingTicks = 0;

        var pos = _executor.GetPosition(_config.Symbol);
        var hasPosition = pos != null;

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
                    var h = history.Select(c => c.High).ToArray();
                    var l = history.Select(c => c.Low).ToArray();
                    var c = history.Select(c => c.Close).ToArray();

                    var atrSeries = Technicals.Atr(h, l, c, 14);
                    atr = atrSeries[^1];
                }

                if (atr <= 0 || double.IsNaN(atr))
                    atr = candle.Close * 0.001;

                var symInfo = _executor.GetSymbolInfo(_config.Symbol)
                    ?? new SymbolInfo(_config.Symbol, 0.00001, 100000, 1, 0.00001, "USD", "USD", 5);

                var bid = _executor.GetBid(_config.Symbol);

                var sizing = _portfolioManager.CalculatePositionSize(
                    _config.Symbol, entryType.Value, RiskLevel.Moderate,
                    _executor.GetEquity(), bid, atr, symInfo);

                await _executor.ExecuteAsync(_config.Symbol, entryType.Value, sizing.Volume, sizing.StopLoss, sizing.TakeProfit, "RL Agent", RiskLevel.Moderate);
                _positionOpenTick = _currentTick;
                _peakUnrealizedPnl = 0;
            }
        }

        var moreData = await _executor.AdvanceAsync();
        _currentTick++;

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

        pos = _executor.GetPosition(_config.Symbol);
        var unrealized = pos?.UnrealizedPnlPercent / 100.0 ?? 0;
        if (unrealized > _peakUnrealizedPnl) _peakUnrealizedPnl = unrealized;

        var currentEquity = _executor.GetEquity();
        if (currentEquity > _peakEquity) _peakEquity = currentEquity;

        double maxDrawdownPct = 0;
        if (_peakEquity > 0)
            maxDrawdownPct = (_peakEquity - currentEquity) / _peakEquity;

        double holdingTimeNorm = 0;
        if (pos != null)
        {
            var heldTicks = _currentTick - _positionOpenTick;
            var maxSteps = _config.MaxHoldingSteps > 0 ? _config.MaxHoldingSteps : 1000.0;
             holdingTimeNorm = Math.Min(1.0, heldTicks / maxSteps);
        }

        var reward = _rewardCalculator.Calculate(
            tradeClosed, tradeProfit, holdingTicks,
            pos != null, unrealized, _peakUnrealizedPnl,
            _initialBalance, maxDrawdownPct);

        var done = !moreData;
        if (currentEquity < _initialBalance * (1 - _config.MaxLossPercent/100.0))
            done = true;
            
        _isDone = done;

        var currentCandle = _executor.GetLastKnownCandle(_config.Symbol);
        if (currentCandle != null) 
        {
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
        _currentTick = 0;
        _isDone = false;
        return Task.FromResult(new float[StateSize]);
    }
}

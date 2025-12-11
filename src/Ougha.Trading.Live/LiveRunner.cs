using Ougha.Trading.Core.Abstractions;
using Ougha.Trading.Core.Models;
using Ougha.Trading.Data;
using Ougha.Trading.RL;
using Ougha.Trading.Risk;

namespace Ougha.Trading.Live;

/// <summary>
/// Orchestrates Live Trading.
/// Fetches latest data, builds features, queries Agent, executes orders via Mt5Executor.
/// </summary>
public class LiveRunner
{
    private readonly Mt5Executor _executor;
    private readonly IFeatureBuilder _featureBuilder;
    private readonly EnvironmentConfig _config;
    private readonly StateBuilder _stateBuilder;
    private readonly PortfolioManager _portfolioManager;
    private double _peakUnrealizedPnl = 0;
    
    private Dictionary<string, Position> _localState = new();

    public LiveRunner(
        Mt5Executor executor,
        IFeatureBuilder featureBuilder,
        EnvironmentConfig config,
        PortfolioManager? portfolioManager = null)
    {
        _executor = executor;
        _featureBuilder = featureBuilder;
        _config = config;
        _stateBuilder = new StateBuilder(featureBuilder, config.WindowSize);
        _portfolioManager = portfolioManager ?? new PortfolioManager();
    }

    public async Task RunLoopAsync(Func<float[], int> agentPolicy, CancellationToken ct)
    {
        Console.WriteLine($"Starting Live Loop for {_config.Symbol}...");
        
        while (!ct.IsCancellationRequested)
        {
            try 
            {
                 // 1. Fetch Data (Candles)
                 var candles = await FetchLiveCandles(_config.Symbol, _config.WindowSize + 50);
                 
                 // 2. Build State
                 var serverPos = _executor.GetPosition(_config.Symbol);
                 
                 // Sync with Local State
                 Position? position = null;
                 if (serverPos != null)
                 {
                     if (_localState.TryGetValue(serverPos.Symbol, out var local) && local.Ticket == serverPos.Ticket)
                     {
                         // Update local with server latest
                         local.CurrentPrice = serverPos.CurrentPrice;
                         local.Profit = serverPos.Profit;
                         local.StopLoss = serverPos.StopLoss;
                         local.TakeProfit = serverPos.TakeProfit;
                         if (serverPos.RiskLevel != RiskLevel.Conservative) local.RiskLevel = serverPos.RiskLevel;
                         position = local;
                     }
                     else
                     {
                         // New position found
                         _localState[serverPos.Symbol] = serverPos;
                         position = serverPos;
                     }
                 }
                 else
                 {
                     _localState.Remove(_config.Symbol);
                 }

                 // State Tracking
                 if (position == null)
                 {
                     _peakUnrealizedPnl = 0;
                 }
                 else
                 {
                     double currentPnlPct = position.UnrealizedPnlPercent / 100.0;
                     if (currentPnlPct > _peakUnrealizedPnl)
                         _peakUnrealizedPnl = currentPnlPct;
                 }
                 
                 double unrealizedPnl = position?.UnrealizedPnlPercent / 100.0 ?? 0;
                 double drawdown = (_peakUnrealizedPnl - unrealizedPnl);
                 
                 // Holding Time
                 double holdingTimeNorm = 0;
                 if (position != null)
                 {
                      var duration = DateTime.UtcNow - position.OpenTime; 
                      double maxSeconds = _config.MaxHoldingSteps * 5.0;
                      holdingTimeNorm = Math.Min(1.0, duration.TotalSeconds / maxSeconds);
                 }
                 
                 var state = _stateBuilder.BuildState(
                     candles, 
                     _config.Symbol, 
                     position != null, 
                     position?.Type ?? TradeType.Buy, 
                     unrealizedPnl, 
                     holdingTimeNorm, 
                     drawdown
                 );
                 
                 int action = agentPolicy(state);

                 var entryType = ActionDecoder.Decode(action);

                 if (ActionDecoder.IsHold(action))
                 {
                     // Do nothing
                 }
                 else if (entryType.HasValue)
                 {
                      if (position != null && position.Type != entryType.Value)
                      {
                           await _executor.ClosePositionAsync(_config.Symbol);
                           _localState.Remove(_config.Symbol);
                           position = null;
                      }

                      if (position != null)
                      {
                           double currentPrice = candles.Last().Close;
                           var symInfo = _executor.GetSymbolInfo(_config.Symbol);
                           if (symInfo != null)
                           {
                               double? newSl = _portfolioManager.CalculateTrailingStop(position, currentPrice, symInfo);
                               if (newSl.HasValue)
                               {
                                    Console.WriteLine($"Updating TS: {position.StopLoss} -> {newSl.Value}");
                                    bool success = await _executor.ModifyPositionAsync(_config.Symbol, newSl.Value, position.TakeProfit);
                                    if(success) position.StopLoss = newSl.Value;
                               }
                           }
                      }

                      if (position == null)
                      {
                          var currentPositions = _executor.GetPositions().ToList();
                          var (canOpen, reason) = _portfolioManager.CanOpenDirection(
                              _config.Symbol, entryType.Value, currentPositions);

                          if (!canOpen)
                          {
                              Console.WriteLine($"Cannot open position: {reason}");
                          }
                          else
                          {
                              double currentPrice = candles.Last().Close;
                              double atr = currentPrice * 0.001;
                              var symInfo = _executor.GetSymbolInfo(_config.Symbol)
                                            ?? new SymbolInfo(_config.Symbol, 0.00001, 100000, 1, 0.00001, "USD", "USD", 5);

                              var sizing = _portfolioManager.CalculatePositionSize(
                                  _config.Symbol, entryType.Value, RiskLevel.Moderate,
                                  _executor.GetEquity(), currentPrice, atr, symInfo);

                              var (isValid, validateReason, adjustedLot) = _portfolioManager.ValidateTradeRisk(
                                  _config.Symbol, sizing.Volume, currentPrice, sizing.StopLoss,
                                  _executor.GetEquity(), symInfo);

                              if (!isValid)
                              {
                                  Console.WriteLine($"Trade risk validation failed: {validateReason}");
                              }
                              else
                              {
                                  if (!string.IsNullOrEmpty(validateReason))
                                      Console.WriteLine($"Risk adjusted: {validateReason}");

                                  await _executor.ExecuteAsync(_config.Symbol, entryType.Value, adjustedLot,
                                      sizing.StopLoss, sizing.TakeProfit, "RL Agent", RiskLevel.Moderate);
                              }
                          }
                      }
                 }
                 
                 // Wait for next tick
                 await Task.Delay(1000, ct);
            }
            catch (Exception ex)
            {
                Console.WriteLine($"Error in loop: {ex.Message}");
                await Task.Delay(5000, ct); 
            }
        }
    }

    private async Task<List<Candle>> FetchLiveCandles(string symbol, int count, int timeframe = 1)
    {
         return await _executor.GetRecentCandlesAsync(symbol, count, timeframe);
    }
}

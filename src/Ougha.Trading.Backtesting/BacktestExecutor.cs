using Ougha.Trading.Core.Abstractions;
using Ougha.Trading.Core.Models;
using Ougha.Trading.Risk;

namespace Ougha.Trading.Backtesting;

public class BacktestExecutor : IOrderExecutor
{
    private readonly CandleTimeline? _candleTimeline;
    private readonly Dictionary<string, Position> _positions = new();
    private readonly List<TradeRecord> _tradeLog = new();
    private readonly Dictionary<string, SymbolInfo> _symbolInfo;
    private readonly PortfolioManager _portfolioManager;

    private double _balance;
    private double _initialBalance;
    private double _equity;

    private readonly List<double> _equityHistory = new();
    private DateTime _lastSampleDate;

    private int _currentTickIndex;

    public DateTime CurrentTime { get; private set; }

    private IAsyncEnumerator<(string Symbol, Candle Candle)>? _candleStream;
    private readonly bool _isStreaming;
    private string? _lastTickedSymbol;

    private readonly Dictionary<string, int> _positionOpenTicks = new();
    private readonly List<PendingCloseInfo> _pendingCloses = new();

    /// <summary>
    /// The symbol that received the most recent tick (from last AdvanceAsync call).
    /// </summary>
    public string? LastTickedSymbol => _lastTickedSymbol;

    public BacktestExecutor(
        CandleTimeline candleTimeline,
        Dictionary<string, SymbolInfo> symbolInfo,
        PortfolioManager? portfolioManager = null,
        double initialBalance = 10000.0)
    {
        _candleTimeline = candleTimeline;
        _symbolInfo = symbolInfo;
        _portfolioManager = portfolioManager ?? new PortfolioManager();
        _initialBalance = initialBalance;
        _balance = initialBalance;
        _equity = initialBalance;
        _currentTickIndex = -1;
        _isStreaming = false;
        
        if (_candleTimeline.Count > 0)
            CurrentTime = _candleTimeline.GetAtIndex(0).Time;

        _lastSampleDate = CurrentTime.Date;
        _equityHistory.Add(_equity);
    }

    public BacktestExecutor(
        IAsyncEnumerable<(string Symbol, Candle Candle)> candleStream,
        Dictionary<string, SymbolInfo> symbolInfo,
        PortfolioManager? portfolioManager = null,
        double initialBalance = 10000.0)
    {
        _candleStream = candleStream.GetAsyncEnumerator();
        _symbolInfo = symbolInfo;
        _portfolioManager = portfolioManager ?? new PortfolioManager();
        _initialBalance = initialBalance;
        _balance = initialBalance;
        _equity = initialBalance;
        _currentTickIndex = -1;
        _isStreaming = true;
        _candleTimeline = null;
        
        _lastSampleDate = DateTime.MinValue;
        _equityHistory.Add(_equity);
    }

    /// <summary>
    /// Advance to the next candle (synchronous, for non-streaming mode only).
    /// </summary>
    private bool Advance()
    {
        if (_isStreaming)
            throw new InvalidOperationException("Use AdvanceAsync() in streaming mode");

        if (_candleTimeline == null || _currentTickIndex >= _candleTimeline.Count - 1)
            return false;

        _currentTickIndex++;
        var (time, symbol, candle) = _candleTimeline.GetAtIndex(_currentTickIndex);
        ProcessCandle(symbol, candle, time);
        return true;
    }

    /// <summary>
    /// Peek which symbol will receive the next candle WITHOUT advancing.
    /// Returns null if no more data.
    /// </summary>
    public string? PeekNextTickedSymbol()
    {
        if (_isStreaming)
            throw new InvalidOperationException("PeekNextTickedSymbol not supported in streaming mode");

        if (_candleTimeline == null || _currentTickIndex >= _candleTimeline.Count - 1)
            return null;

        var (_, symbol, _) = _candleTimeline.GetAtIndex(_currentTickIndex + 1);
        return symbol;
    }

    /// <summary>
    /// Advance to next candle (asynchronous, for streaming mode).
    /// </summary>
    public async ValueTask<bool> AdvanceAsync()
    {
        if (_isStreaming)
        {
            if (_candleStream == null || !await _candleStream.MoveNextAsync())
                return false;

            var (symbol, candle) = _candleStream.Current;
            _currentTickIndex++;
            ProcessCandle(symbol, candle, candle.Time);
            return true;
        }
        else
        {
            return Advance();
        }
    }

    /// <summary>
    /// Common candle processing logic used by both Advance and AdvanceAsync.
    /// </summary>
    private void ProcessCandle(string symbol, Candle candle, DateTime time)
    {
        CurrentTime = time;
        _lastTickedSymbol = symbol;

        CheckStopLossTakeProfit(symbol, candle);

        UpdateEquity(symbol, candle);

        if (CurrentTime.Date > _lastSampleDate)
        {
            _equityHistory.Add(_equity);
            _lastSampleDate = CurrentTime.Date;
        }
    }


    public Task<OrderResult> ExecuteAsync(
        string symbol, TradeType type, double volume,
        double sl = 0, double tp = 0, string comment = "", RiskLevel riskLevel = RiskLevel.Conservative)
    {
        if (_positions.TryGetValue(symbol, out var existing))
        {
            ClosePositionInternal(symbol, existing, CurrentTime, ExitReason.Signal);
        }

        var candle = GetLastKnownCandle(symbol);
        if (candle == null) 
             return Task.FromResult(new OrderResult(false, 0, 0, 0, "No price data for symbol"));

        var mid = candle.Close;
        var spread = CalculateSpread(symbol);
        var halfSpread = spread / 2.0;

        var price = type == TradeType.Buy ? mid + halfSpread : mid - halfSpread;

        var slippage = CalculateSlippage(symbol);
        price += type == TradeType.Buy ? slippage : -slippage;

        // Use symbol's minimum lot size from SymbolInfo
        var minLot = _symbolInfo.TryGetValue(symbol, out var symInfo) ? symInfo.MinLotSize : 0.01;
        
        var position = new Position
        {
            Symbol = symbol,
            Type = type,
            Volume = Math.Max(minLot, volume),
            OpenPrice = price,
            CurrentPrice = price,
            BestPrice = price,
            OpenTime = CurrentTime,
            StopLoss = sl,
            InitialStopLoss = sl,
            TakeProfit = tp,
            Ticket = GenerateTicket(),
            RiskLevel = riskLevel
        };

        _positions[symbol] = position;
        _positionOpenTicks[symbol] = _currentTickIndex;

        return Task.FromResult(new OrderResult(true, position.Ticket, price, volume));
    }

    public Task<CloseResult> ClosePositionAsync(string symbol)
    {
        if (!_positions.TryGetValue(symbol, out var position))
            return Task.FromResult(new CloseResult(false, 0, 0, "No position"));

        var profit = ClosePositionInternal(symbol, position, CurrentTime);

        return Task.FromResult(new CloseResult(true, profit, 0)); 
    }

    private double ClosePositionInternal(string symbol, Position position, DateTime time, ExitReason exitReason = ExitReason.Manual)
    {
        var candle = GetLastKnownCandle(symbol);

        var closePrice = position.CurrentPrice; 

        if (candle != null)
        {
             var mid = candle.Close;
             var spread = CalculateSpread(symbol);
             var halfSpread = spread / 2.0;
             closePrice = position.Type == TradeType.Buy 
                ? mid - halfSpread
                : mid + halfSpread;
        }

        var priceDiff = position.Type == TradeType.Buy
            ? closePrice - position.OpenPrice
            : position.OpenPrice - closePrice;

        if (!_symbolInfo.TryGetValue(symbol, out var info))
        {
            info = new SymbolInfo(symbol, 0.00001, 100000, 1, 0.00001, "USD", "USD", 5);
        }

        var profit = (priceDiff / info.Point) * info.TickValue * position.Volume;

        _balance += profit;
        _positions.Remove(symbol);

        _tradeLog.Add(new TradeRecord
        {
            Symbol = symbol,
            Type = position.Type,
            Volume = position.Volume,
            OpenPrice = position.OpenPrice,
            ClosePrice = closePrice,
            OpenTime = position.OpenTime,
            CloseTime = time,
            Profit = profit,
            ExitReason = exitReason
        });

        return profit;
    }

    private readonly Dictionary<string, Candle> _lastCandles = new();

    private void UpdateCandleCache(string symbol, Candle candle)
    {
        _lastCandles[symbol] = candle;
    }

    public Candle? GetLastKnownCandle(string symbol)
    {
        if (_lastCandles.TryGetValue(symbol, out var candle)) return candle;
        return null; 
    }
    
    private void CheckStopLossTakeProfit(string currentSymbol, Candle currentCandle)
    {
        UpdateCandleCache(currentSymbol, currentCandle);

        if (_positions.TryGetValue(currentSymbol, out var position))
        {
            position.CurrentPrice = currentCandle.Close;

            var high = currentCandle.High;
            var low = currentCandle.Low;

            if (_symbolInfo.TryGetValue(currentSymbol, out var info))
            {
                if (position.Type == TradeType.Buy)
                    position.BestPrice = Math.Max(position.BestPrice, high);
                else
                    position.BestPrice = Math.Min(position.BestPrice, low);
                    
                var newSl = _portfolioManager.CalculateTrailingStop(position, currentCandle.Close, info);
                if (newSl.HasValue)
                {
                    ModifyPositionAsync(currentSymbol, newSl.Value, position.TakeProfit);
                }
            }

            var slHit = false;
            if (position.StopLoss > 0)
            {
                slHit = position.Type == TradeType.Buy
                    ? low <= position.StopLoss
                    : high >= position.StopLoss;
            }

            if (slHit)
            {
                var holdingTicks = _currentTickIndex - _positionOpenTicks.GetValueOrDefault(currentSymbol);
                ClosePositionAtPrice(currentSymbol, position, position.StopLoss, currentCandle.Time, ExitReason.StopLoss);
                _pendingCloses.Add(new PendingCloseInfo(currentSymbol, 0, holdingTicks, ExitReason.StopLoss));
                _positionOpenTicks.Remove(currentSymbol);
                return;
            }

            var tpHit = false;
            if (position.TakeProfit > 0)
            {
                tpHit = position.Type == TradeType.Buy
                    ? high >= position.TakeProfit
                    : low <= position.TakeProfit;
            }

            if (tpHit)
            {
                 var holdingTicks = _currentTickIndex - _positionOpenTicks.GetValueOrDefault(currentSymbol);
                 ClosePositionAtPrice(currentSymbol, position, position.TakeProfit, currentCandle.Time, ExitReason.TakeProfit);
                 
                 _pendingCloses.Add(new PendingCloseInfo(currentSymbol, 0, holdingTicks, ExitReason.TakeProfit));
                 _positionOpenTicks.Remove(currentSymbol);
            }
        }
    }

    private void ClosePositionAtPrice(string symbol, Position position, double exitPrice, DateTime time, ExitReason reason)
    {
         var priceDiff = position.Type == TradeType.Buy
            ? exitPrice - position.OpenPrice
            : position.OpenPrice - exitPrice;

        if (!_symbolInfo.TryGetValue(symbol, out var info))
        {
            info = new SymbolInfo(symbol, 0.00001, 100000, 1, 0.00001, "USD", "USD", 5);
        }

        var profit = (priceDiff / info.Point) * info.TickValue * position.Volume;

        _balance += profit;
        _positions.Remove(symbol);

        _tradeLog.Add(new TradeRecord
        {
            Symbol = symbol,
            Type = position.Type,
            Volume = position.Volume,
            OpenPrice = position.OpenPrice,
            ClosePrice = exitPrice,
            OpenTime = position.OpenTime,
            CloseTime = time,
            Profit = profit,
            ExitReason = reason
        });
    }

    public Task<bool> ModifyPositionAsync(string symbol, double sl, double tp)
    {
        if (_positions.TryGetValue(symbol, out var position))
        {
            position.StopLoss = sl;
            position.TakeProfit = tp;
            return Task.FromResult(true);
        }
        return Task.FromResult(false);
    }

    private void UpdateEquity(string currentSymbol, Candle currentCandle)
    {
        double unrealized = 0;
        
        foreach (var (symbol, pos) in _positions)
        {
            var candle = (symbol == currentSymbol) ? currentCandle : GetLastKnownCandle(symbol);
            
            if (candle == null) continue;
            
            var mid = candle.Close;
             var spread = CalculateSpread(symbol);
             var halfSpread = spread / 2.0;

            var price = pos.Type == TradeType.Buy 
                ? mid - halfSpread
                : mid + halfSpread;

            var diff = pos.Type == TradeType.Buy ? price - pos.OpenPrice : pos.OpenPrice - price;
            
            if (_symbolInfo.TryGetValue(symbol, out var info))
            {
                unrealized += (diff / info.Point) * info.TickValue * pos.Volume;
            }
        }
        
        _equity = _balance + unrealized;
    }

    public BacktestResults GetResults() => new()
    {
        InitialBalance = _initialBalance,
        FinalBalance = _balance,
        FinalEquity = _equity,
        TotalTrades = _tradeLog.Count,
        TradeLog = _tradeLog.ToList(),
        EquityCurve = _equityHistory.ToList()
    };

    public Task CloseAllPositionsAsync()
    {
        foreach (var symbol in _positions.Keys.ToList())
             ClosePositionAsync(symbol);
        return Task.CompletedTask;
    }

    public IEnumerable<Position> GetPositions() => _positions.Values;
    public Position? GetPosition(string symbol) => _positions.TryGetValue(symbol, out var p) ? p : null;
    public SymbolInfo? GetSymbolInfo(string symbol) => _symbolInfo.TryGetValue(symbol, out var info) ? info : null;
    public double GetEquity() => _equity;
    public double GetBalance() => _balance;

    public List<PendingCloseInfo> GetAndClearPendingCloses()
    {
        var closes = _pendingCloses.ToList();
        _pendingCloses.Clear();
        return closes;
    }

    public int GetPositionOpenTick(string symbol) => _positionOpenTicks.GetValueOrDefault(symbol);

    /// <summary>
    /// Reset the executor to initial state for a new training episode.
    /// Clears positions, trade log, and resets balance.
    /// </summary>
    public void Reset()
    {
        _positions.Clear();
        _tradeLog.Clear();
        _lastCandles.Clear();
        _pendingCloses.Clear();
        _positionOpenTicks.Clear();
        _balance = _initialBalance;
        _equity = _initialBalance;
        _equityHistory.Clear();
        _equityHistory.Add(_equity);
        _currentTickIndex = -1;
        _lastSampleDate = DateTime.MinValue;
        _lastTickedSymbol = null;

        if (!_isStreaming && _candleTimeline != null && _candleTimeline.Count > 0)
        {
            CurrentTime = _candleTimeline.GetAtIndex(0).Time;
            _lastSampleDate = CurrentTime.Date;
        }
    }
    
    /// <summary>
    /// Seek to a random point in the timeline for varied training data.
    /// Only works in non-streaming mode.
    /// </summary>
    public void SeekToRandomPoint(double minProgressPct = 0.0, double maxProgressPct = 0.7)
    {
        if (_isStreaming || _candleTimeline == null)
            return;
            
        var random = new Random();
        var minIdx = (int)(_candleTimeline.Count * minProgressPct);
        var maxIdx = (int)(_candleTimeline.Count * maxProgressPct);
        
        if (maxIdx <= minIdx)
            maxIdx = minIdx + 1;
            
        _currentTickIndex = random.Next(minIdx, maxIdx);
        
        if (_currentTickIndex < _candleTimeline.Count)
        {
            var (time, _, _) = _candleTimeline.GetAtIndex(_currentTickIndex);
            CurrentTime = time;
            _lastSampleDate = CurrentTime.Date;
        }
    }
    
    public double GetFreeMargin()
    {
        double usedMargin = 0;
        foreach(var kvp in _positions)
        {
            var pos = kvp.Value;
            var symbol = kvp.Key;
            
             var leverage = 100.0;
            
            if (_symbolInfo.TryGetValue(symbol, out var info))
            {
                 usedMargin += (pos.Volume * info.ContractSize * pos.CurrentPrice) / leverage;
            }
            else
            {
                 usedMargin += (pos.Volume * 100000 * pos.CurrentPrice) / leverage;
            }
        }
        
        return _equity - usedMargin;
    }
    
    private double CalculateSpread(string symbol)
    {
        if (_symbolInfo.TryGetValue(symbol, out var info))
             return info.Point * 10;
        return 0.0001;
    }

    public double GetBid(string symbol) 
    {
        var c = GetLastKnownCandle(symbol); 
        if (c==null) return 0;
        return c.Close - (CalculateSpread(symbol)/2);
    }
    
    public double GetAsk(string symbol)
    {
        var c = GetLastKnownCandle(symbol); 
        if (c==null) return 0;
        return c.Close + (CalculateSpread(symbol)/2);
    }
    
    public bool IsMarketOpen(string symbol) => true;

    private const double BaseSlippagePoints = 0.5;

    private double CalculateSlippage(string symbol)
    {
        if (!_symbolInfo.TryGetValue(symbol, out var info))
            return 0.0001;

        return BaseSlippagePoints * info.Point;
    }
    
    private long GenerateTicket() => DateTime.UtcNow.Ticks;
}

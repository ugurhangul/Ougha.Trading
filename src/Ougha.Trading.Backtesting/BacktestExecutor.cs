using Ougha.Trading.Core.Abstractions;
using Ougha.Trading.Core.Models;
using Ougha.Trading.Risk;

namespace Ougha.Trading.Backtesting;

public record PendingCloseInfo(string Symbol, double Profit, int HoldingTicks, ExitReason ExitReason);

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

    // We need to track current index in timeline
    private int _currentTickIndex;

    // We need to track current time
    public DateTime CurrentTime { get; private set; }

    // Streaming mode fields
    private IAsyncEnumerator<(string Symbol, Candle Candle)>? _candleStream;
    private readonly bool _isStreaming;
    private string? _lastTickedSymbol;

    private readonly Dictionary<string, int> _positionOpenTicks = new();
    private readonly List<PendingCloseInfo> _pendingCloses = new();

    /// <summary>
    /// The symbol that received the most recent tick (from last AdvanceAsync call).
    /// </summary>
    public string? LastTickedSymbol => _lastTickedSymbol;

    // New CandleTimeline Constructor
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
        _currentTickIndex = -1; // Not started
        _isStreaming = false;
        
        if (_candleTimeline.Count > 0)
            CurrentTime = _candleTimeline.GetAtIndex(0).Time; // Init with first candle time
            
        _lastSampleDate = CurrentTime.Date;
        _equityHistory.Add(_equity);
    }

    // New streaming constructor
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
    /// Advance to next candle (synchronous, for non-streaming mode only).
    /// </summary>
    public bool Advance()
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
            // Non-streaming: delegate to sync method
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

        // In a real multi-symbol simulator, we'd update the "Latest Known Price" for 'symbol'.
        // And check SL/TP for *all* positions using their latest known prices.
        // For simplicity and speed in this strictly sequential tick stream:
        // We only definitely know the price of 'symbol' changed.

        // However, checking SL/TP acts on the price.
        CheckStopLossTakeProfit(symbol, candle);

        // Update equity (mark-to-market).
        // Ideal: Update equity for all positions. But we only have new price for 'symbol'.
        // We assume other prices haven't changed since their last tick? YES.
        UpdateEquity(symbol, candle);

        // Track Equity Daily
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
        // Net position mode
        if (_positions.TryGetValue(symbol, out var existing))
        {
            ClosePositionInternal(symbol, existing, CurrentTime, ExitReason.Signal);
        }

        // Get execution price using Candle Close (or spread adjusted)
        var candle = GetLastKnownCandle(symbol);
        if (candle == null) 
             return Task.FromResult(new OrderResult(false, 0, 0, 0, "No price data for symbol"));

        // Use Close price for market execution assumption
        // Or if we want more realism, simulate spread around Close
        double mid = candle.Close;
        double spread = CalculateSpread(symbol);
        double halfSpread = spread / 2.0;

        double price = type == TradeType.Buy ? mid + halfSpread : mid - halfSpread;

        // Slippage
        double slippage = CalculateSlippage(symbol);
        price += type == TradeType.Buy ? slippage : -slippage;

        var position = new Position
        {
            Symbol = symbol,
            Type = type,
            Volume = 0.01,
            OpenPrice = price,
            CurrentPrice = price, // Init
            BestPrice = price,    // Init BestPrice
            OpenTime = CurrentTime,
            StopLoss = sl,
            InitialStopLoss = sl, // Track initial
            TakeProfit = tp,
            Ticket = GenerateTicket(),
            RiskLevel = riskLevel // Track Risk Level
        };

        _positions[symbol] = position;
        _positionOpenTicks[symbol] = _currentTickIndex;

        return Task.FromResult(new OrderResult(true, position.Ticket, price, volume));
    }

    public Task<CloseResult> ClosePositionAsync(string symbol)
    {
        if (!_positions.TryGetValue(symbol, out var position))
            return Task.FromResult(new CloseResult(false, 0, 0, "No position"));

        double profit = ClosePositionInternal(symbol, position, CurrentTime);
        // Position removed in Internal
        
        return Task.FromResult(new CloseResult(true, profit, 0)); 
    }

    private double ClosePositionInternal(string symbol, Position position, DateTime time, ExitReason exitReason = ExitReason.Manual)
    {
        var candle = GetLastKnownCandle(symbol);
        
        // Use Close price for closing logic by default 
        // (unless we knew we hit partial candle, but for simple backtest Close is safe proxy for "current" price)
        double closePrice = position.CurrentPrice; 

        if (candle != null)
        {
             double mid = candle.Close;
             double spread = CalculateSpread(symbol);
             double halfSpread = spread / 2.0;
             closePrice = position.Type == TradeType.Buy 
                ? mid - halfSpread  // Sell to close Buy
                : mid + halfSpread; // Buy to close Sell
        }

        double priceDiff = position.Type == TradeType.Buy
            ? closePrice - position.OpenPrice
            : position.OpenPrice - closePrice;

        if (!_symbolInfo.TryGetValue(symbol, out var info))
        {
            info = new SymbolInfo(symbol, 0.00001, 100000, 1, 0.00001, "USD", "USD", 5);
        }

        double profit = (priceDiff / info.Point) * info.TickValue * position.Volume;

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

    // Cache logic
    private readonly Dictionary<string, Candle> _lastCandles = new();
    
    // Called by Advance when a new candle arrives
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
        // Update cache
        UpdateCandleCache(currentSymbol, currentCandle);
        
        // Only check the symbol that just ticked? Yes.
        if (_positions.TryGetValue(currentSymbol, out var position))
        {
            // Update current price - Close is safest single-point proxy
            position.CurrentPrice = currentCandle.Close; 

            // OHLC Logic for SL/TP
            double high = currentCandle.High;
            double low = currentCandle.Low;
            
            // --- Trailing Stop Logic ---
            if (_symbolInfo.TryGetValue(currentSymbol, out var info))
            {
                // We use High for Buy (best possible price to trigger trailing calc off?)
                // Or Close? Let's use Close to be conservative on trailing updates?
                // Actually, trailing usually tracks 'best reached price'.
                if (position.Type == TradeType.Buy)
                    position.BestPrice = Math.Max(position.BestPrice, high);
                else
                    position.BestPrice = Math.Min(position.BestPrice, low);
                    
                double? newSl = _portfolioManager.CalculateTrailingStop(position, currentCandle.Close, info); // Calc off Close?
                if (newSl.HasValue)
                {
                    ModifyPositionAsync(currentSymbol, newSl.Value, position.TakeProfit);
                }
            }
            // ---------------------------

            // SL Check (Hit if price touches SL)
            bool slHit = false;
            if (position.StopLoss > 0)
            {
                slHit = position.Type == TradeType.Buy
                    ? low <= position.StopLoss  // Low touched SL
                    : high >= position.StopLoss; // High touched SL
            }

            if (slHit)
            {
                int holdingTicks = _currentTickIndex - _positionOpenTicks.GetValueOrDefault(currentSymbol);
                // Assume filled at SL price (slippage ignored for SL here for simplicity, or could add)
                // Actually, we should fill at SL price exactly if gap didn't jump over it.
                // For simplicity, fill at SL.
                
                // Close Internal recalculates profit based on "current price" or passed close price.
                // We need to simulate the exit price.
                double exitPrice = position.StopLoss;
                
                // Just use internal helper but we need to trick it or refactor it to accept price.
                // Use a 'simulated' exit candle? Or just refactor Internal takes price?
                // Internal takes 'time' and uses 'GetLastKnownCandle' -> 'CurrentPrice'. 
                // We should probably update the position 'CurrentPrice' to exitPrice temporarily?
                // Or assume Internal uses 'Close' which might be wrong for SL.
                
                // Fix: Let's pass 'exitPrice' to ClosePositionInternal if we want manual override?
                // For now, let's just use the Internal which pulls from `GetLastKnownCandle`.
                // That uses 'Close'. That is WRONG for SL hit on a wick.
                // Refactoring ClosePositionInternal to accept 'overridePrice' seems best.
                // But as quick fix/logic in existing constrained method:
                // We will manually calculate profit and remove position here to be precise.
                
                ClosePositionAtPrice(currentSymbol, position, position.StopLoss, currentCandle.Time, ExitReason.StopLoss);
                
                _pendingCloses.Add(new PendingCloseInfo(currentSymbol, 0, holdingTicks, ExitReason.StopLoss)); // Profit in CloseAtPrice
                _positionOpenTicks.Remove(currentSymbol);
                return;
            }

            // TP Check
            bool tpHit = false;
            if (position.TakeProfit > 0)
            {
                tpHit = position.Type == TradeType.Buy
                    ? high >= position.TakeProfit
                    : low <= position.TakeProfit;
            }

            if (tpHit)
            {
                 int holdingTicks = _currentTickIndex - _positionOpenTicks.GetValueOrDefault(currentSymbol);
                 ClosePositionAtPrice(currentSymbol, position, position.TakeProfit, currentCandle.Time, ExitReason.TakeProfit);
                 
                 _pendingCloses.Add(new PendingCloseInfo(currentSymbol, 0, holdingTicks, ExitReason.TakeProfit));
                 _positionOpenTicks.Remove(currentSymbol);
                 return;
            }
        }
    }

    private double ClosePositionAtPrice(string symbol, Position position, double exitPrice, DateTime time, ExitReason reason)
    {
         double priceDiff = position.Type == TradeType.Buy
            ? exitPrice - position.OpenPrice
            : position.OpenPrice - exitPrice;

        if (!_symbolInfo.TryGetValue(symbol, out var info))
        {
            info = new SymbolInfo(symbol, 0.00001, 100000, 1, 0.00001, "USD", "USD", 5);
        }

        double profit = (priceDiff / info.Point) * info.TickValue * position.Volume;

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
        
        return profit;
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
        // Equity = Balance + Sum(Unrealized PnL)
        double unrealized = 0;
        
        foreach (var kvp in _positions)
        {
            var symbol = kvp.Key;
            var pos = kvp.Value;
            
            // Use current tick for 'currentSymbol', cached for others
            Candle? candle = (symbol == currentSymbol) ? currentCandle : GetLastKnownCandle(symbol);
            
            if (candle == null) continue;
            
            double mid = candle.Close;
             double spread = CalculateSpread(symbol);
             double halfSpread = spread / 2.0;

            double price = pos.Type == TradeType.Buy 
                ? mid - halfSpread // Bid
                : mid + halfSpread; // Ask
                
            double diff = pos.Type == TradeType.Buy ? price - pos.OpenPrice : pos.OpenPrice - price;
            
            if (_symbolInfo.TryGetValue(symbol, out var info))
            {
                unrealized += (diff / info.Point) * info.TickValue * pos.Volume;
            }
        }
        
        _equity = _balance + unrealized;
    }

    public BacktestResults GetResults() => new BacktestResults
    {
        InitialBalance = _initialBalance,
        FinalBalance = _balance,
        FinalEquity = _equity,
        TotalTrades = _tradeLog.Count,
        TradeLog = _tradeLog.ToList(), // Copy
        EquityCurve = _equityHistory.ToList() // Copy history
    };

    public Task CloseAllPositionsAsync()
    {
        foreach (var symbol in _positions.Keys.ToList())
             ClosePositionAsync(symbol); // Sync wait?
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

        // For non-streaming mode, reset to beginning of timeline
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
        int minIdx = (int)(_candleTimeline.Count * minProgressPct);
        int maxIdx = (int)(_candleTimeline.Count * maxProgressPct);
        
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
        // Equity - UsedMargin
        double usedMargin = 0;
        foreach(var kvp in _positions)
        {
            var pos = kvp.Value;
            var symbol = kvp.Key;
            
             double leverage = 100.0;
            
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
        // Simple fixed spread simulation
        // 1 Pip?
        if (_symbolInfo.TryGetValue(symbol, out var info))
             return info.Point * 10; // 1 pip (10 points)
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
    
    public bool IsMarketOpen(string symbol) => true; // Simulated always open
    
    // Slippage: Points-based model matching Python simulated_broker.py
    // Base slippage: 0.5 points for normal conditions
    private const double BaseSlippagePoints = 0.5;

    private double CalculateSlippage(string symbol)
    {
        if (!_symbolInfo.TryGetValue(symbol, out var info))
            return 0.0001; // Default 1 pip fallback

        // Base slippage in price terms
        return BaseSlippagePoints * info.Point;
    }
    
    private long GenerateTicket() => DateTime.UtcNow.Ticks; // Only unique ID needed
}

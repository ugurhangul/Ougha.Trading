using Ougha.Trading.Core.Abstractions;
using Ougha.Trading.Core.Models;
using Ougha.Trading.Risk;

namespace Ougha.Trading.Backtesting;

public class BacktestExecutor : IOrderExecutor
{
    private readonly TickTimeline? _tickTimeline;
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
    private IAsyncEnumerator<(string Symbol, Tick Tick)>? _tickStream;
    private readonly bool _isStreaming;
    private string? _lastTickedSymbol;

    /// <summary>
    /// The symbol that received the most recent tick (from last AdvanceAsync call).
    /// </summary>
    public string? LastTickedSymbol => _lastTickedSymbol;

    // Original constructor (non-streaming, backward compatible)
    public BacktestExecutor(
        TickTimeline tickTimeline,
        Dictionary<string, SymbolInfo> symbolInfo,
        PortfolioManager? portfolioManager = null,
        double initialBalance = 10000.0)
    {
        _tickTimeline = tickTimeline;
        _symbolInfo = symbolInfo;
        _portfolioManager = portfolioManager ?? new PortfolioManager();
        _initialBalance = initialBalance;
        _balance = initialBalance;
        _equity = initialBalance;
        _currentTickIndex = -1; // Not started
        _isStreaming = false;
        
        if (_tickTimeline.Count > 0)
            CurrentTime = _tickTimeline.GetAtIndex(0).Time; // Init with first tick time or StartDate
            
        _lastSampleDate = CurrentTime.Date;
        // Add initial equity point
        _equityHistory.Add(_equity);
    }

    // New streaming constructor
    public BacktestExecutor(
        IAsyncEnumerable<(string Symbol, Tick Tick)> tickStream,
        Dictionary<string, SymbolInfo> symbolInfo,
        PortfolioManager? portfolioManager = null,
        double initialBalance = 10000.0)
    {
        _tickStream = tickStream.GetAsyncEnumerator();
        _symbolInfo = symbolInfo;
        _portfolioManager = portfolioManager ?? new PortfolioManager();
        _initialBalance = initialBalance;
        _balance = initialBalance;
        _equity = initialBalance;
        _currentTickIndex = -1;
        _isStreaming = true;
        _tickTimeline = null;
        
        _lastSampleDate = DateTime.MinValue;
        _equityHistory.Add(_equity);
    }

    /// <summary>
    /// Advance to next tick (synchronous, for non-streaming mode only).
    /// </summary>
    public bool Advance()
    {
        if (_isStreaming)
            throw new InvalidOperationException("Use AdvanceAsync() in streaming mode");

        if (_tickTimeline == null || _currentTickIndex >= _tickTimeline.Count - 1)
            return false;

        _currentTickIndex++;
        var (time, symbol, tick) = _tickTimeline.GetAtIndex(_currentTickIndex);
        ProcessTick(symbol, tick, time);
        return true;
    }

    /// <summary>
    /// Peek which symbol will receive the next tick WITHOUT advancing.
    /// Returns null if no more data.
    /// </summary>
    public string? PeekNextTickedSymbol()
    {
        if (_isStreaming)
            throw new InvalidOperationException("PeekNextTickedSymbol not supported in streaming mode");

        if (_tickTimeline == null || _currentTickIndex >= _tickTimeline.Count - 1)
            return null;

        var (_, symbol, _) = _tickTimeline.GetAtIndex(_currentTickIndex + 1);
        return symbol;
    }

    /// <summary>
    /// Advance to next tick (asynchronous, for streaming mode).
    /// </summary>
    public async ValueTask<bool> AdvanceAsync()
    {
        if (_isStreaming)
        {
            if (_tickStream == null || !await _tickStream.MoveNextAsync())
                return false;

            var (symbol, tick) = _tickStream.Current;
            _currentTickIndex++;
            ProcessTick(symbol, tick, tick.Time);
            return true;
        }
        else
        {
            // Non-streaming: delegate to sync method
            return Advance();
        }
    }

    /// <summary>
    /// Common tick processing logic used by both Advance and AdvanceAsync.
    /// </summary>
    private void ProcessTick(string symbol, Tick tick, DateTime time)
    {
        CurrentTime = time;
        _lastTickedSymbol = symbol;

        // In a real multi-symbol simulator, we'd update the "Latest Known Price" for 'symbol'.
        // And check SL/TP for *all* positions using their latest known prices.
        // For simplicity and speed in this strictly sequential tick stream:
        // We only definitely know the price of 'symbol' changed.

        // However, checking SL/TP acts on the price.
        // We must CheckStopLossTakeProfit(symbol, tick).
        CheckStopLossTakeProfit(symbol, tick);

        // Update equity (mark-to-market).
        // Ideal: Update equity for all positions. But we only have new price for 'symbol'.
        // We assume other prices haven't changed since their last tick? YES.
        UpdateEquity(symbol, tick);

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
        // Net position mode: Close existing opposite? Or close existing same?
        // Design doc: "Close existing position if any (net position mode)"
        if (_positions.TryGetValue(symbol, out var existing))
        {
            ClosePositionInternal(symbol, existing, CurrentTime, ExitReason.Signal);
        }

        // Get execution price
        var tick = GetLastKnownTick(symbol);
        if (tick == null) 
             return Task.FromResult(new OrderResult(false, 0, 0, 0, "No price data for symbol"));

        double price = type == TradeType.Buy ? tick.Ask : tick.Bid;

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
        
        return Task.FromResult(new OrderResult(true, position.Ticket, price, volume));
    }

    public Task<CloseResult> ClosePositionAsync(string symbol)
    {
        if (!_positions.TryGetValue(symbol, out var position))
            return Task.FromResult(new CloseResult(false, 0, 0, "No position"));

        double profit = ClosePositionInternal(symbol, position, CurrentTime);
        // Position removed in Internal
        
        return Task.FromResult(new CloseResult(true, profit, 0)); // ClosePrice inside Internal?
    }

    private double ClosePositionInternal(string symbol, Position position, DateTime time, ExitReason exitReason = ExitReason.Manual)
    {
        var tick = GetLastKnownTick(symbol);
        double closePrice = tick != null
            ? (position.Type == TradeType.Buy ? tick.Bid : tick.Ask)
            : position.CurrentPrice;

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
    private readonly Dictionary<string, Tick> _lastTicks = new();
    
    // Called by Advance when a new tick arrives
    private void UpdateTickCache(string symbol, Tick tick)
    {
        _lastTicks[symbol] = tick;
    }

    public Tick? GetLastKnownTick(string symbol)
    {
        if (_lastTicks.TryGetValue(symbol, out var tick)) return tick;
        // Should we search back? In backtesting, if we rely on Advance(), we verify strict time order.
        return null; 
    }
    
    private void CheckStopLossTakeProfit(string currentSymbol, Tick currentTick)
    {
        // Update cache
        UpdateTickCache(currentSymbol, currentTick);
        
        // Only check the symbol that just ticked?
        // Yes, prices of others haven't changed.
        if (_positions.TryGetValue(currentSymbol, out var position))
        {
            // Update current price
            position.CurrentPrice = currentTick.Bid; // or Bid/Ask?

            double checkPrice = position.Type == TradeType.Buy ? currentTick.Bid : currentTick.Ask;

            // --- Trailing Stop Logic ---
            if (_symbolInfo.TryGetValue(currentSymbol, out var info))
            {
                // Update BestPrice locally first? PortfolioManager assumes we pass "candidate" best price implicitly?
                // Logic in PortfolioManager used "currentPrice" vs "pos.BestPrice".
                // We should update pos.BestPrice if we are processing this tick.
                if (position.Type == TradeType.Buy)
                    position.BestPrice = Math.Max(position.BestPrice, checkPrice);
                else
                    position.BestPrice = Math.Min(position.BestPrice, checkPrice);
                    
                double? newSl = _portfolioManager.CalculateTrailingStop(position, checkPrice, info);
                if (newSl.HasValue)
                {
                    ModifyPositionAsync(currentSymbol, newSl.Value, position.TakeProfit);
                }
            }
            // ---------------------------

            // SL
            if (position.StopLoss > 0)
            {
                bool slHit = position.Type == TradeType.Buy
                    ? checkPrice <= position.StopLoss
                    : checkPrice >= position.StopLoss;

                if (slHit)
                {
                    ClosePositionInternal(currentSymbol, position, currentTick.Time, ExitReason.StopLoss);
                    return;
                }
            }

            // TP
            if (position.TakeProfit > 0)
            {
                bool tpHit = position.Type == TradeType.Buy
                    ? checkPrice >= position.TakeProfit
                    : checkPrice <= position.TakeProfit;

                 if (tpHit)
                 {
                     ClosePositionInternal(currentSymbol, position, currentTick.Time, ExitReason.TakeProfit);
                     return;
                 }
            }
        }
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

    private void UpdateEquity(string currentSymbol, Tick currentTick)
    {
        // Equity = Balance + Sum(Unrealized PnL)
        double unrealized = 0;
        
        foreach (var kvp in _positions)
        {
            var symbol = kvp.Key;
            var pos = kvp.Value;
            
            // Use current tick for 'currentSymbol', cached for others
            Tick? tick = (symbol == currentSymbol) ? currentTick : GetLastKnownTick(symbol);
            
            if (tick == null) continue; // Should not happen for open position
            
            double price = pos.Type == TradeType.Buy ? tick.Bid : tick.Ask;
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
    
    public double GetFreeMargin()
    {
        // Equity - UsedMargin
        double usedMargin = 0;
        foreach(var kvp in _positions)
        {
            var pos = kvp.Value;
            var symbol = kvp.Key;
            
            // Margin = (Volume * ContractSize * Price) / Leverage
            // Assuming default leverage 100 if not in info
            // SymbolInfo doesn't store leverage directly yet.
            // We use global leverage 100.
            double leverage = 100.0;
            
            if (_symbolInfo.TryGetValue(symbol, out var info))
            {
                 // Standard Forex Margin Calc
                 // Or CFD? 
                 // Margin = Volume * ContractSize * CurrentPrice / Leverage
                 usedMargin += (pos.Volume * info.ContractSize * pos.CurrentPrice) / leverage;
            }
            else
            {
                // Fallback: Assume Forex-like (100k contract) or raw?
                // Fallback: Volume * 100000 * Price / 100
                 usedMargin += (pos.Volume * 100000 * pos.CurrentPrice) / leverage;
            }
        }
        
        return _equity - usedMargin;
    }
    public double GetBid(string symbol) => GetLastKnownTick(symbol)?.Bid ?? 0;
    public double GetAsk(string symbol) => GetLastKnownTick(symbol)?.Ask ?? 0;
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

using Microsoft.Extensions.Configuration;
using Python.Runtime;
using Ougha.Trading.Core.Abstractions;
using Ougha.Trading.Core.Models;

namespace Ougha.Trading.Data;

/// <summary>
/// MT5 connector for symbol info and order execution.
/// Uses pythonnet to connect to MetaTrader5 Python API.
/// </summary>
public class Mt5Executor : IOrderExecutor, ISymbolInfoProvider, IDisposable
{
    private dynamic? _mt5;
    private bool _initialized;

    /// <summary>
    /// Create Mt5Executor from IConfiguration.
    /// Reads credentials from the "MT5" section in appsettings.
    /// </summary>
    public Mt5Executor(IConfiguration configuration)
    {
        var mt5Config = new Mt5Config();
        configuration.GetSection("MT5").Bind(mt5Config);
        
        var pythonDllPath = mt5Config.PythonDllPath;

        if (!PythonEngine.IsInitialized)
        {
            Runtime.PythonDLL = pythonDllPath;
            PythonEngine.Initialize();
            PythonEngine.BeginAllowThreads();
        }

        var initTask = InitializeAsync(mt5Config.Login, mt5Config.Password, mt5Config.Server);
        initTask.Wait();
        
        if (!_initialized)
        {
            throw new Exception("MT5 Initialize failed. Check credentials in appsettings.");
        }
    }

    private async Task InitializeAsync(string login, string password, string server)
    {
        await Task.Run(() =>
        {
            using (Py.GIL())
            {
                _mt5 = Py.Import("MetaTrader5");

                dynamic result;
                if (string.IsNullOrEmpty(login))
                {
                    result = _mt5.initialize();
                }
                else
                {
                    result = _mt5.initialize(
                        login: int.Parse(login),
                        password: password,
                        server: server
                    );
                }

                if (!(bool)result)
                {
                    var err = _mt5.last_error();
                    throw new Exception($"MT5 Initialize failed: {err}");
                }

                _initialized = true;
            }
        });
    }

    public void Dispose()
    {
        if (!_initialized || _mt5 == null) return;
        using (Py.GIL())
        {
            _mt5?.shutdown();
        }
    }

    public SymbolInfo GetSymbolInfo(string symbol)
    {
        using (Py.GIL())
        {
            var info = _mt5!.symbol_info(symbol);

            return new SymbolInfo(
                (string)info.name,
                (double)info.point,
                (double)info.trade_contract_size,
                (double)info.trade_tick_value,
                (double)info.trade_tick_size,
                (string)info.currency_base,
                (string)info.currency_profit,
                (int)info.digits,
                (double)info.volume_min,
                DetectCategory(symbol)
            );
        }
    }
    
    private static SymbolCategory DetectCategory(string symbol)
    {
        var s = symbol.ToUpperInvariant();
        
        // Crypto
        if (s.Contains("BTC") || s.Contains("ETH") || s.Contains("XRP") || 
            s.Contains("LTC") || s.Contains("DOGE") || s.Contains("SOL") ||
            s.Contains("ADA") || s.Contains("CRYPTO"))
            return SymbolCategory.Crypto;
            
        // Indices
        if (s.Contains("US30") || s.Contains("US500") || s.Contains("NAS") ||
            s.Contains("DAX") || s.Contains("FTSE") || s.Contains("NDX") ||
            s.Contains("SPX") || s.Contains("DJI") || s.Contains("UK100") ||
            s.Contains("DE40") || s.Contains("JP225"))
            return SymbolCategory.Indices;
            
        // Commodities
        if (s.Contains("XAUUSD") || s.Contains("GOLD") || s.Contains("SILVER") ||
            s.Contains("XAGUSD") || s.Contains("OIL") || s.Contains("BRENT") ||
            s.Contains("WTI") || s.Contains("NATGAS"))
            return SymbolCategory.Commodities;
            
        // Default to Forex
        return SymbolCategory.Forex;
    }

    public async Task<OrderResult> ExecuteAsync(
        string symbol, TradeType type, double volume,
        double sl = 0, double tp = 0, string comment = "", RiskLevel riskLevel = RiskLevel.Conservative)
    {
        return await Task.Run(() =>
        {
            using (Py.GIL())
            {
                var orderType = type == TradeType.Buy ? 0 : 1;
                var price = type == TradeType.Buy 
                    ? (double)_mt5!.symbol_info_tick(symbol).ask 
                    : (double)_mt5!.symbol_info_tick(symbol).bid;

                var request = new PyDict();
                request["action"] = 1.ToPython();
                request["symbol"] = symbol.ToPython();
                request["volume"] = volume.ToPython();
                request["type"] = orderType.ToPython();
                request["price"] = price.ToPython();
                request["deviation"] = 10.ToPython();
                request["magic"] = 123456.ToPython();

                var metaComment = $"{comment}|{(int)riskLevel}|{sl}";
                request["comment"] = metaComment.ToPython();

                request["type_time"] = 0.ToPython();
                request["type_filling"] = 1.ToPython();

                if (sl > 0) request["sl"] = sl.ToPython();
                if (tp > 0) request["tp"] = tp.ToPython();

                var result = _mt5!.order_send(request);

                if (result == null)
                    return new OrderResult(false, 0, 0, 0, "OrderSend returned null");

                return (int)result.retcode != 10009 ? new OrderResult(false, 0, 0, 0, $"RetCode: {result.retcode}, Comment: {result.comment}") : new OrderResult(true, (long)result.order, (double)result.price, (double)result.volume);
            }
        });
    }

    public async Task<CloseResult> ClosePositionAsync(string symbol)
    {
        var pos = GetPosition(symbol);
        if (pos == null) return new CloseResult(false, 0, 0, "No position found");

        return await Task.Run(() =>
        {
            using (Py.GIL())
            {
                var type = pos.Type == TradeType.Buy ? 1 : 0;
                var price = pos.Type == TradeType.Buy 
                    ? (double)_mt5!.symbol_info_tick(symbol).bid 
                    : (double)_mt5!.symbol_info_tick(symbol).ask;

                var request = new PyDict();
                request["action"] = 1.ToPython();
                request["symbol"] = symbol.ToPython();
                request["volume"] = pos.Volume.ToPython();
                request["type"] = type.ToPython();
                request["position"] = pos.Ticket.ToPython();
                request["price"] = price.ToPython();
                request["deviation"] = 10.ToPython();
                request["magic"] = 123456.ToPython();

                var result = _mt5!.order_send(request);

                return (int)result.retcode != 10009 ? new CloseResult(false, 0, 0, $"RetCode: {result.retcode}") : new CloseResult(true, 0, (double)result.price);
            }
        });
    }

    public async Task<bool> ModifyPositionAsync(string symbol, double sl, double tp)
    {
        var pos = GetPosition(symbol);
        if (pos == null) return false;

        return await Task.Run(() =>
        {
            using (Py.GIL())
            {
                var request = new PyDict();
                request["action"] = 6.ToPython();
                request["symbol"] = symbol.ToPython();
                request["position"] = pos.Ticket.ToPython();
                request["sl"] = sl.ToPython();
                request["tp"] = tp.ToPython();
                request["magic"] = 123456.ToPython();

                var result = _mt5!.order_send(request);
                return (int)result.retcode == 10009;
            }
        });
    }

    public Task CloseAllPositionsAsync()
    {
        var positions = GetPositions();
        return Task.Run(async () =>
        {
            foreach (var p in positions.ToList())
            {
                await ClosePositionAsync(p.Symbol);
            }
        });
    }

    public IEnumerable<Position> GetPositions()
    {
        using (Py.GIL())
        {
            var positions = _mt5!.positions_get();
            if (positions == null) return Enumerable.Empty<Position>();

            var list = new List<Position>();
            foreach (dynamic p in positions)
            {
                var pos = new Position
                {
                    Symbol = (string)p.symbol,
                    Type = (int)p.type == 0 ? TradeType.Buy : TradeType.Sell,
                    Volume = (double)p.volume,
                    OpenPrice = (double)p.price_open,
                    CurrentPrice = (double)p.price_current,
                    Ticket = (long)p.ticket,
                    StopLoss = (double)p.sl,
                    TakeProfit = (double)p.tp,
                    Profit = (double)p.profit,
                };

                try
                {
                    var comment = (string)p.comment;
                    if (!string.IsNullOrEmpty(comment))
                    {
                        var parts = comment.Split('|');
                        if (parts.Length >= 3)
                        {
                            if (Enum.TryParse<RiskLevel>(parts[1], out var rl))
                                pos.RiskLevel = rl;

                            if (double.TryParse(parts[2], out var isl))
                                pos.InitialStopLoss = isl;
                        }
                    }
                }
                catch
                {
                    // ignored
                }

                list.Add(pos);
            }

            return list;
        }
    }

    public Position? GetPosition(string symbol)
        => GetPositions().FirstOrDefault(p => p.Symbol == symbol);

    public double GetEquity()
    {
        using (Py.GIL())
            return (double)_mt5!.account_info().equity;
    }

    public double GetBalance()
    {
        using (Py.GIL())
            return (double)_mt5!.account_info().balance;
    }

    public double GetFreeMargin()
    {
        using (Py.GIL())
            return (double)_mt5!.account_info().margin_free;
    }

    public double GetBid(string symbol)
    {
        using (Py.GIL())
            return (double)_mt5!.symbol_info_tick(symbol).bid;
    }

    public double GetAsk(string symbol)
    {
        using (Py.GIL())
            return (double)_mt5!.symbol_info_tick(symbol).ask;
    }

    public bool IsMarketOpen(string symbol)
    {
        using (Py.GIL())
        {
            var info = _mt5!.symbol_info(symbol);
            if (info == null) return false;
            return (int)info.trade_mode == 4;
        }
    }

    public async Task<List<Candle>> GetRecentCandlesAsync(string symbol, int count, int timeframe = 1)
    {
        return await Task.Run(() =>
        {
            using (Py.GIL())
            {
                var rates = _mt5!.copy_rates_from_pos(symbol, timeframe, 0, count);

                if (rates == null) return new List<Candle>();

                var list = new List<Candle>();
                foreach (dynamic r in rates)
                {
                    var timeSec = (long)r[0];
                    var time = DateTimeOffset.FromUnixTimeSeconds(timeSec).UtcDateTime;

                    var open = (double)r[1];
                    var high = (double)r[2];
                    var low = (double)r[3];
                    var close = (double)r[4];
                    var vol = (long)r[5];

                    list.Add(new Candle(time, open, high, low, close, vol));
                }

                return list;
            }
        });
    }
}

using Ougha.Trading.Core.Abstractions;
using Ougha.Trading.Core.Models;

namespace Ougha.Trading.Data.Services;

/// <summary>
/// Provides symbol information with optional MT5 integration and inference fallback.
/// </summary>
public class SymbolInfoService
{
    private static readonly HashSet<string> ForexMajors = new() { "EURUSD", "GBPUSD", "USDJPY", "USDCHF", "AUDUSD", "USDCAD", "NZDUSD" };
    private static readonly HashSet<string> ForexMinors = new() { "EURGBP", "EURJPY", "GBPJPY", "AUDJPY", "EURAUD", "EURCHF", "GBPCHF" };
    private static readonly HashSet<string> Metals = new() { "XAUUSD", "XAGUSD", "XPTUSD", "XPDUSD" };
    private static readonly HashSet<string> Crypto = new() { "BTCUSD", "ETHUSD", "LTCUSD", "XRPUSD" };
    private static readonly HashSet<string> JpyPairs = new() { "USDJPY", "EURJPY", "GBPJPY", "AUDJPY", "NZDJPY", "CADJPY", "CHFJPY" };

    private ISymbolInfoProvider? _provider;
    private readonly Dictionary<string, SymbolInfo> _cache = new();

    /// <summary>
    /// Create SymbolInfoService with optional symbol info provider.
    /// </summary>
    public SymbolInfoService(ISymbolInfoProvider? provider = null)
    {
        _provider = provider;
    }

    /// <summary>
    /// Set the symbol info provider (e.g., MT5Executor) at runtime.
    /// </summary>
    public void SetProvider(ISymbolInfoProvider provider)
    {
        _provider = provider;
        ClearCache(); // Clear cache when provider changes
    }

    /// <summary>
    /// Get symbol info from provider if available, otherwise infer from symbol name.
    /// Results are cached for performance.
    /// </summary>
    public SymbolInfo GetSymbolInfo(string symbol)
    {
        if (_provider == null)
        {
            throw new InvalidOperationException("Symbol info provider not set");
        }

        // Check cache first
        if (_cache.TryGetValue(symbol, out var cached))
            return cached;


        try
        {
            var info = _provider.GetSymbolInfo(symbol);

            _cache[symbol] = info;
            return info;
        }
        catch (Exception ex)
        {
            Console.WriteLine($"Warning: Provider symbol info failed for {symbol}: {ex.Message}. Using inference.");
            throw;
        }
    }

    /// <summary>
    /// Clear the symbol info cache.
    /// </summary>
    public void ClearCache() => _cache.Clear();

    /// <summary>
    /// Preload symbol info for multiple symbols.
    /// </summary>
    public void Preload(IEnumerable<string> symbols)
    {
        foreach (var symbol in symbols)
        {
            GetSymbolInfo(symbol);
        }
    }
}
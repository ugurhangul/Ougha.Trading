using Ougha.Trading.Core.Abstractions;
using Ougha.Trading.Core.Models;

namespace Ougha.Trading.Data.Services;

/// <summary>
/// Provides symbol information with optional MT5 integration and inference fallback.
/// </summary>
public class SymbolInfoService
{


    private readonly ISymbolInfoProvider? _provider;
    private readonly Dictionary<string, SymbolInfo> _cache = new();

    /// <summary>
    /// Create SymbolInfoService with an optional symbol info provider.
    /// </summary>
    public SymbolInfoService(ISymbolInfoProvider? provider = null)
    {
        _provider = provider;
    }


    /// <summary>
    /// Get symbol info from the provider if available, otherwise infer from the symbol name.
    /// Results are cached for performance.
    /// </summary>
    public SymbolInfo GetSymbolInfo(string symbol)
    {
        if (_provider == null)
        {
            throw new InvalidOperationException("Symbol info provider not set");
        }

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

}
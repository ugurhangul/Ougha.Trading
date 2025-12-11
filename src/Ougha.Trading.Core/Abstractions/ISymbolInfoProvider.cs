using Ougha.Trading.Core.Models;

namespace Ougha.Trading.Core.Abstractions;

/// <summary>
/// Interface for providing symbol information.
/// Allows SymbolInfoService to use MT5 without circular dependency.
/// </summary>
public interface ISymbolInfoProvider
{
    /// <summary>
    /// Get symbol info from the broker/connector.
    /// Returns null if the symbol is not found.
    /// </summary>
    SymbolInfo GetSymbolInfo(string symbol);
}

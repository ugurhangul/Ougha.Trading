using Ougha.Trading.Core.Abstractions;
using Ougha.Trading.Core.Models;

namespace Ougha.Trading.RL;

/// <summary>
/// Builds combined state for a portfolio of symbols.
/// State structure per symbol: [market_features * window_size] + [portfolio_features(4)] + [symbol_id(1)]
/// </summary>
public class PortfolioStateBuilder
{
    private readonly int _windowSize;
    private readonly IFeatureBuilder _featureBuilder;
    private readonly int _featuresPerCandle;
    private readonly int _statePerSymbol;

    public PortfolioStateBuilder(IFeatureBuilder featureBuilder, int windowSize)
    {
        _featureBuilder = featureBuilder;
        _windowSize = windowSize;
        _featuresPerCandle = featureBuilder.FeatureCount;
        _statePerSymbol = _windowSize * _featuresPerCandle + 4 + 1;
    }

    /// <summary>
    /// Get total state size for the given number of symbols.
    /// </summary>
    public int GetStateSize(int numSymbols) => _statePerSymbol * numSymbols;

    /// <summary>
    /// Get state size for a single symbol.
    /// </summary>
    public int StatePerSymbol => _statePerSymbol;

    /// <summary>
    /// Build combined state for all symbols in the portfolio.
    /// </summary>
    public float[] BuildPortfolioState(
        Dictionary<string, IReadOnlyList<Candle>> symbolCandles,
        Dictionary<string, SymbolPortfolioState> portfolioStates,
        string[] symbols)
    {
        var state = new float[GetStateSize(symbols.Length)];
        var offset = 0;

        foreach (var symbol in symbols)
        {
            if (!symbolCandles.TryGetValue(symbol, out var candles))
                candles = Array.Empty<Candle>();

            if (!portfolioStates.TryGetValue(symbol, out var portfolioState))
                portfolioState = SymbolPortfolioState.Flat;

            var symbolState = BuildSymbolState(candles, symbol, portfolioState);

            Array.Copy(symbolState, 0, state, offset, symbolState.Length);
            offset += _statePerSymbol;
        }

        return state;
    }

    private float[] BuildSymbolState(
        IReadOnlyList<Candle> candles,
        string symbol,
        SymbolPortfolioState portfolioState)
    {
        var state = new float[_statePerSymbol];
        var idx = 0;

        if (candles.Count >= _windowSize)
        {
            var marketFeatures = _featureBuilder.BuildFlattenedFeatures(candles, symbol, _windowSize);
            Array.Copy(marketFeatures, 0, state, idx, marketFeatures.Length);
            idx += marketFeatures.Length;
        }
        else
        {
            idx += _windowSize * _featuresPerCandle;
        }

        state[idx++] = portfolioState.HasPosition 
            ? (portfolioState.PositionType == TradeType.Buy ? 1f : -1f) 
            : 0f;
        state[idx++] = (float)portfolioState.UnrealizedPnlPct * 100f;
        state[idx++] = (float)portfolioState.HoldingTimeNorm;
        state[idx++] = (float)portfolioState.DrawdownPct * 100f;

        state[idx] = SymbolIdMapper.GetSymbolId(symbol);

        return state;
    }
}

/// <summary>
/// Portfolio state for a single symbol.
/// </summary>
public record struct SymbolPortfolioState(
    bool HasPosition,
    TradeType PositionType,
    double UnrealizedPnlPct,
    double HoldingTimeNorm,
    double DrawdownPct)
{
    public static readonly SymbolPortfolioState Flat = new(false, TradeType.Buy, 0, 0, 0);
}

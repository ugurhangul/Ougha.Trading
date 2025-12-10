using Ougha.Trading.Core.Models;

namespace Ougha.Trading.Core.Abstractions;

/// <summary>
/// Builds normalized feature vectors from market data.
/// Mirrors Python MLFeatureEngineer with 47 features.
/// </summary>
public interface IFeatureBuilder
{
    /// <summary>
    /// Number of features per candle.
    /// </summary>
    int FeatureCount { get; }

    /// <summary>
    /// Feature column names for debugging/logging.
    /// </summary>
    IReadOnlyList<string> FeatureNames { get; }

    /// <summary>
    /// Build features from OHLCV candles.
    /// </summary>
    /// <param name="candles">Historical candles (newest last)</param>
    /// <param name="symbol">Symbol for spread lookup</param>
    /// <returns>Feature matrix [windowSize x featureCount]</returns>
    float[,] BuildFeatures(IReadOnlyList<Candle> candles, string symbol);

    /// <summary>
    /// Build flattened feature vector for model input.
    /// </summary>
    /// <param name="candles">Historical candles</param>
    /// <param name="symbol">Symbol name</param>
    /// <param name="windowSize">Number of candles to include</param>
    /// <returns>Flattened feature vector</returns>
    float[] BuildFlattenedFeatures(
        IReadOnlyList<Candle> candles,
        string symbol,
        int windowSize);
}

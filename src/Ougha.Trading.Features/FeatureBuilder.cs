using Ougha.Trading.Core.Abstractions;
using Ougha.Trading.Core.Models;
using Ougha.Trading.Features.Indicators;

namespace Ougha.Trading.Features;

public class FeatureBuilder : IFeatureBuilder
{
    public int FeatureCount => 45;

    public IReadOnlyList<string> FeatureNames { get; } = new[]
    {
        "close_open_diff_pct", "price_range_pct", "previous_close_diff_pct", "typical_price_pct",
        "sma_7_pct", "sma_14_pct", "sma_21_pct", "ema_7_pct",
        "close_lag_1_pct", "close_lag_2_pct", "close_lag_3_pct", "close_lag_4_pct",
        "close_lag_5_pct", "close_lag_6_pct", "close_lag_7_pct",
        "log_volume", "log_volume_sma_7", "volume_ratio",
        "log_volume_lag_1", "log_volume_lag_2", "log_volume_lag_3",
        "log_volume_lag_4", "log_volume_lag_5", "log_volume_lag_6", "log_volume_lag_7",
        "atr_pct", "atr_7_pct", "bollinger_width", "bb_position",
        "rsi_14", "rsi_7",
        "macd_pct", "macd_signal_pct", "macd_hist_pct",
        "stoch_k", "stoch_d",
        "roc_14",
        "adx_14", "trend_strength", "volatility_regime",
        "spread_pct", "hour_sin", "hour_cos", "day_sin", "day_cos"
    };

    public float[] BuildFlattenedFeatures(
        IReadOnlyList<Candle> candles,
        string symbol,
        int windowSize)
    {
        var features = BuildFeatures(candles, symbol);
        int rows = Math.Min(windowSize, features.GetLength(0));
        int cols = features.GetLength(1);

        var result = new float[rows * cols];
        int idx = 0;
        // Take the last 'rows' features
        for (int i = features.GetLength(0) - rows; i < features.GetLength(0); i++)
        {
            for (int j = 0; j < cols; j++)
            {
                result[idx++] = features[i, j];
            }
        }
        return result;
    }

    public float[,] BuildFeatures(IReadOnlyList<Candle> candles, string symbol)
    {
        int n = candles.Count;
        var features = new float[n, FeatureCount];

        // Extract arrays for calculation
        // Ensure we handle conversions from List to Array efficiently or allow Indicators to take ReadOnlyList
        var close = new double[n];
        var open = new double[n];
        var high = new double[n];
        var low = new double[n];
        var volume = new double[n];
        
        for (int i = 0; i < n; i++)
        {
            close[i] = candles[i].Close;
            open[i] = candles[i].Open;
            high[i] = candles[i].High;
            low[i] = candles[i].Low;
            volume[i] = candles[i].Volume;
        }

        // Calculate Indicators
        var sma7 = Technicals.Sma(close, 7);
        var sma14 = Technicals.Sma(close, 14);
        var sma21 = Technicals.Sma(close, 21);
        var ema7 = Technicals.EmaTaLib(close, 7);

        var atr14 = Technicals.Atr(high, low, close, 14);
        var atr7 = Technicals.Atr(high, low, close, 7);
        
        // Bollinger Bands (20, 2)
        var (bbUpper, bbMiddle, bbLower) = Technicals.BollingerBands(close, 20, 2);

        var rsi14 = Technicals.Rsi(close, 14);
        var rsi7 = Technicals.Rsi(close, 7);

        // MACD (12, 26, 9)
        var (macd, signal, hist) = Technicals.Macd(close, 12, 26, 9);

        // Stochastic (14, 3, 3) - fastk=14, slowk=3, slowd=3
        var (stochK, stochD) = Technicals.Stochastic(high, low, close, 14, 3, 3);

        var roc14 = Technicals.Roc(close, 14);
        
        var adx14 = Technicals.Adx(high, low, close, 14);
        
        // Volume SMA
        var volSma7 = Technicals.Sma(volume, 7);
        
        // ATR ma for regime
        var atr14Ma = Technicals.Sma(atr14, 50);

        // Fill features
        for (int i = 0; i < n; i++)
        {
            double c = close[i];
            int col = 0;

            // 1. Price features
            features[i, col++] = (float)((c - open[i]) / open[i]);
            features[i, col++] = (float)((high[i] - low[i]) / c);
            features[i, col++] = i > 0 ? (float)((c - close[i - 1]) / close[i - 1]) : 0f;
            
            double typicalPrice = (high[i] + low[i] + c) / 3.0;
            features[i, col++] = (float)((typicalPrice - c) / c);

            // 2. MA features
            features[i, col++] = (float)((sma7[i] - c) / c);
            features[i, col++] = (float)((sma14[i] - c) / c);
            features[i, col++] = (float)((sma21[i] - c) / c);
            features[i, col++] = (float)((ema7[i] - c) / c);

            // 3. Lag features (Returns)
            for (int lag = 1; lag <= 7; lag++)
            {
                features[i, col++] = i >= lag 
                    ? (float)((c - close[i - lag]) / close[i - lag]) 
                    : 0f;
            }

            // 4. Volume features
            features[i, col++] = (float)Math.Log(volume[i] + 1);
            features[i, col++] = (float)Math.Log(Math.Max(volSma7[i], 0) + 1);
            
            // Volume ratio
            double vSma = volSma7[i];
            features[i, col++] = vSma > 0 ? (float)(volume[i] / vSma) : 1f;

            // Lag volume
            for (int lag = 1; lag <= 7; lag++)
            {
                features[i, col++] = i >= lag 
                    ? (float)Math.Log(Math.Max(volume[i - lag], 0) + 1) 
                    : 0f;
            }

            // 5. Volatility
            features[i, col++] = (float)(atr14[i] / c);
            features[i, col++] = (float)(atr7[i] / c);
            
            double bbRange = bbUpper[i] - bbLower[i];
            features[i, col++] = (float)(bbRange / c);
            
            // BB Position
            if (bbRange > 0)
                features[i, col++] = (float)((c - bbLower[i]) / bbRange);
            else
                features[i, col++] = 0.5f;

            // 6. RSI
            features[i, col++] = (float)rsi14[i];
            features[i, col++] = (float)rsi7[i];

            // 7. MACD
            features[i, col++] = (float)(macd[i] / c);
            features[i, col++] = (float)(signal[i] / c);
            features[i, col++] = (float)(hist[i] / c);

            // 8. Stoch
            features[i, col++] = (float)stochK[i];
            features[i, col++] = (float)stochD[i];

            // 9. ROC
            features[i, col++] = (float)roc14[i];

            // 10. Trend/Regime
            features[i, col++] = (float)adx14[i];
            features[i, col++] = (float)(adx14[i] / 100.0);
            features[i, col++] = atr14[i] > atr14Ma[i] ? 1f : 0f;

            // 11. Spread (Assuming 0.1 * ATR if provider null, matching Python default)
            // Python: spread = atr_14 * 0.1 ... data['spread_pct'] = spread / close
            double spread = atr14[i] * 0.1;
            features[i, col++] = (float)(spread / c);

            // 12. Time (Sin/Cos)
            // Python: hour_sin, hour_cos, day_sin, day_cos
            // Hour 0-23, Day 0-6
            var t = candles[i].Time;
            features[i, col++] = (float)Math.Sin(2 * Math.PI * t.Hour / 24.0);
            features[i, col++] = (float)Math.Cos(2 * Math.PI * t.Hour / 24.0);
            features[i, col++] = (float)Math.Sin(2 * Math.PI * (int)t.DayOfWeek / 7.0);
            features[i, col++] = (float)Math.Cos(2 * Math.PI * (int)t.DayOfWeek / 7.0);
        }

        return features;
    }
}

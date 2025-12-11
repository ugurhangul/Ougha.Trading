using Ougha.Trading.Core.Abstractions;
using Ougha.Trading.Core.Models;
using Ougha.Trading.Features.Indicators;

namespace Ougha.Trading.Features;

public class FeatureBuilder : IFeatureBuilder
{
    public int FeatureCount => 45;

    public IReadOnlyList<string> FeatureNames { get; } =
    [
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
    ];

    public float[] BuildFlattenedFeatures(
        IReadOnlyList<Candle> candles,
        string symbol,
        int windowSize)
    {
        var features = BuildFeatures(candles, symbol);
        var rows = Math.Min(windowSize, features.GetLength(0));
        var cols = features.GetLength(1);

        var result = new float[rows * cols];
        var idx = 0;
        for (var i = features.GetLength(0) - rows; i < features.GetLength(0); i++)
        {
            for (var j = 0; j < cols; j++)
            {
                result[idx++] = features[i, j];
            }
        }
        return result;
    }

    public float[,] BuildFeatures(IReadOnlyList<Candle> candles, string symbol)
    {
        var n = candles.Count;
        var features = new float[n, FeatureCount];

        var close = new double[n];
        var open = new double[n];
        var high = new double[n];
        var low = new double[n];
        var volume = new double[n];
        
        for (var i = 0; i < n; i++)
        {
            close[i] = candles[i].Close;
            open[i] = candles[i].Open;
            high[i] = candles[i].High;
            low[i] = candles[i].Low;
            volume[i] = candles[i].Volume;
        }

        var sma7 = Technicals.Sma(close, 7);
        var sma14 = Technicals.Sma(close, 14);
        var sma21 = Technicals.Sma(close, 21);
        var ema7 = Technicals.EmaTaLib(close, 7);

        var atr14 = Technicals.Atr(high, low, close, 14);
        var atr7 = Technicals.Atr(high, low, close, 7);

        var (bbUpper, _, bbLower) = Technicals.BollingerBands(close, 20, 2);

        var rsi14 = Technicals.Rsi(close, 14);
        var rsi7 = Technicals.Rsi(close, 7);

        var (macd, signal, hist) = Technicals.Macd(close, 12, 26, 9);

        var (stochK, stochD) = Technicals.Stochastic(high, low, close, 14, 3, 3);

        var roc14 = Technicals.Roc(close, 14);
        
        var adx14 = Technicals.Adx(high, low, close, 14);

        var volSma7 = Technicals.Sma(volume, 7);

        var atr14Ma = Technicals.Sma(atr14, 50);

        for (var i = 0; i < n; i++)
        {
            var c = close[i];
            var col = 0;

            features[i, col++] = (float)((c - open[i]) / open[i]);
            features[i, col++] = (float)((high[i] - low[i]) / c);
            features[i, col++] = i > 0 ? (float)((c - close[i - 1]) / close[i - 1]) : 0f;
            
            var typicalPrice = (high[i] + low[i] + c) / 3.0;
            features[i, col++] = (float)((typicalPrice - c) / c);

            features[i, col++] = (float)((sma7[i] - c) / c);
            features[i, col++] = (float)((sma14[i] - c) / c);
            features[i, col++] = (float)((sma21[i] - c) / c);
            features[i, col++] = (float)((ema7[i] - c) / c);

            for (var lag = 1; lag <= 7; lag++)
            {
                features[i, col++] = i >= lag 
                    ? (float)((c - close[i - lag]) / close[i - lag]) 
                    : 0f;
            }

            features[i, col++] = (float)Math.Log(volume[i] + 1);
            features[i, col++] = (float)Math.Log(Math.Max(volSma7[i], 0) + 1);

            var vSma = volSma7[i];
            features[i, col++] = vSma > 0 ? (float)(volume[i] / vSma) : 1f;

            for (var lag = 1; lag <= 7; lag++)
            {
                features[i, col++] = i >= lag 
                    ? (float)Math.Log(Math.Max(volume[i - lag], 0) + 1) 
                    : 0f;
            }

            features[i, col++] = (float)(atr14[i] / c);
            features[i, col++] = (float)(atr7[i] / c);
            
            var bbRange = bbUpper[i] - bbLower[i];
            features[i, col++] = (float)(bbRange / c);

            if (bbRange > 0)
                features[i, col++] = (float)((c - bbLower[i]) / bbRange);
            else
                features[i, col++] = 0.5f;

            features[i, col++] = (float)rsi14[i];
            features[i, col++] = (float)rsi7[i];

            features[i, col++] = (float)(macd[i] / c);
            features[i, col++] = (float)(signal[i] / c);
            features[i, col++] = (float)(hist[i] / c);

            features[i, col++] = (float)stochK[i];
            features[i, col++] = (float)stochD[i];

            features[i, col++] = (float)roc14[i];

            features[i, col++] = (float)adx14[i];
            features[i, col++] = (float)(adx14[i] / 100.0);
            features[i, col++] = atr14[i] > atr14Ma[i] ? 1f : 0f;

            var spread = atr14[i] * 0.1;
            features[i, col++] = (float)(spread / c);

            var t = candles[i].Time;
            features[i, col++] = (float)Math.Sin(2 * Math.PI * t.Hour / 24.0);
            features[i, col++] = (float)Math.Cos(2 * Math.PI * t.Hour / 24.0);
            features[i, col++] = (float)Math.Sin(2 * Math.PI * (int)t.DayOfWeek / 7.0);
            features[i, col] = (float)Math.Cos(2 * Math.PI * (int)t.DayOfWeek / 7.0);
        }

        return features;
    }
}

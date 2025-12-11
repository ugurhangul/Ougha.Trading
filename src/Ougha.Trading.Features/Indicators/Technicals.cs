namespace Ougha.Trading.Features.Indicators;

public static class Technicals
{
    public static double[] Sma(double[] data, int period)
    {
        if (data.Length == 0) return [];
        
        var result = new double[data.Length];
        double sum = 0;
        for (var i = 0; i < data.Length; i++)
        {
            sum += data[i];
            if (i >= period)
                sum -= data[i - period];
            
            if (i >= period - 1)
                result[i] = sum / period;
            else
                result[i] = double.NaN;
        }
        return result;
    }

    /// <summary>
    /// Matches Pandas ewm(span=period, adjust=False).mean()
    /// Alpha = 2 / (period + 1)
    /// Init = data[0]
    /// </summary>
    public static double[] EmaPandas(double[] data, int period)
    {
        var result = new double[data.Length];
        if (data.Length == 0) return result;

        var alpha = 2.0 / (period + 1.0);
        result[0] = data[0];

        for (var i = 1; i < data.Length; i++)
        {
            result[i] = (data[i] * alpha) + (result[i - 1] * (1 - alpha));
        }
        return result;
    }

    /// <summary>
    /// TA-Lib style EMA.
    /// The first value is SMA of the first 'period' values.
    /// Then Wilder/EMA smoothing.
    /// </summary>
    public static double[] EmaTaLib(double[] data, int period)
    {
        var result = new double[data.Length];
        if (data.Length < period) 
        {
            Array.Fill(result, double.NaN);
            return result;
        }

        for(var i = 0; i < period - 1; i++) result[i] = double.NaN;

        double sum = 0;
        for (var i = 0; i < period; i++) sum += data[i];
        result[period - 1] = sum / period;

        var multiplier = 2.0 / (period + 1.0);
        
        for (var i = period; i < data.Length; i++)
        {
            result[i] = (data[i] - result[i - 1]) * multiplier + result[i - 1];
        }
        return result;
    }

    /// <summary>
    /// Wilder's Smoothing (used for ATR, ADX, RSI in TA-Lib).
    /// Effectively EMA with alpha = 1/period.
    /// </summary>
    private static double[] WilderSmooth(double[] data, int period)
    {
        var result = new double[data.Length];
        if (data.Length < period)
        {
            Array.Fill(result, double.NaN);
            return result;
        }

        for (var i = 0; i < period - 1; i++) result[i] = double.NaN;

        double sum = 0;
        for (var i = 0; i < period; i++) 
            sum += double.IsNaN(data[i]) ? 0 : data[i];

        result[period - 1] = sum / period;

        for (var i = period; i < data.Length; i++)
        {
            result[i] = (result[i - 1] * (period - 1) + data[i]) / period;
        }
        return result;
    }

    public static double[] Atr(double[] high, double[] low, double[] close, int period)
    {
        if (high.Length == 0) return [];
        
        var tr = new double[high.Length];
        tr[0] = high[0] - low[0];
        for (var i = 1; i < high.Length; i++)
        {
            var hl = high[i] - low[i];
            var hc = Math.Abs(high[i] - close[i - 1]);
            var lc = Math.Abs(low[i] - close[i - 1]);
            tr[i] = Math.Max(hl, Math.Max(hc, lc));
        }

        return WilderSmooth(tr, period);
    }

    public static (double[] upper, double[] middle, double[] lower) BollingerBands(double[] data, int period, double nbDev)
    {
        if (data.Length == 0) return ([], [], []);
        
        var middle = Sma(data, period);
        var upper = new double[data.Length];
        var lower = new double[data.Length];

        for (var i = 0; i < data.Length; i++)
        {
            if (double.IsNaN(middle[i]))
            {
                upper[i] = double.NaN;
                lower[i] = double.NaN;
                continue;
            }

            double sumSqDiff = 0;
            for (var k = 0; k < period; k++)
            {
                var diff = data[i - k] - middle[i];
                sumSqDiff += diff * diff;
            }
            var stdDev = Math.Sqrt(sumSqDiff / period);

            upper[i] = middle[i] + (nbDev * stdDev);
            lower[i] = middle[i] - (nbDev * stdDev);
        }

        return (upper, middle, lower);
    }

    public static double[] Rsi(double[] data, int period)
    {
        if (data.Length == 0) return [];
        
        var gains = new double[data.Length];
        var losses = new double[data.Length];

        for (var i = 1; i < data.Length; i++)
        {
            var diff = data[i] - data[i - 1];
            if (diff > 0)
                gains[i] = diff;
            else
                losses[i] = -diff;
        }

        var avgGain = WilderSmooth(gains, period);
        var avgLoss = WilderSmooth(losses, period);
        
        var rsi = new double[data.Length];
        for (var i = 0; i < data.Length; i++)
        {
            if (double.IsNaN(avgGain[i]) || double.IsNaN(avgLoss[i]))
            {
                rsi[i] = double.NaN;
                continue;
            }

            if (avgLoss[i] == 0)
            {
                rsi[i] = 100;
            }
            else
            {
                var rs = avgGain[i] / avgLoss[i];
                rsi[i] = 100 - (100 / (1 + rs));
            }
        }
        return rsi;
    }

    public static (double[] macd, double[] signal, double[] hist) Macd(double[] data, int fastPeriod, int slowPeriod, int signalPeriod)
    {
        var fastEma = EmaTaLib(data, fastPeriod);
        var slowEma = EmaTaLib(data, slowPeriod);
        
        var macdLine = new double[data.Length];
        for (var i = 0; i < data.Length; i++)
        {
            macdLine[i] = fastEma[i] - slowEma[i];
        }

        var signalLine = EmaTaLibRobust(macdLine, signalPeriod);
        
        var hist = new double[data.Length];
        for (var i = 0; i < data.Length; i++)
        {
            hist[i] = macdLine[i] - signalLine[i];
        }
        return (macdLine, signalLine, hist);
    }

    /// <summary>
    /// EMA that skips leading NaNs.
    /// </summary>
    private static double[] EmaTaLibRobust(double[] data, int period)
    {
        var result = new double[data.Length];
        Array.Fill(result, double.NaN);

        var firstValid = 0;
        while (firstValid < data.Length && double.IsNaN(data[firstValid]))
            firstValid++;
        
        if (data.Length - firstValid < period)
            return result;

        double sum = 0;
        for (var i = 0; i < period; i++)
            sum += data[firstValid + i];
        
        result[firstValid + period - 1] = sum / period;

        var multiplier = 2.0 / (period + 1.0);
        
        for (var i = firstValid + period; i < data.Length; i++)
        {
            result[i] = (data[i] - result[i - 1]) * multiplier + result[i - 1];
        }
        return result;
    }

    public static (double[] k, double[] d) Stochastic(double[] high, double[] low, double[] close, int fastKPeriod, int slowKPeriod, int slowDPeriod)
    {
        if (high.Length == 0) return ([], []);
        
        var fastK = new double[high.Length];
        
        for (var i = 0; i < high.Length; i++)
        {
            if (i < fastKPeriod - 1)
            {
                fastK[i] = double.NaN;
                continue;
            }

            var lowestLow = double.MaxValue;
            var highestHigh = double.MinValue;

            for (var j = 0; j < fastKPeriod; j++)
            {
                lowestLow = Math.Min(lowestLow, low[i - j]);
                highestHigh = Math.Max(highestHigh, high[i - j]);
            }

            var range = highestHigh - lowestLow;
            if (range == 0)
                fastK[i] = 100;
            else
                fastK[i] = 100 * (close[i] - lowestLow) / range;
        }

        var slowK = SmaRobust(fastK, slowKPeriod);

        var slowD = SmaRobust(slowK, slowDPeriod);

        return (slowK, slowD);
    }

    private static double[] SmaRobust(double[] data, int period)
    {
        var result = new double[data.Length];
        Array.Fill(result, double.NaN);
        
        var firstValid = 0;
        while (firstValid < data.Length && double.IsNaN(data[firstValid]))
            firstValid++;
        
        if (data.Length - firstValid < period) return result;

        double sum = 0;
        for (var i = firstValid; i < firstValid + period; i++)
            sum += data[i];
            
        result[firstValid + period - 1] = sum / period;

        for (var i = firstValid + period; i < data.Length; i++)
        {
            sum += data[i];
            sum -= data[i - period];
            result[i] = sum / period;
        }
        return result;
    }

    public static double[] Roc(double[] data, int period)
    {
        if (data.Length == 0) return [];
        
        var result = new double[data.Length];
        for (var i = 0; i < data.Length; i++)
        {
            if (i < period)
                result[i] = double.NaN;
            else
            {
                var prev = data[i - period];
                if (prev != 0)
                    result[i] = ((data[i] - prev) / prev) * 100;
                else
                    result[i] = 0; 
            }
        }
        return result;
    }

    public static double[] Adx(double[] high, double[] low, double[] close, int period)
    {
        if (high.Length == 0) return [];
        
        var tr = new double[high.Length];
        var plusDm = new double[high.Length];
        var minusDm = new double[high.Length];

        tr[0] = 0;
        tr[0] = high[0] - low[0];
        plusDm[0] = 0;
        minusDm[0] = 0;

        for (var i = 1; i < high.Length; i++)
        {
            var h = high[i];
            var l = low[i];
            var prevH = high[i - 1];
            var prevL = low[i - 1];
            var prevC = close[i - 1];

            tr[i] = Math.Max(h - l, Math.Max(Math.Abs(h - prevC), Math.Abs(l - prevC)));
            
            var diffP = h - prevH;
            var diffM = prevL - l;

            if (diffP > 0 && diffP > diffM)
                plusDm[i] = diffP;
            else
                plusDm[i] = 0;

            if (diffM > 0 && diffM > diffP)
                minusDm[i] = diffM;
            else
                minusDm[i] = 0;
        }

        var trSmoothed = WilderSmoothResult(tr, period);
        var plusDmSmoothed = WilderSmoothResult(plusDm, period);
        var minusDmSmoothed = WilderSmoothResult(minusDm, period);

        var dx = new double[high.Length];
        for (var i = 0; i < high.Length; i++)
        {
            if (double.IsNaN(trSmoothed[i]) || trSmoothed[i] == 0)
            {
                dx[i] = double.NaN;
                continue;
            }

            var plusDi = 100 * plusDmSmoothed[i] / trSmoothed[i];
            var minusDi = 100 * minusDmSmoothed[i] / trSmoothed[i];
            
            var sum = plusDi + minusDi;
            if (sum == 0)
                dx[i] = 0;
            else
                dx[i] = 100 * Math.Abs(plusDi - minusDi) / sum;
        }

        return WilderSmoothResultRobust(dx, period);
    }

    private static double[] WilderSmoothResult(double[] data, int period)
    {
        return WilderSmooth(data, period);
    }
    
    private static double[] WilderSmoothResultRobust(double[] data, int period)
    {
        var result = new double[data.Length];
        Array.Fill(result, double.NaN);

        var firstValid = 0;
        while (firstValid < data.Length && double.IsNaN(data[firstValid]))
            firstValid++;

        if (data.Length - firstValid < period) return result;

        double sum = 0;
        for (var i = firstValid; i < firstValid + period; i++)
            sum += data[i];

        result[firstValid + period - 1] = sum / period;

        for (var i = firstValid + period; i < data.Length; i++)
        {
            result[i] = (result[i - 1] * (period - 1) + data[i]) / period;
        }
        return result;
    }
}

namespace Ougha.Trading.Features.Indicators;

public static class Technicals
{
    public static double[] Sma(double[] data, int period)
    {
        if (data.Length == 0) return [];
        
        var result = new double[data.Length];
        double sum = 0;
        for (int i = 0; i < data.Length; i++)
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

        double alpha = 2.0 / (period + 1.0);
        result[0] = data[0]; // Pandas adjust=False inits with first value

        for (int i = 1; i < data.Length; i++)
        {
            result[i] = (data[i] * alpha) + (result[i - 1] * (1 - alpha));
        }
        return result;
    }

    /// <summary>
    /// TA-Lib style EMA.
    /// First value is SMA of first 'period' values.
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

        // Initialize with NaNs
        for(int i = 0; i < period - 1; i++) result[i] = double.NaN;

        // First valid value is SMA
        double sum = 0;
        for (int i = 0; i < period; i++) sum += data[i];
        result[period - 1] = sum / period;

        double multiplier = 2.0 / (period + 1.0);
        
        for (int i = period; i < data.Length; i++)
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

        for (int i = 0; i < period - 1; i++) result[i] = double.NaN;

        // First value is SMA
        double sum = 0;
        for (int i = 0; i < period; i++) 
            sum += double.IsNaN(data[i]) ? 0 : data[i]; // careful with NaNs in input?
        
        result[period - 1] = sum / period;

        for (int i = period; i < data.Length; i++)
        {
            // Previous * (N-1) + Current  / N
            result[i] = (result[i - 1] * (period - 1) + data[i]) / period;
        }
        return result;
    }

    public static double[] Atr(double[] high, double[] low, double[] close, int period)
    {
        if (high.Length == 0) return [];
        
        var tr = new double[high.Length];
        tr[0] = high[0] - low[0];
        for (int i = 1; i < high.Length; i++)
        {
            double hl = high[i] - low[i];
            double hc = Math.Abs(high[i] - close[i - 1]);
            double lc = Math.Abs(low[i] - close[i - 1]);
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

        for (int i = 0; i < data.Length; i++)
        {
            if (double.IsNaN(middle[i]))
            {
                upper[i] = double.NaN;
                lower[i] = double.NaN;
                continue;
            }

            // Calculate StdDev (Population or Sample? TA-Lib uses ? Usually Population for indicators)
            // TA-Lib sources: variance += diff * diff; ... stddev = sqrt(variance / timePeriod);
            // This is population std dev (div by N).
            double sumSqDiff = 0;
            for (int k = 0; k < period; k++)
            {
                double diff = data[i - k] - middle[i]; // Wait, standard deviation around the MA? Or around the window mean?
                // BB uses standard deviation of the window. The MA *is* the mean of the window.
                sumSqDiff += diff * diff;
            }
            double stdDev = Math.Sqrt(sumSqDiff / period);

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
        
        // Calculate differences
        for (int i = 1; i < data.Length; i++)
        {
            double diff = data[i] - data[i - 1];
            if (diff > 0)
                gains[i] = diff;
            else
                losses[i] = -diff;
        }

        var avgGain = WilderSmooth(gains, period);
        var avgLoss = WilderSmooth(losses, period);
        
        var rsi = new double[data.Length];
        for (int i = 0; i < data.Length; i++)
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
                double rs = avgGain[i] / avgLoss[i];
                rsi[i] = 100 - (100 / (1 + rs));
            }
        }
        return rsi;
    }

    public static (double[] macd, double[] signal, double[] hist) Macd(double[] data, int fastPeriod, int slowPeriod, int signalPeriod)
    {
        // TA-Lib MACD uses its specific EMA (with SMA init).
        // It calculates EMA(fast) and EMA(slow).
        var fastEma = EmaTaLib(data, fastPeriod);
        var slowEma = EmaTaLib(data, slowPeriod);
        
        var macdLine = new double[data.Length];
        for (int i = 0; i < data.Length; i++)
        {
            macdLine[i] = fastEma[i] - slowEma[i];
        }

        // Signal line is EMA of MACD line
        // TA-Lib takes the valid part of MACD line and applies EMA.
        // The first valid MACD value is at index (slowPeriod - 1).
        // So Signal line calculation starts there.
        // We can pass the whole macdLine to EmaTaLib, but EmaTaLib expects continuous data?
        // NaNs at start of macdLine will be handled by EmaTaLib returning NaNs until it has enough valid data?
        // My EmaTaLib implementation returns NaNs if data has NaNs? No, "sum += data[i]".
        // I need to modify EmaTaLib or wrap it to handle NaNs at start.
        
        // Actually simpler: pass MACD line to EmaTaLib. But EmaTaLib needs to skip NaNs.
        // Let's make EmaTaLib robust to leading NaNs.
        var signalLine = EmaTaLibRobust(macdLine, signalPeriod);
        
        var hist = new double[data.Length];
        for (int i = 0; i < data.Length; i++)
        {
            hist[i] = macdLine[i] - signalLine[i];
        }
        return (macdLine, signalLine, hist);
    }

    /// <summary>
    /// EMA that skips leading NaNs.
    /// </summary>
    public static double[] EmaTaLibRobust(double[] data, int period)
    {
        var result = new double[data.Length];
        Array.Fill(result, double.NaN);

        // Find first valid index
        int firstValid = 0;
        while (firstValid < data.Length && double.IsNaN(data[firstValid]))
            firstValid++;
        
        if (data.Length - firstValid < period)
            return result;

        // SMA of first 'period' valid values
        double sum = 0;
        for (int i = 0; i < period; i++)
            sum += data[firstValid + i];
        
        result[firstValid + period - 1] = sum / period;

        double multiplier = 2.0 / (period + 1.0);
        
        for (int i = firstValid + period; i < data.Length; i++)
        {
            result[i] = (data[i] - result[i - 1]) * multiplier + result[i - 1];
        }
        return result;
    }

    public static (double[] k, double[] d) Stochastic(double[] high, double[] low, double[] close, int fastK_Period, int slowK_Period, int slowD_Period)
    {
        if (high.Length == 0) return ([], []);
        
        var fastK = new double[high.Length];
        
        for (int i = 0; i < high.Length; i++)
        {
            if (i < fastK_Period - 1)
            {
                fastK[i] = double.NaN;
                continue;
            }

            double lowestLow = double.MaxValue;
            double highestHigh = double.MinValue;

            for (int j = 0; j < fastK_Period; j++)
            {
                lowestLow = Math.Min(lowestLow, low[i - j]);
                highestHigh = Math.Max(highestHigh, high[i - j]);
            }

            double range = highestHigh - lowestLow;
            if (range == 0)
                fastK[i] = 100; // Or 50?
            else
                fastK[i] = 100 * (close[i] - lowestLow) / range;
        }

        // SlowK = SMA(FastK, slowK_Period)
        // Need SmaRobust for NaNs
        var slowK = SmaRobust(fastK, slowK_Period);
        
        // SlowD = SMA(SlowK, slowD_Period)
        var slowD = SmaRobust(slowK, slowD_Period);

        return (slowK, slowD);
    }

    public static double[] SmaRobust(double[] data, int period)
    {
        var result = new double[data.Length];
        Array.Fill(result, double.NaN);
        
        int firstValid = 0;
        while (firstValid < data.Length && double.IsNaN(data[firstValid]))
            firstValid++;
        
        if (data.Length - firstValid < period) return result;

        double sum = 0;
        for (int i = firstValid; i < firstValid + period; i++)
            sum += data[i];
            
        result[firstValid + period - 1] = sum / period;

        for (int i = firstValid + period; i < data.Length; i++)
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
        for (int i = 0; i < data.Length; i++)
        {
            if (i < period)
                result[i] = double.NaN;
            else
            {
                double prev = data[i - period];
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

        tr[0] = 0; // First is usually skipped in Wilder's? TA-Lib sets first TR to H-L?
        // TA-Lib: First TR is High-Low.
        tr[0] = high[0] - low[0];
        plusDm[0] = 0;
        minusDm[0] = 0;

        for (int i = 1; i < high.Length; i++)
        {
            double h = high[i];
            double l = low[i];
            double prevH = high[i - 1];
            double prevL = low[i - 1];
            double prevC = close[i - 1];

            tr[i] = Math.Max(h - l, Math.Max(Math.Abs(h - prevC), Math.Abs(l - prevC)));
            
            double diffP = h - prevH;
            double diffM = prevL - l;

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
        for (int i = 0; i < high.Length; i++)
        {
            if (double.IsNaN(trSmoothed[i]) || trSmoothed[i] == 0)
            {
                dx[i] = double.NaN;
                continue;
            }

            double plusDi = 100 * plusDmSmoothed[i] / trSmoothed[i];
            double minusDi = 100 * minusDmSmoothed[i] / trSmoothed[i];
            
            double sum = plusDi + minusDi;
            if (sum == 0)
                dx[i] = 0;
            else
                dx[i] = 100 * Math.Abs(plusDi - minusDi) / sum;
        }

        // ADX is usually SMA(DX, period)? No, Wilder's smoothing of DX!
        // TA-Lib DX: SMA? 
        // TA-Lib ADX function:
        // Returns Wilder's smoothed DX.
        // Wait, TA-Lib might just use SMA for the ADX part.
        // Checking sources... ADX is the *Wilder smoothed* average of DX.
        // So I need WilderSmoothResult(dx, period).
        // But WilderSmoothResult needs to handle the NaNs at the start of DX.
        
        return WilderSmoothResultRobust(dx, period);
    }
    
    // Helper to allow re-entry of WilderSmooth logic
    // Same as WilderSmooth but named differently to avoid confusion or merge?
    // I'll reuse my logic.
    private static double[] WilderSmoothResult(double[] data, int period)
    {
         // Copied logic from WilderSmooth above
         // Better to expose WilderSmooth if I can use it.
         return WilderSmooth(data, period);
    }
    
    private static double[] WilderSmoothResultRobust(double[] data, int period)
    {
        // Skip NaNs
        var result = new double[data.Length];
        Array.Fill(result, double.NaN);

        int firstValid = 0;
        while (firstValid < data.Length && double.IsNaN(data[firstValid]))
            firstValid++;

        if (data.Length - firstValid < period) return result;

        double sum = 0;
        for (int i = firstValid; i < firstValid + period; i++)
            sum += data[i];

        result[firstValid + period - 1] = sum / period;

        for (int i = firstValid + period; i < data.Length; i++)
        {
            result[i] = (result[i - 1] * (period - 1) + data[i]) / period;
        }
        return result;
    }
}

namespace Ougha.Trading.Data.Services;

/// <summary>
/// Service for calculating cross-symbol correlation features (20D).
/// Uses rolling return correlations to identify market regime and diversification.
/// </summary>
public class CorrelationService
{
    private readonly Dictionary<string, Queue<double>> _returnHistory = new();
    private const int MaxHistory = 100;
    private const int CorrelationWindow = 20;
    
    private readonly Dictionary<string, double> _lastPrices = new();
    
    /// <summary>
    /// Update price history and calculate correlation features.
    /// Feature vector (20D):
    /// [0-5]: Pairwise correlations for up to 6 pairs
    /// [6-11]: Correlation regime flags per pair
    /// [12-14]: Average correlation for 1H/4H/1D windows
    /// [15-17]: Correlation momentum (change vs past)
    /// [18]: Market regime (-1=risk-off, 0=uncorrelated, 1=risk-on)
    /// [19]: Diversification benefit score
    /// </summary>
    public float[] BuildCorrelationFeatures(Dictionary<string, double> currentPrices, string[] symbols)
    {
        var features = new float[20];
        
        // Update return history for each symbol
        foreach (var symbol in symbols)
        {
            if (!currentPrices.TryGetValue(symbol, out var price) || price <= 0)
                continue;
                
            if (!_returnHistory.ContainsKey(symbol))
                _returnHistory[symbol] = new Queue<double>();
            
            if (_lastPrices.TryGetValue(symbol, out var lastPrice) && lastPrice > 0)
            {
                var ret = (price - lastPrice) / lastPrice;
                _returnHistory[symbol].Enqueue(ret);
                
                while (_returnHistory[symbol].Count > MaxHistory)
                    _returnHistory[symbol].Dequeue();
            }
            
            _lastPrices[symbol] = price;
        }
        
        // Build symbol pairs
        var pairs = BuildPairs(symbols);
        
        // Calculate pairwise correlations (features 0-5)
        var correlations = new List<double>();
        for (var i = 0; i < Math.Min(pairs.Count, 6); i++)
        {
            var (sym1, sym2) = pairs[i];
            var corr = CalculateCorrelation(sym1, sym2, CorrelationWindow);
            features[i] = (float)corr;
            correlations.Add(corr);
        }
        
        // Correlation regime flags (features 6-11)
        // >0.7 = trending together, <-0.7 = diverging
        for (var i = 0; i < Math.Min(pairs.Count, 6); i++)
        {
            var corr = i < correlations.Count ? correlations[i] : 0;
            features[6 + i] = corr > 0.7f ? 1f : (corr < -0.7f ? -1f : 0f);
        }
        
        // Average correlations for different windows (features 12-14)
        if (correlations.Count > 0)
        {
            var avgCorr = correlations.Average();
            features[12] = (float)avgCorr;  // Current (proxy for 1H)
            features[13] = (float)avgCorr;  // 4H (same for now, could track history)
            features[14] = (float)avgCorr;  // 1D (same for now)
        }
        
        // Correlation momentum (features 15-17) - simplified: 0 for now
        features[15] = 0f;
        features[16] = 0f;
        features[17] = 0f;
        
        // Market regime (feature 18)
        // Risk-on: high positive correlations (markets moving together up)
        // Risk-off: high negative correlations (flight to safety)
        if (correlations.Count > 0)
        {
            var avgCorr = correlations.Average();
            features[18] = avgCorr > 0.5 ? 1f : (avgCorr < -0.5 ? -1f : 0f);
        }
        
        // Diversification benefit (feature 19)
        // Lower average absolute correlation = better diversification
        if (correlations.Count > 0)
        {
            var avgAbsCorr = correlations.Average(c => Math.Abs(c));
            features[19] = (float)(1.0 - avgAbsCorr);  // 0=no benefit, 1=max benefit
        }
        
        // Clamp all features
        for (var i = 0; i < features.Length; i++)
            features[i] = Math.Clamp(features[i], -1f, 1f);
        
        return features;
    }
    
    private static List<(string, string)> BuildPairs(string[] symbols)
    {
        var pairs = new List<(string, string)>();
        for (var i = 0; i < symbols.Length; i++)
        {
            for (var j = i + 1; j < symbols.Length; j++)
            {
                pairs.Add((symbols[i], symbols[j]));
            }
        }
        return pairs;
    }
    
    private double CalculateCorrelation(string sym1, string sym2, int window)
    {
        if (!_returnHistory.TryGetValue(sym1, out var returns1) ||
            !_returnHistory.TryGetValue(sym2, out var returns2))
            return 0;
        
        var r1 = returns1.TakeLast(window).ToArray();
        var r2 = returns2.TakeLast(window).ToArray();
        
        var n = Math.Min(r1.Length, r2.Length);
        if (n < 5) return 0;
        
        // Pearson correlation
        var mean1 = r1.Take(n).Average();
        var mean2 = r2.Take(n).Average();
        
        double cov = 0, var1 = 0, var2 = 0;
        for (var i = 0; i < n; i++)
        {
            var d1 = r1[i] - mean1;
            var d2 = r2[i] - mean2;
            cov += d1 * d2;
            var1 += d1 * d1;
            var2 += d2 * d2;
        }
        
        var denom = Math.Sqrt(var1 * var2);
        return denom > 0 ? cov / denom : 0;
    }
    
    /// <summary>
    /// Reset service state for new episode.
    /// </summary>
    public void Reset()
    {
        _returnHistory.Clear();
        _lastPrices.Clear();
    }
}

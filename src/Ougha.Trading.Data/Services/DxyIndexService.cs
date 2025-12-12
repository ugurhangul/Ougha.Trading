namespace Ougha.Trading.Data.Services;

/// <summary>
/// Service for calculating DXY (US Dollar Index) features.
/// Can load actual DXY data or calculate synthetic DXY from component pairs.
/// 
/// DXY composition: EUR (57.6%), JPY (13.6%), GBP (11.9%), CAD (9.1%), SEK (4.2%), CHF (3.6%)
/// Formula: DXY = 50.14348 × EURUSD^(-0.576) × USDJPY^(0.136) × GBPUSD^(-0.119) × USDCAD^(0.091) × USDSEK^(0.042) × USDCHF^(0.036)
/// </summary>
public class DxyIndexService
{
    private readonly Queue<float> _dxyHistory = new();
    private const int MaxHistory = 100;  // For RSI and SMA calculations
    
    // DXY component weights
    private static readonly Dictionary<string, (float Weight, bool Inverted)> DxyComponents = new()
    {
        ["EURUSD"] = (-0.576f, true),   // EUR weight, inverted (EURUSD down = DXY up)
        ["USDJPY"] = (0.136f, false),   // JPY weight
        ["GBPUSD"] = (-0.119f, true),   // GBP weight, inverted
        ["USDCAD"] = (0.091f, false),   // CAD weight
        ["USDSEK"] = (0.042f, false),   // SEK weight
        ["USDCHF"] = (0.036f, false)    // CHF weight
    };
    
    private float _lastDxy = 100f;
    private float _dxyHourAgo = 100f;
    private float _dxy4HoursAgo = 100f;
    private float _dxyDayAgo = 100f;
    
    private DateTime _lastHourUpdate = DateTime.MinValue;
    private DateTime _last4HourUpdate = DateTime.MinValue;
    private DateTime _lastDayUpdate = DateTime.MinValue;
    
    /// <summary>
    /// Calculate synthetic DXY value from available forex pairs
    /// </summary>
    private static float CalculateSyntheticDxy(Dictionary<string, double> currentPrices)
    {
        // Base DXY constant
        var dxy = 50.14348;
        
        foreach (var (pair, (weight, _)) in DxyComponents)
        {
            if (currentPrices.TryGetValue(pair, out var price) && price > 0)
            {
                dxy *= Math.Pow(price, weight);
            }
        }
        
        return (float)Math.Max(80, Math.Min(120, dxy));  // Reasonable DXY bounds
    }
    
    /// <summary>
    /// Update DXY history and calculate features
    /// </summary>
    public float[] BuildDxyFeatures(Dictionary<string, double> currentPrices, DateTime timestamp)
    {
        var dxy = CalculateSyntheticDxy(currentPrices);
        
        // Update history
        _dxyHistory.Enqueue(dxy);
        while (_dxyHistory.Count > MaxHistory)
            _dxyHistory.Dequeue();
        
        // Track hourly/4-hourly/daily changes
        if ((timestamp - _lastHourUpdate).TotalHours >= 1)
        {
            _dxyHourAgo = _lastDxy;
            _lastHourUpdate = timestamp;
        }
        if ((timestamp - _last4HourUpdate).TotalHours >= 4)
        {
            _dxy4HoursAgo = _lastDxy;
            _last4HourUpdate = timestamp;
        }
        if ((timestamp - _lastDayUpdate).TotalHours >= 24)
        {
            _dxyDayAgo = _lastDxy;
            _lastDayUpdate = timestamp;
        }
        
        _lastDxy = dxy;
        
        // Build features
        var features = new float[8];
        
        // 0: DXY normalized (centered around 100)
        features[0] = (dxy - 100f) / 10f;  // Normalize to roughly [-2, 2]
        
        // 1: 1-hour change %
        features[1] = _dxyHourAgo > 0 ? (dxy - _dxyHourAgo) / _dxyHourAgo * 100f : 0f;
        
        // 2: 4-hour change %
        features[2] = _dxy4HoursAgo > 0 ? (dxy - _dxy4HoursAgo) / _dxy4HoursAgo * 100f : 0f;
        
        // 3: 1-day change %
        features[3] = _dxyDayAgo > 0 ? (dxy - _dxyDayAgo) / _dxyDayAgo * 100f : 0f;
        
        // 4: RSI-like (simplified)
        features[4] = CalculateRsi() / 50f - 1f;  // Normalize to [-1, 1]
        
        // 5: SMA deviation (current vs 20-period SMA)
        var sma = CalculateSma(20);
        features[5] = sma > 0 ? (dxy - sma) / sma * 100f : 0f;
        
        // 6: Volatility (normalized ATR-like)
        features[6] = CalculateVolatility();
        
        // 7: Trend direction (-1 down, 0 range, +1 up)
        features[7] = CalculateTrendDirection();
        
        // Clamp all features
        for (var i = 0; i < features.Length; i++)
            features[i] = Math.Clamp(features[i], -3f, 3f);
        
        return features;
    }
    
    private float CalculateRsi(int period = 14)
    {
        if (_dxyHistory.Count < period + 1)
            return 50f;
        
        var values = _dxyHistory.ToArray();
        float gains = 0, losses = 0;
        
        for (var i = values.Length - period; i < values.Length; i++)
        {
            var change = values[i] - values[i - 1];
            if (change > 0) gains += change;
            else losses -= change;
        }
        
        if (losses == 0) return 100f;
        var rs = gains / losses;
        return 100f - (100f / (1f + rs));
    }
    
    private float CalculateSma(int period)
    {
        if (_dxyHistory.Count < period)
            return _lastDxy;
        
        return _dxyHistory.TakeLast(period).Average();
    }
    
    private float CalculateVolatility()
    {
        if (_dxyHistory.Count < 10)
            return 0f;
        
        var values = _dxyHistory.TakeLast(10).ToArray();
        var changes = new List<float>();
        
        for (var i = 1; i < values.Length; i++)
            changes.Add(Math.Abs(values[i] - values[i - 1]));
        
        var avgChange = changes.Average();
        return avgChange;  // Already small value, no need to normalize much
    }
    
    private float CalculateTrendDirection()
    {
        if (_dxyHistory.Count < 5)
            return 0f;
        
        var shortSma = _dxyHistory.TakeLast(5).Average();
        var longSma = _dxyHistory.TakeLast(Math.Min(20, _dxyHistory.Count)).Average();
        
        var diff = (shortSma - longSma) / longSma * 100f;

        return diff switch
        {
            > 0.1f => 1f,
            < -0.1f => -1f,
            _ => 0f
        };
    }
    
    /// <summary>
    /// Reset service state (call at start of new episodes)
    /// </summary>
    public void Reset()
    {
        _dxyHistory.Clear();
        _lastDxy = 100f;
        _dxyHourAgo = 100f;
        _dxy4HoursAgo = 100f;
        _dxyDayAgo = 100f;
        _lastHourUpdate = DateTime.MinValue;
        _last4HourUpdate = DateTime.MinValue;
        _lastDayUpdate = DateTime.MinValue;
    }
}

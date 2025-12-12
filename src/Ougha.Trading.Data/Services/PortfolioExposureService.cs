using Ougha.Trading.Core.Models;

namespace Ougha.Trading.Data.Services;

/// <summary>
/// Service for calculating portfolio exposure features (12D).
/// Tracks currency exposures, concentration, and margin usage.
/// </summary>
public class PortfolioExposureService
{
    // Currency mappings for major forex pairs
    private static readonly Dictionary<string, (string Base, string Quote)> CurrencyPairs = new()
    {
        ["EURUSD"] = ("EUR", "USD"),
        ["GBPUSD"] = ("GBP", "USD"),
        ["USDJPY"] = ("USD", "JPY"),
        ["USDCHF"] = ("USD", "CHF"),
        ["AUDUSD"] = ("AUD", "USD"),
        ["USDCAD"] = ("USD", "CAD"),
        ["NZDUSD"] = ("NZD", "USD"),
        ["EURJPY"] = ("EUR", "JPY"),
        ["GBPJPY"] = ("GBP", "JPY"),
        ["EURGBP"] = ("EUR", "GBP"),
        ["USDTRY"] = ("USD", "TRY"),
        ["XAUUSD"] = ("XAU", "USD"),
    };

    /// <summary>
    /// Build portfolio exposure features (12D).
    /// Features:
    /// [0]: Net USD exposure (-1 to 1)
    /// [1]: Net EUR exposure
    /// [2]: Net GBP exposure
    /// [3]: Net JPY exposure
    /// [4]: Gross exposure (total position value / equity)
    /// [5]: Long exposure ratio
    /// [6]: Short exposure ratio
    /// [7]: Concentration score (HHI-based)
    /// [8]: Max single position weight
    /// [9]: Open positions count normalized
    /// [10]: Margin usage ratio
    /// [11]: Available margin ratio
    /// </summary>
    public float[] BuildExposureFeatures(
        IReadOnlyList<Position> positions,
        double equity,
        double freeMargin,
        double usedMargin,
        int maxPositions = 10)
    {
        var features = new float[12];
        
        if (equity <= 0)
            return features;
        
        // Track currency exposures
        var currencyExposure = new Dictionary<string, double>
        {
            ["USD"] = 0, ["EUR"] = 0, ["GBP"] = 0, ["JPY"] = 0
        };
        
        double totalLongExposure = 0;
        double totalShortExposure = 0;
        var positionWeights = new List<double>();
        
        foreach (var pos in positions)
        {
            if (!CurrencyPairs.TryGetValue(pos.Symbol, out var currencies))
                continue;
            
            // Calculate position value (simplified: use volume as proxy)
            var posValue = pos.Volume * 100000;  // Standard lot = 100,000 units
            var posWeight = posValue / equity;
            positionWeights.Add(Math.Abs(posWeight));
            
            // Direction multiplier
            var direction = pos.Type == TradeType.Buy ? 1.0 : -1.0;
            var exposureAmount = posWeight * direction;
            
            // Track long/short exposure
            if (pos.Type == TradeType.Buy)
                totalLongExposure += Math.Abs(posWeight);
            else
                totalShortExposure += Math.Abs(posWeight);
            
            // Update currency exposures
            // For EURUSD Buy: long EUR, short USD
            // For USDJPY Buy: long USD, short JPY
            if (currencyExposure.ContainsKey(currencies.Base))
                currencyExposure[currencies.Base] += exposureAmount;
            if (currencyExposure.ContainsKey(currencies.Quote))
                currencyExposure[currencies.Quote] -= exposureAmount;
        }
        
        // Normalize currency exposures to [-1, 1]
        var maxExposure = currencyExposure.Values.Max(v => Math.Abs(v));
        var normFactor = maxExposure > 0 ? 1.0 / Math.Max(maxExposure, 1.0) : 1.0;
        
        features[0] = (float)(currencyExposure["USD"] * normFactor);
        features[1] = (float)(currencyExposure["EUR"] * normFactor);
        features[2] = (float)(currencyExposure["GBP"] * normFactor);
        features[3] = (float)(currencyExposure["JPY"] * normFactor);
        
        // Gross exposure (total / equity)
        var grossExposure = totalLongExposure + totalShortExposure;
        features[4] = (float)Math.Min(grossExposure, 5.0) / 5.0f;  // Cap at 5x leverage
        
        // Long/short ratios
        var totalExposure = totalLongExposure + totalShortExposure;
        features[5] = totalExposure > 0 ? (float)(totalLongExposure / totalExposure) : 0.5f;
        features[6] = totalExposure > 0 ? (float)(totalShortExposure / totalExposure) : 0.5f;
        
        // Concentration (Herfindahl-Hirschman Index)
        // HHI = sum of squared weights, 0=diversified, 1=concentrated
        if (positionWeights.Count > 0)
        {
            var totalWeight = positionWeights.Sum();
            if (totalWeight > 0)
            {
                var hhi = positionWeights.Sum(w => Math.Pow(w / totalWeight, 2));
                features[7] = (float)hhi;
            }
        }
        
        // Max single position weight
        features[8] = positionWeights.Count > 0 
            ? (float)Math.Min(positionWeights.Max() / Math.Max(grossExposure, 0.01), 1.0) 
            : 0f;
        
        // Open positions normalized
        features[9] = (float)positions.Count / maxPositions;
        
        // Margin usage ratio
        var totalMargin = usedMargin + freeMargin;
        features[10] = totalMargin > 0 ? (float)(usedMargin / totalMargin) : 0f;
        
        // Available margin ratio
        features[11] = totalMargin > 0 ? (float)(freeMargin / totalMargin) : 1f;
        
        // Clamp all features
        for (var i = 0; i < features.Length; i++)
            features[i] = Math.Clamp(features[i], -1f, 1f);
        
        return features;
    }
    
    /// <summary>
    /// Reset service state for new episode.
    /// </summary>
    public void Reset()
    {
        // No persistent state to reset currently
    }
}

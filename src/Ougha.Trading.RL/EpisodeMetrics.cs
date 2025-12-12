namespace Ougha.Trading.RL;

/// <summary>
/// Tracks episode-level metrics for PF and Sharpe calculation.
/// </summary>
public class EpisodeMetrics
{
    private double _grossProfit;
    private double _grossLoss;
    private int _tradeCount;
    private readonly List<double> _returns = [];

    public int TradeCount => _tradeCount;

    public void UpdateTrade(double tradeProfit)
    {
        _tradeCount++;
        if (tradeProfit > 0)
            _grossProfit += tradeProfit;
        else
            _grossLoss += Math.Abs(tradeProfit);
        
        _returns.Add(tradeProfit);
    }

    public double CalculateProfitFactor()
    {
        if (_grossLoss <= 0) return _grossProfit > 0 ? 10.0 : 1.0;
        return _grossProfit / _grossLoss;
    }

    public double CalculateSharpeRatio(float annualizationFactor = 252f)
    {
        if (_returns.Count < 2) return 0;
        
        var mean = _returns.Average();
        var variance = _returns.Sum(r => Math.Pow(r - mean, 2)) / (_returns.Count - 1);
        var stdDev = Math.Sqrt(variance);
        
        if (stdDev <= 0) return mean > 0 ? 5.0 : 0;
        
        return (mean / stdDev) * Math.Sqrt(annualizationFactor);
    }

    public void Reset()
    {
        _grossProfit = 0;
        _grossLoss = 0;
        _tradeCount = 0;
        _returns.Clear();
    }
}

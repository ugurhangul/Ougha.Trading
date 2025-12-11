using Ougha.Trading.Backtesting;

namespace Ougha.Trading.Analysis;

public class ResultsAnalyzer
{
    public static PerformanceMetrics Analyze(BacktestResults results)
    {
        var trades = results.TradeLog;
        if (trades.Count == 0) return new PerformanceMetrics();
        
        var profits = trades.Select(t => t.Profit).ToArray();
        var winning = profits.Where(p => p > 0).ToArray();
        var losing = profits.Where(p => p < 0).ToArray();

        double[] returns;
        if (results.EquityCurve.Count > 1)
        {
            returns = new double[results.EquityCurve.Count - 1];
            for (var i = 1; i < results.EquityCurve.Count; i++)
            {
                var prev = results.EquityCurve[i-1];
                if (prev != 0)
                    returns[i-1] = (results.EquityCurve[i] - prev) / prev;
            }
        }
        else
        {
            returns = [];
        }

        var totalProfit = profits.Sum();
        var startBalance = results.InitialBalance;
        
        return new PerformanceMetrics
        {
            TotalReturn = (results.FinalEquity - startBalance) / startBalance * 100.0,
            TotalProfit = totalProfit,
            TotalTrades = trades.Count,
            WinRate = trades.Count > 0 ? (double)winning.Length / trades.Count * 100 : 0,
            ProfitFactor = losing.Length > 0 ? winning.Sum() / Math.Abs(losing.Sum()) : double.PositiveInfinity,
            SharpeRatio = CalculateSharpeRatio(returns),
            MaxDrawdown = CalculateMaxDrawdown(results.EquityCurve),
            AverageWin = winning.Length > 0 ? winning.Average() : 0,
            AverageLoss = losing.Length > 0 ? losing.Average() : 0,
            LargestWin = winning.Length > 0 ? winning.Max() : 0,
            LargestLoss = losing.Length > 0 ? losing.Min() : 0
        };
    }

    private static double CalculateSharpeRatio(double[] returns, double riskFreeRate = 0)
    {
        if (returns.Length < 2) return 0;
        
        if (returns.Length < 2) return 0;

        var mean = returns.Average();
        var sumSq = returns.Sum(d => Math.Pow(d - mean, 2));
        var std = Math.Sqrt(sumSq / (returns.Length - 1));

        if (std == 0) return 0;
        return (mean - riskFreeRate) / std; 
    }

    private static double CalculateMaxDrawdown(List<double>? equityCurve)
    {
        if (equityCurve == null || equityCurve.Count < 2) return 0;

        var peak = equityCurve[0];
        double maxDrawdown = 0;

        foreach (var equity in equityCurve)
        {
            if (equity > peak) peak = equity;
            if (peak == 0) continue;

            var drawdown = (peak - equity) / peak * 100;
            if (drawdown > maxDrawdown) maxDrawdown = drawdown;
        }

        return maxDrawdown;
    }
}

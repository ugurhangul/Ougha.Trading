using Ougha.Trading.Backtesting;

namespace Ougha.Trading.Analysis;

public class ResultsAnalyzer
{
    public PerformanceMetrics Analyze(BacktestResults results)
    {
        var trades = results.TradeLog;
        if (trades.Count == 0) return new PerformanceMetrics();
        
        var profits = trades.Select(t => t.Profit).ToArray();
        var winning = profits.Where(p => p > 0).ToArray();
        var losing = profits.Where(p => p < 0).ToArray();
        
        // Calculate Returns from Equity Curve (Daily) if available
        double[] returns;
        if (results.EquityCurve != null && results.EquityCurve.Count > 1)
        {
            returns = new double[results.EquityCurve.Count - 1];
            for (int i = 1; i < results.EquityCurve.Count; i++)
            {
                // Simple Return: (Current - Prev) / Prev
                double prev = results.EquityCurve[i-1];
                if (prev != 0)
                    returns[i-1] = (results.EquityCurve[i] - prev) / prev;
            }
        }
        else
        {
             // Fallback to per-trade returns? Or just 0.
             // If per-trade, we need basis. 
             // Let's fallback to empty
             returns = new double[0];
        }

        double totalProfit = profits.Sum();
        double startBalance = results.InitialBalance;
        
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

    public double CalculateSharpeRatio(double[] returns, double riskFreeRate = 0)
    {
        if (returns.Length < 2) return 0;
        
        if (returns.Length < 2) return 0;
        
        // Sharpe calculated on periodic returns (daily)
        double mean = returns.Average();
        // Sample std dev
        double sumSq = returns.Sum(d => Math.Pow(d - mean, 2));
        double std = Math.Sqrt(sumSq / (returns.Length - 1));

        if (std == 0) return 0;
        // Annualize? Assume daily returns
        // Sharpe = (Average Return - Rfr) / StdDev * Sqrt(252)
        // Design doc uses simple formula, but we add Sqrt(252) for Standard Annualized Sharpe if desired.
        // For now, return Daily Sharpe or simple Sharpe
        return (mean - riskFreeRate) / std; 
    }

    public double CalculateMaxDrawdown(List<double> equityCurve)
    {
        if (equityCurve == null || equityCurve.Count < 2) return 0;

        double peak = equityCurve[0];
        double maxDrawdown = 0;

        foreach (var equity in equityCurve)
        {
            if (equity > peak) peak = equity;
            if (peak == 0) continue; // Avoid div by zero
            
            double drawdown = (peak - equity) / peak * 100;
            if (drawdown > maxDrawdown) maxDrawdown = drawdown;
        }

        return maxDrawdown;
    }

    public string GenerateReport(PerformanceMetrics metrics)
    {
        return $"""
            ═══════════════════════════════════════════════════════════
                              BACKTEST RESULTS
            ═══════════════════════════════════════════════════════════
            Total Return:     {metrics.TotalReturn:F2}%
            Total Profit:     ${metrics.TotalProfit:F2}
            Total Trades:     {metrics.TotalTrades}
            Win Rate:         {metrics.WinRate:F1}%
            Profit Factor:    {metrics.ProfitFactor:F2}
            Sharpe Ratio:     {metrics.SharpeRatio:F2}
            Max Drawdown:     {metrics.MaxDrawdown:F2}%
            ───────────────────────────────────────────────────────────
            Average Win:      ${metrics.AverageWin:F2}
            Average Loss:     ${metrics.AverageLoss:F2}
            Largest Win:      ${metrics.LargestWin:F2}
            Largest Loss:     ${metrics.LargestLoss:F2}
            ═══════════════════════════════════════════════════════════
            """;
    }
}

using System.Collections.Concurrent;
using System.Text.Json;
using Serilog;

namespace Ougha.Trading.RL.Training;

/// <summary>
/// Logs detailed action decisions with full context for analysis.
/// Helps identify flaws in agent behavior by capturing state-action-reward tuples.
/// </summary>
public class ActionLogger : IDisposable
{
    private readonly ConcurrentQueue<ActionLogEntry> _entries = new();
    private readonly string _logDirectory;
    private readonly int _maxEntriesInMemory;
    private readonly bool _enabled;
    private int _entriesWritten;
    private StreamWriter? _writer;
    private readonly object _writerLock = new();
    
    /// <summary>
    /// Rolling window statistics for quick analysis during training.
    /// </summary>
    public ActionAnalysisStats RollingStats { get; } = new();
    
    public ActionLogger(string logDirectory = "action_logs", int maxEntriesInMemory = 10000, bool enabled = true)
    {
        _logDirectory = logDirectory;
        _maxEntriesInMemory = maxEntriesInMemory;
        _enabled = enabled;
        
        if (_enabled)
        {
            Directory.CreateDirectory(_logDirectory);
            var logPath = Path.Combine(_logDirectory, $"actions_{DateTime.UtcNow:yyyyMMdd_HHmmss}.jsonl");
            _writer = new StreamWriter(logPath, append: true);
            Log.Information("[ActionLogger] Logging actions to {Path}", logPath);
        }
    }

    /// <summary>
    /// Log an action with full context for later analysis.
    /// </summary>
    public void LogAction(ActionLogEntry entry)
    {
        if (!_enabled) return;
        
        // Update rolling stats
        RollingStats.Update(entry);
        
        // Add to memory queue
        _entries.Enqueue(entry);
        
        // Write to disk periodically
        if (_entries.Count >= 100)
        {
            FlushToDisk();
        }
        
        // Trim memory if too large
        while (_entries.Count > _maxEntriesInMemory)
        {
            _entries.TryDequeue(out _);
        }
    }

    /// <summary>
    /// Log a batch of actions efficiently.
    /// </summary>
    public void LogActionBatch(
        int episode,
        int step,
        DateTime timestamp,
        string[] symbols,
        int[] actions,
        float[] rewards,
        bool[] hasPositions,
        float[] unrealizedPnls,
        double[] prices,
        float entropy,
        float[]? actionProbabilities = null)
    {
        if (!_enabled) return;
        
        for (var i = 0; i < symbols.Length; i++)
        {
            var entry = new ActionLogEntry
            {
                Episode = episode,
                Step = step,
                Timestamp = timestamp,
                Symbol = symbols[i],
                Action = actions[i],
                ActionName = GetActionName(actions[i]),
                Reward = rewards[i],
                HasPosition = hasPositions[i],
                UnrealizedPnl = unrealizedPnls[i],
                CurrentPrice = prices[i],
                Entropy = entropy,
                ActionProbabilities = actionProbabilities
            };
            
            LogAction(entry);
        }
    }

    /// <summary>
    /// Log a trade close event with detailed outcome.
    /// </summary>
    public void LogTradeClose(
        int episode,
        int step,
        string symbol,
        double profit,
        int holdingTicks,
        int entryAction,
        double entryPrice,
        double closePrice,
        string closeReason)
    {
        if (!_enabled) return;
        
        var entry = new ActionLogEntry
        {
            Episode = episode,
            Step = step,
            Timestamp = DateTime.UtcNow,
            Symbol = symbol,
            Action = -1, // Special marker for trade close
            ActionName = "CLOSE",
            Reward = (float)profit,
            TradeProfit = profit,
            HoldingTicks = holdingTicks,
            EntryAction = entryAction,
            EntryPrice = entryPrice,
            ClosePrice = closePrice,
            CloseReason = closeReason,
            IsTradeClosed = true
        };
        
        LogAction(entry);
        RollingStats.RecordTrade(profit, holdingTicks, closeReason);
    }

    /// <summary>
    /// Flush all pending entries to disk.
    /// </summary>
    public void FlushToDisk()
    {
        if (!_enabled || _writer == null) return;
        
        lock (_writerLock)
        {
            while (_entries.TryDequeue(out var entry))
            {
                try
                {
                    var json = JsonSerializer.Serialize(entry, new JsonSerializerOptions 
                    { 
                        WriteIndented = false,
                        DefaultIgnoreCondition = System.Text.Json.Serialization.JsonIgnoreCondition.WhenWritingNull
                    });
                    _writer.WriteLine(json);
                    _entriesWritten++;
                }
                catch (Exception ex)
                {
                    Log.Warning(ex, "[ActionLogger] Failed to write entry");
                }
            }
            _writer.Flush();
        }
    }

    /// <summary>
    /// Generate a summary report of action patterns.
    /// </summary>
    public ActionAnalysisReport GenerateReport()
    {
        return RollingStats.GenerateReport();
    }

    private static string GetActionName(int action) => action switch
    {
        0 => "HOLD",
        1 => "BUY",
        2 => "SELL",
        _ => $"ACT_{action}"
    };

    public void Dispose()
    {
        FlushToDisk();
        lock (_writerLock)
        {
            _writer?.Dispose();
            _writer = null;
        }
        Log.Information("[ActionLogger] Wrote {Count} entries to disk", _entriesWritten);
    }
}

/// <summary>
/// Single action log entry with full context.
/// </summary>
public class ActionLogEntry
{
    public int Episode { get; init; }
    public int Step { get; init; }
    public DateTime Timestamp { get; init; }
    public string Symbol { get; init; } = "";
    public int Action { get; init; }
    public string ActionName { get; init; } = "";
    public float Reward { get; init; }
    
    // Position context
    public bool HasPosition { get; init; }
    public float UnrealizedPnl { get; init; }
    public double CurrentPrice { get; init; }
    
    // Agent state
    public float Entropy { get; init; }
    public float[]? ActionProbabilities { get; init; }
    
    // Trade close details (if applicable)
    public bool IsTradeClosed { get; init; }
    public double TradeProfit { get; init; }
    public int HoldingTicks { get; init; }
    public int EntryAction { get; init; }
    public double EntryPrice { get; init; }
    public double ClosePrice { get; init; }
    public string? CloseReason { get; init; }
    
    // Market context (optional)
    public float? Atr { get; init; }
    public float? Volatility { get; init; }
    public int? TrendDirection { get; init; }
}

/// <summary>
/// Rolling window statistics for action analysis.
/// </summary>
public class ActionAnalysisStats
{
    private readonly object _lock = new();
    
    // Action counts
    public int TotalActions { get; private set; }
    public int[] ActionCounts { get; } = new int[8]; // Up to 8 action types
    
    // Action patterns
    public int ConsecutiveHolds { get; private set; }
    public int MaxConsecutiveHolds { get; private set; }
    public int HoldAfterProfit { get; private set; } // HOLDs after profitable close
    public int HoldAfterLoss { get; private set; }   // HOLDs after loss
    
    // Position-specific actions
    public int BuysWithoutPosition { get; private set; }
    public int SellsWithoutPosition { get; private set; }
    public int BuysWithBuyPosition { get; private set; }  // Trying to buy when already long
    public int SellsWithSellPosition { get; private set; } // Trying to sell when already short
    public int HoldsWithProfit { get; private set; }  // Holding in profit
    public int HoldsWithLoss { get; private set; }    // Holding in loss
    
    // Reward tracking
    public float TotalReward { get; private set; }
    public float[] RewardByAction { get; } = new float[8];
    public int[] RewardCountByAction { get; } = new int[8];
    
    // Trade outcomes
    public int TotalTrades { get; private set; }
    public int WinningTrades { get; private set; }
    public int LosingTrades { get; private set; }
    public double TotalProfit { get; private set; }
    public double TotalLoss { get; private set; }
    public long TotalHoldingTicks { get; private set; }
    public int TpHits { get; private set; }
    public int SlHits { get; private set; }
    public int ManualCloses { get; private set; }
    
    // Anomaly detection
    public int ZeroRewardActions { get; private set; }
    public int ExtremeRewards { get; private set; } // |reward| > 0.5
    public int QuickExits { get; private set; }     // Exits < 60 ticks
    public int VeryLongHolds { get; private set; }  // Holds > 3600 ticks
    
    private int _lastAction = -1;
    private double _lastTradeProfit;

    public void Update(ActionLogEntry entry)
    {
        lock (_lock)
        {
            TotalActions++;
            
            if (entry.Action >= 0 && entry.Action < ActionCounts.Length)
            {
                ActionCounts[entry.Action]++;
                RewardByAction[entry.Action] += entry.Reward;
                RewardCountByAction[entry.Action]++;
            }
            
            TotalReward += entry.Reward;
            
            // Track consecutive holds
            if (entry.Action == 0) // HOLD
            {
                ConsecutiveHolds++;
                MaxConsecutiveHolds = Math.Max(MaxConsecutiveHolds, ConsecutiveHolds);
                
                if (_lastTradeProfit > 0) HoldAfterProfit++;
                else if (_lastTradeProfit < 0) HoldAfterLoss++;
                
                if (entry.HasPosition)
                {
                    if (entry.UnrealizedPnl > 0) HoldsWithProfit++;
                    else if (entry.UnrealizedPnl < 0) HoldsWithLoss++;
                }
            }
            else
            {
                ConsecutiveHolds = 0;
            }
            
            // Position-based action analysis
            if (entry.Action == 1) // BUY
            {
                if (!entry.HasPosition) BuysWithoutPosition++;
                else BuysWithBuyPosition++; // Already has position
            }
            else if (entry.Action == 2) // SELL
            {
                if (!entry.HasPosition) SellsWithoutPosition++;
                else SellsWithSellPosition++;
            }
            
            // Reward anomalies
            if (Math.Abs(entry.Reward) < 0.0001f) ZeroRewardActions++;
            if (Math.Abs(entry.Reward) > 0.5f) ExtremeRewards++;
            
            _lastAction = entry.Action;
        }
    }

    public void RecordTrade(double profit, int holdingTicks, string closeReason)
    {
        lock (_lock)
        {
            TotalTrades++;
            TotalHoldingTicks += holdingTicks;
            
            if (profit > 0)
            {
                WinningTrades++;
                TotalProfit += profit;
            }
            else
            {
                LosingTrades++;
                TotalLoss += Math.Abs(profit);
            }
            
            if (closeReason.Contains("TP", StringComparison.OrdinalIgnoreCase)) TpHits++;
            else if (closeReason.Contains("SL", StringComparison.OrdinalIgnoreCase)) SlHits++;
            else ManualCloses++;
            
            if (holdingTicks < 60) QuickExits++;
            if (holdingTicks > 3600) VeryLongHolds++;
            
            _lastTradeProfit = profit;
        }
    }

    public ActionAnalysisReport GenerateReport()
    {
        lock (_lock)
        {
            var report = new ActionAnalysisReport
            {
                TotalActions = TotalActions,
                ActionDistribution = new Dictionary<string, double>
                {
                    ["HOLD"] = TotalActions > 0 ? (double)ActionCounts[0] / TotalActions * 100 : 0,
                    ["BUY"] = TotalActions > 0 ? (double)ActionCounts[1] / TotalActions * 100 : 0,
                    ["SELL"] = TotalActions > 0 ? (double)ActionCounts[2] / TotalActions * 100 : 0
                },
                AverageRewardByAction = new Dictionary<string, double>
                {
                    ["HOLD"] = RewardCountByAction[0] > 0 ? RewardByAction[0] / RewardCountByAction[0] : 0,
                    ["BUY"] = RewardCountByAction[1] > 0 ? RewardByAction[1] / RewardCountByAction[1] : 0,
                    ["SELL"] = RewardCountByAction[2] > 0 ? RewardByAction[2] / RewardCountByAction[2] : 0
                },
                MaxConsecutiveHolds = MaxConsecutiveHolds,
                HoldAfterProfit = HoldAfterProfit,
                HoldAfterLoss = HoldAfterLoss,
                HoldsWithProfit = HoldsWithProfit,
                HoldsWithLoss = HoldsWithLoss,
                BuysWithoutPosition = BuysWithoutPosition,
                SellsWithoutPosition = SellsWithoutPosition,
                TotalTrades = TotalTrades,
                WinRate = TotalTrades > 0 ? (double)WinningTrades / TotalTrades * 100 : 0,
                ProfitFactor = TotalLoss > 0 ? TotalProfit / TotalLoss : TotalProfit > 0 ? 999 : 0,
                AvgHoldingTicks = TotalTrades > 0 ? (double)TotalHoldingTicks / TotalTrades : 0,
                TpHitRate = TotalTrades > 0 ? (double)TpHits / TotalTrades * 100 : 0,
                SlHitRate = TotalTrades > 0 ? (double)SlHits / TotalTrades * 100 : 0,
                QuickExitRate = TotalTrades > 0 ? (double)QuickExits / TotalTrades * 100 : 0,
                ZeroRewardRate = TotalActions > 0 ? (double)ZeroRewardActions / TotalActions * 100 : 0,
                ExtremeRewardRate = TotalActions > 0 ? (double)ExtremeRewards / TotalActions * 100 : 0
            };

            // Identify potential flaws
            report.PotentialFlaws = IdentifyFlaws(report);
            
            return report;
        }
    }

    private List<string> IdentifyFlaws(ActionAnalysisReport report)
    {
        var flaws = new List<string>();
        
        // Action distribution issues
        if (report.ActionDistribution["HOLD"] > 70)
            flaws.Add($"⚠️ HOLD bias: {report.ActionDistribution["HOLD"]:F1}% - Agent rarely trades");
            
        if (report.ActionDistribution["BUY"] > 60 || report.ActionDistribution["SELL"] > 60)
            flaws.Add($"⚠️ Directional bias: BUY={report.ActionDistribution["BUY"]:F1}%, SELL={report.ActionDistribution["SELL"]:F1}%");
        
        // Consecutive hold issues
        if (MaxConsecutiveHolds > 500)
            flaws.Add($"⚠️ Dead periods: Max {MaxConsecutiveHolds} consecutive HOLDs - Agent may be stuck");
        
        // Holding behavior issues
        if (HoldsWithLoss > HoldsWithProfit * 2 && TotalTrades > 10)
            flaws.Add($"⚠️ Loss aversion: Holding losers {HoldsWithLoss}x vs winners {HoldsWithProfit}x - Cut losses earlier");
            
        if (HoldsWithProfit > 0 && HoldsWithLoss > 0)
        {
            var holdRatio = (double)HoldsWithLoss / HoldsWithProfit;
            if (holdRatio < 0.3)
                flaws.Add($"⚠️ Profit taking too early: Holding profits only {HoldsWithProfit}x vs losses {HoldsWithLoss}x");
        }
        
        // Trade execution issues
        if (report.QuickExitRate > 30)
            flaws.Add($"⚠️ Quick exits: {report.QuickExitRate:F1}% trades close < 1 min - Consider MIN_HOLDING_TICKS");
            
        if (report.SlHitRate > 70)
            flaws.Add($"⚠️ High SL hit rate: {report.SlHitRate:F1}% - SL may be too tight or entries poor");
            
        if (report.TpHitRate < 10 && TotalTrades > 20)
            flaws.Add($"⚠️ Low TP hit rate: {report.TpHitRate:F1}% - TP may be too ambitious or not reached");
        
        // Reward issues
        if (report.ZeroRewardRate > 80)
            flaws.Add($"⚠️ Zero reward rate: {report.ZeroRewardRate:F1}% - Sparse reward signal problem");
            
        if (report.ExtremeRewardRate > 20)
            flaws.Add($"⚠️ Extreme rewards: {report.ExtremeRewardRate:F1}% - May cause gradient instability");
        
        // Average reward analysis
        if (report.AverageRewardByAction["HOLD"] > report.AverageRewardByAction["BUY"] * 2 &&
            report.AverageRewardByAction["HOLD"] > report.AverageRewardByAction["SELL"] * 2)
            flaws.Add($"⚠️ HOLD over-rewarded: avg HOLD={report.AverageRewardByAction["HOLD"]:F4} vs BUY={report.AverageRewardByAction["BUY"]:F4}");
        
        // Win rate vs profit factor mismatch
        if (report.WinRate > 50 && report.ProfitFactor < 1.0)
            flaws.Add($"⚠️ Win rate/PF mismatch: {report.WinRate:F1}% WR but PF={report.ProfitFactor:F2} - Losses too large");
            
        if (report.WinRate < 40 && report.ProfitFactor > 1.5)
            flaws.Add($"✅ Low WR but good PF: Risk/Reward working well");
        
        return flaws;
    }
}

/// <summary>
/// Comprehensive action analysis report.
/// </summary>
public class ActionAnalysisReport
{
    public int TotalActions { get; init; }
    public Dictionary<string, double> ActionDistribution { get; init; } = new();
    public Dictionary<string, double> AverageRewardByAction { get; init; } = new();
    
    public int MaxConsecutiveHolds { get; init; }
    public int HoldAfterProfit { get; init; }
    public int HoldAfterLoss { get; init; }
    public int HoldsWithProfit { get; init; }
    public int HoldsWithLoss { get; init; }
    public int BuysWithoutPosition { get; init; }
    public int SellsWithoutPosition { get; init; }
    
    public int TotalTrades { get; init; }
    public double WinRate { get; init; }
    public double ProfitFactor { get; init; }
    public double AvgHoldingTicks { get; init; }
    public double TpHitRate { get; init; }
    public double SlHitRate { get; init; }
    public double QuickExitRate { get; init; }
    
    public double ZeroRewardRate { get; init; }
    public double ExtremeRewardRate { get; init; }
    
    public List<string> PotentialFlaws { get; set; } = new();

    public override string ToString()
    {
        var sb = new System.Text.StringBuilder();
        sb.AppendLine("═══════════════════════════════════════════════════════");
        sb.AppendLine("              ACTION ANALYSIS REPORT                   ");
        sb.AppendLine("═══════════════════════════════════════════════════════");
        
        sb.AppendLine("\n📊 ACTION DISTRIBUTION:");
        foreach (var (action, pct) in ActionDistribution)
            sb.AppendLine($"   {action,-6}: {pct,6:F1}%");
            
        sb.AppendLine("\n💰 AVERAGE REWARD BY ACTION:");
        foreach (var (action, reward) in AverageRewardByAction)
            sb.AppendLine($"   {action,-6}: {reward:+0.0000;-0.0000;0.0000}");
        
        sb.AppendLine("\n📈 TRADING METRICS:");
        sb.AppendLine($"   Total Trades:    {TotalTrades}");
        sb.AppendLine($"   Win Rate:        {WinRate:F1}%");
        sb.AppendLine($"   Profit Factor:   {ProfitFactor:F2}");
        sb.AppendLine($"   Avg Hold Time:   {AvgHoldingTicks:F0} ticks ({AvgHoldingTicks/60:F1} min)");
        sb.AppendLine($"   TP Hit Rate:     {TpHitRate:F1}%");
        sb.AppendLine($"   SL Hit Rate:     {SlHitRate:F1}%");
        sb.AppendLine($"   Quick Exit Rate: {QuickExitRate:F1}%");
        
        sb.AppendLine("\n🔍 BEHAVIORAL PATTERNS:");
        sb.AppendLine($"   Max Consecutive HOLDs: {MaxConsecutiveHolds}");
        sb.AppendLine($"   HOLDs after profit:    {HoldAfterProfit}");
        sb.AppendLine($"   HOLDs after loss:      {HoldAfterLoss}");
        sb.AppendLine($"   HOLDs in profit:       {HoldsWithProfit}");
        sb.AppendLine($"   HOLDs in loss:         {HoldsWithLoss}");
        
        sb.AppendLine("\n⚠️ REWARD SIGNAL HEALTH:");
        sb.AppendLine($"   Zero Reward Rate:    {ZeroRewardRate:F1}%");
        sb.AppendLine($"   Extreme Reward Rate: {ExtremeRewardRate:F1}%");
        
        if (PotentialFlaws.Count > 0)
        {
            sb.AppendLine("\n🚨 POTENTIAL FLAWS DETECTED:");
            foreach (var flaw in PotentialFlaws)
                sb.AppendLine($"   {flaw}");
        }
        else
        {
            sb.AppendLine("\n✅ No obvious flaws detected");
        }
        
        sb.AppendLine("\n═══════════════════════════════════════════════════════");
        
        return sb.ToString();
    }
}

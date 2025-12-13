using System.Text.Json;
using Ougha.Trading.RL.Training;
using Serilog;

namespace Ougha.Trading.RL.Training;

/// <summary>
/// Analyzes action log files to find behavioral patterns and potential flaws.
/// Can be run after training to get deeper insights.
/// </summary>
public static class ActionLogAnalyzer
{
    /// <summary>
    /// Analyze action logs from a directory and generate a comprehensive report.
    /// </summary>
    public static ActionAnalysisReport AnalyzeFromDirectory(string logDirectory)
    {
        var stats = new ActionAnalysisStats();
        var entryCount = 0;
        
        if (!Directory.Exists(logDirectory))
        {
            Log.Warning("[ActionLogAnalyzer] Log directory not found: {Directory}", logDirectory);
            return stats.GenerateReport();
        }
        
        var files = Directory.GetFiles(logDirectory, "actions_*.jsonl")
            .OrderByDescending(f => f)
            .ToList();
        
        if (files.Count == 0)
        {
            Log.Warning("[ActionLogAnalyzer] No action log files found in {Directory}", logDirectory);
            return stats.GenerateReport();
        }
        
        Log.Information("[ActionLogAnalyzer] Found {Count} log files to analyze", files.Count);
        
        foreach (var file in files)
        {
            try
            {
                foreach (var line in File.ReadLines(file))
                {
                    if (string.IsNullOrWhiteSpace(line)) continue;
                    
                    try
                    {
                        var entry = JsonSerializer.Deserialize<ActionLogEntry>(line);
                        if (entry != null)
                        {
                            stats.Update(entry);
                            
                            if (entry.IsTradeClosed)
                            {
                                stats.RecordTrade(
                                    entry.TradeProfit, 
                                    entry.HoldingTicks, 
                                    entry.CloseReason ?? "Unknown");
                            }
                            
                            entryCount++;
                        }
                    }
                    catch (JsonException)
                    {
                        // Skip malformed lines
                    }
                }
            }
            catch (Exception ex)
            {
                Log.Warning(ex, "[ActionLogAnalyzer] Error reading file: {File}", file);
            }
        }
        
        Log.Information("[ActionLogAnalyzer] Analyzed {Count} action entries", entryCount);
        
        return stats.GenerateReport();
    }

    /// <summary>
    /// Analyze and find specific behavioral patterns for debugging.
    /// </summary>
    public static DebugAnalysisResult AnalyzeForDebugging(string logDirectory, int maxEntries = 10000)
    {
        var result = new DebugAnalysisResult();
        
        if (!Directory.Exists(logDirectory))
            return result;
        
        var files = Directory.GetFiles(logDirectory, "actions_*.jsonl")
            .OrderByDescending(f => f)
            .ToList();
        
        if (files.Count == 0)
            return result;
        
        var entries = new List<ActionLogEntry>();
        
        foreach (var file in files)
        {
            try
            {
                foreach (var line in File.ReadLines(file))
                {
                    if (entries.Count >= maxEntries) break;
                    if (string.IsNullOrWhiteSpace(line)) continue;
                    
                    try
                    {
                        var entry = JsonSerializer.Deserialize<ActionLogEntry>(line);
                        if (entry != null)
                            entries.Add(entry);
                    }
                    catch { /* Skip */ }
                }
            }
            catch { /* Skip */ }
            
            if (entries.Count >= maxEntries) break;
        }
        
        result.TotalEntries = entries.Count;
        
        // Find consecutive hold sequences
        var currentHoldStreak = 0;
        ActionLogEntry? holdStreakStart = null;
        
        foreach (var entry in entries)
        {
            if (entry.Action == 0) // HOLD
            {
                if (currentHoldStreak == 0)
                    holdStreakStart = entry;
                currentHoldStreak++;
            }
            else
            {
                if (currentHoldStreak > 100 && holdStreakStart != null)
                {
                    result.LongHoldSequences.Add(new HoldSequence
                    {
                        StartEpisode = holdStreakStart.Episode,
                        StartStep = holdStreakStart.Step,
                        Length = currentHoldStreak,
                        Symbol = holdStreakStart.Symbol
                    });
                }
                currentHoldStreak = 0;
                holdStreakStart = null;
            }
        }
        
        // Find actions with extreme rewards
        result.ExtremeRewardActions = entries
            .Where(e => Math.Abs(e.Reward) > 0.3f)
            .Select(e => new ExtremeRewardEntry
            {
                Episode = e.Episode,
                Step = e.Step,
                Symbol = e.Symbol,
                Action = e.ActionName,
                Reward = e.Reward,
                HasPosition = e.HasPosition,
                UnrealizedPnl = e.UnrealizedPnl
            })
            .Take(50)
            .ToList();
        
        // Find trade patterns
        var tradeCloses = entries.Where(e => e.IsTradeClosed).ToList();
        result.TradeClosePatterns = new TradeClosePatterns
        {
            TotalTrades = tradeCloses.Count,
            AvgHoldingTicks = tradeCloses.Count > 0 ? tradeCloses.Average(t => t.HoldingTicks) : 0,
            MinHoldingTicks = tradeCloses.Count > 0 ? tradeCloses.Min(t => t.HoldingTicks) : 0,
            MaxHoldingTicks = tradeCloses.Count > 0 ? tradeCloses.Max(t => t.HoldingTicks) : 0,
            QuickExitCount = tradeCloses.Count(t => t.HoldingTicks < 60),
            TpHitCount = tradeCloses.Count(t => t.CloseReason?.Contains("TP") == true),
            SlHitCount = tradeCloses.Count(t => t.CloseReason?.Contains("SL") == true),
            ManualCloseCount = tradeCloses.Count(t => t.CloseReason?.Contains("Manual") == true)
        };
        
        // Find action distribution by position state
        var withPosition = entries.Where(e => e.HasPosition).ToList();
        var withoutPosition = entries.Where(e => !e.HasPosition).ToList();
        
        result.ActionDistributionWithPosition = new Dictionary<string, int>
        {
            ["HOLD"] = withPosition.Count(e => e.Action == 0),
            ["BUY"] = withPosition.Count(e => e.Action == 1),
            ["SELL"] = withPosition.Count(e => e.Action == 2)
        };
        
        result.ActionDistributionWithoutPosition = new Dictionary<string, int>
        {
            ["HOLD"] = withoutPosition.Count(e => e.Action == 0),
            ["BUY"] = withoutPosition.Count(e => e.Action == 1),
            ["SELL"] = withoutPosition.Count(e => e.Action == 2)
        };
        
        // Find positions where agent took opposite action (potential conflict)
        result.ConflictingActions = entries
            .Where(e => e.HasPosition && e.UnrealizedPnl > 0 && e.Action != 0)
            .Take(20)
            .Select(e => $"Ep{e.Episode}/Step{e.Step}: {e.ActionName} while holding profitable position ({e.UnrealizedPnl:F2}%)")
            .ToList();
        
        return result;
    }
}

/// <summary>
/// Detailed debug analysis results.
/// </summary>
public class DebugAnalysisResult
{
    public int TotalEntries { get; set; }
    public List<HoldSequence> LongHoldSequences { get; set; } = new();
    public List<ExtremeRewardEntry> ExtremeRewardActions { get; set; } = new();
    public TradeClosePatterns TradeClosePatterns { get; set; } = new();
    public Dictionary<string, int> ActionDistributionWithPosition { get; set; } = new();
    public Dictionary<string, int> ActionDistributionWithoutPosition { get; set; } = new();
    public List<string> ConflictingActions { get; set; } = new();
    
    public override string ToString()
    {
        var sb = new System.Text.StringBuilder();
        sb.AppendLine("═══════════════════════════════════════════════════════");
        sb.AppendLine("              DEBUG ANALYSIS RESULTS                    ");
        sb.AppendLine("═══════════════════════════════════════════════════════");
        
        sb.AppendLine($"\n📊 Total Entries Analyzed: {TotalEntries}");
        
        sb.AppendLine("\n🔄 LONG HOLD SEQUENCES (>100 consecutive HOLDs):");
        if (LongHoldSequences.Count == 0)
            sb.AppendLine("   None found");
        else
            foreach (var seq in LongHoldSequences.Take(10))
                sb.AppendLine($"   Ep{seq.StartEpisode}/Step{seq.StartStep}: {seq.Length} holds on {seq.Symbol}");
        
        sb.AppendLine("\n💥 EXTREME REWARD ACTIONS (|reward| > 0.3):");
        if (ExtremeRewardActions.Count == 0)
            sb.AppendLine("   None found");
        else
            foreach (var action in ExtremeRewardActions.Take(10))
                sb.AppendLine($"   Ep{action.Episode}/Step{action.Step}: {action.Action} => {action.Reward:+0.0000;-0.0000}");
        
        sb.AppendLine("\n📈 TRADE CLOSE PATTERNS:");
        sb.AppendLine($"   Total Trades:     {TradeClosePatterns.TotalTrades}");
        sb.AppendLine($"   Avg Hold Time:    {TradeClosePatterns.AvgHoldingTicks:F0} ticks");
        sb.AppendLine($"   Min/Max Hold:     {TradeClosePatterns.MinHoldingTicks}-{TradeClosePatterns.MaxHoldingTicks} ticks");
        sb.AppendLine($"   Quick Exits (<1m): {TradeClosePatterns.QuickExitCount}");
        sb.AppendLine($"   TP Hits:          {TradeClosePatterns.TpHitCount}");
        sb.AppendLine($"   SL Hits:          {TradeClosePatterns.SlHitCount}");
        sb.AppendLine($"   Manual Closes:    {TradeClosePatterns.ManualCloseCount}");
        
        sb.AppendLine("\n📍 ACTION DISTRIBUTION BY POSITION STATE:");
        sb.AppendLine("   WITH position:");
        foreach (var (action, count) in ActionDistributionWithPosition)
            sb.AppendLine($"      {action,-6}: {count}");
        sb.AppendLine("   WITHOUT position:");
        foreach (var (action, count) in ActionDistributionWithoutPosition)
            sb.AppendLine($"      {action,-6}: {count}");
        
        if (ConflictingActions.Count > 0)
        {
            sb.AppendLine("\n⚠️ CONFLICTING ACTIONS (trading while in profit):");
            foreach (var conflict in ConflictingActions.Take(10))
                sb.AppendLine($"   {conflict}");
        }
        
        sb.AppendLine("\n═══════════════════════════════════════════════════════");
        
        return sb.ToString();
    }
}

public class HoldSequence
{
    public int StartEpisode { get; set; }
    public int StartStep { get; set; }
    public int Length { get; set; }
    public string Symbol { get; set; } = "";
}

public class ExtremeRewardEntry
{
    public int Episode { get; set; }
    public int Step { get; set; }
    public string Symbol { get; set; } = "";
    public string Action { get; set; } = "";
    public float Reward { get; set; }
    public bool HasPosition { get; set; }
    public float UnrealizedPnl { get; set; }
}

public class TradeClosePatterns
{
    public int TotalTrades { get; set; }
    public double AvgHoldingTicks { get; set; }
    public int MinHoldingTicks { get; set; }
    public int MaxHoldingTicks { get; set; }
    public int QuickExitCount { get; set; }
    public int TpHitCount { get; set; }
    public int SlHitCount { get; set; }
    public int ManualCloseCount { get; set; }
}

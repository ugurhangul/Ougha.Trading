using Spectre.Console;
using Spectre.Console.Rendering;
using Ougha.Trading.RL.Training;

namespace Ougha.Trading.App.Runners;

/// <summary>
/// Rich console display utilities for RL training progress using Spectre.Console.
/// Mirrors the Python display.py functionality.
/// </summary>
public static class TrainingDisplay
{
    /// <summary>
    /// Build a complete training display with all panels.
    /// </summary>
    public static IRenderable BuildDisplay(TrainingStats stats, TrainingBudget budget)
    {
        var rows = new List<IRenderable>
        {
            BuildHeaderPanel(stats),
            BuildStatusPanel(stats),
            BuildPerformancePanel(stats),
            BuildBudgetPanel(budget),
            BuildGlobalStatsPanel(stats),
            BuildSymbolPerformanceTable(stats),
            BuildProgressBar(stats)
        };
        
        return new Rows(rows);
    }

    /// <summary>
    /// Build the header panel with episode progress, chunk info, and date ranges.
    /// </summary>
    private static Panel BuildHeaderPanel(TrainingStats stats)
    {
        var elapsed = stats.Elapsed;
        var eta = stats.EstimatedTimeRemaining;
        var progress = stats.TotalEpisodes > 0 
            ? (double)stats.Episode / stats.TotalEpisodes * 100 
            : 0;
        
        var table = new Table().Border(TableBorder.None).HideHeaders();
        table.AddColumn("L1").AddColumn("V1").AddColumn("L2").AddColumn("V2").AddColumn("L3").AddColumn("V3");

        table.AddRow(
            "[bold]Episode:[/]", $"[bold yellow]{stats.Episode}/{stats.TotalEpisodes}[/] ({progress:F1}%)",
            "[bold]Elapsed:[/]", $@"[cyan]{elapsed:hh\:mm\:ss}[/]",
            "[bold]ETA:[/]", $@"[yellow]{eta:hh\:mm\:ss}[/]"
        );

        var chunkDateRange = stats.ChunkStartDate != DateTime.MinValue 
            ? $"{stats.ChunkStartDate:MM/dd} - {stats.ChunkEndDate:MM/dd}"
            : "-";
        table.AddRow(
            "[bold]Chunk:[/]", $"[magenta]{stats.CurrentChunk}/{stats.TotalChunks}[/]",
            "[bold]Chunk Dates:[/]", $"[dim]{chunkDateRange}[/]",
            "[bold]Eps/Chunk:[/]", $"[dim]{stats.EpisodeInChunk}/{stats.EpisodesPerChunk}[/]"
        );

        var episodeDateRange = stats.EpisodeStartDate != DateTime.MinValue 
            ? $"{stats.EpisodeStartDate:yyyy-MM-dd HH:mm} → {stats.EpisodeEndDate:yyyy-MM-dd HH:mm}"
            : "-";
        table.AddRow(
            "[bold]Episode Range:[/]", $"[cyan]{episodeDateRange}[/]",
            "", "", "", ""
        );
        
        return new Panel(table)
            .Header("[bold cyan]Multi-Symbol RL Training[/]")
            .Border(BoxBorder.Rounded)
            .BorderColor(Color.Cyan1);
    }

    /// <summary>
    /// Build the current status panel with live training metrics.
    /// </summary>
    private static Panel BuildStatusPanel(TrainingStats stats)
    {
        var pnlStyle = stats.PnLPercent >= 0 ? "green" : "red";
        var pnlSign = stats.PnLPercent >= 0 ? "+" : "";

        var safeSymbol = Markup.Escape(stats.CurrentSymbol);
        var safeAction = Markup.Escape(stats.CurrentAction);
        
        var table = new Table().Border(TableBorder.None).HideHeaders();
        table.AddColumn("L1").AddColumn("V1").AddColumn("L2").AddColumn("V2").AddColumn("L3").AddColumn("V3");
        
        table.AddRow(
            "[bold]Symbol:[/]", $"[cyan]{safeSymbol}[/]",
            "[bold]Step:[/]", $"[yellow]{stats.CurrentStep}[/]/[dim]{stats.MaxSteps}[/]",
            "[bold]Positions:[/]", $"[blue]{stats.Positions}[/]"
        );

        var explorationLabel = stats.IsPpoAgent ? "Entropy:" : "Epsilon:";
        var explorationValue = stats.IsPpoAgent ? stats.Entropy : stats.Epsilon;
        
        table.AddRow(
            "[bold]Action:[/]", $"[yellow]{safeAction}[/]",
            $"[bold]{explorationLabel}[/]", $"[magenta]{explorationValue:F4}[/]",
            "[bold]Equity:[/]", $"[{pnlStyle}]${stats.Equity:N2} ({pnlSign}{stats.PnLPercent:F2}%)[/]"
        );
        
        table.AddRow(
            "[bold]Ep Reward:[/]", $"[green]{stats.EpisodeReward:F4}[/]",
            "[bold]Avg (100):[/]", $"{stats.AverageReward100:F4}",
            "[bold]Avg (10):[/]", $"{stats.AverageReward10:F4}"
        );
        
        table.AddRow(
            "[bold]Best:[/]", $"[bold green]{stats.BestReward:F4}[/]",
            "[bold]Buffer:[/]", $"{stats.BufferSize:N0}",
            "[bold]GPU:[/]", stats.GpuAvailable ? "[green]Active[/]" : "[red]CPU[/]"
        );
        
        return new Panel(table)
            .Header("[bold]Current Status[/]")
            .Border(BoxBorder.Rounded)
            .BorderColor(Color.Blue);
    }

    /// <summary>
    /// Build a performance metrics panel with timing breakdowns.
    /// </summary>
    private static Panel BuildPerformancePanel(TrainingStats stats)
    {
        var totalTime = stats.ActionTimeMs + stats.EnvStepTimeMs + stats.TrainTimeMs + stats.BufferAddTimeMs;

        var actionPct = totalTime > 0 ? stats.ActionTimeMs / totalTime * 100 : 0;
        var envPct = totalTime > 0 ? stats.EnvStepTimeMs / totalTime * 100 : 0;
        var trainPct = totalTime > 0 ? stats.TrainTimeMs / totalTime * 100 : 0;
        var bufferPct = totalTime > 0 ? stats.BufferAddTimeMs / totalTime * 100 : 0;
        
        var actionStyle = actionPct > 30 ? "red" : actionPct > 15 ? "yellow" : "green";
        var envStyle = envPct > 50 ? "red" : envPct > 30 ? "yellow" : "green";
        var trainStyle = trainPct > 50 ? "red" : trainPct > 30 ? "yellow" : "green";
        var bufferStyle = bufferPct > 20 ? "red" : bufferPct > 10 ? "yellow" : "green";
        
        var table = new Table().Border(TableBorder.None).HideHeaders();
        table.AddColumn("L1").AddColumn("V1").AddColumn("L2").AddColumn("V2").AddColumn("L3").AddColumn("V3").AddColumn("L4").AddColumn("V4");
        
        table.AddRow(
            "[bold]Steps/sec:[/]", $"[bold cyan]{stats.StepsPerSecond:F1}[/]",
            "[bold]Total Steps:[/]", $"{stats.TotalSteps:N0}",
            "[bold]Train Calls:[/]", $"{stats.TrainCalls:N0}",
            "", ""
        );
        
        table.AddRow(
            "[bold]Action:[/]", $"[{actionStyle}]{actionPct:F1}% ({stats.ActionTimeMs:F2}ms)[/]",
            "[bold]Env Step:[/]", $"[{envStyle}]{envPct:F1}% ({stats.EnvStepTimeMs:F2}ms)[/]",
            "[bold]Train:[/]", $"[{trainStyle}]{trainPct:F1}% ({stats.TrainTimeMs:F2}ms)[/]",
            "[bold]Buffer:[/]", $"[{bufferStyle}]{bufferPct:F1}% ({stats.BufferAddTimeMs:F2}ms)[/]"
        );
        
        return new Panel(table)
            .Header("[bold]Performance Metrics[/]")
            .Border(BoxBorder.Rounded)
            .BorderColor(Color.Magenta1);
    }

    /// <summary>
    /// Build training budget panel showing configuration.
    /// </summary>
    private static Panel BuildBudgetPanel(TrainingBudget budget)
    {
        var table = new Table().Border(TableBorder.None).HideHeaders();
        table.AddColumn("L1").AddColumn("V1").AddColumn("L2").AddColumn("V2").AddColumn("L3").AddColumn("V3").AddColumn("L4").AddColumn("V4");

        var safeGpuName = budget.Hardware.GpuName != null ? Markup.Escape(budget.Hardware.GpuName) : "Unknown";
        
        table.AddRow(
            "[bold]Batch:[/]", $"[cyan]{budget.BatchSize:N0}[/]",
            "[bold]Buffer:[/]", $"[cyan]{budget.MemorySize:N0}[/]",
            "[bold]GPU:[/]", budget.Hardware.GpuAvailable ? $"[green]{safeGpuName}[/]" : "[dim]None[/]",
            "[bold]VRAM:[/]", budget.Hardware.GpuAvailable ? $"[yellow]{budget.Hardware.GpuMemoryGb:F1}GB[/]" : "-"
        );
        
        table.AddRow(
            "[bold]Episodes:[/]", $"[yellow]{budget.Episodes:N0}[/]",
            "[bold]Max Steps:[/]", $"[yellow]{budget.MaxSteps:N0}[/]",
            "[bold]LR:[/]", $"[yellow]{budget.LearningRate:F6}[/]",
            "[bold]ε Decay:[/]", $"[yellow]{budget.EpsilonDecay:F6}[/]"
        );
        
        return new Panel(table)
            .Header("[bold]Training Budget[/]")
            .Border(BoxBorder.Rounded)
            .BorderColor(Color.Cyan1);
    }

    /// <summary>
    /// Build a global training stats panel with action distribution and trade counts.
    /// </summary>
    private static Panel BuildGlobalStatsPanel(TrainingStats stats)
    {
        var totalActions = stats.TotalActions > 0 ? stats.TotalActions : 1;

        var holdCount = stats.ActionCounts.GetValueOrDefault(0, 0);
        var buyCount = stats.ActionCounts.GetValueOrDefault(1, 0);
        var sellCount = stats.ActionCounts.GetValueOrDefault(2, 0);
        
        var holdPct = (double)holdCount / totalActions * 100;
        var buyPct = (double)buyCount / totalActions * 100;
        var sellPct = (double)sellCount / totalActions * 100;
        
        var holdStyle = holdPct > 90 ? "red" : holdPct > 70 ? "yellow" : "green";
        var wrStyle = stats.WinRate > 50 ? "green" : "red";
        var pfStyle = stats.ProfitFactor > 1 ? "green" : "red";
        
        var table = new Table().Border(TableBorder.None).HideHeaders();
        table.AddColumn("L1").AddColumn("V1").AddColumn("L2").AddColumn("V2").AddColumn("L3").AddColumn("V3");
        
        table.AddRow(
            "[bold]HOLD:[/]", $"[{holdStyle}]{holdPct:F1}% ({holdCount:N0})[/]",
            "[bold]BUY:[/]", $"[cyan]{buyPct:F1}% ({buyCount:N0})[/]",
            "[bold]SELL:[/]", $"[magenta]{sellPct:F1}% ({sellCount:N0})[/]"
        );
        
        table.AddRow(
            "[bold]Opened:[/]", $"[blue]{stats.TradesOpened:N0}[/]",
            "[bold]Closed:[/]", $"[yellow]{stats.TradesClosed:N0}[/]",
            "[bold]Win Rate:[/]", $"[{wrStyle}]{stats.WinRate:F1}% ({stats.Wins}W/{stats.Losses}L)[/]"
        );
        
        table.AddRow(
            "[bold]Profit:[/]", $"[green]${stats.TotalProfit:N2}[/]",
            "[bold]Loss:[/]", $"[red]${stats.TotalLoss:N2}[/]",
            "[bold]PF:[/]", $"[{pfStyle}]{stats.ProfitFactor:F2}[/]"
        );
        
        return new Panel(table)
            .Header("[bold]Global Training Stats[/]")
            .Border(BoxBorder.Rounded)
            .BorderColor(Color.Green);
    }

    /// <summary>
    /// Build a symbol performance table with per-symbol metrics.
    /// </summary>
    private static Table BuildSymbolPerformanceTable(TrainingStats stats)
    {
        var table = new Table()
            .Title("[bold magenta]Symbol Performance (Best Episode Stats)[/]")
            .Border(TableBorder.Rounded)
            .Expand();
        
        table.AddColumn(new TableColumn("[cyan]Symbol[/]").Width(8));
        table.AddColumn(new TableColumn("Eps").RightAligned().Width(6));
        table.AddColumn(new TableColumn("Avg R").RightAligned().Width(8));
        table.AddColumn(new TableColumn("Best R").RightAligned().Width(8));
        table.AddColumn(new TableColumn("PF").RightAligned().Width(6));
        table.AddColumn(new TableColumn("Sharpe").RightAligned().Width(6));
        table.AddColumn(new TableColumn("MDD").RightAligned().Width(6));
        table.AddColumn(new TableColumn("Win%").RightAligned().Width(6));
        table.AddColumn(new TableColumn("#Trd").RightAligned().Width(5));
        table.AddColumn(new TableColumn("No Imp").RightAligned().Width(8));
        table.AddColumn(new TableColumn("Status").Centered().Width(12));
        
        var sortedSymbols = stats.SymbolPerformance
            .OrderByDescending(x => x.Value.BestReward)
            .ToList();
        
        foreach (var (symbol, s) in sortedSymbols)
        {
            var safeSymbol = Markup.Escape(symbol);
            var avgStyle = s.AverageReward > 0 ? "green" : s.AverageReward < 0 ? "red" : "dim";
            var bestStr = s.BestReward > double.MinValue ? $"[green]{s.BestReward:F2}[/]" : "-";
            var pfStyle = s.BestProfitFactor > 1.5 ? "green" : s.BestProfitFactor > 1 ? "yellow" : "red";
            var sharpeStyle = s.BestSharpe > 1 ? "green" : s.BestSharpe > 0 ? "yellow" : "red";
            var mddStyle = s.BestMaxDrawdown < 10 ? "green" : s.BestMaxDrawdown < 20 ? "yellow" : "red";
            var wrStyle = s.BestWinRate > 50 ? "green" : "yellow";
            
            string statusStr;
            string symDisplay;
            
            if (s.EarlyStopped)
            {
                statusStr = "[bold red]EARLY STOP[/]";
                symDisplay = $"[dim]{safeSymbol}[/]";
            }
            else if (s.NoImprovementCount > stats.EarlyStopPatience * 0.7)
            {
                statusStr = $"[yellow]! {s.NoImprovementCount}/{stats.EarlyStopPatience}[/]";
                symDisplay = safeSymbol;
            }
            else
            {
                statusStr = $"[green]ACTIVE[/] [dim]({s.NoImprovementCount}/{stats.EarlyStopPatience})[/]";
                symDisplay = safeSymbol;
            }
            
            table.AddRow(
                symDisplay,
                s.Episodes.ToString(),
                $"[{avgStyle}]{s.AverageReward:F2}[/]",
                bestStr,
                $"[{pfStyle}]{s.BestProfitFactor:F2}[/]",
                $"[{sharpeStyle}]{s.BestSharpe:F2}[/]",
                $"[{mddStyle}]{s.BestMaxDrawdown:F1}%[/]",
                $"[{wrStyle}]{s.BestWinRate:F0}%[/]",
                s.BestTrades.ToString(),
                s.NoImprovementCount.ToString(),
                statusStr
            );
        }

        if (sortedSymbols.Count == 0)
        {
            table.AddRow("-", "-", "-", "-", "-", "-", "-", "-", "-", "-", "[dim]Waiting...[/]");
        }
        
        return table;
    }

    /// <summary>
    /// Build a progress bar component.
    /// </summary>
    private static IRenderable BuildProgressBar(TrainingStats stats)
    {
        var progress = stats.TotalEpisodes > 0 
            ? (double)stats.Episode / stats.TotalEpisodes * 100 
            : 0;

        var barWidth = 40;
        var filled = (int)(progress / 100.0 * barWidth);
        var bar = new string('█', filled) + new string('░', barWidth - filled);
        
        return new Panel(
            new Markup($"[bold blue]Training Progress:[/] [{(progress > 50 ? "green" : "yellow")}]{bar}[/] {progress:F1}%")
        ).Border(BoxBorder.None);
    }

    /// <summary>
    /// Generate a final training result report.
    /// </summary>
    public static string GenerateResultReport(TrainingStats stats, TrainingBudget budget, string modelDir)
    {
        var elapsed = stats.Elapsed;
        var lines = new List<string>
        {
            "",
            new('=', 80),
            "                    RL TRAINING RESULT REPORT",
            new('=', 80),
            "",
            "📊 TRAINING SUMMARY",
            new('-', 40),
            $"  Total Episodes:        {stats.Episode:N0}",
            $@"  Training Duration:     {elapsed:hh\:mm\:ss}",
            $"  Model Directory:       {modelDir}",
            "",
            "📈 REWARD STATISTICS",
            new('-', 40),
            $"  Best Episode Reward:   {stats.BestReward:F2}",
            $"  Final Avg (last 100):  {stats.AverageReward100:F2}",
            "",
            "🎯 TRADE STATISTICS",
            new('-', 40),
            $"  Trades Opened:         {stats.TradesOpened:N0}",
            $"  Trades Closed:         {stats.TradesClosed:N0}",
            $"  Win Rate:              {stats.WinRate:F1}%",
            $"  Profit Factor:         {stats.ProfitFactor:F2}",
            "",
            "📊 SYMBOL PERFORMANCE",
            new('-', 40)
        };
        
        foreach (var (symbol, s) in stats.SymbolPerformance.OrderByDescending(x => x.Value.BestReward))
        {
            lines.Add($"  {symbol,-12} Eps: {s.Episodes,6} Avg: {s.AverageReward,8:F2} Best: {s.BestReward,8:F2}");
        }
        
        lines.Add("");
        lines.Add(new string('=', 80));
        lines.Add("                         END OF TRAINING REPORT");
        lines.Add(new string('=', 80));
        lines.Add("");
        
        return string.Join(Environment.NewLine, lines);
    }
}

using Spectre.Console;
using Spectre.Console.Rendering;
using Ougha.Trading.RL.Training;

namespace Ougha.Trading.App.Runners;

/// <summary>
/// Rich console display utilities for RL training progress using Spectre.Console.
/// Uses fixed heights and padding to prevent display flickering.
/// </summary>
public static class TrainingDisplay
{
    private const int MaxSymbolRows = 10;
    
    public static IRenderable BuildDisplay(TrainingStats stats, TrainingBudget budget)
    {
        // Use Grid instead of Columns to prevent duplication when terminal is narrow
        var headerGrid = new Grid();
        headerGrid.AddColumn(new GridColumn().NoWrap());
        headerGrid.AddColumn(new GridColumn().NoWrap());
        headerGrid.AddRow(BuildHeaderPanel(stats), BuildBudgetPanel(budget));
        
        var statusGrid = new Grid();
        statusGrid.AddColumn(new GridColumn().NoWrap());
        statusGrid.AddColumn(new GridColumn().NoWrap());
        statusGrid.AddColumn(new GridColumn().NoWrap());
        statusGrid.AddColumn(new GridColumn().NoWrap());
        statusGrid.AddRow(
            BuildStatusPanel(stats), 
            BuildRewardPanel(stats),
            BuildPerformancePanel(stats), 
            BuildTradeStatsPanel(stats)
        );
        
        var rows = new List<IRenderable>
        {
            headerGrid,
            statusGrid,
            BuildSymbolPerformanceTable(stats),
            BuildProgressBar(stats)
        };
        
        return new Rows(rows);
    }

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
            "[bold]Episode:[/]", $"[yellow]{stats.Episode,6}/{stats.TotalEpisodes,-6}[/] ({progress,5:F1}%)",
            "[bold]Elapsed:[/]", $@"[cyan]{elapsed:hh\:mm\:ss}[/]",
            "[bold]ETA:[/]", $@"[yellow]{eta:hh\:mm\:ss}[/]"
        );

        var chunkDateRange = stats.ChunkStartDate != DateTime.MinValue 
            ? $"{stats.ChunkStartDate:MM/dd} - {stats.ChunkEndDate:MM/dd}"
            : "---";
        table.AddRow(
            "[bold]Chunk:[/]", $"[magenta]{stats.CurrentChunk,3}/{stats.TotalChunks,-3}[/]",
            "[bold]Chunk Dates:[/]", $"[dim]{chunkDateRange,-15}[/]",
            "[bold]Eps/Chunk:[/]", $"[dim]{stats.EpisodeInChunk,3}/{stats.EpisodesPerChunk,-3}[/]"
        );

        var episodeDateRange = stats.EpisodeStartDate != DateTime.MinValue 
            ? $"{stats.EpisodeStartDate:yyyy-MM-dd HH:mm} to {stats.EpisodeEndDate:yyyy-MM-dd HH:mm}"
            : "---";
        table.AddRow(
            "[bold]Episode Range:[/]", $"[cyan]{episodeDateRange}[/]",
            "", "", "", ""
        );
        
        return new Panel(table)
            .Header("[bold cyan]Multi-Symbol RL Training[/]")
            .Border(BoxBorder.Double)
            .BorderColor(Color.Cyan1);
    }

    private static Panel BuildStatusPanel(TrainingStats stats)
    {
        var pnlStyle = stats.PnLPercent >= 0 ? "green" : "red";

        var safeSymbol = stats.CurrentSymbol.Length > 10 ? stats.CurrentSymbol[..10] : stats.CurrentSymbol.PadRight(10);
        var safeAction = stats.CurrentAction.PadRight(6);
        
        var table = new Table().Border(TableBorder.None).HideHeaders().Expand();
        table.AddColumn(new TableColumn("L").Width(8));
        table.AddColumn(new TableColumn("V").Width(14).NoWrap());
        
        table.AddRow("[dim]Symbol:[/]", $"[cyan]{Markup.Escape(safeSymbol)}[/]");
        table.AddRow("[dim]Step:[/]", $"[yellow]{stats.CurrentStep,6:N0}[/]/[dim]{stats.MaxSteps:N0}[/]");
        table.AddRow("[dim]Action:[/]", $"[yellow]{Markup.Escape(safeAction)}[/]");
        table.AddRow("[dim]Pos:[/]", $"[blue]{stats.Positions,6}[/]");
        
        var expLabel = stats.IsPpoAgent ? "Entropy:" : "Epsilon:";
        var expValue = stats.IsPpoAgent ? stats.Entropy : stats.Epsilon;
        table.AddRow($"[dim]{expLabel}[/]", $"[magenta]{expValue,12:F4}[/]");
        table.AddRow("[dim]Equity:[/]", $"[{pnlStyle}]${stats.Equity,12:N2}[/]");
        
        return new Panel(table)
            .Header("[bold blue]Status[/]")
            .Border(BoxBorder.Rounded)
            .BorderColor(Color.Blue)
            .Expand();
    }

    private static Panel BuildRewardPanel(TrainingStats stats)
    {
        var table = new Table().Border(TableBorder.None).HideHeaders().Expand();
        table.AddColumn(new TableColumn("L").Width(8));
        table.AddColumn(new TableColumn("V").Width(14).NoWrap());
        
        var epStyle = stats.EpisodeReward > 0 ? "green" : stats.EpisodeReward < 0 ? "red" : "dim";
        var avg100Style = stats.AverageReward100 > 0 ? "green" : stats.AverageReward100 < 0 ? "red" : "dim";
        var avg10Style = stats.AverageReward10 > 0 ? "green" : stats.AverageReward10 < 0 ? "red" : "dim";
        
        var trend = stats.RewardTrend;
        var trendColor = trend == "UP" ? "green" : trend == "DOWN" ? "red" : "yellow";
        
        table.AddRow("[dim]Ep R:[/]", $"[{epStyle}]{stats.EpisodeReward,12:F4}[/]");
        table.AddRow("[dim]Avg100:[/]", $"[{avg100Style}]{stats.AverageReward100,12:F4}[/]");
        table.AddRow("[dim]Avg10:[/]", $"[{avg10Style}]{stats.AverageReward10,12:F4}[/]");
        table.AddRow("[dim]Best:[/]", $"[green]{stats.BestReward,12:F4}[/]");
        table.AddRow("[dim]Trend:[/]", $"[{trendColor}]{trend,12}[/]");
        table.AddRow("[dim]Buf/GPU:[/]", $"[dim]{stats.BufferSize,10:N0}[/] " + (stats.GpuAvailable ? "[green]ON[/]" : "[red]--[/]"));
        
        return new Panel(table)
            .Header("[bold green]Rewards[/]")
            .Border(BoxBorder.Rounded)
            .BorderColor(Color.Green)
            .Expand();
    }

    private static Panel BuildPerformancePanel(TrainingStats stats)
    {
        var totalTime = stats.ActionTimeMs + stats.EnvStepTimeMs + stats.TrainTimeMs + stats.BufferAddTimeMs;
        totalTime = totalTime > 0 ? totalTime : 1;

        var actionPct = stats.ActionTimeMs / totalTime * 100;
        var envPct = stats.EnvStepTimeMs / totalTime * 100;
        var trainPct = stats.TrainTimeMs / totalTime * 100;
        
        var speedStyle = stats.StepsPerSecond > 100 ? "green" : stats.StepsPerSecond > 50 ? "yellow" : "red";
        var epsHrStyle = stats.EpisodesPerHour > 100 ? "green" : stats.EpisodesPerHour > 50 ? "yellow" : "dim";
        
        var table = new Table().Border(TableBorder.None).HideHeaders().Expand();
        table.AddColumn(new TableColumn("L").Width(8));
        table.AddColumn(new TableColumn("V").Width(14).NoWrap());
        
        table.AddRow("[dim]Steps/s:[/]", $"[{speedStyle}]{stats.StepsPerSecond,12:F1}[/]");
        table.AddRow("[dim]Eps/Hr:[/]", $"[{epsHrStyle}]{stats.EpisodesPerHour,12:F1}[/]");
        table.AddRow("[dim]Sec/Ep:[/]", $"[dim]{stats.SecondsPerEpisode,12:F1}[/]");
        table.AddRow("[dim]Trains:[/]", $"[cyan]{stats.TrainCalls,12:N0}[/]");
        table.AddRow("[dim]TSteps:[/]", $"[dim]{stats.TotalSteps,12:N0}[/]");
        table.AddRow("[dim]Timing:[/]", $"[dim]A{actionPct,2:F0} E{envPct,2:F0} T{trainPct,2:F0}%[/]");
        
        return new Panel(table)
            .Header("[bold magenta]Perf[/]")
            .Border(BoxBorder.Rounded)
            .BorderColor(Color.Magenta1)
            .Expand();
    }

    private static Panel BuildTradeStatsPanel(TrainingStats stats)
    {
        var totalActions = stats.TotalActions > 0 ? stats.TotalActions : 1;

        var holdPct = (double)stats.ActionCounts.GetValueOrDefault(0, 0) / totalActions * 100;
        var buyPct = (double)stats.ActionCounts.GetValueOrDefault(1, 0) / totalActions * 100;
        var sellPct = (double)stats.ActionCounts.GetValueOrDefault(2, 0) / totalActions * 100;
        
        var holdStyle = holdPct > 90 ? "red" : holdPct > 70 ? "yellow" : "green";
        var wrStyle = stats.WinRate > 50 ? "green" : stats.WinRate > 40 ? "yellow" : "red";
        var pfStyle = stats.ProfitFactor > 1.5 ? "green" : stats.ProfitFactor > 1 ? "yellow" : "red";
        
        var netPnL = stats.TotalProfit - stats.TotalLoss;
        var pnlStyle = netPnL > 0 ? "green" : netPnL < 0 ? "red" : "dim";
        
        var table = new Table().Border(TableBorder.None).HideHeaders().Expand();
        table.AddColumn(new TableColumn("L").Width(8));
        table.AddColumn(new TableColumn("V").Width(14).NoWrap());
        
        table.AddRow("[dim]Actions:[/]", $"[{holdStyle}]H{holdPct,2:F0}[/] [cyan]B{buyPct,2:F0}[/] [magenta]S{sellPct,2:F0}[/]");
        table.AddRow("[dim]Trades:[/]", $"[dim]{stats.TradesOpened,6:N0}/{stats.TradesClosed,-6:N0}[/]");
        table.AddRow("[dim]WinRate:[/]", $"[{wrStyle}]{stats.WinRate,6:F1}% {stats.Wins,4}W/{stats.Losses}L[/]");
        table.AddRow("[dim]PF:[/]", $"[{pfStyle}]{stats.ProfitFactor,12:F2}[/]");
        table.AddRow("[dim]Gross:[/]", $"[green]+{stats.TotalProfit,7:N0}[/] [red]-{stats.TotalLoss,7:N0}[/]");
        table.AddRow("[dim]Net:[/]", $"[{pnlStyle}]${netPnL,12:N2}[/]");
        
        return new Panel(table)
            .Header("[bold yellow]Trades[/]")
            .Border(BoxBorder.Rounded)
            .BorderColor(Color.Yellow)
            .Expand();
    }

    private static Panel BuildBudgetPanel(TrainingBudget budget)
    {
        var table = new Table().Border(TableBorder.None).HideHeaders();
        table.AddColumn("L1").AddColumn("V1").AddColumn("L2").AddColumn("V2").AddColumn("L3").AddColumn("V3").AddColumn("L4").AddColumn("V4");

        var gpuName = budget.Hardware.GpuName ?? "Unknown";
        if (gpuName.Length > 25) gpuName = gpuName[..25];
        
        table.AddRow(
            "[dim]Batch:[/]", $"[cyan]{budget.BatchSize:N0}[/]",
            "[dim]Buffer:[/]", $"[cyan]{budget.MemorySize:N0}[/]",
            "[dim]GPU:[/]", budget.Hardware.GpuAvailable ? $"[green]{Markup.Escape(gpuName)}[/]" : "[dim]None[/]",
            "[dim]VRAM:[/]", budget.Hardware.GpuAvailable ? $"[yellow]{budget.Hardware.GpuMemoryGb:F0}GB[/]" : "---"
        );
        
        table.AddRow(
            "[dim]Episodes:[/]", $"[yellow]{budget.Episodes:N0}[/]",
            "[dim]MaxSteps:[/]", $"[yellow]{budget.MaxSteps:N0}[/]",
            "[dim]LR:[/]", $"[yellow]{budget.LearningRate:F6}[/]",
            "[dim]Decay:[/]", $"[yellow]{budget.EpsilonDecay:F6}[/]"
        );
        
        return new Panel(table)
            .Header("[bold grey]Config[/]")
            .Border(BoxBorder.Rounded)
            .BorderColor(Color.Grey)
            .Expand();
    }

    private static Table BuildSymbolPerformanceTable(TrainingStats stats)
    {
        var table = new Table()
            .Title("[bold magenta]Symbol Performance (Cumulative)[/]")
            .Border(TableBorder.Rounded)
            .BorderColor(Color.Magenta1)
            .Expand();
        
        table.AddColumn(new TableColumn("[cyan]Symbol[/]").Width(10));
        table.AddColumn(new TableColumn("Eps").RightAligned().Width(6));
        table.AddColumn(new TableColumn("Avg R").RightAligned().Width(9));
        table.AddColumn(new TableColumn("PF").RightAligned().Width(7));
        table.AddColumn(new TableColumn("Win%").RightAligned().Width(7));
        table.AddColumn(new TableColumn("Trades").RightAligned().Width(7));
        table.AddColumn(new TableColumn("$/Trade").RightAligned().Width(9));
        table.AddColumn(new TableColumn("Net P&L").RightAligned().Width(12));
        table.AddColumn(new TableColumn("Status").Centered().Width(8));
        
        var sortedSymbols = stats.SymbolPerformance
            .OrderByDescending(x => x.Value.NetProfit)
            .Take(MaxSymbolRows)
            .ToList();
        
        var rowCount = 0;
        foreach (var (symbol, s) in sortedSymbols)
        {
            var avgStyle = s.AverageReward > 0 ? "green" : s.AverageReward < 0 ? "red" : "dim";
            var pfStyle = s.CumulativeProfitFactor > 1.5 ? "green" : s.CumulativeProfitFactor > 1 ? "yellow" : "red";
            var wrStyle = s.CumulativeWinRate > 50 ? "green" : s.CumulativeWinRate > 40 ? "yellow" : "red";
            var profitStyle = s.NetProfit > 0 ? "green" : s.NetProfit < 0 ? "red" : "dim";
            var avgTradeStyle = s.AvgProfitPerTrade > 0 ? "green" : s.AvgProfitPerTrade < 0 ? "red" : "dim";
            
            string statusStr;
            if (s.EarlyStopped)
                statusStr = "[red]STOP[/]";
            else if (s.NoImprovementCount > stats.EarlyStopPatience * 0.7)
                statusStr = $"[yellow]{s.NoImprovementCount}[/]";
            else
                statusStr = $"[green]{s.NoImprovementCount}[/]";
            
            table.AddRow(
                Markup.Escape(symbol),
                $"{s.Episodes}",
                $"[{avgStyle}]{s.AverageReward:F2}[/]",
                $"[{pfStyle}]{s.CumulativeProfitFactor:F2}[/]",
                $"[{wrStyle}]{s.CumulativeWinRate:F0}%[/]",
                $"{s.CumulativeTrades}",
                $"[{avgTradeStyle}]{s.AvgProfitPerTrade:F2}[/]",
                $"[{profitStyle}]{s.NetProfit:N2}[/]",
                statusStr
            );
            rowCount++;
        }

        while (rowCount < MaxSymbolRows)
        {
            table.AddRow("", "", "", "", "", "", "", "", "");
            rowCount++;
        }
        
        return table;
    }

    private static IRenderable BuildProgressBar(TrainingStats stats)
    {
        var progress = stats.TotalEpisodes > 0 
            ? (double)stats.Episode / stats.TotalEpisodes * 100 
            : 0;

        var barWidth = 60;
        var filled = (int)(progress / 100.0 * barWidth);
        var bar = new string('█', filled) + new string('░', barWidth - filled);
        
        var progressColor = progress > 75 ? "green" : progress > 50 ? "cyan" : progress > 25 ? "yellow" : "red";
        
        return new Markup($"  [bold]Progress:[/] [{progressColor}]{bar}[/] [bold]{progress:F1}%[/]");
    }

    public static string GenerateResultReport(TrainingStats stats, TrainingBudget budget, string modelDir)
    {
        var elapsed = stats.Elapsed;
        var netPnL = stats.TotalProfit - stats.TotalLoss;
        
        var lines = new List<string>
        {
            "",
            new('=', 80),
            "                    RL TRAINING RESULT REPORT",
            new('=', 80),
            "",
            "TRAINING SUMMARY",
            new('-', 40),
            $"  Total Episodes:        {stats.Episode:N0}",
            $@"  Training Duration:     {elapsed:hh\:mm\:ss}",
            $"  Model Directory:       {modelDir}",
            "",
            "REWARD STATISTICS",
            new('-', 40),
            $"  Best Episode Reward:   {stats.BestReward:F2}",
            $"  Final Avg (last 100):  {stats.AverageReward100:F2}",
            "",
            "TRADE STATISTICS",
            new('-', 40),
            $"  Trades Opened:         {stats.TradesOpened:N0}",
            $"  Trades Closed:         {stats.TradesClosed:N0}",
            $"  Win Rate:              {stats.WinRate:F1}%",
            $"  Profit Factor:         {stats.ProfitFactor:F2}",
            $"  Net P&L:               ${netPnL:N2}",
            "",
            "SYMBOL PERFORMANCE",
            new('-', 40)
        };
        
        foreach (var (symbol, s) in stats.SymbolPerformance.OrderByDescending(x => x.Value.NetProfit))
        {
            var trend = s.NetProfit > 0 ? "+" : "";
            lines.Add($"  {symbol,-12} Eps:{s.Episodes,5} PF:{s.CumulativeProfitFactor,6:F2} Win:{s.CumulativeWinRate,5:F1}% P&L:{trend}${s.NetProfit:N2}");
        }
        
        lines.Add("");
        lines.Add(new string('=', 80));
        lines.Add("                         END OF TRAINING REPORT");
        lines.Add(new string('=', 80));
        lines.Add("");
        
        return string.Join(Environment.NewLine, lines);
    }
}

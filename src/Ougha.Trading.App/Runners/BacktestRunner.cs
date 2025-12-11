using Ougha.Trading.Backtesting;
using Ougha.Trading.Core.Models;
using Ougha.Trading.Data;
using Ougha.Trading.Data.Services;
using Ougha.Trading.Data.Downloaders; // Ex2ArchiveDownloader
using Ougha.Trading.Data.Streamers; // QuestDBTickTimeline
using Ougha.Trading.Analysis;
using Ougha.Trading.RL;
using Ougha.Trading.RL.Agents;
using Ougha.Trading.Risk;
using Ougha.Trading.Features;
using Ougha.Trading.Features.Indicators;

using Microsoft.Extensions.Configuration;
using Spectre.Console;
using System.Diagnostics;

namespace Ougha.Trading.App.Runners;

public static class BacktestRunner
{
    public static async Task RunAsync(string symbolArg, IConfiguration config)
    {
        // 0. Setup Console
        AnsiConsole.Write(new FigletText("Ougha Backtest").Color(Color.Cyan1));
        var symbols = symbolArg.Split(',', StringSplitOptions.RemoveEmptyEntries | StringSplitOptions.TrimEntries);
        
        // 1. Config Loading
        
        var initialBalance = config.GetValue<double>("Backtest:InitialBalance", 1000);
        var riskConfig = new RiskConfiguration(); // Load from config section in future
        config.GetSection("Risk").Bind(riskConfig);

        // ML Config
        var modelPath = config.GetValue<string>("ML:ModelPath", "data/models/unified_rl_agent.onnx");
        var useCuda = config.GetValue("ML:UseCuda", true);

        // Portfolio Config for all symbols
        var portfolioConfig = new PortfolioEnvironmentConfig(symbols);
        // Can bind additional settings from config if needed

        // 2. Services Init
        var httpClient = new HttpClient();
        var downloader = new Ex2ArchiveDownloader(httpClient);
        var dbLoader = new QuestDbDataLoader();
        
        // Initialize MT5 executor from config and inject into SymbolInfoService
        var mt5Executor = new Mt5Executor(config);
        var symbolService = new SymbolInfoService(mt5Executor);
        var dataService = new DataService(dbLoader, downloader, symbolService);

        // 3. Data Readiness
        var end = config.GetValue("Backtest:End", new DateTime(DateTime.UtcNow.Year, DateTime.UtcNow.Month, DateTime.UtcNow.Day));
        var start = config.GetValue("Backtest:Start", new DateTime(2025, 01, 01));
        
        await AnsiConsole.Status()
            .StartAsync("Checking Data Readiness...", async ctx => 
            {
                foreach (var sym in symbols)
                {
                    ctx.Status($"Syncing data for [bold]{sym}[/] ({start:yyyy-MM-dd} to {end:yyyy-MM-dd})...");
                    await dataService.EnsureDataReadyAsync(sym, start, end);
                }
            });

        // 4. Stream Data using QuestDBS1Timeline (heap-based merge for multi-symbol)
        var symbolInfos = new Dictionary<string, SymbolInfo>();
        foreach (var sym in symbols)
        {
            symbolInfos[sym] = symbolService.GetSymbolInfo(sym);
        }

        var candleTimeline = new QuestDBS1Timeline(dbLoader, symbols, start, end);
        
        // Get total count for progress display (quick query, doesn't load data)
        long totalCandles = 0;
        await AnsiConsole.Status()
            .StartAsync("[green]Counting candles...[/]", async ctx =>
            {
                totalCandles = await candleTimeline.GetCountAsync();
                ctx.Status($"[green]Found {totalCandles:N0} candles to stream[/]");
            });

        if (totalCandles == 0)
        {
            AnsiConsole.MarkupLine("[red]No data found![/]");
            return;
        }

        AnsiConsole.MarkupLine($"[grey]Streaming {totalCandles:N0} candles (heap-merged chronologically)[/]");
        
        // 5. Build Engine Components with STREAMING mode
        var portfolioManager = new PortfolioManager(riskConfig);
        
        // Use streaming constructor - candles loaded on-demand during backtest
        var executor = new BacktestExecutor(candleTimeline.StreamAsync(), symbolInfos, portfolioManager, initialBalance);
        
        var featureBuilder = new FeatureBuilder();
        
        // Reward Shape
        var rewardCalc = new RewardCalculator(new RewardConfig());

        // Load Agent (ONNX or Torch)
        IAgent? agent = null;
        bool isTraining = config.GetValue<bool>("Backtest:Train", false);

        if (isTraining)
        {
             AnsiConsole.MarkupLine($"[bold yellow]Initializing TorchAgent for Online Training...[/]");
             // Use same params as TrainingRunner for now, or load from config
             agent = new TorchAgent(
                batchSize: 64,
                gamma: 0.99f,
                epsilon: 0.5f, // Start with lower exploration for online adaptation? Or high?
                epsilonMin: 0.05f,
                epsilonDecay: 0.999f,
                bufferSize: 10000,
                useCuda: useCuda
             );
             agent.ResetOnlineLearning();
             AnsiConsole.MarkupLine("[green]TorchAgent ready for online learning (state reset)![/]");
        }
        else if (File.Exists(modelPath))
        {
            try
            {
                AnsiConsole.MarkupLine($"[cyan]Loading ONNX model: {modelPath}[/]");
                agent = new OnnxAgent(modelPath, useCuda);
                AnsiConsole.MarkupLine("[green]ONNX agent loaded successfully![/]");
            }
            catch (Exception ex)
            {
                AnsiConsole.MarkupLine($"[yellow]Warning: Failed to load ONNX model: {ex.Message}[/]");
                AnsiConsole.MarkupLine("[yellow]Falling back to HOLD action[/]");
            }
        }
        else
        {
            AnsiConsole.MarkupLine($"[yellow]ONNX model not found at: {modelPath}[/]");
            AnsiConsole.MarkupLine("[yellow]Falling back to HOLD action[/]");
        }

        var env = new PortfolioTradingEnvironment(
            executor, featureBuilder, portfolioManager, rewardCalc, portfolioConfig
        );

        AnsiConsole.MarkupLine($"[cyan]Portfolio Environment: {symbols.Length} symbols, State Size: {env.StateSize}[/]");

        // 6. Run Simulation Loop
        AnsiConsole.MarkupLine("[bold green]Starting Simulation...[/]");
        
        // Track action distribution per symbol and performance
        var actionCounts = new Dictionary<string, int[]>();
        foreach (var s in symbols) actionCounts[s] = new int[8];
        var peakEquity = initialBalance;

        await AnsiConsole.Live(new Panel("Starting..."))
            .AutoClear(false)
            .StartAsync(async ctx => 
            {
                var state = await env.ResetAsync();
                
                // Preload historical candles from QuestDB materialized views
                // This replaces the old tick-based warmup - candles are already pre-built
                AnsiConsole.MarkupLine("[grey]Preloading historical candles from materialized views...[/]");
                await env.PreloadHistoricalCandlesAsync(dbLoader, start);
                AnsiConsole.MarkupLine("[grey]Historical candles loaded.[/]");
                
                // For online training, we need previous inputs to store transitions
                AgentInput[]? prevInputs = null;
                int[]? prevActions = null;
                
                bool done = false;
                int step = 0;
                var stopwatch = Stopwatch.StartNew();

                const int UI_UPDATE_INTERVAL = 1;

                int[] cachedActions = new int[symbols.Length];
                AgentInput[]? cachedInputs = null;

                while (!done)
                {
                    step++;

                    var (rewards, dones, m1CandleClosed) = await env.StepFastAsync(cachedActions);

                    // No warmup needed - we preloaded historical candles from QuestDB
                    bool shouldDecide = m1CandleClosed;

                    if (agent != null && shouldDecide)
                    {
                        cachedInputs = env.BuildAgentInputs();

                        for(int i=0; i<symbols.Length; i++)
                        {
                            cachedActions[i] = agent.Act(cachedInputs[i], training: isTraining);
                            actionCounts[symbols[i]][cachedActions[i]]++;
                        }
                    }

                    if (isTraining && shouldDecide && agent != null && prevInputs != null && cachedInputs != null)
                    {
                        bool episodeDone = dones.All(d => d);
                        var nextInputs = env.BuildAgentInputs();

                        for(int i=0; i<symbols.Length; i++)
                        {
                             agent.AddExperience(cachedInputs[i], cachedActions[i], rewards[i], nextInputs[i], episodeDone);
                        }

                        agent.Train();
                    }

                    if (shouldDecide)
                        prevInputs = cachedInputs;

                    done = dones.All(d => d);

                    if (step % UI_UPDATE_INTERVAL != 0 && !done) continue;

                        var eq = executor.GetEquity();
                        peakEquity = Math.Max(peakEquity, eq);
                        var balance = executor.GetBalance();
                        var currentTime = executor.CurrentTime;
                        var elapsed = stopwatch.Elapsed.TotalSeconds;
                        var ticksPerSec = step / Math.Max(elapsed, 0.001);
                        
                        // Calculate realtime metrics
                        var results = executor.GetResults();
                        var (wins, losses, grossProfit, grossLoss) = CalculateTradeMetrics(results.TradeLog);
                        var totalTrades = wins + losses;
                        var winRate = totalTrades > 0 ? (wins * 100.0 / totalTrades) : 0;
                        var profitFactor = grossLoss > 0 ? grossProfit / grossLoss : (grossProfit > 0 ? double.PositiveInfinity : 0);
                        var pfDisplay = double.IsInfinity(profitFactor) ? "∞" : $"{profitFactor:F2}";
                        var drawdownPct = peakEquity > 0 ? ((peakEquity - eq) / peakEquity * 100) : 0;
                        
                        // Create Stats Panel as a single row table (full width)
                        var statsTable = new Table().Border(TableBorder.None).HideHeaders().Expand();
                        statsTable.AddColumn(new TableColumn("Time"));
                        statsTable.AddColumn(new TableColumn("Balance"));
                        statsTable.AddColumn(new TableColumn("Equity"));
                        statsTable.AddColumn(new TableColumn("WinRate"));
                        statsTable.AddColumn(new TableColumn("PF"));
                        statsTable.AddColumn(new TableColumn("MaxDD"));
                        statsTable.AddColumn(new TableColumn("Speed"));
                        statsTable.AddColumn(new TableColumn("Trades"));
                        statsTable.AddColumn(new TableColumn("Actions"));

                        var wrColor = winRate >= 50 ? "green" : winRate >= 40 ? "yellow" : "red";
                        var pfColor = profitFactor >= 1.5 ? "green" : profitFactor >= 1.0 ? "yellow" : "red";
                        var ddColor = drawdownPct < 5 ? "green" : drawdownPct < 10 ? "yellow" : "red";

                        statsTable.AddRow(
                            $"[bold]Time:[/] [white]{currentTime:yyyy-MM-dd HH:mm:ss}[/]",
                            $"[bold]Bal:[/] [green]${balance:N2}[/]",
                            $"[bold]Eq:[/] [cyan]${eq:N2}[/]",
                            $"[bold]WR:[/] [{wrColor}]{winRate:F1}%[/]",
                            $"[bold]PF:[/] [{pfColor}]{pfDisplay}[/]",
                            $"[bold]DD:[/] [{ddColor}]{drawdownPct:F2}%[/]",
                            $"[bold]Spd:[/] [yellow]{ticksPerSec:N0} candles/s[/]",
                            $"[bold]Trd:[/] [white]{totalTrades} ({wins}W/{losses}L)[/]",
                            $"[bold]Act:[/] [magenta]{string.Join(", ", symbols.Select((s, i) => $"{s}:{GetActionName(cachedActions[i])}"))}[/]"
                        );

                        var statsPanel = new Panel(statsTable)
                            .Header("[bold]Backtest Stats[/]")
                            .Border(BoxBorder.Rounded)
                            .BorderColor(Color.Blue)
                            .Expand();
                        
                        // Create Positions Table
                        var positionsTable = CreatePositionsTable(executor.GetPositions(), currentTime, symbolInfos);
                        
                        // Create Action Distribution Table
                        var actionTable = new Table().Border(TableBorder.Simple);
                        actionTable.AddColumn("Action");
                        actionTable.AddColumn("Count");
                        actionTable.AddColumn("%");
                        var totalActionCounts = new int[8];
                        foreach (var counts in actionCounts.Values)
                            for (int i = 0; i < 8; i++)
                                totalActionCounts[i] += counts[i];

                        int totalActions = totalActionCounts.Sum();
                        for (int i = 0; i < 8; i++)
                        {
                            if (totalActionCounts[i] > 0)
                            {
                                var pct = totalActions > 0 ? (totalActionCounts[i] * 100.0 / totalActions) : 0;
                                actionTable.AddRow(GetActionName(i), $"{totalActionCounts[i]:N0}", $"{pct:F1}%");
                            }
                        }

                        // Create Exit Reason Table
                        var exitTable = new Table().Border(TableBorder.Simple);
                        exitTable.AddColumn("Exit");
                        exitTable.AddColumn("Count");
                        exitTable.AddColumn("%");
                        var exitCounts = results.TradeLog
                            .GroupBy(t => t.ExitReason)
                            .ToDictionary(g => g.Key, g => g.Count());
                        int totalExits = exitCounts.Values.Sum();
                        foreach (var (reason, count) in exitCounts.OrderByDescending(x => x.Value))
                        {
                            var pct = totalExits > 0 ? (count * 100.0 / totalExits) : 0;
                            var color = reason switch
                            {
                                ExitReason.TakeProfit => "green",
                                ExitReason.StopLoss => "red",
                                _ => "white"
                            };
                            exitTable.AddRow($"[{color}]{reason}[/]", $"{count}", $"{pct:F1}%");
                        }

                        // Build Layout Grid - stats panel on its own row (full width)
                        var layout = new Grid();
                        layout.AddColumn();
                        layout.AddRow(statsPanel);

                        var innerGrid = new Grid();
                        innerGrid.AddColumn(new GridColumn().Width(30));
                        innerGrid.AddColumn(new GridColumn().NoWrap());
                        innerGrid.AddRow(new Panel(actionTable).Header("[bold]Action Distribution[/]").Border(BoxBorder.Rounded), positionsTable);

                        var bottomGrid = new Grid();
                        bottomGrid.AddColumn();
                        bottomGrid.AddColumn();
                        bottomGrid.AddRow(new Panel(exitTable).Header("[bold]Exit Reasons[/]").Border(BoxBorder.Rounded), CreatePerformanceTable(results.TradeLog));

                        layout.AddRow(innerGrid);
                        layout.AddRow(bottomGrid);
                        
                        ctx.UpdateTarget(layout);
                    
                }
                stopwatch.Stop();
                AnsiConsole.MarkupLine($"[bold]Simulation Finished in {stopwatch.Elapsed.TotalSeconds:F1}s ({step:N0} steps)[/]");
            });

        // Dispose agent
        agent?.Dispose();

        // 7. Results
        var results = executor.GetResults();
        
        // Analyze using ResultsAnalyzer
        var analyzer = new ResultsAnalyzer();
        var analysis = analyzer.Analyze(results);

        var resultTable = new Table();
        resultTable.AddColumn("Metric");
        resultTable.AddColumn("Value");
        
        double pnl = results.FinalBalance - initialBalance;
        string pnlColor = pnl >= 0 ? "green" : "red";
        
        resultTable.AddRow("Final Balance", $"[green]{results.FinalBalance:F2}[/]");
        resultTable.AddRow("P&L", $"[{pnlColor}]${pnl:+0.00;-0.00;0.00}[/]");
        resultTable.AddRow("Total Return", $"{analysis.TotalReturn:F2}%");
        resultTable.AddRow("Max Drawdown", $"[red]{analysis.MaxDrawdown:F2}%[/]");
        resultTable.AddRow("Total Trades", $"{results.TotalTrades}");
        resultTable.AddRow("Win Rate", $"{analysis.WinRate:F1}%");
        resultTable.AddRow("Profit Factor", $"{analysis.ProfitFactor:F2}");
        resultTable.AddRow("Sharpe Ratio", $"{analysis.SharpeRatio:F2}");

        AnsiConsole.Write(resultTable);

        // Action Distribution
        AnsiConsole.MarkupLine("\n[bold]Action Distribution:[/]");
        var actionTable = new Table();
        actionTable.AddColumn("Action");
        actionTable.AddColumn("Count");
        actionTable.AddColumn("%");
        
        // Sum action counts across all symbols for final display
        var finalActionCounts = new int[8];
        foreach (var counts in actionCounts.Values)
            for (int a = 0; a < 8; a++)
                finalActionCounts[a] += counts[a];
        
        int totalActions = finalActionCounts.Sum();
        for (int i = 0; i < 8; i++)
        {
            var pct = totalActions > 0 ? (finalActionCounts[i] * 100.0 / totalActions) : 0;
            actionTable.AddRow(GetActionName(i), finalActionCounts[i].ToString("N0"), $"{pct:F1}%");
        }
        AnsiConsole.Write(actionTable);
    }

    private static string GetActionName(int action) => action switch
    {
        0 => "HOLD      ",
        1 => "BUY (Cons)",
        2 => "BUY  (Mod)",
        3 => "BUY  (Agg)",
        4 => "SELL(Cons)",
        5 => "SELL (Mod)",
        6 => "SELL (Agg)",
        7 => "CLOSE     ",
        _ => "UNKNOWN"
    };

    private static (int wins, int losses, double grossProfit, double grossLoss) CalculateTradeMetrics(List<TradeRecord> trades)
    {
        int wins = 0, losses = 0;
        double grossProfit = 0, grossLoss = 0;
        
        foreach (var trade in trades)
        {
            if (trade.Profit > 0)
            {
                wins++;
                grossProfit += trade.Profit;
            }
            else if (trade.Profit < 0)
            {
                losses++;
                grossLoss += Math.Abs(trade.Profit);
            }
        }
        
        return (wins, losses, grossProfit, grossLoss);
    }

    private static Panel CreatePositionsTable(IEnumerable<Position> positions, DateTime currentTime, Dictionary<string, SymbolInfo> symbolInfos)
    {
        var table = new Table()
            .Border(TableBorder.Simple)
            .AddColumn("Ticket")
            .AddColumn("Symbol")
            .AddColumn("Type")
            .AddColumn("Volume")
            .AddColumn("Entry")
            .AddColumn("Current")
            .AddColumn("P&L")
            .AddColumn("SL")
            .AddColumn("TP")
            .AddColumn("Time");
        
        var positionList = positions.ToList();
        
        if (positionList.Count == 0)
        {
            table.AddRow("—", "—", "—", "—", "—", "—", "—", "—", "—", "—");
        }
        else
        {
            foreach (var pos in positionList.OrderByDescending(p => p.OpenTime))
            {
                var priceDiff = pos.Type == TradeType.Buy
                    ? pos.CurrentPrice - pos.OpenPrice
                    : pos.OpenPrice - pos.CurrentPrice;

                double pnl;
                if (symbolInfos.TryGetValue(pos.Symbol, out var symInfo))
                {
                    pnl = (priceDiff / symInfo.Point) * symInfo.TickValue * pos.Volume;
                }
                else
                {
                    pnl = priceDiff * pos.Volume * 100000.0;
                }
                
                var pnlColor = pnl > 0 ? "green" : pnl < 0 ? "red" : "white";
                var typeColor = pos.Type == TradeType.Buy ? "green" : "red";
                
                // Calculate duration
                var duration = currentTime - pos.OpenTime;
                var durationStr = duration.TotalHours >= 1 
                    ? $"{(int)duration.TotalHours}h{duration.Minutes}m"
                    : $"{duration.Minutes}m";
                
                table.AddRow(
                    $"{pos.Ticket}",
                    pos.Symbol,
                    $"[{typeColor}]{pos.Type}[/]",
                    $"{pos.Volume:F2}",
                    $"{pos.OpenPrice:F5}",
                    $"{pos.CurrentPrice:F5}",
                    $"[{pnlColor}]${pnl:+0.00;-0.00;0.00}[/]",
                    pos.StopLoss > 0 ? $"{pos.StopLoss:F5}" : "—",
                    pos.TakeProfit > 0 ? $"{pos.TakeProfit:F5}" : "—",
                    durationStr
                );
            }
        }
        
        return new Panel(table)
            .Header($"[bold]Open Positions ({positionList.Count})[/]")
            .Border(BoxBorder.Rounded)
            .BorderColor(Color.Cyan1);
    }

    private static Panel CreatePerformanceTable(List<TradeRecord> trades)
    {
        var table = new Table()
            .Border(TableBorder.Simple)
            .AddColumn("Symbol")
            .AddColumn("Trades")
            .AddColumn("Win%")
            .AddColumn("Profit")
            .AddColumn("Avg");
        
        if (trades.Count == 0)
        {
            table.AddRow("—", "—", "—", "—", "—");
        }
        else
        {
            // Group by symbol
            var bySymbol = trades.GroupBy(t => t.Symbol);
            
            foreach (var group in bySymbol.OrderByDescending(g => g.Sum(t => t.Profit)))
            {
                var tradeList = group.ToList();
                var wins = tradeList.Count(t => t.Profit > 0);
                var total = tradeList.Count;
                var winRate = total > 0 ? (wins * 100.0 / total) : 0;
                var profit = tradeList.Sum(t => t.Profit);
                var avgProfit = total > 0 ? profit / total : 0;
                
                var profitColor = profit > 0 ? "green" : profit < 0 ? "red" : "white";
                var wrColor = winRate >= 50 ? "green" : winRate >= 40 ? "yellow" : "red";
                
                table.AddRow(
                    group.Key,
                    $"{total}",
                    $"[{wrColor}]{winRate:F1}%[/]",
                    $"[{profitColor}]${profit:N2}[/]",
                    $"${avgProfit:N2}"
                );
            }
        }
        
        return new Panel(table)
            .Header("[bold]Performance by Symbol[/]")
            .Border(BoxBorder.Rounded)
            .BorderColor(Color.Green);
    }
}


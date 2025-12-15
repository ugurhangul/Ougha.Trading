using Microsoft.Extensions.Configuration;
using Spectre.Console;
using Ougha.Trading.Analysis;
using Ougha.Trading.Core.Models;
using Ougha.Trading.Data;
using Ougha.Trading.Data.Downloaders;
using Ougha.Trading.Data.Services;
using Ougha.Trading.Data.Streamers;
using Ougha.Trading.RL;
using Ougha.Trading.RL.Agents;
using Ougha.Trading.RL.Training;
using Serilog;

namespace Ougha.Trading.App.Runners;

public static class TrainingRunner
{
    public static async Task RunAsync(string symbolArg, int? episodes, IConfiguration config)
    {
        Log.Information("=== Training session started ===");

        var symbols = symbolArg.Split(',', StringSplitOptions.RemoveEmptyEntries | StringSplitOptions.TrimEntries).ToList();

        var qdbSection = config.GetSection("QuestDB");
        var host = qdbSection["Host"] ?? "localhost";
        var port = qdbSection.GetValue("Port", 8812);
        var dbLoader = new QuestDbDataLoader(host, port);

        var trainWindow = config.GetValue("Training:WindowSize", 20);

        var endDate = config.GetValue("Training:End", new DateTime(DateTime.UtcNow.Year, DateTime.UtcNow.Month, DateTime.UtcNow.Day));
        var startDate = config.GetValue("Training:Start", new DateTime(2025, 01, 01));

        // Ensure tick data is ready for all symbols before training
        AnsiConsole.MarkupLine("[bold cyan]Checking tick data readiness...[/]");
        using var httpClient = new HttpClient();
        var downloader = new Ex2ArchiveDownloader(httpClient);
        var dataService = new DataService(dbLoader, downloader);
        
        var tickReport = await dataService.EnsureTicksReadyAsync(symbols, startDate, endDate);
        
        if (!tickReport.AllReady)
        {
            AnsiConsole.MarkupLine($"[yellow]Warning: {tickReport.MissingSymbols.Count} symbols have no data available[/]");
            foreach (var missing in tickReport.MissingSymbols)
            {
                AnsiConsole.MarkupLine($"  [yellow]• {missing}[/]");
            }
            
            // Filter out symbols with no data
            var originalCount = symbols.Count;
            symbols = tickReport.ReadySymbols;
            
            if (symbols.Count == 0)
            {
                AnsiConsole.MarkupLine("[red]No symbols have available data. Exiting.[/]");
                return;
            }
            
            AnsiConsole.MarkupLine($"[grey]Continuing with {symbols.Count}/{originalCount} symbols that have data[/]");
        }
        else
        {
            AnsiConsole.MarkupLine($"[green]All {symbols.Count} symbols have tick data ready ({tickReport.TotalCandleCount:N0} total candles)[/]");
        }

        AnsiConsole.MarkupLine("[bold cyan]Calculating optimal training budget...[/]");

        var budget = TrainingBudgetCalculator.CalculateTrainingBudget(
            startDate: startDate,
            endDate: endDate,
            numSymbols: symbols.Count,
            episodes: episodes > 0 ? episodes : null,
            batchSize: config.GetValue<int?>("Training:BatchSize", null),
            learningRate: config.GetValue<double?>("Training:LearningRate", null),
            chunkDays: config.GetValue<int?>("Training:ChunkDays", null),
            chunkPrefetchCount: config.GetValue<int?>("Training:ChunkPrefetchCount", null),
            chunkHistoryBufferDays: config.GetValue<int?>("Training:ChunkHistoryBufferDays", null)
        );

        var trainMaxSteps = budget.MaxSteps;
        var batchSize = budget.BatchSize;
        var bufferSize = budget.MemorySize;


        AnsiConsole.MarkupLine($"Training date range: {startDate:yyyy-MM-dd} to {endDate:yyyy-MM-dd}");

        var totalDays = (int)(endDate - startDate).TotalDays;
        var episodeDays = Math.Max(1, (int)Math.Ceiling(budget.DaysPerEpisode));
        var maxStartOffset = Math.Max(0, totalDays - episodeDays - 1);

        AnsiConsole.MarkupLine($"[grey]Episode length: ~{episodeDays} days, {maxStartOffset + 1} possible start positions[/]");

        var questTimeline = new QuestDbs1Timeline(dbLoader, symbols, startDate, endDate);
        var totalCandles = await questTimeline.GetCountAsync();

        if (totalCandles == 0)
        {
            AnsiConsole.MarkupLine("[red]No S1 data found. Exiting.[/]");
            return;
        }

        AnsiConsole.MarkupLine($"[grey]Found {totalCandles:N0} total S1 candles in date range[/]");

        var chunkConfig = new ChunkConfig(
            ChunkDays: budget.ChunkDays,
            PrefetchChunks: budget.ChunkPrefetchCount,
            HistoryBufferDays: budget.ChunkHistoryBufferDays,
            EpisodeDays: episodeDays
        );

        using var chunkProvider = new ChunkBasedDataProvider(
            dbLoader, symbols, startDate, endDate, chunkConfig);

        AnsiConsole.MarkupLine($"[bold cyan]Chunk-based loading enabled: {chunkProvider.TotalChunks} chunks of {budget.ChunkDays} days each[/]");
        AnsiConsole.MarkupLine($"[grey]Prefetch buffer: {budget.ChunkPrefetchCount} chunks, History buffer: {budget.ChunkHistoryBufferDays} days[/]");

        chunkProvider.StartPrefetching();

        var mt5Executor = new Mt5Executor(config);
        var symbolService = new SymbolInfoService(mt5Executor);

        // Pre-fetch all economic calendar events for the entire training period
        AnsiConsole.MarkupLine("[grey]Pre-fetching economic calendar events...[/]");
        var calendarService = new EconomicCalendarService();
        await calendarService.LoadEventsAsync(startDate, endDate);
        AnsiConsole.MarkupLine("[grey]Economic calendar events loaded.[/]");

        var symbolInfo = new Dictionary<string, SymbolInfo>();
        foreach (var sym in symbols)
        {
            symbolInfo[sym] = symbolService.GetSymbolInfo(sym);
        }

        var envConfig = new PortfolioEnvironmentConfig(
            Symbols: symbols.ToArray(),
            WindowSize: trainWindow,
            MaxSteps: trainMaxSteps
        );

        var envPoolConfig = new EnvironmentPoolConfig(
            PoolSize: config.GetValue("Training:EnvPoolSize", 3)
        );

        using var envPool = new EnvironmentPool(
            chunkProvider,
            symbolInfo,
            envConfig,
            budget.Episodes,
            envPoolConfig
        );

        envPool.StartPrefetching();
        AnsiConsole.MarkupLine($"[grey]Environment pool started with {envPoolConfig.PoolSize} prefetch slots[/]");

        using var agent = CreateAgent(config, budget, batchSize, bufferSize, envConfig.NewsFeatureSize);

        // Action logger for behavior analysis
        var actionLoggingEnabled = config.GetValue("Training:EnableActionLogging", true);
        using var actionLogger = new ActionLogger(
            logDirectory: Path.Combine(Environment.CurrentDirectory, "action_logs"),
            maxEntriesInMemory: 50000,
            enabled: actionLoggingEnabled);
        
        if (actionLoggingEnabled)
            AnsiConsole.MarkupLine("[grey]Action logging enabled - analyzing behavior patterns[/]");

        var earlyStop = new EarlyStopTracker(
            patience: budget.EarlyStopPatience,
            minEpisodes: budget.EarlyStopMinEpisodes);
        var modelDir = Path.Combine(Environment.CurrentDirectory, "models_dotnet");
        Directory.CreateDirectory(modelDir);

        AnsiConsole.MarkupLine($"[green]Environment & Agent Ready. Starting in-memory training loop.[/]");

        var stats = new TrainingStats
        {
            TotalEpisodes = budget.Episodes,
            MaxSteps = trainMaxSteps,
            InitialBalance = 10000,
            GpuAvailable = budget.Hardware.GpuAvailable,
            EarlyStopPatience = budget.EarlyStopPatience,
            EarlyStopMinEpisodes = budget.EarlyStopMinEpisodes,
            IsPpoAgent = agent is PpoAgent,
            TotalChunks = envPool.TotalChunks
        };

        foreach (var sym in symbols) stats.GetOrCreateSymbolStats(sym);

        try
        {
            await AnsiConsole.Live(TrainingDisplay.BuildDisplay(stats, budget))
                .AutoClear(false)
                .StartAsync(async ctx =>
                {
                    try
                    {
                        var preparedEnv = await envPool.GetNextEnvironmentAsync();
                        if (preparedEnv == null)
                        {
                            AnsiConsole.MarkupLine("[red]Failed to get initial environment from pool[/]");
                            return;
                        }

                        var env = preparedEnv.Env;

                        // Preload historical candles and economic calendar events
                        await env.PreloadHistoricalCandlesAsync(dbLoader, preparedEnv.EpisodeStart);

                        var lastChunkIndex = preparedEnv.ChunkIndex;
                        stats.CurrentChunk = lastChunkIndex;
                        stats.ChunkStartDate = preparedEnv.ChunkStartDate;
                        stats.ChunkEndDate = preparedEnv.ChunkEndDate;
                        stats.EpisodesPerChunk = preparedEnv.EpisodesPerChunk;
                        stats.EpisodeInChunk = preparedEnv.EpisodeInChunk;
                        stats.EpisodeStartDate = preparedEnv.EpisodeStart;
                        stats.EpisodeEndDate = preparedEnv.EpisodeEnd;

                        var actionTimer = new System.Diagnostics.Stopwatch();
                        var envTimer = new System.Diagnostics.Stopwatch();
                        var trainTimer = new System.Diagnostics.Stopwatch();
                        var bufferTimer = new System.Diagnostics.Stopwatch();
                        var symbolEarlyStops = new Dictionary<string, EarlyStopTracker>();
                        Task<float>? trainingTask = null;

                        for (var ep = 1; ep <= budget.Episodes; ep++)
                        {
                            // Log progress every 50 episodes (writes to file, not console)
                            if (ep % 50 == 0 || ep == 1)
                            {
                                Log.Information("Episode {Episode}/{Total} ({Percent:F1}%) - Chunk {Chunk}/{TotalChunks}",
                                    ep, budget.Episodes, (double)ep / budget.Episodes * 100,
                                    stats.CurrentChunk, envPool.TotalChunks);
                            }

                            stats.Episode = ep;
                            stats.EpisodeReward = 0;
                            stats.EpisodeLoss = 0;

                            if (ep > 1)
                            {
                                var nextEnv = await envPool.GetNextEnvironmentAsync();
                                if (nextEnv != null)
                                {
                                    env = nextEnv.Env;

                                    // Preload for new environment
                                    await env.PreloadHistoricalCandlesAsync(dbLoader, nextEnv.EpisodeStart);

                                    stats.EpisodeInChunk = nextEnv.EpisodeInChunk;
                                    stats.EpisodeStartDate = nextEnv.EpisodeStart;
                                    stats.EpisodeEndDate = nextEnv.EpisodeEnd;

                                    if (nextEnv.ChunkIndex != lastChunkIndex)
                                    {
                                        lastChunkIndex = nextEnv.ChunkIndex;
                                        stats.CurrentChunk = lastChunkIndex;
                                        stats.ChunkStartDate = nextEnv.ChunkStartDate;
                                        stats.ChunkEndDate = nextEnv.ChunkEndDate;
                                    }
                                }
                            }

                            await env.ResetAsync();
                            
                            // Signal episode boundary to PPO agent for sequence tracking
                            if (agent is PpoAgent ppoAgentEp)
                            {
                                ppoAgentEp.StartNewEpisode();
                            }
                            
                            var stateInputs = env.BuildAgentInputs();

                            var done = false;
                            double episodeReward = 0;
                            var step = 0;

                            var symbolRewards = new Dictionary<string, double>();
                            foreach (var sym in symbols) symbolRewards[sym] = 0;

                            // M1-based decision loop: decide once per minute, step S1 internally
                            while (!done && step < trainMaxSteps)
                            {
                                stats.CurrentStep = step;
                                stats.StartStepTimer();

                                // Get position state for action masking (prevents CLOSE when no position)
                                var currentPositions = symbols.Select(s => env.Executor.GetPosition(s) != null).ToArray();
                                
                                // Get action at M1 decision point (with action masking)
                                actionTimer.Restart();
                                var (actions, tpSlMults, logProbs) = agent.ActBatchWithTpSlAndLogProbs(stateInputs, training: true, hasPositions: currentPositions);
                                foreach (var a in actions) stats.RecordAction(a);
                                actionTimer.Stop();
                                stats.ActionTimeMs = actionTimer.Elapsed.TotalMilliseconds;

                                env.SetTpSlMultipliersBatch(tpSlMults);
                                
                                // Store entry predictions for accuracy tracking when trades open
                                var predictions = (agent as PpoAgent)?.LastPredictions;
                                if (predictions != null && predictions.Length == symbols.Count)
                                {
                                    for (var i = 0; i < actions.Length; i++)
                                    {
                                        // If opening a new trade (BUY=1 or SELL=2) and didn't have position
                                        if ((actions[i] == 1 || actions[i] == 2) && !currentPositions[i])
                                        {
                                            env.SetEntryPrediction(symbols[i], predictions[i]);
                                        }
                                    }
                                }

                                if (actions.Length > 0)
                                {
                                    stats.CurrentAction = GetActionName(actions[0]);
                                    stats.CurrentSymbol = symbols[0];
                                }

                                // Step until M1 closes (accumulates ~60 S1 steps internally)
                                envTimer.Restart();
                                var (nextStates, rewards, dones, stepsTaken) = await env.StepUntilM1CloseAsync(actions);
                                envTimer.Stop();
                                stats.EnvStepTimeMs = envTimer.Elapsed.TotalMilliseconds;
                                
                                // Update entropy stats BEFORE logging so logged values are current
                                stats.EntropyCoefficient = GetEntropyCoefficient(agent);
                                stats.PolicyEntropy = GetPolicyEntropy(agent);
                                
                                // Log actions for analysis (every 10 steps to reduce overhead)
                                if (step % 10 == 0 && actionLoggingEnabled)
                                {
                                    // Note: Use currentPositions (captured BEFORE action) to correctly log
                                    // whether agent had position when deciding the action
                                    var positions = symbols.Select(s => env.Executor.GetPosition(s)).ToArray();
                                    var unrealizedPnls = positions.Select(p => (float)(p?.UnrealizedPnlPercent ?? 0)).ToArray();
                                    var prices = symbols.Select(s => env.Executor.GetBid(s)).ToArray();
                                    
                                    actionLogger.LogActionBatch(
                                        episode: ep,
                                        step: step,
                                        timestamp: env.Executor.CurrentTime,
                                        symbols: symbols.ToArray(),
                                        actions: actions,
                                        rewards: rewards,
                                        hasPositions: currentPositions,  // Use PRE-action state for correct analysis
                                        unrealizedPnls: unrealizedPnls,
                                        prices: prices,
                                        entropyCoefficient: stats.EntropyCoefficient,
                                        policyEntropy: stats.PolicyEntropy);
                                }

                                // Update step counter with actual S1 steps taken
                                step += stepsTaken;
                                stats.TotalSteps += stepsTaken;

                                var episodeDone = dones.All(d => d);

                                bufferTimer.Restart();

                                var doneFlags = new bool[stateInputs.Length];
                                Array.Fill(doneFlags, episodeDone);

                                // Extract TP/SL multipliers from 2D array for training
                                var tpMults = new float[stateInputs.Length];
                                var slMults = new float[stateInputs.Length];
                                var hindsightSlMults = new float[stateInputs.Length];
                                var actualPriceChanges = new float[stateInputs.Length];
                                for (var i = 0; i < stateInputs.Length; i++)
                                {
                                    tpMults[i] = tpSlMults[i, 0];
                                    slMults[i] = tpSlMults[i, 1];
                                    
                                    // Get hindsight SL from environment (computed from MAE tracking)
                                    // Only populated for symbols where a trade was closed
                                    var symbol = symbols[i];
                                    hindsightSlMults[i] = env.GetHindsightSlMultiplier(symbol);
                                    actualPriceChanges[i] = env.GetActualPriceChange(symbol);
                                    
                                    // Record prediction accuracy when a trade closed
                                    if (actualPriceChanges[i] > -900f)  // Trade closed
                                    {
                                        var entryPred = env.GetEntryPrediction(symbol);
                                        stats.RecordPrediction(entryPred, actualPriceChanges[i]);
                                    }
                                    
                                    // Clear tracking after retrieving (ready for next trade)
                                    if (hindsightSlMults[i] > 0)
                                        env.ClearHindsightTracking(symbol);
                                }

                                // Add experience with accumulated M1 rewards, hindsight SL, position state, and actual price change
                                agent.AddExperienceBatchWithLogProbs(stateInputs, actions, rewards, nextStates, doneFlags, logProbs, tpMults, slMults, hindsightSlMults, currentPositions, actualPriceChanges);

                                for (var i = 0; i < stateInputs.Length; i++)
                                {
                                    episodeReward += rewards[i];

                                    if (i < symbols.Count)
                                        symbolRewards[symbols[i]] += rewards[i];
                                }

                                bufferTimer.Stop();
                                stats.BufferAddTimeMs = bufferTimer.Elapsed.TotalMilliseconds;

                                if (trainingTask is { IsCompleted: true })
                                {
                                    try
                                    {
                                        await trainingTask;
                                        stats.TrainCalls++;
                                    }
                                    catch (Exception ex)
                                    {
                                        Log.Error(ex, "[TrainingRunner] Training task error");
                                        stats.LastError = ex.Message;
                                    }
                                    finally
                                    {
                                        trainingTask = null;
                                    }
                                }

                                if (trainingTask == null && step % budget.TrainFreq == 0)
                                {
                                    var shouldTrain = agent switch
                                    {
                                        PpoAgent ppo => ppo.HasPendingRollout(),
                                        _ => GetBufferSize(agent) >= batchSize
                                    };

                                    if (shouldTrain)
                                    {
                                        trainTimer.Restart();

                                        // PPO requires synchronous training (on-policy algorithm)
                                        // Data must be collected by the current policy
                                        // PPO requires synchronous training (on-policy algorithm)
                                        var loss = agent.TrainMultipleBatches(budget.TrainBatches);
                                        trainTimer.Stop();
                                        stats.TrainTimeMs = trainTimer.Elapsed.TotalMilliseconds;
                                        stats.TrainCalls++;
                                    }
                                }

                                stats.EndStepTimer(stepsTaken);
                                stateInputs = nextStates;

                                done = episodeDone;


                                stats.EpisodeReward = episodeReward;
                                stats.Epsilon = GetEpsilon(agent);
                                stats.EntropyCoefficient = GetEntropyCoefficient(agent);
                                stats.PolicyEntropy = GetPolicyEntropy(agent);
                                stats.BufferSize = GetBufferSize(agent);
                                stats.Positions = env.Executor.GetPositions().Count();
                                stats.Equity = env.Executor.GetBalance();

                                ctx.UpdateTarget(TrainingDisplay.BuildDisplay(stats, budget));
                            }


                            if (trainingTask != null)
                            {
                                await trainingTask;
                                stats.TrainCalls++;
                                trainingTask = null;
                            }

                            // Train every episode: flush buffer and train on accumulated experiences
                            if (agent is PpoAgent ppoAgentTrain)
                            {
                                trainTimer.Restart();
                                var episodeLoss = await ppoAgentTrain.TrainEpisodeAsync();
                                trainTimer.Stop();
                                
                                if (episodeLoss > 0)
                                {
                                    stats.TrainTimeMs = trainTimer.Elapsed.TotalMilliseconds;
                                    stats.TrainCalls++;
                                    stats.EpisodeLoss = episodeLoss;
                                }
                            }

                            agent.SyncInferenceNetwork();

                            stats.EpisodeReward = episodeReward;
                            stats.AddReward(episodeReward);
                            stats.IncrementEpisode();

                            if (episodeReward > stats.BestReward)
                            {
                                stats.BestReward = episodeReward;
                            }

                            earlyStop.Update(episodeReward, ep);

                            var allResults = env.Executor.GetResults();
                            var episodeTrades = allResults.TradeLog;
                            var totalHoldingSeconds = episodeTrades.Sum(t => (t.CloseTime - t.OpenTime).TotalSeconds);
                            
                            // Log trade closes for detailed analytics
                            if (actionLoggingEnabled)
                            {
                                foreach (var trade in episodeTrades)
                                {
                                    var holdingTicks = (int)(trade.CloseTime - trade.OpenTime).TotalSeconds;
                                    var entryAction = trade.Type == TradeType.Buy ? 1 : 2;
                                    var closeReason = trade.ExitReason.ToString();
                                    
                                    actionLogger.LogTradeClose(
                                        episode: ep,
                                        step: stats.CurrentStep,
                                        symbol: trade.Symbol,
                                        profit: trade.Profit,
                                        holdingTicks: holdingTicks,
                                        entryAction: entryAction,
                                        entryPrice: trade.OpenPrice,
                                        closePrice: trade.ClosePrice,
                                        closeReason: closeReason);
                                }
                            }
                            
                            // Calculate episode max drawdown from equity curve
                            var episodeMaxDrawdown = Analysis.ResultsAnalyzer.Analyze(allResults).MaxDrawdown;

                            stats.TradesOpened += allResults.TotalTrades;
                            stats.TradesClosed += episodeTrades.Count;
                            stats.Wins += episodeTrades.Count(t => t.Profit > 0);
                            stats.Losses += episodeTrades.Count(t => t.Profit <= 0);
                            stats.TotalProfit += episodeTrades.Where(t => t.Profit > 0).Sum(t => t.Profit);
                            stats.TotalLoss += Math.Abs(episodeTrades.Where(t => t.Profit < 0).Sum(t => t.Profit));
                            stats.TotalHoldingTimeSeconds += totalHoldingSeconds;

                            foreach (var symbol in symbols)
                            {
                                var symStats = stats.GetOrCreateSymbolStats(symbol);
                                symStats.Episodes++;

                                double symbolReward = 0;
                                if (symbolRewards.TryGetValue(symbol, out var reward))
                                    symbolReward = reward;

                                symStats.TotalReward += symbolReward;

                                // Track reward trend
                                symStats.PrevEpisodeReward = symStats.LastEpisodeReward;
                                symStats.LastEpisodeReward = symbolReward;

                                // Cumulative stats for display
                                var symbolTrades = allResults.TradeLog.Where(t => t.Symbol == symbol).ToList();
                                var grossProfit = symbolTrades.Where(t => t.Profit > 0).Sum(t => t.Profit);
                                var grossLoss = Math.Abs(symbolTrades.Where(t => t.Profit < 0).Sum(t => t.Profit));
                                var episodeWins = symbolTrades.Count(t => t.Profit > 0);
                                var episodeLosses = symbolTrades.Count(t => t.Profit <= 0);
                                var buys = symbolTrades.Count(t => t.Type == TradeType.Buy);
                                var sells = symbolTrades.Count(t => t.Type == TradeType.Sell);
                                var symbolHoldingSeconds = symbolTrades.Sum(t => (t.CloseTime - t.OpenTime).TotalSeconds);

                                symStats.CumulativeTrades += symbolTrades.Count;
                                symStats.CumulativeWins += episodeWins;
                                symStats.CumulativeLosses += episodeLosses;
                                symStats.CumulativeProfit += grossProfit;
                                symStats.CumulativeLoss += grossLoss;
                                symStats.CumulativeBuys += buys;
                                symStats.CumulativeSells += sells;
                                symStats.TotalHoldingTimeSeconds += symbolHoldingSeconds;
                                
                                // Track min/max holding times per symbol and update rolling window
                                foreach (var trade in symbolTrades)
                                {
                                    var holdSeconds = (trade.CloseTime - trade.OpenTime).TotalSeconds;
                                    if (holdSeconds > 0)
                                    {
                                        if (holdSeconds < symStats.MinHoldingTimeSeconds)
                                            symStats.MinHoldingTimeSeconds = holdSeconds;
                                        if (holdSeconds > symStats.MaxHoldingTimeSeconds)
                                            symStats.MaxHoldingTimeSeconds = holdSeconds;
                                    }
                                    
                                    // Add to rolling window for recent performance tracking
                                    var isWin = trade.Profit > 0;
                                    var profit = trade.Profit > 0 ? trade.Profit : 0;
                                    var loss = trade.Profit < 0 ? Math.Abs(trade.Profit) : 0;
                                    symStats.AddRollingTrade(isWin, profit, loss, holdSeconds);
                                }
                                
                                // Track worst max drawdown per symbol
                                if (symbolTrades.Count > 0 && episodeMaxDrawdown > symStats.CumulativeMaxDrawdown)
                                    symStats.CumulativeMaxDrawdown = episodeMaxDrawdown;

                                if (symbolReward > symStats.BestReward)
                                {
                                    symStats.BestReward = symbolReward;
                                    symStats.BestProfitFactor = grossLoss > 0 ? grossProfit / grossLoss : grossProfit > 0 ? 999.0 : 0.0;
                                    symStats.BestWinRate = symbolTrades.Count > 0 ? (double)episodeWins / symbolTrades.Count * 100 : 0;
                                    symStats.BestTrades = symbolTrades.Count;
                                }

                                if (!symbolEarlyStops.TryGetValue(symbol, out var symbolEarlyStop))
                                {
                                    symbolEarlyStop = new EarlyStopTracker(
                                        patience: budget.EarlyStopPatience,
                                        minEpisodes: budget.EarlyStopMinEpisodes);
                                    symbolEarlyStops[symbol] = symbolEarlyStop;
                                }

                                symbolEarlyStop.Update(symbolReward, symStats.Episodes);
                                symStats.NoImprovementCount = symbolEarlyStop.NoImprovementCount;
                                symStats.EarlyStopped = symbolEarlyStop.ShouldStop;
                            }

                            agent.DecayEpsilon();
                            
                            // Check for entropy reset every 200 episodes (less frequent for supervised mode)
                            // Higher threshold since prediction-based actions are expected to be skewed
                            if (ep % 200 == 0 && agent is PpoAgent ppoAgent)
                            {
                                var actionCountsArr = stats.GetActionCountsAsArray();
                                if (ppoAgent.CheckAndResetEntropy(actionCountsArr, skewThreshold: 0.85f))  // High threshold for supervised
                                {
                                    Log.Information("[TrainingRunner] Entropy reset triggered at episode {Episode}", ep);
                                    stats.EntropyResetCount++;
                                    stats.ResetActionCounts();  // Reset counts after check
                                }
                            }

                            stats.Epsilon = GetEpsilon(agent);
                            stats.EntropyCoefficient = GetEntropyCoefficient(agent);
                            stats.PolicyEntropy = GetPolicyEntropy(agent);
                            ctx.UpdateTarget(TrainingDisplay.BuildDisplay(stats, budget));

                            // Disable checkpointing for now
                            // if (ep % budget.SaveFrequency == 0)
                            // {
                            //     var checkpointPath = Path.Combine(modelDir, $"checkpoint_ep{ep}.pt");
                            //     agent.Save(checkpointPath);
                            // }

                            // Early stop disabled - train for full episodes
                            // if (shouldStop)
                            // {
                            //     AnsiConsole.MarkupLine($"[yellow]Early stopping triggered at episode {ep}[/]");
                            //     break;
                            // }
                        }
                    }
                    catch (Exception ex)
                    {
                        Log.Error(ex, "[TrainingRunner] Inner training error");
                        throw; // Re-throw to outer handler
                    }
                });
        }
        catch (Exception ex)
        {
            Log.Fatal(ex, "[TrainingRunner] FATAL ERROR in training loop");
            AnsiConsole.MarkupLine($"[red]Training error: {Markup.Escape(ex.Message)}[/]");

            // Check if producer failed
            if (envPool.HasProducerFailed)
            {
                Log.Error(envPool.ProducerException, "[TrainingRunner] Producer also failed");
            }
        }
        finally
        {
            // Generate and display action analysis report
            if (actionLoggingEnabled)
            {
                var report = actionLogger.GenerateReport();
                AnsiConsole.MarkupLine("\n[bold cyan]Action Analysis Report:[/]");
                AnsiConsole.WriteLine(report.ToString());
                Log.Information("Action Analysis Report:\n{Report}", report.ToString());
                
                // Log potential flaws as warnings
                foreach (var flaw in report.PotentialFlaws)
                {
                    Log.Warning("[ActionAnalysis] {Flaw}", flaw);
                }
            }
            
            Log.Information("=== Training session ended ===");
            await Log.CloseAndFlushAsync();
        }

        var finalModelPath = Path.Combine(modelDir, $"model_{DateTime.UtcNow:yyyyMMdd_HHmmss}.pt");
        agent.Save(finalModelPath);
        AnsiConsole.MarkupLine($"[bold green]Model saved to: {Markup.Escape(finalModelPath)}[/]");

        var latestPath = Path.Combine(modelDir, "model_latest.pt");
        agent.Save(latestPath);

        AnsiConsole.MarkupLine("[bold green]Training Complete![/]");
    }

    private static float GetEpsilon(IAgent agent)
    {
        // PPO uses entropy instead of epsilon
        return 0f;
    }

    private static float GetEntropyCoefficient(IAgent agent)
    {
        if (agent is PpoAgent ppo)
        {
            return ppo.GetEntropyCoefficient();
        }

        return 0f;
    }
    
    private static float GetPolicyEntropy(IAgent agent)
    {
        if (agent is PpoAgent ppo)
        {
            return ppo.GetPolicyEntropy();
        }

        return 0f;
    }

    private static int GetBufferSize(IAgent agent)
    {
        if (agent is PpoAgent ppo)
        {
            return ppo.GetActiveBufferCount();
        }

        return 0;
    }

    private static string GetActionName(int action)
    {
        return action switch
        {
            0 => "HOLD",
            1 => "BUY",
            2 => "SELL",
            3 => "CLOSE",
            _ => $"ACT_{action}"
        };
    }

    private static IAgent CreateAgent(IConfiguration config, TrainingBudget budget, int batchSize, int bufferSize, int newsFeatureSize)
    {
        // Only PPO is supported now (DQN removed)
        // Calculate expected updates for LR scheduling
        var stepsPerRollout = 2048; // Reduced from 4096 for fresher samples
        var expectedRollouts = budget.Episodes * budget.MaxSteps / stepsPerRollout;
        
        // Debug mode uses ~16x smaller model for fast hyperparameter tuning
        var debugMode = config.GetValue("Training:DebugMode", false);

        var modeLabel = debugMode ? "[yellow]DEBUG MODE (small model)[/]" : "[green]FULL MODE (large model)[/]";
        AnsiConsole.MarkupLine($"[bold cyan]Using PPO Strategy[/] | {modeLabel} | rollout=2048, batch=256, LR schedule over {expectedRollouts} updates");
        
        return new PpoAgent(
            batchSize: 256, // Increased for RTX 3090
            rolloutHorizon: 8192, // Reduced from 4096 for fresher samples
            gamma: 0.95f,  // Reduced from 0.99 for day trading (shorter horizon)
            learningRate: 3e-4f,
            useCuda: budget.Hardware.GpuAvailable,
            newsFeatureSize: newsFeatureSize,
            debugMode: debugMode
        );
    }
}
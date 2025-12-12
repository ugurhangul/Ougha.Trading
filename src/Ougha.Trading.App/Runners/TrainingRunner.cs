using Microsoft.Extensions.Configuration;
using Spectre.Console;
using Ougha.Trading.Core.Models;
using Ougha.Trading.Data;
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

                                // Get action at M1 decision point
                                actionTimer.Restart();
                                var (actions, tpSlMults, logProbs) = agent.ActBatchWithTpSlAndLogProbs(stateInputs, training: true);
                                foreach (var a in actions) stats.RecordAction(a);
                                actionTimer.Stop();
                                stats.ActionTimeMs = actionTimer.Elapsed.TotalMilliseconds;

                                env.SetTpSlMultipliersBatch(tpSlMults);

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

                                // Update step counter with actual S1 steps taken
                                step += stepsTaken;
                                stats.TotalSteps += stepsTaken;

                                var episodeDone = dones.All(d => d);

                                bufferTimer.Restart();

                                var doneFlags = new bool[stateInputs.Length];
                                Array.Fill(doneFlags, episodeDone);

                                // Add experience with accumulated M1 rewards
                                agent.AddExperienceBatchWithLogProbs(stateInputs, actions, rewards, nextStates, doneFlags, logProbs);

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

                                        trainingTask = Task.Run(() =>
                                        {
                                            // ReSharper disable once AccessToDisposedClosure
                                            var loss = agent.TrainMultipleBatches(budget.TrainBatches);
                                            trainTimer.Stop();
                                            stats.TrainTimeMs = trainTimer.Elapsed.TotalMilliseconds;
                                            return loss;
                                        });
                                    }
                                }

                                stats.EndStepTimer(stepsTaken);
                                stateInputs = nextStates;

                                done = episodeDone;


                                stats.EpisodeReward = episodeReward;
                                stats.Epsilon = GetEpsilon(agent);
                                stats.Entropy = GetEntropy(agent);
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
                            stats.TradesOpened += allResults.TotalTrades;
                            stats.TradesClosed += episodeTrades.Count;
                            stats.Wins += episodeTrades.Count(t => t.Profit > 0);
                            stats.Losses += episodeTrades.Count(t => t.Profit <= 0);
                            stats.TotalProfit += episodeTrades.Where(t => t.Profit > 0).Sum(t => t.Profit);
                            stats.TotalLoss += Math.Abs(episodeTrades.Where(t => t.Profit < 0).Sum(t => t.Profit));

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

                                symStats.CumulativeTrades += symbolTrades.Count;
                                symStats.CumulativeWins += episodeWins;
                                symStats.CumulativeLosses += episodeLosses;
                                symStats.CumulativeProfit += grossProfit;
                                symStats.CumulativeLoss += grossLoss;
                                symStats.CumulativeBuys += buys;
                                symStats.CumulativeSells += sells;

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
                            
                            // Check for entropy reset every 100 episodes if PPO
                            if (ep % 100 == 0 && agent is PpoAgent ppoAgent)
                            {
                                var actionCountsArr = stats.GetActionCountsAsArray();
                                if (ppoAgent.CheckAndResetEntropy(actionCountsArr, skewThreshold: 0.70f))
                                {
                                    Log.Information("[TrainingRunner] Entropy reset triggered at episode {Episode}", ep);
                                    stats.EntropyResetCount++;
                                    stats.ResetActionCounts();  // Reset counts after check
                                }
                            }

                            stats.Epsilon = GetEpsilon(agent);
                            stats.Entropy = GetEntropy(agent);
                            ctx.UpdateTarget(TrainingDisplay.BuildDisplay(stats, budget));

                            if (ep % budget.SaveFrequency == 0)
                            {
                                var checkpointPath = Path.Combine(modelDir, $"checkpoint_ep{ep}.pt");
                                agent.Save(checkpointPath);
                            }

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
        if (agent is TorchAgent)
        {
            var field = typeof(TorchAgent).GetField("_epsilon", System.Reflection.BindingFlags.NonPublic | System.Reflection.BindingFlags.Instance);
            return (float)(field?.GetValue(agent) ?? 0f);
        }

        return 0f;
    }

    private static float GetEntropy(IAgent agent)
    {
        if (agent is PpoAgent ppo)
        {
            return ppo.GetEntropyCoef();
        }

        return 0f;
    }

    private static int GetBufferSize(IAgent agent)
    {
        if (agent is TorchAgent)
        {
            var field = typeof(TorchAgent).GetField("_buffer", System.Reflection.BindingFlags.NonPublic | System.Reflection.BindingFlags.Instance);
            var buffer = field?.GetValue(agent);
            if (buffer == null) return 0;

            var countProp = buffer.GetType().GetProperty("Count");
            return (int)(countProp?.GetValue(buffer) ?? 0);
        }
        else if (agent is PpoAgent ppo)
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
            _ => $"ACT_{action}"
        };
    }

    private static IAgent CreateAgent(IConfiguration config, TrainingBudget budget, int batchSize, int bufferSize, int newsFeatureSize)
    {
        var strategy = config.GetValue<string>("Training:Strategy", "DQN");

        if (strategy.Equals("PPO", StringComparison.OrdinalIgnoreCase))
        {
            // Calculate expected updates for LR scheduling
            var stepsPerRollout = 4096; // Increased rollout horizon
            var expectedRollouts = budget.Episodes * budget.MaxSteps / stepsPerRollout;

            AnsiConsole.MarkupLine($"[bold cyan]Using PPO Strategy (rollout=4096, batch=256, LR schedule over {expectedRollouts} updates)[/]");
            return new PpoAgent(
                batchSize: 256, // Increased for RTX 3090
                rolloutHorizon: 4096, // Increased from 2048
                gamma: 0.99f,
                learningRate: 3e-4f,
                useCuda: budget.Hardware.GpuAvailable,
                newsFeatureSize: newsFeatureSize
            );
        }

        AnsiConsole.MarkupLine("[bold cyan]Using DQN Strategy[/]");
        return new TorchAgent(
            batchSize: batchSize,
            gamma: 0.99f,
            epsilon: 1.0f,
            epsilonMin: 0.01f,
            epsilonDecay: (float)budget.EpsilonDecay,
            bufferSize: bufferSize,
            useCuda: budget.Hardware.GpuAvailable,
            newsFeatureSize: newsFeatureSize
        );
    }
}
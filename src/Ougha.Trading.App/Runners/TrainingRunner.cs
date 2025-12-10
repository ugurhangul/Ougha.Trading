using System;
using System.IO;
using System.Collections.Generic;
using System.Linq;
using System.Threading.Tasks;
using Microsoft.Extensions.Configuration;
using Spectre.Console;
using Ougha.Trading.Core.Models;
using Ougha.Trading.Data;
using Ougha.Trading.Data.Streamers;
using Ougha.Trading.Features;
using Ougha.Trading.Risk;
using Ougha.Trading.RL;
using Ougha.Trading.RL.Agents;
using Ougha.Trading.RL.Training;
using Ougha.Trading.Backtesting;

namespace Ougha.Trading.App.Runners;

public class EarlyStopTracker
{
    private readonly int _patience;
    private readonly int _minEpisodes;
    private readonly int _window;
    
    private readonly Queue<double> _rewardHistory = new();
    private double _bestMovingAverage = double.MinValue;
    private int _noImprovementCount = 0;

    public EarlyStopTracker(int patience = 300, int minEpisodes = 500, int window = 50)
    {
        _patience = patience;
        _minEpisodes = minEpisodes;
        _window = window;
    }

    public bool ShouldStop => _noImprovementCount >= _patience;
    public double BestAverage => _bestMovingAverage;
    public int NoImprovementCount => _noImprovementCount;

    public bool Update(double reward, int episode)
    {
        _rewardHistory.Enqueue(reward);
        if (_rewardHistory.Count > _window)
            _rewardHistory.Dequeue();

        if (episode < _minEpisodes)
            return false;

        var movingAvg = _rewardHistory.Average();
        
        if (movingAvg > _bestMovingAverage)
        {
            _bestMovingAverage = movingAvg;
            _noImprovementCount = 0;
        }
        else
        {
            _noImprovementCount++;
        }

        return ShouldStop;
    }
}

public static class TrainingRunner
{
    public static async Task RunAsync(string symbolArg, int episodes, IConfiguration config)
    {
        var symbols = symbolArg.Split(',', StringSplitOptions.RemoveEmptyEntries | StringSplitOptions.TrimEntries).ToList();
        AnsiConsole.MarkupLine($"[bold yellow]Starting Native .NET RL Training for: {string.Join(", ", symbols)} ({episodes} episodes)[/]");

        // 1. Setup Data Loader
        var qdbSection = config.GetSection("QuestDB");
        var host = qdbSection["Host"] ?? "localhost";
        var port = qdbSection.GetValue<int>("Port", 8812);
        var dbLoader = new QuestDBDataLoader(host, port);
        
        var lookbackDays = config.GetValue<int>("Training:LookbackDays", 7);
        var trainWindow = config.GetValue<int>("Training:WindowSize", 20);
        var trainMaxSteps = config.GetValue<int>("Training:MaxSteps", 2000);
        var batchSize = config.GetValue<int>("Training:BatchSize", 64);
        var bufferSize = config.GetValue<int>("Training:BufferSize", 100000);
        
        var startDate = DateTime.UtcNow.Date.AddDays(-lookbackDays); 
        var endDate = DateTime.UtcNow;

        // 2. Load Data
        AnsiConsole.MarkupLine($"Loading tick data ({startDate:yyyy-MM-dd} to {endDate:yyyy-MM-dd})...");
        
        var tickTimeline = new QuestDBTickTimeline(dbLoader, symbols, startDate, endDate);
        await tickTimeline.LoadAllAsync();
        
        if (tickTimeline.Count == 0)
        {
            AnsiConsole.MarkupLine("[red]No data found. Exiting.[/]");
            return;
        }

        // Convert to TickTimeline
        var cachedTicks = tickTimeline.GetCachedTicks();
        var timeline = new TickTimeline(cachedTicks);

        // 3. Create Environment
        var symbolInfo = new Dictionary<string, SymbolInfo>();
        foreach (var sym in symbols)
        {
            // Default symbol info - ideally fetch from DB or Config
            symbolInfo[sym] = new SymbolInfo(sym, 0.00001, 100000, 1, 0.00001, "USD", "USD", 5);
        }

        var executor = new BacktestExecutor(timeline, symbolInfo);
        
        // Use PortfolioEnvironmentConfig
        var envConfig = new PortfolioEnvironmentConfig(
            Symbols: symbols.ToArray(),
            WindowSize: trainWindow,
            MaxSteps: trainMaxSteps
        );
        
        var env = new PortfolioTradingEnvironment(
            executor,
            new FeatureBuilder(),
            new PortfolioManager(),
            new RewardCalculator(),
            envConfig
        );

        // 4. Setup Agent
        // Use TorchAgent
        using var agent = new TorchAgent(
            batchSize: batchSize,
            gamma: 0.99f, 
            epsilon: 1.0f,
            epsilonMin: 0.05f,
            epsilonDecay: 0.998f, // Slow decay for long training
            bufferSize: bufferSize,
            useCuda: true // Try CUDA
        );
        
        var earlyStop = new EarlyStopTracker(patience: 300, minEpisodes: episodes / 10);
        var modelDir = Path.Combine(Environment.CurrentDirectory, "models_dotnet");
        Directory.CreateDirectory(modelDir);

        AnsiConsole.MarkupLine($"[green]Environment & Agent Ready.[/]");

        // 5. Training Loop
        var table = new Table().Border(TableBorder.Rounded);
        table.AddColumn("Episode");
        table.AddColumn("Total Reward");
        table.AddColumn("Loss");
        table.AddColumn("Epsilon");
        table.AddColumn("Trades");

        await AnsiConsole.Live(table)
            .AutoClear(false)
            .StartAsync(async ctx => 
            {
                var totalRewards = new List<double>();
                
                for (int ep = 1; ep <= episodes; ep++)
                {
                    // Reset
                    await env.ResetAsync();
                    
                    // Warmup
                    int warmupSteps = 1500;
                    for(int w=0; w<warmupSteps; w++)
                    {
                        var actions = new int[symbols.Count]; // all 0 (HOLD)
                        await env.StepTrainingAsync(actions);
                    }
                    
                    var stateInputs = env.BuildAgentInputs();
                    
                    bool done = false;
                    double episodeReward = 0;
                    float episodeLoss = 0;
                    int trainSteps = 0;
                    int step = 0;
                    
                    while (!done && step < trainMaxSteps)
                    {
                        step++;
                        
                        // Select actions
                        int[] actions = new int[stateInputs.Length];
                        for(int i=0; i<stateInputs.Length; i++)
                        {
                            actions[i] = agent.Act(stateInputs[i], training: true);
                        }
                        
                        // Step using specialized training method
                        var (nextStates, rewards, dones, infos) = await env.StepTrainingAsync(actions);
                        
                        bool episodeDone = dones.All(d => d);
                        
                        // Store & Train
                        for(int i=0; i<stateInputs.Length; i++)
                        {
                            agent.AddExperience(stateInputs[i], actions[i], rewards[i], nextStates[i], episodeDone);
                            episodeReward += rewards[i];
                        }
                        
                        // Train Step
                        float loss = agent.TrainStep();
                        if (loss > 0)
                        {
                            episodeLoss += loss;
                            trainSteps++;
                        }
                        
                        stateInputs = nextStates;
                        if (episodeDone) done = true;
                    }
                    
                    float avgLoss = trainSteps > 0 ? episodeLoss / trainSteps : 0;
                    
                    table.AddRow(
                        new Markup($"[blue]{ep}[/]"),
                        new Markup($"[green]{episodeReward:F2}[/]"),
                        new Markup($"[red]{avgLoss:F4}[/]"),
                        new Markup($"{GetEpsilon(agent):F4}"),
                        new Markup($"{executor.GetResults().TotalTrades}") 
                    );
                    ctx.Refresh();
                    
                    // Periodically save
                    if (ep % 20 == 0)
                    {
                         // TBD: agent.Save(...)
                    }
                }
            });
            
        AnsiConsole.MarkupLine("[bold green]Training Complete![/]");
    }
    
    private static float GetEpsilon(TorchAgent agent)
    {
        // Reflection hack to get epsilon for display
        var field = typeof(TorchAgent).GetField("_epsilon", System.Reflection.BindingFlags.NonPublic | System.Reflection.BindingFlags.Instance);
        return (float)(field?.GetValue(agent) ?? 0f);
    }
}

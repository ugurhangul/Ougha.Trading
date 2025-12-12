using System.CommandLine;
using Microsoft.Extensions.Configuration;
using Ougha.Trading.App.Runners;
using Serilog;

namespace Ougha.Trading.App;

abstract class Program
{
    private static async Task<int> Main(string[] args)
    {
        // Setup global Serilog logger immediately
        var logPath = Path.Combine(Environment.CurrentDirectory, "logs", "training-.log");
        Directory.CreateDirectory(Path.GetDirectoryName(logPath)!);
        
        Log.Logger = new LoggerConfiguration()
            .MinimumLevel.Debug()
            .WriteTo.File(logPath, 
                rollingInterval: RollingInterval.Day,
                flushToDiskInterval: TimeSpan.FromSeconds(1), // Flush frequently!
                outputTemplate: "{Timestamp:HH:mm:ss.fff} [{Level:u3}] {Message:lj}{NewLine}{Exception}")
            .CreateLogger();

        // Global exception handlers to catch ANY crash
        AppDomain.CurrentDomain.UnhandledException += (sender, e) =>
        {
            var ex = e.ExceptionObject as Exception;
            Log.Fatal(ex, "!!! UNHANDLED EXCEPTION - IsTerminating: {IsTerminating}", e.IsTerminating);
            Log.CloseAndFlush();
        };

        TaskScheduler.UnobservedTaskException += (sender, e) =>
        {
            Log.Error(e.Exception, "!!! UNOBSERVED TASK EXCEPTION");
            e.SetObserved(); // Prevent app crash
        };

        AppDomain.CurrentDomain.ProcessExit += (sender, e) =>
        {
            Log.Information("!!! PROCESS EXIT EVENT");
            Log.CloseAndFlush();
        };

        Log.Information("=== Application started ===");

        var configBuilder = new ConfigurationBuilder()
            .SetBasePath(AppContext.BaseDirectory)
            .AddJsonFile("appsettings.json", optional: true, reloadOnChange: true);

        IConfiguration configuration = configBuilder.Build();

        var rootCommand = new RootCommand("Ougha Trading Application");

        var modeOption = new Option<string>(
            name: "--mode",
            description: "Execution mode: backtest, train, live",
            getDefaultValue: () => "backtest");

        var symbolOption = new Option<string?>(
            name: "--symbol",
            description: "Symbol to trade/test (overrides appsettings.json)",
            getDefaultValue: () => null);

        var episodesOption = new Option<int?>(
            name: "--episodes",
            description: "Number of episodes for training (null = auto-calculate)");

        var trainOption = new Option<bool>(
            name: "--train",
            description: "Enable online training during backtest",
            getDefaultValue: () => true);

        rootCommand.AddOption(modeOption);
        rootCommand.AddOption(symbolOption);
        rootCommand.AddOption(episodesOption);
        rootCommand.AddOption(trainOption);

        rootCommand.SetHandler(async (mode, symbolArg, episodes, train) =>
        {
            configuration["Backtest:Train"] = train.ToString();

            string targetSymbols;
            if (!string.IsNullOrEmpty(symbolArg))
            {
                targetSymbols = symbolArg;
            }
            else
            {
                var configSymbols = configuration.GetSection("Trading:Symbols").Get<string[]>();

                if (configSymbols != null) targetSymbols = string.Join(",", configSymbols);
                else throw new ArgumentException("No symbols specified");
            }

            Console.WriteLine($"Starting Ougha Trading App in {mode.ToUpper()} mode for {targetSymbols}...");
            Log.Information("Mode: {Mode}, Symbols: {Symbols}", mode, targetSymbols);

            try
            {
                switch (mode.ToLower())
                {
                    case "backtest":
                        await BacktestRunner.RunAsync(targetSymbols, configuration);
                        break;
                    case "train":
                        await TrainingRunner.RunAsync(targetSymbols, episodes, configuration);
                        break;
                    case "live":
                        Console.WriteLine("Live mode not yet implemented in Console App. Use existing mechanism.");
                        break;
                    default:
                        Console.WriteLine($"Unknown mode: {mode}");
                        break;
                }
            }
            catch (Exception ex)
            {
                Log.Fatal(ex, "!!! TOP-LEVEL EXCEPTION in {Mode} mode", mode);
                throw;
            }
        }, modeOption, symbolOption, episodesOption, trainOption);

        var result = await rootCommand.InvokeAsync(args);
        
        Log.Information("=== Application exiting normally with code {ExitCode} ===", result);
        await Log.CloseAndFlushAsync();
        
        return result;
    }
}
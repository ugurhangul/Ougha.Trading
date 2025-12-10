using System.CommandLine;
using Microsoft.Extensions.Configuration;
using Ougha.Trading.App.Runners;

namespace Ougha.Trading.App;

class Program
{
    static async Task<int> Main(string[] args)
    {
        var configBuilder = new ConfigurationBuilder()
            .SetBasePath(AppContext.BaseDirectory)
            .AddJsonFile("appsettings.json", optional: true, reloadOnChange: true);

        IConfiguration configuration = configBuilder.Build();

        var rootCommand = new RootCommand("Ougha Trading Application");

        var modeOption = new Option<string>(
            name: "--mode",
            description: "Execution mode: backtest, train, live",
            getDefaultValue: () => "backtest");

        // Optional: Default null so we can check if passed
        var symbolOption = new Option<string?>(
            name: "--symbol",
            description: "Symbol to trade/test (overrides appsettings.json)",
            getDefaultValue: () => null);

        var episodesOption = new Option<int>(
            name: "--episodes",
            description: "Number of episodes for training",
            getDefaultValue: () => 1000);

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
            // Inject train flag into config
            configuration["Backtest:Train"] = train.ToString();

            // Determine Symbols
            string targetSymbols;
            if (!string.IsNullOrEmpty(symbolArg))
            {
                targetSymbols = symbolArg;
            }
            else
            {
                // Fallback to Config
                var configSymbols = configuration.GetSection("Trading:Symbols").Get<string[]>();

                if (configSymbols != null) targetSymbols = string.Join(",", configSymbols);
                else throw new ArgumentException("No symbols specified");
            }

            Console.WriteLine($"Starting Ougha Trading App in {mode.ToUpper()} mode for {targetSymbols}...");

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
        }, modeOption, symbolOption, episodesOption, trainOption);

        return await rootCommand.InvokeAsync(args);
    }
}
using System.Diagnostics;
using Ougha.Trading.Data;

namespace Ougha.Trading.Benchmark;

class Program
{
    static async Task Main(string[] args)
    {
        Console.WriteLine("Starting Benchmark...");
        
        // Defaults from QuestDBDataLoader
        var loader = new QuestDBDataLoader(); 
        var symbol = "BTCUSD"; // Common symbol, can be changed via args if needed
        if (args.Length > 0) symbol = args[0];
        
        // Use a wide range to force high load
        var end = DateTime.UtcNow;
        var start = end.AddDays(-30); // 30 days of ticks is A LOT of data usually

        Console.WriteLine($"Benchmarking LoadTicksAsync for {symbol} from {start} to {end}...");

        try 
        {
            var sw = Stopwatch.StartNew();
            // Test with parallelism = 1 (simulating old behavior roughly, though chunked)
            // Wait, I updated the code to default to 4. I should expose it or validte default.
            
            // To properly compare, I'd need the old code, but since I overwrote it, I will just test current performance.
            // If it runs fast/doesn't crash, that's a pass.
            
            Console.WriteLine("Loading with Parallelism = 4 (Default)...");
            var ticks = await loader.LoadTicksAsync(symbol, start, end, parallelism: 4);
            sw.Stop();
            
            Console.WriteLine($"Loaded {ticks.Count} ticks in {sw.ElapsedMilliseconds} ms.");
            
            if (ticks.Count > 0)
            {
                Console.WriteLine($"Sample Tick: {ticks[0]}");
            }
            else
            {
                Console.WriteLine("No ticks found. Ensure QuestDB has data for this symbol/range.");
            }
        }
        catch (Exception ex)
        {
            Console.WriteLine($"Error: {ex.Message}");
        }
    }
}

using Ougha.Trading.Core.Models;

namespace Ougha.Trading.Backtesting;

public class TickTimeline
{
    private readonly IReadOnlyList<(DateTime Time, string Symbol, Tick Tick)> _allTicks;

    public int Count => _allTicks.Count;

    public TickTimeline(IEnumerable<(DateTime Time, string Symbol, Tick Tick)> ticks)
    {
        // Simple sort by time
        _allTicks = ticks.OrderBy(x => x.Time).ToList();
    }

    public (DateTime Time, string Symbol, Tick Tick) this[int index] => _allTicks[index];
    
    public (DateTime Time, string Symbol, Tick Tick) GetAtIndex(int index)
    {
        return _allTicks[index];
    }
}

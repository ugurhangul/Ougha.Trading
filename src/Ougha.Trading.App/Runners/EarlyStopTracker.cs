namespace Ougha.Trading.App.Runners;

public class EarlyStopTracker(int patience = 300, int minEpisodes = 500, int window = 50)
{
    private readonly Queue<double> _rewardHistory = new();
    private double _bestMovingAverage = double.MinValue;

    public bool ShouldStop => NoImprovementCount >= patience;
    public int NoImprovementCount { get; private set; }

    public bool Update(double reward, int episode)
    {
        _rewardHistory.Enqueue(reward);
        if (_rewardHistory.Count > window)
            _rewardHistory.Dequeue();

        if (episode < minEpisodes)
            return false;

        var movingAvg = _rewardHistory.Average();

        if (movingAvg > _bestMovingAverage)
        {
            _bestMovingAverage = movingAvg;
            NoImprovementCount = 0;
        }
        else
        {
            NoImprovementCount++;
        }

        return ShouldStop;
    }
}

using Ougha.Trading.Core.Models;
using Ougha.Trading.RL.Agents;

namespace Ougha.Trading.RL;

public record VectorizedEnvConfig(
    int NEnvs = 8,
    int WindowSize = 20,
    double InitialBalance = 10000.0,
    double SlPips = 20.0,
    double TpPips = 40.0,
    int MaxEpisodeTicks = 50000,
    double MaxLossPercent = 50.0
);

public class VectorizedPortfolioTradingEnvironment : IDisposable
{
    public static readonly string[] SupportedTimeframes = { "M1", "M5", "M15", "H1", "H4" };

    private readonly VectorizedEnvConfig _config;
    private readonly string[] _symbols;
    private readonly int _nEnvs;
    private readonly int _nSymbols;

    private readonly Dictionary<string, float[,]> _tfFeatureMatrices;
    private readonly Dictionary<string, long[]> _tfTimestamps;
    private readonly int[] _tickToM1Candle;
    private readonly Dictionary<string, int[]> _tickToCandleMappings;

    private readonly int _nTicks;
    private readonly int _nFeatures;

    private readonly int[] _currentTicks;
    private readonly int[] _episodeStartTicks;
    private readonly int[] _positionTypes;
    private readonly double[] _openPrices;
    private readonly double[] _slPrices;
    private readonly double[] _tpPrices;
    private readonly int[] _positionOpenTicks;
    private readonly double[] _peakPnls;
    private readonly double[] _balances;
    private readonly double[] _episodeProfits;
    private readonly int[] _episodeTrades;
    private readonly bool[] _dones;

    private readonly double _pipValue;
    private readonly Random _random = new();

    private readonly float[] _rewardBuffer;
    private readonly bool[] _doneBuffer;

    public int NEnvs => _nEnvs;
    public int NSymbols => _nSymbols;
    public string[] Symbols => _symbols;

    public VectorizedPortfolioTradingEnvironment(
        Dictionary<string, float[,]> tfFeatureMatrices,
        Dictionary<string, long[]> tfTimestamps,
        long[] tickTimestamps,
        string[] symbols,
        VectorizedEnvConfig? config = null)
    {
        _config = config ?? new VectorizedEnvConfig();
        _symbols = symbols;
        _nEnvs = _config.NEnvs;
        _nSymbols = symbols.Length;

        _tfFeatureMatrices = tfFeatureMatrices;
        _tfTimestamps = tfTimestamps;

        var firstTf = tfFeatureMatrices.Values.First();
        _nFeatures = firstTf.GetLength(1);
        _nTicks = tickTimestamps.Length;

        _tickToCandleMappings = BuildTickToCandleMappings(tickTimestamps, tfTimestamps);
        _tickToM1Candle = _tickToCandleMappings.GetValueOrDefault("M1") ?? new int[_nTicks];

        int totalSlots = _nEnvs * _nSymbols;
        _currentTicks = new int[_nEnvs];
        _episodeStartTicks = new int[_nEnvs];
        _positionTypes = new int[totalSlots];
        _openPrices = new double[totalSlots];
        _slPrices = new double[totalSlots];
        _tpPrices = new double[totalSlots];
        _positionOpenTicks = new int[totalSlots];
        _peakPnls = new double[totalSlots];
        _balances = new double[_nEnvs];
        _episodeProfits = new double[_nEnvs];
        _episodeTrades = new int[_nEnvs];
        _dones = new bool[_nEnvs];

        _pipValue = symbols.Any(s => s.Contains("JPY")) ? 0.01 : 0.0001;

        _rewardBuffer = new float[totalSlots];
        _doneBuffer = new bool[_nEnvs];
    }

    private static Dictionary<string, int[]> BuildTickToCandleMappings(
        long[] tickTimestamps,
        Dictionary<string, long[]> tfTimestamps)
    {
        var mappings = new Dictionary<string, int[]>();

        foreach (var (tf, candleTimestamps) in tfTimestamps)
        {
            var mapping = new int[tickTimestamps.Length];
            int candleIdx = 0;

            for (int tickIdx = 0; tickIdx < tickTimestamps.Length; tickIdx++)
            {
                long tickTime = tickTimestamps[tickIdx];

                while (candleIdx < candleTimestamps.Length - 1 &&
                       candleTimestamps[candleIdx + 1] <= tickTime)
                {
                    candleIdx++;
                }

                mapping[tickIdx] = candleIdx;
            }

            mappings[tf] = mapping;
        }

        return mappings;
    }
    
    public AgentInput[,] Reset(int[]? envIndices = null)
    {
        envIndices ??= Enumerable.Range(0, _nEnvs).ToArray();
        
        int minStart = Math.Max(_config.WindowSize + 50, 1000);
        int maxStart = Math.Min(_nTicks - _config.MaxEpisodeTicks, 10000);
        if (maxStart <= minStart) maxStart = minStart + 1000;
        
        foreach (int envIdx in envIndices)
        {
            int startTick = _random.Next(minStart, maxStart);
            _currentTicks[envIdx] = startTick;
            _episodeStartTicks[envIdx] = startTick;
            _balances[envIdx] = _config.InitialBalance;
            _episodeProfits[envIdx] = 0;
            _episodeTrades[envIdx] = 0;
            _dones[envIdx] = false;
            
            for (int s = 0; s < _nSymbols; s++)
            {
                int slot = envIdx * _nSymbols + s;
                _positionTypes[slot] = 0;
                _openPrices[slot] = 0;
                _slPrices[slot] = 0;
                _tpPrices[slot] = 0;
                _positionOpenTicks[slot] = 0;
                _peakPnls[slot] = 0;
            }
        }
        
        return GetStates(envIndices);
    }
    
    private AgentInput[,] GetStates(int[] envIndices)
    {
        var states = new AgentInput[envIndices.Length, _nSymbols];
        
        for (int i = 0; i < envIndices.Length; i++)
        {
            int envIdx = envIndices[i];
            int tickIdx = _currentTicks[envIdx];
            
            for (int s = 0; s < _nSymbols; s++)
            {
                int slot = envIdx * _nSymbols + s;
                states[i, s] = BuildAgentInputForSlot(slot, tickIdx, _symbols[s]);
            }
        }
        
        return states;
    }
    
    private AgentInput BuildAgentInputForSlot(int slot, int tickIdx, string symbol)
    {
        var tfFeatures = new Dictionary<string, float[,]>();

        foreach (var tf in SupportedTimeframes)
        {
            var features = new float[_config.WindowSize, _nFeatures];

            if (_tfFeatureMatrices.TryGetValue(tf, out var tfMatrix) &&
                _tickToCandleMappings.TryGetValue(tf, out var tickToCandle))
            {
                int safeTickIdx = Math.Clamp(tickIdx, 0, tickToCandle.Length - 1);
                int candleIdx = tickToCandle[safeTickIdx];
                int nCandles = tfMatrix.GetLength(0);

                int startCandle = Math.Max(0, candleIdx - _config.WindowSize);
                int endCandle = Math.Min(candleIdx, nCandles);
                int available = endCandle - startCandle;

                if (available > 0)
                {
                    int destOffset = _config.WindowSize - available;
                    for (int w = 0; w < available; w++)
                    {
                        int srcRow = startCandle + w;
                        int destRow = destOffset + w;
                        for (int f = 0; f < _nFeatures; f++)
                        {
                            float val = tfMatrix[srcRow, f];
                            features[destRow, f] = float.IsNaN(val) || float.IsInfinity(val) ? 0f : val;
                        }
                    }
                }
            }

            tfFeatures[tf] = features;
        }

        var portfolioFeatures = BuildPortfolioFeatures(slot);
        var riskState = new float[9];
        if (_positionTypes[slot] != 0)
        {
            riskState[0] = 1.0f;
            riskState[1] = _positionTypes[slot] == 1 ? 1.0f : -1.0f;
        }

        return new AgentInput
        {
            TimeframeFeatures = tfFeatures,
            SymbolId = Math.Abs(symbol.GetHashCode()) % 256,
            TriggerContext = new float[5],
            ConfluenceFeatures = new float[10],
            PortfolioFeatures = portfolioFeatures,
            RiskState = riskState
        };
    }

    private float[] BuildPortfolioFeatures(int slot)
    {
        var features = new float[5];
        int envIdx = slot / _nSymbols;

        if (_positionTypes[slot] != 0 && _openPrices[slot] > 0)
        {
            features[0] = _positionTypes[slot];
            double unrealizedPnl = 0;
            features[1] = (float)(unrealizedPnl * 100);

            int holdingTicks = _currentTicks[envIdx] - _positionOpenTicks[slot];
            features[2] = (float)Math.Min(1.0, holdingTicks / 1000.0);

            double drawdown = Math.Max(0, _peakPnls[slot] - unrealizedPnl);
            features[3] = (float)(drawdown * 100);
        }

        features[4] = (float)((_balances[envIdx] / _config.InitialBalance) - 1.0);
        return features;
    }

    public (AgentInput[,] NextStates, float[,] Rewards, bool[] Dones) Step(int[,] actions)
    {
        Array.Clear(_rewardBuffer, 0, _rewardBuffer.Length);

        Parallel.For(0, _nEnvs, envIdx =>
        {
            if (_dones[envIdx]) return;

            for (int s = 0; s < _nSymbols; s++)
            {
                int slot = envIdx * _nSymbols + s;
                int action = actions[envIdx, s];
                ProcessAction(envIdx, slot, action);
            }

            _currentTicks[envIdx]++;

            int episodeLength = _currentTicks[envIdx] - _episodeStartTicks[envIdx];
            if (episodeLength >= _config.MaxEpisodeTicks || _currentTicks[envIdx] >= _nTicks - 1)
                _dones[envIdx] = true;

            if (_balances[envIdx] < _config.InitialBalance * (1 - _config.MaxLossPercent / 100.0))
                _dones[envIdx] = true;
        });

        var rewards = new float[_nEnvs, _nSymbols];
        for (int e = 0; e < _nEnvs; e++)
            for (int s = 0; s < _nSymbols; s++)
                rewards[e, s] = _rewardBuffer[e * _nSymbols + s];

        var allEnvs = Enumerable.Range(0, _nEnvs).ToArray();
        return (GetStates(allEnvs), rewards, (bool[])_dones.Clone());
    }

    private void ProcessAction(int envIdx, int slot, int action)
    {
        if (action == 0) return;

        bool isBuy = action == 1;
        bool isSell = action == 2;

        if (_positionTypes[slot] == 0)
        {
            if (isBuy)
                OpenPosition(envIdx, slot, 1);
            else if (isSell)
                OpenPosition(envIdx, slot, -1);
        }
        else if ((isBuy && _positionTypes[slot] == -1) || (isSell && _positionTypes[slot] == 1))
        {
            ClosePosition(envIdx, slot);
            if (isBuy)
                OpenPosition(envIdx, slot, 1);
            else
                OpenPosition(envIdx, slot, -1);
        }
    }

    private void OpenPosition(int envIdx, int slot, int posType)
    {
        _positionTypes[slot] = posType;
        _openPrices[slot] = 1.0;
        _positionOpenTicks[slot] = _currentTicks[envIdx];
        _peakPnls[slot] = 0;

        double slDistance = _config.SlPips * _pipValue;
        double tpDistance = _config.TpPips * _pipValue;

        if (posType == 1)
        {
            _slPrices[slot] = _openPrices[slot] - slDistance;
            _tpPrices[slot] = _openPrices[slot] + tpDistance;
        }
        else
        {
            _slPrices[slot] = _openPrices[slot] + slDistance;
            _tpPrices[slot] = _openPrices[slot] - tpDistance;
        }
    }

    private void ClosePosition(int envIdx, int slot)
    {
        double profit = _random.NextDouble() * 20 - 10;
        _balances[envIdx] += profit;
        _episodeProfits[envIdx] += profit;
        _episodeTrades[envIdx]++;

        float reward = (float)(profit / _config.InitialBalance * 100);
        _rewardBuffer[slot] = reward;

        _positionTypes[slot] = 0;
        _openPrices[slot] = 0;
        _slPrices[slot] = 0;
        _tpPrices[slot] = 0;
    }

    public void Dispose() { }
}


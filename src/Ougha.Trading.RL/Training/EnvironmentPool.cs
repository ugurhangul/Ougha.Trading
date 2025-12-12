using System.Threading.Channels;
using Ougha.Trading.Backtesting;
using Ougha.Trading.Core.Models;
using Ougha.Trading.Features;
using Ougha.Trading.Risk;

namespace Ougha.Trading.RL.Training;

public record PreparedEnvironment(
    PortfolioTradingEnvironment Env,
    DateTime EpisodeStart,
    DateTime EpisodeEnd,
    int ChunkIndex,
    DateTime ChunkStartDate,
    DateTime ChunkEndDate,
    int EpisodeInChunk,
    int EpisodesPerChunk
);

public record EnvironmentPoolConfig(
    int PoolSize = 3
);

public class EnvironmentPool : IDisposable
{
    private readonly ChunkBasedDataProvider _chunkProvider;
    private readonly Dictionary<string, SymbolInfo> _symbolInfo;
    private readonly PortfolioEnvironmentConfig _envConfig;
    private readonly string[] _symbols;
    private readonly int _episodesPerChunk;

    private readonly Channel<PreparedEnvironment> _envChannel;
    private readonly CancellationTokenSource _cts;
    private Task? _producerTask;

    private readonly Lock _lock = new();

    public int TotalChunks => _chunkProvider.TotalChunks;

    public EnvironmentPool(
        ChunkBasedDataProvider chunkProvider,
        Dictionary<string, SymbolInfo> symbolInfo,
        PortfolioEnvironmentConfig envConfig,
        int totalEpisodes,
        EnvironmentPoolConfig? poolConfig = null)
    {
        _chunkProvider = chunkProvider;
        _symbolInfo = symbolInfo;
        _envConfig = envConfig;
        var poolConfig1 = poolConfig ?? new EnvironmentPoolConfig();
        _symbols = envConfig.Symbols;
        _episodesPerChunk = Math.Max(1, totalEpisodes / Math.Max(1, _chunkProvider.TotalChunks));

        _envChannel = Channel.CreateBounded<PreparedEnvironment>(
            new BoundedChannelOptions(poolConfig1.PoolSize)
            {
                FullMode = BoundedChannelFullMode.Wait,
                SingleReader = true,
                SingleWriter = true
            });

        _cts = new CancellationTokenSource();
    }

    public void StartPrefetching()
    {
        _producerTask = Task.Run(() => ProduceEnvironmentsAsync(_cts.Token));
    }

    public async Task<PreparedEnvironment?> GetNextEnvironmentAsync()
    {
        try
        {
            if (await _envChannel.Reader.WaitToReadAsync(_cts.Token))
            {
                if (_envChannel.Reader.TryRead(out var preparedEnv))
                {
                    lock (_lock)
                    {
                    }
                    return preparedEnv;
                }
            }
        }
        catch (OperationCanceledException) { }
        return null;
    }

    private async Task ProduceEnvironmentsAsync(CancellationToken ct)
    {
        var producedInChunk = 0;

        try
        {
            var chunk = await _chunkProvider.GetNextChunkAsync();
            if (chunk == null) return;
            var chunkIndex = _chunkProvider.CurrentChunkIndex;
            var chunkStart = chunk.StartDate;
            var chunkEnd = chunk.EndDate;

            while (!ct.IsCancellationRequested)
            {
                var env = CreateEnvironmentFromChunk(out var episodeStart, out var episodeEnd);
                producedInChunk++;
                
                var prepared = new PreparedEnvironment(
                    env, 
                    episodeStart, 
                    episodeEnd,
                    chunkIndex,
                    chunkStart,
                    chunkEnd,
                    producedInChunk,
                    _episodesPerChunk
                );

                await _envChannel.Writer.WriteAsync(prepared, ct);

                if (producedInChunk >= _episodesPerChunk)
                {
                    var nextChunk = await _chunkProvider.GetNextChunkAsync();
                    if (nextChunk == null) break;
                    chunk = nextChunk;
                    chunkIndex = _chunkProvider.CurrentChunkIndex;
                    chunkStart = chunk.StartDate;
                    chunkEnd = chunk.EndDate;
                    producedInChunk = 0;
                }
            }
        }
        catch (OperationCanceledException) { }
        finally
        {
            _envChannel.Writer.Complete();
        }
    }

    private PortfolioTradingEnvironment CreateEnvironmentFromChunk(out DateTime episodeStart, out DateTime episodeEnd)
    {
        var (episodeData, historyBySymbol, start, end) = _chunkProvider.CreateEpisodeFromCurrentChunk();
        episodeStart = start;
        episodeEnd = end;

        if (episodeData.Count == 0)
            throw new InvalidOperationException("No episode data available from current chunk");

        var candleTimeline = new CandleTimeline(episodeData, alreadySorted: true);
        var episodeExecutor = new BacktestExecutor(candleTimeline, _symbolInfo, new PortfolioManager());

        var env = new PortfolioTradingEnvironment(
            episodeExecutor,
            new FeatureBuilder(),
            new RewardCalculator(),
            _envConfig
        );

        foreach (var sym in _symbols)
        {
            if (historyBySymbol.TryGetValue(sym, out var history) && history.Count > 0)
            {
                env.GetMtFAggregator(sym).PreloadCandles("M1", history);
            }
        }

        return env;
    }

    public void Dispose()
    {
        _cts.Cancel();
        _producerTask?.Wait(TimeSpan.FromSeconds(5));
        _cts.Dispose();
    }
}


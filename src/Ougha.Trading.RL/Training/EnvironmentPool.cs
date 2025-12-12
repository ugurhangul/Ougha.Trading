using System.Threading.Channels;
using Ougha.Trading.Backtesting;
using Ougha.Trading.Core.Models;
using Ougha.Trading.Features;
using Ougha.Trading.Risk;
using Serilog;

namespace Ougha.Trading.RL.Training;

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
    
    // Track producer failures for diagnostics
    private Exception? _producerException;
    private bool _producerCompleted;
    
    public bool HasProducerFailed => _producerException != null;
    public Exception? ProducerException => _producerException;
    public bool IsProducerCompleted => _producerCompleted;

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
            Log.Debug("[EnvironmentPool] Channel returned false from WaitToReadAsync (no more environments)");
        }
        catch (OperationCanceledException)
        {
            Log.Debug("[EnvironmentPool] GetNextEnvironmentAsync cancelled");
        }
        catch (Exception ex)
        {
            Log.Error(ex, "[EnvironmentPool] Error getting next environment");
        }
        return null;
    }

    private async Task ProduceEnvironmentsAsync(CancellationToken ct)
    {
        var producedInChunk = 0;

        try
        {
            var chunk = await _chunkProvider.GetNextChunkAsync();
            if (chunk == null)
            {
                Log.Warning("[EnvironmentPool] No initial chunk available, producer exiting");
                return;
            }
            var chunkIndex = _chunkProvider.CurrentChunkIndex;
            var chunkStart = chunk.StartDate;
            var chunkEnd = chunk.EndDate;
            
            Log.Information("[EnvironmentPool] Producer started, first chunk: {ChunkStart:yyyy-MM-dd} to {ChunkEnd:yyyy-MM-dd}", chunkStart, chunkEnd);

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
                    if (nextChunk == null)
                    {
                        Log.Information("[EnvironmentPool] No more chunks, produced {ChunkIndex} chunks total", chunkIndex);
                        break;
                    }
                    chunk = nextChunk;
                    chunkIndex = _chunkProvider.CurrentChunkIndex;
                    chunkStart = chunk.StartDate;
                    chunkEnd = chunk.EndDate;
                    producedInChunk = 0;
                    Log.Debug("[EnvironmentPool] Moved to chunk {ChunkIndex}: {ChunkStart:yyyy-MM-dd} to {ChunkEnd:yyyy-MM-dd}", chunkIndex, chunkStart, chunkEnd);
                }
            }
            
            Log.Information("[EnvironmentPool] Producer completed normally after {ChunkIndex} chunks", chunkIndex);
        }
        catch (OperationCanceledException)
        {
            Log.Information("[EnvironmentPool] Producer cancelled");
        }
        catch (Exception ex)
        {
            _producerException = ex;
            Log.Error(ex, "[EnvironmentPool] PRODUCER ERROR: {ErrorType}: {ErrorMessage}", ex.GetType().Name, ex.Message);
        }
        finally
        {
            _producerCompleted = true;
            _envChannel.Writer.Complete();
            Log.Debug("[EnvironmentPool] Producer finished, channel completed");
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


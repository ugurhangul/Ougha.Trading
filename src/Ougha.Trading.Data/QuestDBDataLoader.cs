using Npgsql;
using Dapper;
using Ougha.Trading.Core.Models;

namespace Ougha.Trading.Data;

public class QuestDBDataLoader
{
    private readonly string _connectionString;

    public QuestDBDataLoader(string host = "localhost", int port = 8812, string username = "admin", string password = "quest", string database = "qdb")
    {
        _connectionString = $"Host={host};Port={port};Database={database};Username={username};Password={password};ServerCompatibilityMode=NoTypeLoading;CommandTimeout=300;";
        DefaultTypeMap.MatchNamesWithUnderscores = true;
    }

    public async Task<List<Tick>> LoadTicksAsync(
        string symbol, DateTime startDate, DateTime endDate, int parallelism = 16)
    {
        var chunks = CreateTimeChunks(startDate, endDate, parallelism);
        var tasks = new List<Task<IEnumerable<Tick>>>();

        foreach (var (chunkStart, chunkEnd) in chunks)
        {
            tasks.Add(LoadTicksChunkAsync(symbol, chunkStart, chunkEnd));
        }

        await Task.WhenAll(tasks);

        return tasks.SelectMany(t => t.Result).OrderBy(t => t.Time).ToList();
    }

    public async IAsyncEnumerable<Tick> StreamTicksAsync(
        string symbol, DateTime startDate, DateTime endDate)
    {
        await using var conn = new NpgsqlConnection(_connectionString);
        await conn.OpenAsync();

        var sql = @"
            SELECT timestamp as Time, bid, ask, tick_type, has_no_data, volume
            FROM ticks
            WHERE symbol = @symbol
              AND timestamp >= @start
              AND timestamp < @end
              AND bid is not null
            ORDER BY timestamp";

        var startParam = NormalizeDateTime(startDate);
        var endParam = NormalizeDateTime(endDate);

        await foreach (var tick in conn.QueryUnbufferedAsync<Tick>(sql, new { symbol, start = startParam, end = endParam }))
        {
            yield return tick;
        }
    }

    private static DateTime NormalizeDateTime(DateTime dt) =>
        dt.Kind == DateTimeKind.Utc ? DateTime.SpecifyKind(dt, DateTimeKind.Unspecified) : dt;

    private async Task<IEnumerable<Tick>> LoadTicksChunkAsync(string symbol, DateTime startDate, DateTime endDate)
    {
        await using var conn = new NpgsqlConnection(_connectionString);
        await conn.OpenAsync();

        var sql = @"
            SELECT timestamp as Time, bid, ask, tick_type, has_no_data, volume
            FROM ticks
            WHERE symbol = @symbol
              AND timestamp >= @start
              AND timestamp < @end
              AND bid is not null
            ORDER BY timestamp";

        var startParam = NormalizeDateTime(startDate);
        var endParam = NormalizeDateTime(endDate);

        return await conn.QueryAsync<Tick>(sql, new { symbol, start = startParam, end = endParam });
    }

    public async Task<List<Candle>> LoadCandlesAsync(
        string symbol, string timeframe, DateTime startDate, DateTime endDate, int parallelism = 4)
    {
        var chunks = CreateTimeChunks(startDate, endDate, parallelism);
        var tasks = new List<Task<IEnumerable<Candle>>>();

        foreach (var (chunkStart, chunkEnd) in chunks)
        {
            tasks.Add(LoadCandlesChunkAsync(symbol, timeframe, chunkStart, chunkEnd));
        }

        await Task.WhenAll(tasks);

        return tasks.SelectMany(t => t.Result).OrderBy(c => c.Time).ToList();
    }

    private async Task<IEnumerable<Candle>> LoadCandlesChunkAsync(string symbol, string timeframe, DateTime startDate, DateTime endDate)
    {
        var table = timeframe.ToLower();

        await using var conn = new NpgsqlConnection(_connectionString);
        await conn.OpenAsync();

        var sql = $@"
            SELECT timestamp as Time, open, high, low, close, volume
            FROM {table}
            WHERE symbol = @symbol
              AND timestamp >= @start
              AND timestamp < @end
            ORDER BY timestamp";

        var startParam = NormalizeDateTime(startDate);
        var endParam = NormalizeDateTime(endDate);

        return await conn.QueryAsync<Candle>(sql, new { symbol, start = startParam, end = endParam });
    }

    private List<(DateTime Start, DateTime End)> CreateTimeChunks(DateTime start, DateTime end, int chunks)
    {
        var result = new List<(DateTime, DateTime)>();
        var totalDuration = end - start;
        if (totalDuration <= TimeSpan.Zero || chunks <= 1)
        {
            result.Add((start, end));
            return result;
        }

        var chunkDuration = new TimeSpan(totalDuration.Ticks / chunks);
        var currentStart = start;

        for (int i = 0; i < chunks; i++)
        {
            var currentEnd = (i == chunks - 1) ? end : currentStart.Add(chunkDuration);
            result.Add((currentStart, currentEnd));
            currentStart = currentEnd;
        }
        return result;
    }

    public async Task<HashSet<DateTime>> GetExistingDaysAsync(string symbol, DateTime start, DateTime end)
    {
        await using var conn = new NpgsqlConnection(_connectionString);
        await conn.OpenAsync();

        // Use SAMPLE BY 1d to get occupied days efficiently
        var sql = @"
            SELECT timestamp as ts
            FROM times_1D
            WHERE symbol = @symbol
              AND timestamp >= @start
              AND timestamp <= @end";

        var startParam = start.ToUniversalTime();
        var endParam = end.ToUniversalTime();

        var days = await conn.QueryAsync<DateTime>(sql, new { symbol, start = startParam, end = endParam });
        return days.Select(d => d.Date).ToHashSet();
    }

    public async Task InsertTicksAsync(string symbol, List<Tick> ticks)
    {
        if (ticks.Count == 0) return;

        await using var conn = new NpgsqlConnection(_connectionString);
        await conn.OpenAsync();

        // Batch insert - QuestDB prefers smaller batches via PG wire or ILP.
        // We use PG wire compliant batch inserts here for simplicity (no new dependencies).
        var batchSize = 1000;
        for (int i = 0; i < ticks.Count; i += batchSize)
        {
            var batch = ticks.Skip(i).Take(batchSize).ToList();
            var sb = new System.Text.StringBuilder();
            sb.Append("INSERT INTO ticks (timestamp, symbol, tick_type, bid, ask, volume) VALUES ");
            
            var parameters = new DynamicParameters();
            for (int j = 0; j < batch.Count; j++)
            {
                var t = batch[j];
                if (j > 0) sb.Append(",");
                sb.Append($"(@ts{j}, '{symbol}', 1, @bid{j}, @ask{j}, @vol{j})"); // TickType 1 hardcoded

                parameters.Add($"ts{j}", t.Time);
                parameters.Add($"bid{j}", t.Bid);
                parameters.Add($"ask{j}", t.Ask);
                parameters.Add($"vol{j}", t.Volume);
            }
            sb.Append(";");

            await conn.ExecuteAsync(sb.ToString(), parameters);
        }
    }

    public async Task MarkDayNoDataAsync(string symbol, DateTime date)
    {
        await using var conn = new NpgsqlConnection(_connectionString);
        await conn.OpenAsync();

        var sql = "INSERT INTO ticks (timestamp, symbol, tick_type, has_no_data) VALUES (@ts, @sym, 1, true);";
        
        // Ensure UTC midnight
        var ts = date.Date.ToUniversalTime();
        
        await conn.ExecuteAsync(sql, new { ts, sym = symbol });
    }

    public async Task<bool> IsDayMarkedNoDataAsync(string symbol, DateTime date)
    {
        await using var conn = new NpgsqlConnection(_connectionString);
        await conn.OpenAsync();

        var sql = @"
            SELECT 1 FROM ticks 
            WHERE symbol = @symbol 
              AND timestamp >= @start 
              AND timestamp < @end 
              AND has_no_data = true 
            LIMIT 1";

        var start = date.Date.ToUniversalTime();
        var end = start.AddDays(1);

        var result = await conn.ExecuteScalarAsync<int?>(sql, new { symbol, start, end });
        return result.HasValue;
    }

    /// <summary>
    /// Count total ticks for a symbol in date range.
    /// Matches Python count_ticks.
    /// </summary>
    public async Task<long> CountTicksAsync(string symbol, DateTime start, DateTime end)
    {
        await using var conn = new NpgsqlConnection(_connectionString);
        await conn.OpenAsync();

        var sql = @"
            SELECT COUNT(*) FROM ticks 
            WHERE symbol = @symbol 
              AND timestamp >= @start 
              AND timestamp < @end 
              AND (has_no_data IS NULL OR has_no_data = false)";

        var result = await conn.ExecuteScalarAsync<long>(sql, new { symbol, start, end });
        return result;
    }

    /// <summary>
    /// Check if any data exists for a specific day.
    /// </summary>
    public async Task<bool> HasDataForDayAsync(string symbol, DateTime date)
    {
        await using var conn = new NpgsqlConnection(_connectionString);
        await conn.OpenAsync();

        var start = date.Date.ToUniversalTime();
        var end = start.AddDays(1);

        var sql = @"
            SELECT 1 FROM ticks 
            WHERE symbol = @symbol 
              AND timestamp >= @start 
              AND timestamp < @end 
              AND (has_no_data IS NULL OR has_no_data = false)
            LIMIT 1";

        var result = await conn.ExecuteScalarAsync<int?>(sql, new { symbol, start, end });
        return result.HasValue;
    }

    /// <summary>
    /// Get first and last timestamp for a symbol.
    /// Matches Python get_date_range.
    /// </summary>
    public async Task<(DateTime? First, DateTime? Last)> GetDateRangeAsync(string symbol)
    {
        await using var conn = new NpgsqlConnection(_connectionString);
        await conn.OpenAsync();

        var sql = @"
            SELECT MIN(timestamp), MAX(timestamp) FROM ticks 
            WHERE symbol = @symbol 
              AND (has_no_data IS NULL OR has_no_data = false)";

        var result = await conn.QuerySingleAsync<(DateTime?, DateTime?)>(sql, new { symbol });
        return result;
    }

    /// <summary>
    /// Get tick statistics for a symbol.
    /// Matches Python get_stats.
    /// </summary>
    public async Task<TickStats?> GetStatsAsync(string symbol)
    {
        await using var conn = new NpgsqlConnection(_connectionString);
        await conn.OpenAsync();

        var sql = @"
            SELECT 
                COUNT(*) as TotalTicks,
                MIN(timestamp) as FirstTick,
                MAX(timestamp) as LastTick,
                COUNT(DISTINCT DATE(timestamp)) as TradingDays
            FROM ticks 
            WHERE symbol = @symbol 
              AND (has_no_data IS NULL OR has_no_data = false)";

        return await conn.QuerySingleOrDefaultAsync<TickStats>(sql, new { symbol });
    }
}

public record TickStats(
    long TotalTicks,
    DateTime FirstTick,
    DateTime LastTick,
    int TradingDays
);


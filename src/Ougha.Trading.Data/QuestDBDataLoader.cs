using Npgsql;
using Dapper;
using Ougha.Trading.Core.Models;

namespace Ougha.Trading.Data;

public class QuestDbDataLoader
{
    private readonly string _connectionString;

    public QuestDbDataLoader(string host = "localhost", int port = 8812, string username = "admin", string password = "quest", string database = "qdb")
    {
        _connectionString = $"Host={host};Port={port};Database={database};Username={username};Password={password};ServerCompatibilityMode=NoTypeLoading;CommandTimeout=3600;Timeout=120;Pooling=true;MaxPoolSize=100;";
        DefaultTypeMap.MatchNamesWithUnderscores = true;
    }

    private static DateTime NormalizeDateTime(DateTime dt) =>
        dt.Kind == DateTimeKind.Utc ? DateTime.SpecifyKind(dt, DateTimeKind.Unspecified) : dt;


    public async Task<IEnumerable<Candle>> LoadCandlesAsync(string symbol, string timeframe, DateTime startDate, DateTime endDate)
    {
        var table = timeframe.ToLower();

        await using var conn = new NpgsqlConnection(_connectionString);
        await conn.OpenAsync();

        var sql = $@"
            SELECT timestamp as Time, open, high, low, close, volume
            FROM {table}
            WHERE symbol = @symbol and timestamp Between @start and @end";

        var startParam = NormalizeDateTime(startDate);
        var endParam = NormalizeDateTime(endDate);

        return await conn.QueryAsync<Candle>(sql, new { symbol, start = startParam, end = endParam });
    }


    public async Task<HashSet<DateTime>> GetExistingDaysAsync(string symbol, DateTime start, DateTime end)
    {
        await using var conn = new NpgsqlConnection(_connectionString);
        await conn.OpenAsync();

        const string sql = """

                                       SELECT timestamp as ts
                                       FROM times_1D
                                       WHERE symbol = @symbol
                                         AND timestamp >= @start
                                         AND timestamp <= @end
                           """;

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

        const int batchSize = 1000;
        for (var i = 0; i < ticks.Count; i += batchSize)
        {
            var batch = ticks.Skip(i).Take(batchSize).ToList();
            var sb = new System.Text.StringBuilder();
            sb.Append("INSERT INTO ticks (timestamp, symbol, tick_type, bid, ask, volume) VALUES ");
            
            var parameters = new DynamicParameters();
            for (var j = 0; j < batch.Count; j++)
            {
                var t = batch[j];
                if (j > 0) sb.Append(",");
                sb.Append($"(@ts{j}, '{symbol}', 1, @bid{j}, @ask{j}, @vol{j})");

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
    /// Load the N most recent candles BEFORE a given timestamp.
    /// Used for preloading historical context from materialized views.
    /// </summary>
    public async Task<List<Candle>> LoadHistoricalCandlesAsync(
        string symbol, string timeframe, DateTime beforeTimestamp, int count)
    {
        var table = timeframe.ToLower();

        await using var conn = new NpgsqlConnection(_connectionString);
        await conn.OpenAsync();

        var sql = $@"
            SELECT timestamp as Time, open, high, low, close, volume
            FROM {table}
            WHERE symbol = @symbol
              AND timestamp < @before
            ORDER BY timestamp DESC
            LIMIT @count";

        var beforeParam = NormalizeDateTime(beforeTimestamp);

        var candles = await conn.QueryAsync<Candle>(sql, new { symbol, before = beforeParam, count });

        return candles.Reverse().ToList();
    }

    /// <summary>
    /// Stream candles from a materialized view (s1, m1, etc.) for a single symbol.
    /// </summary>
    public async IAsyncEnumerable<Candle> StreamCandlesAsync(
        string symbol, string timeframe, DateTime startDate, DateTime endDate)
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

        await foreach (var candle in conn.QueryUnbufferedAsync<Candle>(sql, new { symbol, start = startParam, end = endParam }))
        {
            yield return candle;
        }
    }

    /// <summary>
    /// Count total candles for a symbol in a date range from a materialized view.
    /// </summary>
    public async Task<long> CountCandlesAsync(string symbol, string timeframe, DateTime start, DateTime end)
    {
        var table = timeframe.ToLower();

        await using var conn = new NpgsqlConnection(_connectionString);
        await conn.OpenAsync();

        var sql = $@"
            SELECT COUNT(*) FROM {table} 
            WHERE symbol = @symbol 
              AND timestamp >= @start 
              AND timestamp < @end";

        var startParam = NormalizeDateTime(start);
        var endParam = NormalizeDateTime(end);

        var result = await conn.ExecuteScalarAsync<long>(sql, new { symbol, start = startParam, end = endParam });
        return result;
    }
}
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

        for (var i = 0; i < chunks; i++)
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

        var batchSize = 1000;
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
    /// Count total ticks for a symbol in the date range.
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
    /// Get the first and last timestamp for a symbol.
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
    
    // ========== Economic Calendar Events ==========
    
    /// <summary>
    /// Create the economic_events table if it doesn't exist.
    /// Call this once during app startup.
    /// </summary>
    public async Task EnsureEconomicEventsTableAsync()
    {
        await using var conn = new NpgsqlConnection(_connectionString);
        await conn.OpenAsync();
        
        var sql = @"
            CREATE TABLE IF NOT EXISTS economic_events (
                timestamp TIMESTAMP,
                currency SYMBOL,
                event_name STRING,
                impact INT,
                forecast DOUBLE,
                previous DOUBLE,
                actual DOUBLE
            ) TIMESTAMP(timestamp) PARTITION BY MONTH;";
        
        await conn.ExecuteAsync(sql);
    }
    
    /// <summary>
    /// Save economic events to QuestDB
    /// </summary>
    public async Task SaveEconomicEventsAsync(IEnumerable<Services.EconomicEvent> events)
    {
        var eventList = events.ToList();
        if (eventList.Count == 0) return;
        
        await using var conn = new NpgsqlConnection(_connectionString);
        await conn.OpenAsync();
        
        const int batchSize = 500;
        for (var i = 0; i < eventList.Count; i += batchSize)
        {
            var batch = eventList.Skip(i).Take(batchSize).ToList();
            var sb = new System.Text.StringBuilder();
            sb.Append("INSERT INTO economic_events (timestamp, currency, event_name, impact, forecast, previous, actual) VALUES ");
            
            var parameters = new DynamicParameters();
            for (var j = 0; j < batch.Count; j++)
            {
                var e = batch[j];
                if (j > 0) sb.Append(",");
                sb.Append($"(@ts{j}, @currency{j}, @name{j}, @impact{j}, @forecast{j}, @previous{j}, @actual{j})");
                
                parameters.Add($"ts{j}", e.Time);
                parameters.Add($"currency{j}", e.Currency);
                parameters.Add($"name{j}", e.EventName);
                parameters.Add($"impact{j}", (int)e.Impact);
                parameters.Add($"forecast{j}", e.Forecast);
                parameters.Add($"previous{j}", e.Previous);
                parameters.Add($"actual{j}", e.Actual);
            }
            sb.Append(';');
            
            await conn.ExecuteAsync(sb.ToString(), parameters);
        }
    }
    
    /// <summary>
    /// Load economic events from QuestDB
    /// </summary>
    public async Task<List<Services.EconomicEvent>> LoadEconomicEventsAsync(DateTime from, DateTime to)
    {
        await using var conn = new NpgsqlConnection(_connectionString);
        await conn.OpenAsync();
        
        var sql = @"
            SELECT timestamp as Time, currency, event_name, impact, forecast, previous, actual
            FROM economic_events
            WHERE timestamp >= @from AND timestamp <= @to
            ORDER BY timestamp";
        
        var fromParam = NormalizeDateTime(from);
        var toParam = NormalizeDateTime(to);
        
        var rows = await conn.QueryAsync<EconomicEventRow>(sql, new { from = fromParam, to = toParam });
        
        return rows.Select(r => new Services.EconomicEvent(
            r.Time,
            r.Currency,
            r.Event_Name,
            (Services.EventImpact)r.Impact,
            r.Forecast,
            r.Previous,
            r.Actual
        )).ToList();
    }
    
    /// <summary>
    /// Check if events exist for a date range
    /// </summary>
    public async Task<bool> HasEconomicEventsAsync(DateTime from, DateTime to)
    {
        await using var conn = new NpgsqlConnection(_connectionString);
        await conn.OpenAsync();
        
        var sql = @"
            SELECT 1 FROM economic_events
            WHERE timestamp >= @from AND timestamp <= @to
            LIMIT 1";
        
        var fromParam = NormalizeDateTime(from);
        var toParam = NormalizeDateTime(to);
        
        var result = await conn.ExecuteScalarAsync<int?>(sql, new { from = fromParam, to = toParam });
        return result.HasValue;
    }
}

internal record EconomicEventRow(
    DateTime Time,
    string Currency,
    string Event_Name,
    int Impact,
    double? Forecast,
    double? Previous,
    double? Actual
);

public record TickStats(
    long TotalTicks,
    DateTime FirstTick,
    DateTime LastTick,
    int TradingDays
);


-- QuestDB Schema Definitions
-- Based on the user provided schema for 'ticks' and 'h1' view.

-- 1 Minute Candles
CREATE MATERIALIZED VIEW 'm1' WITH BASE 'ticks' REFRESH IMMEDIATE AS (
    SELECT
        timestamp, symbol,
        first(bid) AS open,
        max(bid) as high,
        min(bid) as low,
        last(bid) AS close,
        count(*) AS volume
    FROM ticks
    SAMPLE BY 1m
) PARTITION BY MONTH;

-- 5 Minute Candles
CREATE MATERIALIZED VIEW 'm5' WITH BASE 'ticks' REFRESH IMMEDIATE AS (
    SELECT
        timestamp, symbol,
        first(bid) AS open,
        max(bid) as high,
        min(bid) as low,
        last(bid) AS close,
        count(*) AS volume
    FROM ticks
    SAMPLE BY 5m
) PARTITION BY MONTH;

-- 15 Minute Candles
CREATE MATERIALIZED VIEW 'm15' WITH BASE 'ticks' REFRESH IMMEDIATE AS (
    SELECT
        timestamp, symbol,
        first(bid) AS open,
        max(bid) as high,
        min(bid) as low,
        last(bid) AS close,
        count(*) AS volume
    FROM ticks
    SAMPLE BY 15m
) PARTITION BY MONTH;

-- 30 Minute Candles
CREATE MATERIALIZED VIEW 'm30' WITH BASE 'ticks' REFRESH IMMEDIATE AS (
    SELECT
        timestamp, symbol,
        first(bid) AS open,
        max(bid) as high,
        min(bid) as low,
        last(bid) AS close,
        count(*) AS volume
    FROM ticks
    SAMPLE BY 30m
) PARTITION BY MONTH;

-- 1 Hour Candles (Already defined, included for reference)
CREATE MATERIALIZED VIEW 'h1' WITH BASE 'ticks' REFRESH IMMEDIATE AS (
    SELECT
        timestamp, symbol,
        first(bid) AS open,
        max(bid) as high,
        min(bid) as low,
        last(bid) AS close,
        count(*) AS volume
    FROM ticks
    SAMPLE BY 1h
) PARTITION BY MONTH;

-- 4 Hour Candles
CREATE MATERIALIZED VIEW 'h4' WITH BASE 'ticks' REFRESH IMMEDIATE AS (
    SELECT
        timestamp, symbol,
        first(bid) AS open,
        max(bid) as high,
        min(bid) as low,
        last(bid) AS close,
        count(*) AS volume
    FROM ticks
    SAMPLE BY 4h
) PARTITION BY MONTH;

-- 1 Day Candles
CREATE MATERIALIZED VIEW 'd1' WITH BASE 'ticks' REFRESH IMMEDIATE AS (
    SELECT
        timestamp, symbol,
        first(bid) AS open,
        max(bid) as high,
        min(bid) as low,
        last(bid) AS close,
        count(*) AS volume
    FROM ticks
    SAMPLE BY 1d
) PARTITION BY MONTH;

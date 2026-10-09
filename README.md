# Ougha.Trading

A .NET 10 research platform for multi-asset portfolio trading with deep reinforcement learning. A PPO actor-critic agent (TorchSharp) is the sole decision maker: it observes multi-timeframe market state for a basket of symbols and decides position sizing across the portfolio. Tick data is stored in QuestDB and aggregated into candles with materialized views.

> **Status:** experimental research code. Backtesting and training work; live trading is not wired into the console app yet.

## Features

- **PPO agent** with an actor-critic model, async rollout buffer and a pool of parallel environments.
- **Portfolio environment** that trades many symbols at once (FX, metals, indices, stocks, crypto CFDs) with a configurable reward (equity, drawdown and holding penalties).
- **Multi-timeframe state** built from M1 to D1 candles plus technical indicators.
- **Chunked data loading** with prefetch, so multi-year tick histories fit in memory.
- **Backtester** with candle aggregation, slippage and trade records.
- **Data pipeline**: tick downloader, QuestDB loader, economic calendar, DXY index and correlation services.
- **MetaTrader 5 executor** through embedded Python (pythonnet).
- Spectre.Console training dashboard and Serilog file logs.

## Project layout

| Project | Purpose |
| --- | --- |
| `Ougha.Trading.App` | Console entry point (`backtest`, `train` modes) |
| `Ougha.Trading.RL` | Environment, state builders, reward, PPO agent, training infrastructure |
| `Ougha.Trading.Backtesting` | Backtest executor, candle builders, trade records |
| `Ougha.Trading.Data` | QuestDB loader, tick downloader, MT5 executor, market data services |
| `Ougha.Trading.Features` | Feature builder and technical indicators |
| `Ougha.Trading.Risk` | Position sizing and portfolio risk |
| `Ougha.Trading.Analysis` | Performance metrics and results analysis |
| `Ougha.Trading.Core` | Shared models and abstractions |
| `Ougha.Trading.Benchmark` | Benchmarks |

The original architecture write-up is in [`ml_only_portfolio_design.md`](ml_only_portfolio_design.md).

## Requirements

- [.NET 10 SDK](https://dotnet.microsoft.com/download)
- [QuestDB](https://questdb.io/) on `localhost:8812` (PostgreSQL wire protocol)
- An NVIDIA GPU with CUDA for training (the project references `TorchSharp-cuda-windows`)
- Python 3.12 with the `MetaTrader5` and `cloudscraper` packages, only for the MT5 executor and the economic calendar scraper. Point pythonnet at the interpreter:

  ```powershell
  $env:PYTHONNET_PYDLL = "C:\Path\To\Python312\python312.dll"
  ```

## Getting started

1. Start QuestDB and create the candle views from [`QuestDB_Schema.sql`](QuestDB_Schema.sql).
2. Adjust `src/Ougha.Trading.App/appsettings.json`: symbols, date ranges, training options and the QuestDB connection.
3. Build and run:

   ```powershell
   dotnet build -c Release
   dotnet run -c Release --project src/Ougha.Trading.App -- --mode train --symbol EURUSD,XAUUSD
   dotnet run -c Release --project src/Ougha.Trading.App -- --mode backtest
   ```

Missing tick data is downloaded and loaded into QuestDB before training starts. Trained models are saved as `.pt` files, and logs go to `logs/`.

### Command-line options

| Option | Default | Description |
| --- | --- | --- |
| `--mode` | `backtest` | `backtest` or `train` (`live` is a placeholder) |
| `--symbol` | from `Trading:Symbols` | Comma-separated symbol list |
| `--episodes` | auto | Training episodes; computed from the data and hardware budget when omitted |
| `--train` | `true` | Online training during a backtest |

## Configuration

`appsettings.json` sections:

- `Trading:Symbols`: default symbol universe.
- `QuestDB`: connection to the local QuestDB instance. The values shipped are QuestDB's factory defaults. Change them if your instance can be reached from other machines.
- `Backtest`: date range, initial balance, streaming mode.
- `Training`: algorithm, date range, window size, chunking and environment pool size.
- `ML`: model path and CUDA toggle.

Do not commit broker credentials. Keep machine-specific settings out of the tracked `appsettings.json`; `appsettings.Development.json` and `appsettings.Production.json` are git-ignored.

## Data sources

Tick history comes from a public broker tick archive, and economic events are scraped from a public calendar website. Check each provider's terms of use before you download or redistribute data. No market data is included in this repository.

## Disclaimer

This project is for research and education. It is not financial advice. Trading leveraged products carries a high risk of loss, and past or backtested performance does not predict future results. Use it at your own risk.

## License

[MIT](LICENSE)

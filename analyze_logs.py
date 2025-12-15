import json
from collections import defaultdict

# Read and parse JSONL file
trades = []
actions = []
filepath = r'C:\repos\ugurhangul\Ougha.Trading\src\Ougha.Trading.App\bin\Release\net10.0\win-x64\publish\action_logs\actions_20251215_141106.jsonl'

with open(filepath, 'r') as f:
    for line in f:
        try:
            record = json.loads(line.strip())
            actions.append(record)
            if record.get('IsTradeClosed'):
                trades.append(record)
        except:
            pass

print(f'Total Actions: {len(actions)}')
print(f'Total Trades Closed: {len(trades)}')
episodes = set(a["Episode"] for a in actions)
print(f'Total Episodes: {len(episodes)}')
print()

# Analyze by symbol
symbol_stats = defaultdict(lambda: {'trades': 0, 'wins': 0, 'losses': 0, 'profit': 0.0})
for t in trades:
    s = t.get('Symbol', 'Unknown')
    symbol_stats[s]['trades'] += 1
    symbol_stats[s]['profit'] += t.get('TradeProfit', 0)
    if t.get('TradeProfit', 0) > 0:
        symbol_stats[s]['wins'] += 1
    else:
        symbol_stats[s]['losses'] += 1

print('=== BY SYMBOL ===')
for sym, stats in sorted(symbol_stats.items()):
    wr = (stats['wins'] / stats['trades'] * 100) if stats['trades'] > 0 else 0
    print(f'{sym:10} Trades:{stats["trades"]:6}  Wins:{stats["wins"]:6}  Losses:{stats["losses"]:6}  WinRate:{wr:6.2f}%  Profit:{stats["profit"]:12.4f}')
print()

# Analyze by close reason
reason_stats = defaultdict(lambda: {'trades': 0, 'wins': 0, 'losses': 0, 'profit': 0.0})
for t in trades:
    r = t.get('CloseReason', 'Unknown')
    reason_stats[r]['trades'] += 1
    reason_stats[r]['profit'] += t.get('TradeProfit', 0)
    if t.get('TradeProfit', 0) > 0:
        reason_stats[r]['wins'] += 1
    else:
        reason_stats[r]['losses'] += 1

print('=== BY CLOSE REASON ===')
for reason, stats in sorted(reason_stats.items()):
    wr = (stats['wins'] / stats['trades'] * 100) if stats['trades'] > 0 else 0
    print(f'{reason:15} Trades:{stats["trades"]:6}  Wins:{stats["wins"]:6}  Losses:{stats["losses"]:6}  WinRate:{wr:6.2f}%  Profit:{stats["profit"]:12.4f}')
print()

# Analyze by entry action
entry_stats = defaultdict(lambda: {'trades': 0, 'wins': 0, 'losses': 0, 'profit': 0.0})
for t in trades:
    ea = t.get('EntryAction', -1)
    ea_name = {1: 'BUY', 2: 'SELL'}.get(ea, f'Action{ea}')
    entry_stats[ea_name]['trades'] += 1
    entry_stats[ea_name]['profit'] += t.get('TradeProfit', 0)
    if t.get('TradeProfit', 0) > 0:
        entry_stats[ea_name]['wins'] += 1
    else:
        entry_stats[ea_name]['losses'] += 1

print('=== BY ENTRY ACTION ===')
for ea, stats in sorted(entry_stats.items()):
    wr = (stats['wins'] / stats['trades'] * 100) if stats['trades'] > 0 else 0
    print(f'{ea:10} Trades:{stats["trades"]:6}  Wins:{stats["wins"]:6}  Losses:{stats["losses"]:6}  WinRate:{wr:6.2f}%  Profit:{stats["profit"]:12.4f}')
print()

# Analyze holding ticks
holding_ticks = [t.get('HoldingTicks', 0) for t in trades]
if holding_ticks:
    avg_hold = sum(holding_ticks) / len(holding_ticks)
    print(f'=== HOLDING STATS ===')
    print(f'Avg Holding Ticks: {avg_hold:.2f}')
    print(f'Min Holding Ticks: {min(holding_ticks)}')
    print(f'Max Holding Ticks: {max(holding_ticks)}')
    print()

# Analyze action distribution (non-trade actions)
action_counts = defaultdict(int)
for a in actions:
    if not a.get('IsTradeClosed'):
        an = a.get('ActionName', 'Unknown')
        action_counts[an] += 1

print('=== ACTION DISTRIBUTION (Non-Trade) ===')
total_actions = sum(action_counts.values())
for action, count in sorted(action_counts.items(), key=lambda x: -x[1]):
    pct = count / total_actions * 100 if total_actions > 0 else 0
    print(f'{action:10} Count:{count:8}  ({pct:5.2f}%)')
print()

# Analyze TakeProfit performance
tp_trades = [t for t in trades if t.get('CloseReason') == 'TakeProfit']
if tp_trades:
    tp_wins = sum(1 for t in tp_trades if t.get('TradeProfit', 0) > 0)
    tp_losses = len(tp_trades) - tp_wins
    print('=== TAKEPROFIT DETAILED ===')
    print(f'TakeProfit trades with positive P/L: {tp_wins}')
    print(f'TakeProfit trades with negative P/L: {tp_losses}')
    if tp_losses > 0:
        print('(Note: TakeProfit showing losses might indicate spread/slippage issues)')
    print()

# Analyze Manual close performance
manual_trades = [t for t in trades if t.get('CloseReason') == 'Manual']
if manual_trades:
    manual_wins = sum(1 for t in manual_trades if t.get('TradeProfit', 0) > 0)
    manual_losses = len(manual_trades) - manual_wins
    avg_manual_profit = sum(t.get('TradeProfit', 0) for t in manual_trades) / len(manual_trades)
    print('=== MANUAL CLOSE DETAILED ===')
    print(f'Manual close wins: {manual_wins}')
    print(f'Manual close losses: {manual_losses}')
    print(f'Avg profit per manual close: {avg_manual_profit:.6f}')
    print()

# Entropy analysis
entropies = [a.get('Entropy', 0) for a in actions if not a.get('IsTradeClosed') and a.get('Entropy', 0) > 0]
if entropies:
    print('=== ENTROPY STATS ===')
    print(f'Avg Entropy: {sum(entropies)/len(entropies):.6f}')
    print(f'Min Entropy: {min(entropies):.6f}')
    print(f'Max Entropy: {max(entropies):.6f}')
    print()

# Policy entropy analysis
policy_entropies = [a.get('PolicyEntropy', 0) for a in actions if not a.get('IsTradeClosed') and a.get('PolicyEntropy', 0) > 0]
if policy_entropies:
    print('=== POLICY ENTROPY STATS ===')
    print(f'Avg Policy Entropy: {sum(policy_entropies)/len(policy_entropies):.10f}')
    print(f'Min Policy Entropy: {min(policy_entropies):.10f}')
    print(f'Max Policy Entropy: {max(policy_entropies):.10f}')
    print()

# Episode progression analysis
episode_profits = defaultdict(float)
episode_trades = defaultdict(int)
for t in trades:
    ep = t.get('Episode', 0)
    episode_profits[ep] += t.get('TradeProfit', 0)
    episode_trades[ep] += 1

if episode_profits:
    print('=== EPISODE PROGRESSION (First 10 and Last 10) ===')
    sorted_eps = sorted(episode_profits.keys())
    for ep in sorted_eps[:10]:
        print(f'Episode {ep:4}: Trades={episode_trades[ep]:4}  Profit={episode_profits[ep]:10.4f}')
    if len(sorted_eps) > 20:
        print('...')
    for ep in sorted_eps[-10:]:
        if ep not in sorted_eps[:10]:
            print(f'Episode {ep:4}: Trades={episode_trades[ep]:4}  Profit={episode_profits[ep]:10.4f}')

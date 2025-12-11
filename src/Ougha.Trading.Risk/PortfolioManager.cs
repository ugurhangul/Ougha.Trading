using Ougha.Trading.Core.Models;

namespace Ougha.Trading.Risk;

public class PortfolioManager(RiskConfiguration? config = null)
{
    private readonly Dictionary<RiskLevel, RiskProfile> _profiles = new()
    {
        [RiskLevel.Conservative] = new RiskProfile(
            SlAtrMultiplier: 1.0, TpRrRatio: 3.0,
            TrailingTriggerRr: 1.5, MaxPositionPct: 0.5),
        [RiskLevel.Moderate] = new RiskProfile(
            SlAtrMultiplier: 1.5, TpRrRatio: 2.0,
            TrailingTriggerRr: 1.0, MaxPositionPct: 1.0),
        [RiskLevel.Aggressive] = new RiskProfile(
            SlAtrMultiplier: 2.0, TpRrRatio: 1.5,
            TrailingTriggerRr: 0.75, MaxPositionPct: 2.0)
    };

    private readonly RiskConfiguration _config = config ?? new RiskConfiguration();

    public bool CanOpenNewPosition(double totalRiskPct, int currentPositionCount)
    {
        if (currentPositionCount >= _config.MaxOpenPositions) return false;
        if (totalRiskPct >= _config.MaxPortfolioRiskPct) return false;
        return true;
    }

    public PositionSizing CalculatePositionSize(
        string symbol,
        TradeType type,
        RiskLevel riskLevel,
        double equity,
        double currentPrice,
        double atr,
        SymbolInfo symbolInfo) 
    {
        var profile = _profiles[riskLevel];

        var slDistance = atr * profile.SlAtrMultiplier;

        if (slDistance <= 0) slDistance = currentPrice * 0.001; 

        var sl = type == TradeType.Buy
            ? currentPrice - slDistance
            : currentPrice + slDistance;

        var tpDistance = slDistance * profile.TpRrRatio;
        var tp = type == TradeType.Buy
            ? currentPrice + tpDistance
            : currentPrice - tpDistance;

        var riskAmount = equity * (profile.MaxPositionPct / 100.0);

        var slPoints = slDistance / symbolInfo.Point;
        var tickValue = symbolInfo.TickValue; 
        
        if (tickValue == 0 || slPoints == 0) return new PositionSizing(_config.MinLotSize, sl, tp);

        var volume = riskAmount / (tickValue * slPoints);

        volume = Math.Round(volume, 2);
        volume = Math.Max(_config.MinLotSize, Math.Min(volume, _config.MaxLotSize)); 

        return new PositionSizing(volume, sl, tp);
    }

    public double? CalculateTrailingStop(Position position, double currentPrice, SymbolInfo info)
    {
        if (!_config.UseTrailingStop) return null;
        
        var profile = _profiles[position.RiskLevel];

        var bestPrice = position.BestPrice;
        bestPrice = position.Type == TradeType.Buy ? Math.Max(bestPrice, currentPrice) : Math.Min(bestPrice, currentPrice);

        var riskPoints = Math.Abs(position.OpenPrice - position.InitialStopLoss);
        if (riskPoints <= 0) return null;

        var profitPoints = position.Type == TradeType.Buy 
            ? (currentPrice - position.OpenPrice)
            : (position.OpenPrice - currentPrice);
            
        var currentRr = profitPoints / riskPoints;
        
        if (currentRr >= profile.TrailingTriggerRr)
        {
             var distPrice = _config.TrailingStopDistance * info.Point;
             
             if (position.Type == TradeType.Buy)
             {
                 var newSl = bestPrice - distPrice;
                 if (newSl > position.StopLoss) return newSl;
             }
             else
             {
                 var newSl = bestPrice + distPrice;
                 if (newSl < position.StopLoss || position.StopLoss == 0) return newSl;
             }
        }
        return null;
    }

    /// <summary>
    /// Validate if trade meets risk requirements.
    /// If risk exceeds maximum, automatically recalculates a smaller lot size.
    /// Matches Python validate_trade_risk.
    /// </summary>
    /// <returns>Tuple of (isValid, reason, adjustedLotSize)</returns>
    public (bool IsValid, string Reason, double AdjustedLot) ValidateTradeRisk(
        string symbol,
        double lotSize,
        double entryPrice,
        double stopLoss,
        double equity,
        SymbolInfo symbolInfo)
    {
        if (lotSize < _config.MinLotSize)
            return (false, $"Lot size {lotSize:F2} below minimum {_config.MinLotSize:F2}", 0);

        if (lotSize > _config.MaxLotSize)
            return (false, $"Lot size {lotSize:F2} above maximum {_config.MaxLotSize:F2}", 0);

        var slDistance = Math.Abs(entryPrice - stopLoss);
        if (slDistance <= 0)
            return (false, "Invalid stop loss distance", 0);

        var slPoints = slDistance / symbolInfo.Point;
        var tickValue = symbolInfo.TickValue;

        if (tickValue <= 0 || slPoints <= 0)
            return (false, "Invalid tick value or SL distance", 0);

        var riskAmount = slPoints * tickValue * lotSize;
        var riskPercent = (riskAmount / equity) * 100.0;

        var maxRisk = _config.MaxRiskPerTradePct * 1.5;
        
        if (riskPercent > maxRisk)
        {
            var targetRiskAmount = equity * (_config.MaxRiskPerTradePct / 100.0);
            var adjustedLot = targetRiskAmount / (slPoints * tickValue);

            adjustedLot = Math.Round(adjustedLot, 2);
            adjustedLot = Math.Max(_config.MinLotSize, Math.Min(adjustedLot, _config.MaxLotSize));

            if (adjustedLot < _config.MinLotSize)
                return (false, $"Even minimum lot exceeds acceptable risk ({riskPercent:F2}%)", 0);

            return (true, $"Lot adjusted from {lotSize:F2} to {adjustedLot:F2} (risk was {riskPercent:F2}%)", adjustedLot);
        }

        return (true, "", lotSize);
    }

    /// <summary>
    /// Check if we can open a new position in the specified direction.
    /// Matches Python can_open_direction.
    /// </summary>
    /// <returns>Tuple of (canOpen, reason)</returns>
    public (bool CanOpen, string Reason) CanOpenDirection(
        string symbol,
        TradeType type,
        IEnumerable<Position> currentPositions,
        string? strategyKey = null)
    {
        var positions = currentPositions.ToList();

        if (positions.Count >= _config.MaxOpenPositions)
            return (false, $"Maximum positions ({_config.MaxOpenPositions}) reached");

        if (positions.Any(pos => pos.Symbol == symbol && pos.Type == type))
        {
            return string.IsNullOrEmpty(strategyKey) ? (false, $"{type} position already exists for {symbol}") : (false, $"{type} position already exists for {symbol} [{strategyKey}]");
        }

        return (true, "");
    }

}


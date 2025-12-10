using Ougha.Trading.Core.Models;

namespace Ougha.Trading.Risk;

public class PortfolioManager
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

    private readonly RiskConfiguration _config;

    public PortfolioManager(RiskConfiguration? config = null)
    {
        _config = config ?? new RiskConfiguration();
    }

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

        // Calculate stop loss distance
        double slDistance = atr * profile.SlAtrMultiplier;
        
        // Safety check to avoid 0 SL distance
        if (slDistance <= 0) slDistance = currentPrice * 0.001; 

        double sl = type == TradeType.Buy
            ? currentPrice - slDistance
            : currentPrice + slDistance;

        // Calculate take profit
        double tpDistance = slDistance * profile.TpRrRatio;
        double tp = type == TradeType.Buy
            ? currentPrice + tpDistance
            : currentPrice - tpDistance;

        double riskAmount = equity * (profile.MaxPositionPct / 100.0);
        
        // Capping risk amount by global config?
        // if (riskAmount > equity * (_config.MaxRiskPerTradePct / 100.0)) ... 
        
        double slPoints = slDistance / symbolInfo.Point;
        double tickValue = symbolInfo.TickValue; 
        
        if (tickValue == 0 || slPoints == 0) return new PositionSizing(_config.MinLotSize, sl, tp);

        double volume = riskAmount / (tickValue * slPoints);

        // Normalize to lot size 
        volume = Math.Round(volume, 2);
        volume = Math.Max(_config.MinLotSize, Math.Min(volume, _config.MaxLotSize)); 

        return new PositionSizing(volume, sl, tp);
    }

    public double? CalculateTrailingStop(Position position, double currentPrice, SymbolInfo info)
    {
        if (!_config.UseTrailingStop) return null;
        
        var profile = _profiles[position.RiskLevel];
        
        // Update Best Price tracking logic should be outside? 
        // Or we calculate "potential" best price here?
        // We need BestPrice to be stored on Position. 
        // Assuming currentPrice IS the new candidate for BestPrice if better.
        
        double bestPrice = position.BestPrice;
        if (position.Type == TradeType.Buy)
             bestPrice = Math.Max(bestPrice, currentPrice);
        else
             bestPrice = Math.Min(bestPrice, currentPrice);

        // Calculate Risk R
        double riskPoints = Math.Abs(position.OpenPrice - position.InitialStopLoss);
        if (riskPoints <= 0) return null; // No risk defined

        double profitPoints = position.Type == TradeType.Buy 
            ? (currentPrice - position.OpenPrice)
            : (position.OpenPrice - currentPrice);
            
        double currentRr = profitPoints / riskPoints;
        
        if (currentRr >= profile.TrailingTriggerRr)
        {
             double distPrice = _config.TrailingStopDistance * info.Point;
             
             if (position.Type == TradeType.Buy)
             {
                 double newSl = bestPrice - distPrice;
                 if (newSl > position.StopLoss) return newSl;
             }
             else
             {
                 double newSl = bestPrice + distPrice;
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
        // Check lot size bounds
        if (lotSize < _config.MinLotSize)
            return (false, $"Lot size {lotSize:F2} below minimum {_config.MinLotSize:F2}", 0);

        if (lotSize > _config.MaxLotSize)
            return (false, $"Lot size {lotSize:F2} above maximum {_config.MaxLotSize:F2}", 0);

        // Check SL distance
        double slDistance = Math.Abs(entryPrice - stopLoss);
        if (slDistance <= 0)
            return (false, "Invalid stop loss distance", 0);

        // Calculate risk
        double slPoints = slDistance / symbolInfo.Point;
        double tickValue = symbolInfo.TickValue;

        if (tickValue <= 0 || slPoints <= 0)
            return (false, "Invalid tick value or SL distance", 0);

        double riskAmount = slPoints * tickValue * lotSize;
        double riskPercent = (riskAmount / equity) * 100.0;

        // Check if risk exceeds maximum (use 1.5x tolerance like Python)
        double maxRisk = _config.MaxRiskPerTradePct * 1.5;
        
        if (riskPercent > maxRisk)
        {
            // Auto-adjust lot size to target configured risk
            double targetRiskAmount = equity * (_config.MaxRiskPerTradePct / 100.0);
            double adjustedLot = targetRiskAmount / (slPoints * tickValue);
            
            // Round to 2 decimals and clamp
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

        // Check max positions
        if (positions.Count >= _config.MaxOpenPositions)
            return (false, $"Maximum positions ({_config.MaxOpenPositions}) reached");

        // Check for duplicate direction on same symbol-strategy pair
        foreach (var pos in positions)
        {
            if (pos.Symbol == symbol && pos.Type == type)
            {
                // If no strategy key, any same-direction position blocks
                if (string.IsNullOrEmpty(strategyKey))
                    return (false, $"{type} position already exists for {symbol}");

                // With strategy key, check comment for strategy match
                // Simplified: assume position comment contains strategy key
                // In full implementation, parse comment like Python CommentParser
                return (false, $"{type} position already exists for {symbol} [{strategyKey}]");
            }
        }

        return (true, "");
    }

    public RiskProfile GetProfile(RiskLevel level) => _profiles[level];
    public RiskConfiguration GetConfig() => _config;
}


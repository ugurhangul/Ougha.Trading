namespace Ougha.Trading.Risk;

/// <summary>
/// Provides currency conversion for tick values.
/// Matches Python currency_conversion_service.py.
/// </summary>
public class CurrencyConversionService
{
    private readonly Dictionary<string, double> _rates = new();
    private readonly string _accountCurrency;

    public CurrencyConversionService(string accountCurrency = "USD")
    {
        _accountCurrency = accountCurrency;
        
        // Default hardcoded rates (should be updated from broker/API)
        InitializeDefaultRates();
    }

    private void InitializeDefaultRates()
    {
        // Base rates to USD (approximate, should be updated dynamically)
        _rates["EURUSD"] = 1.08;
        _rates["GBPUSD"] = 1.27;
        _rates["USDJPY"] = 0.0067; // 1/150
        _rates["USDCHF"] = 1.12;
        _rates["AUDUSD"] = 0.66;
        _rates["NZDUSD"] = 0.61;
        _rates["USDCAD"] = 0.74;
        
        // Cross pairs derived
        _rates["EURGBP"] = _rates["EURUSD"] / _rates["GBPUSD"];
        _rates["EURJPY"] = _rates["EURUSD"] / _rates["USDJPY"];
    }

    /// <summary>
    /// Update conversion rate for a currency pair.
    /// </summary>
    public void UpdateRate(string pair, double rate)
    {
        _rates[pair] = rate;
    }

    /// <summary>
    /// Convert tick value from profit currency to account currency.
    /// Matches Python _get_tick_value_in_account_currency.
    /// </summary>
    public double ConvertTickValue(double tickValue, string profitCurrency)
    {
        if (profitCurrency == _accountCurrency)
            return tickValue;

        // Try direct conversion
        string directPair = $"{profitCurrency}{_accountCurrency}";
        if (_rates.TryGetValue(directPair, out var directRate))
            return tickValue * directRate;

        // Try inverse
        string inversePair = $"{_accountCurrency}{profitCurrency}";
        if (_rates.TryGetValue(inversePair, out var inverseRate) && inverseRate != 0)
            return tickValue / inverseRate;

        // Fallback: assume 1:1 (log warning in production)
        Console.WriteLine($"Warning: No conversion rate for {profitCurrency} -> {_accountCurrency}, using 1:1");
        return tickValue;
    }

    /// <summary>
    /// Get conversion rate between two currencies.
    /// </summary>
    public double GetRate(string fromCurrency, string toCurrency)
    {
        if (fromCurrency == toCurrency)
            return 1.0;

        string pair = $"{fromCurrency}{toCurrency}";
        if (_rates.TryGetValue(pair, out var rate))
            return rate;

        string inversePair = $"{toCurrency}{fromCurrency}";
        if (_rates.TryGetValue(inversePair, out var inverseRate) && inverseRate != 0)
            return 1.0 / inverseRate;

        return 1.0; // Fallback
    }

    /// <summary>
    /// Update rates from symbol info (called on each symbol load).
    /// </summary>
    public void UpdateFromSymbolInfo(string symbol, string baseCurrency, string profitCurrency, double currentPrice)
    {
        // If the symbol is XXX/YYY, currentPrice represents exchange rate
        string pair = $"{baseCurrency}{profitCurrency}";
        _rates[pair] = currentPrice;
    }
}

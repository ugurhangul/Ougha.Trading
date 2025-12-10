using System.Text;

namespace Ougha.Trading.RL;

/// <summary>
/// Maps symbol names to integer IDs matching Python's symbol_to_id function.
/// Uses CRC32 hash exactly like Python: zlib.crc32(s.encode()) % MAX_SYMBOLS
/// </summary>
public static class SymbolIdMapper
{
    /// <summary>
    /// Maximum number of symbols supported (matches Python's MAX_SYMBOLS = 128).
    /// </summary>
    public const int MaxSymbols = 128;

    /// <summary>
    /// CRC32 lookup table (standard IEEE 802.3 polynomial, same as zlib).
    /// </summary>
    private static readonly uint[] Crc32Table = GenerateCrc32Table();

    private static uint[] GenerateCrc32Table()
    {
        const uint polynomial = 0xEDB88320; // IEEE 802.3 polynomial (reversed)
        var table = new uint[256];
        for (uint i = 0; i < 256; i++)
        {
            uint crc = i;
            for (int j = 0; j < 8; j++)
            {
                if ((crc & 1) == 1)
                    crc = (crc >> 1) ^ polynomial;
                else
                    crc >>= 1;
            }
            table[i] = crc;
        }
        return table;
    }

    /// <summary>
    /// Compute CRC32 hash matching Python's zlib.crc32.
    /// </summary>
    private static uint ComputeCrc32(byte[] data)
    {
        uint crc = 0xFFFFFFFF;
        foreach (byte b in data)
        {
            crc = Crc32Table[(crc ^ b) & 0xFF] ^ (crc >> 8);
        }
        return crc ^ 0xFFFFFFFF;
    }

    /// <summary>
    /// Get numeric ID for a symbol (0 to MaxSymbols-1).
    /// Uses CRC32 hash exactly like Python: zlib.crc32(s.encode()) % MAX_SYMBOLS
    /// </summary>
    /// <param name="symbol">Symbol name (case-sensitive, must match training data)</param>
    /// <returns>Symbol ID in range [0, MaxSymbols-1]</returns>
    public static int GetSymbolId(string symbol)
    {
        if (string.IsNullOrEmpty(symbol))
            return 0;

        byte[] bytes = Encoding.UTF8.GetBytes(symbol);
        uint crc = ComputeCrc32(bytes);
        return (int)(crc % MaxSymbols);
    }
}

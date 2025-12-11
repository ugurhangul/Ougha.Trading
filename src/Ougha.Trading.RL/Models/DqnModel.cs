using System;
using System.Collections.Generic;
using TorchSharp;
using static TorchSharp.torch;
using static TorchSharp.torch.nn;
using TorchSharp.Modules;

namespace Ougha.Trading.RL.Models;

public class DqnModel : Module<Tensor[], (Tensor QValues, Tensor TpSlMultipliers)>
{
    private readonly ModuleList<LSTM> _lstmEncoders;
    private readonly ModuleList<LayerNorm> _inputNorms;
    private readonly MultiheadAttention _attention;
    private readonly LayerNorm _attnNorm;
    private readonly Embedding _symbolEmbedding;

    private readonly Linear? _newsEncoder;
    private readonly Linear? _correlationEncoder;
    private readonly Linear? _exposureEncoder;

    private readonly ModuleList<Linear> _hiddenLayers;
    private readonly ModuleList<Dropout> _dropouts;

    private readonly Linear _actionHead;
    private readonly Linear _tpSlHead;

    private readonly int _windowSize;
    private readonly int _nFeatures;
    private readonly int _lstmUnits;
    private readonly int _embedDim;

    private const int N_TIMEFRAMES = 5;

    public DqnModel(
        string name,
        int windowSize = 20,
        int nFeatures = 45,
        int actionSpace = 3,
        int numSymbols = SymbolIdMapper.MaxSymbols,
        int symbolEmbedDim = 16,
        int lstmUnits = 64,
        int attentionHeads = 8,
        int[]? hiddenSizes = null,
        bool includeNews = true,
        bool includeCrossSymbol = true) : base(name)
    {
        _windowSize = windowSize;
        _nFeatures = nFeatures;
        _lstmUnits = lstmUnits;
        _embedDim = symbolEmbedDim;
        
        hiddenSizes ??= new[] { 256, 128 };
        
        // 1. LSTM Encoders (one per timeframe) with input normalization
        _lstmEncoders = new ModuleList<LSTM>();
        _inputNorms = new ModuleList<LayerNorm>();
        for (int i = 0; i < N_TIMEFRAMES; i++)
        {
            // Input normalization for each timeframe features
            _inputNorms.Add(LayerNorm(new long[] { nFeatures }));
            // batch_first=true is important
            _lstmEncoders.Add(LSTM(nFeatures, lstmUnits, batchFirst: true));
        }
        
        // 2. Attention
        int keyDim = Math.Max(8, lstmUnits / attentionHeads);
        _attention = MultiheadAttention(lstmUnits, attentionHeads, dropout: 0.0, bias: true, add_bias_kv: false, add_zero_attn: false, kdim: null, vdim: null);
        _attnNorm = LayerNorm(new long[] { lstmUnits });
        
        // 3. Symbol Embedding
        _symbolEmbedding = Embedding(numSymbols, symbolEmbedDim);
        
        // 4. Input Dimensions Calculation
        // Fused (lstmUnits) + Symbol (embedDim) + Trigger (5) + Confluence (10) + Portfolio (5) + Risk (9)
        // Portfolio now includes balance (5 instead of 4)
        long totalInputDim = lstmUnits + symbolEmbedDim + 5 + 10 + 5 + 9;

        if (includeNews)
        {
            _newsEncoder = Linear(16, 32);
            totalInputDim += 32;
        }

        if (includeCrossSymbol)
        {
            _correlationEncoder = Linear(20, 32);
            totalInputDim += 32;

            _exposureEncoder = Linear(12, 16);
            totalInputDim += 16;
        }

        // 5. Hidden Layers
        _hiddenLayers = new ModuleList<Linear>();
        _dropouts = new ModuleList<Dropout>();

        long inputDim = totalInputDim;
        foreach (var size in hiddenSizes)
        {
            _hiddenLayers.Add(Linear(inputDim, size));
            _dropouts.Add(Dropout(0.2));
            inputDim = size;
        }

        // 6. Output Heads
        // Action head: 3 discrete actions (Hold, Buy, Sell)
        _actionHead = Linear(inputDim, actionSpace);
        // TP/SL head: 2 continuous values (TP multiplier, SL multiplier)
        _tpSlHead = Linear(inputDim, 2);

        RegisterComponents();
    }
    
    /// <summary>
    /// Forward pass.
    /// Inputs expected in order:
    /// [0-4]: M1, M5, M15, H1, H4 tensors (Batch, Window, Features)
    /// [5]: SymbolID (Batch, 1)
    /// [6]: TriggerContext (Batch, 5)
    /// [7]: Confluence (Batch, 10)
    /// [8]: Portfolio (Batch, 5) - now includes balance
    /// [9]: RiskState (Batch, 9)
    /// [10]: News (Optional)
    /// [11]: Correlation (Optional)
    /// [12]: Exposure (Optional)
    /// Returns: (QValues for 3 actions, TP/SL multipliers [2])
    /// </summary>
    public override (Tensor QValues, Tensor TpSlMultipliers) forward(Tensor[] inputs)
    {
        var encodedTfs = new List<Tensor>();
        for (int i = 0; i < N_TIMEFRAMES; i++)
        {
            var normalized = _inputNorms[i].forward(inputs[i]);
            var (output, hn, cn) = _lstmEncoders[i].forward(normalized);
            encodedTfs.Add(hn.squeeze(0));
        }

        var stacked = torch.stack(encodedTfs, dim: 1);

        var stackedPermuted = stacked.permute(1, 0, 2);
        var (attnOutput, _) = _attention.forward(stackedPermuted, stackedPermuted, stackedPermuted, key_padding_mask: null, need_weights: false, attn_mask: null);
        attnOutput = attnOutput.permute(1, 0, 2);

        attnOutput = _attnNorm.forward(attnOutput + stacked);
        var fused = attnOutput.mean(new long[] { 1 });

        var symEmbed = _symbolEmbedding.forward(inputs[5].to(torch.int64)).flatten(1);

        var toConcat = new List<Tensor>
        {
            fused,
            symEmbed,
            inputs[6], // Trigger
            inputs[7], // Confluence
            inputs[8], // Portfolio (now 5D with balance)
            inputs[9]  // Risk
        };

        int processedIdx = 10;

        if (_newsEncoder != null && inputs.Length > processedIdx)
        {
            var news = functional.relu(_newsEncoder.forward(inputs[processedIdx++]));
            toConcat.Add(news);
        }

        if (_correlationEncoder != null && inputs.Length > processedIdx)
        {
            var corr = functional.relu(_correlationEncoder.forward(inputs[processedIdx++]));
            toConcat.Add(corr);
        }

        if (_exposureEncoder != null && inputs.Length > processedIdx)
        {
            var expo = functional.relu(_exposureEncoder.forward(inputs[processedIdx++]));
            toConcat.Add(expo);
        }

        var combined = torch.cat(toConcat, dim: 1);

        var x = combined;
        for (int i = 0; i < _hiddenLayers.Count; i++)
        {
            x = functional.relu(_hiddenLayers[i].forward(x));
            x = _dropouts[i].forward(x);
        }

        // Action Q-values (3 actions: Hold, Buy, Sell)
        var qValues = _actionHead.forward(x);

        // TP/SL multipliers with sigmoid to bound output [0, 1]
        // These will be scaled to actual ATR multipliers in the environment
        // tpSlRaw[0] = TP multiplier (will be scaled to 1.0-5.0 ATR range)
        // tpSlRaw[1] = SL multiplier (will be scaled to 0.5-3.0 ATR range)
        var tpSlRaw = functional.sigmoid(_tpSlHead.forward(x));

        return (qValues, tpSlRaw);
    }
}

using System;
using System.Collections.Generic;
using TorchSharp;
using static TorchSharp.torch;
using static TorchSharp.torch.nn;
using TorchSharp.Modules;

namespace Ougha.Trading.RL.Models;

public class DqnModel : Module<Tensor[], Tensor>
{
    private readonly ModuleList<LSTM> _lstmEncoders;
    private readonly MultiheadAttention _attention;
    private readonly LayerNorm _attnNorm;
    private readonly Embedding _symbolEmbedding;
    
    // Feature encoders for optional inputs
    private readonly Linear? _newsEncoder;
    private readonly Linear? _correlationEncoder;
    private readonly Linear? _exposureEncoder;
    
    // Main dense layers
    private readonly ModuleList<Linear> _hiddenLayers;
    private readonly ModuleList<Dropout> _dropouts;
    private readonly Linear _outputLayer;
    
    private readonly int _windowSize;
    private readonly int _nFeatures;
    private readonly int _lstmUnits;
    private readonly int _embedDim;
    
    // Constants matching Python
    private const int N_TIMEFRAMES = 5;
    
    public DqnModel(
        string name,
        int windowSize = 20,
        int nFeatures = 45,
        int actionSpace = 8,
        int numSymbols = SymbolIdMapper.MaxSymbols, // 128, matching Python
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
        
        // 1. LSTM Encoders (one per timeframe)
        _lstmEncoders = new ModuleList<LSTM>();
        for (int i = 0; i < N_TIMEFRAMES; i++)
        {
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
        // Fused (lstmUnits) + Symbol (embedDim) + Trigger (5) + Confluence (10) + Portfolio (4) + Risk (9)
        long totalInputDim = lstmUnits + symbolEmbedDim + 5 + 10 + 4 + 9;
        
        // Optional Encoders
        if (includeNews)
        {
            _newsEncoder = Linear(16, 32); // Assuming news dim 16 -> 32
            totalInputDim += 32;
        }
        
        if (includeCrossSymbol)
        {
            _correlationEncoder = Linear(20, 32); // Corr 20 -> 32
            totalInputDim += 32;
            
            _exposureEncoder = Linear(12, 16); // Exposure 12 -> 16
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
        
        // 6. Output Layer
        _outputLayer = Linear(inputDim, actionSpace);
        
        RegisterComponents();
    }
    
    /// <summary>
    /// Forward pass.
    /// Inputs expected in order:
    /// [0-4]: M1, M5, M15, H1, H4 tensors (Batch, Window, Features)
    /// [5]: SymbolID (Batch, 1)
    /// [6]: TriggerContext (Batch, 5)
    /// [7]: Confluence (Batch, 10)
    /// [8]: Portfolio (Batch, 4)
    /// [9]: RiskState (Batch, 9)
    /// [10]: News (Optional)
    /// [11]: Correlation (Optional)
    /// [12]: Exposure (Optional)
    /// </summary>
    public override Tensor forward(Tensor[] inputs)
    {
        // 1. Process Timeframes
        var encodedTfs = new List<Tensor>();
        for (int i = 0; i < N_TIMEFRAMES; i++)
        {
            // LSTM output: (output, (h_n, c_n))
            // We want the last hidden state? 
            // Python: LSTM(return_sequences=False) -> last output
            // Torch: output contains all steps. h_n is the last state.
            // Let's use the output at the last time step for simplicity or h_n
            
            var (output, hn, cn) = _lstmEncoders[i].forward(inputs[i]);
            // hn is (num_layers * num_directions, batch, hidden_size) -> (1, batch, 64)
            // Squeeze first dim -> (batch, 64)
            encodedTfs.Add(hn.squeeze(0));
        }
        
        // Stack: (Batch, 5, 64)
        var stacked = torch.stack(encodedTfs, dim: 1);
        
        // 2. Attention Fusion
        // Default batch_first=False -> (Seq, Batch, Feature)
        var stackedPermuted = stacked.permute(1, 0, 2); 
        var (attnOutput, _) = _attention.forward(stackedPermuted, stackedPermuted, stackedPermuted, key_padding_mask: null, need_weights: false, attn_mask: null);
        attnOutput = attnOutput.permute(1, 0, 2); // Back to (Batch, Seq, Feature)
        
        // Residual + Norm
        attnOutput = _attnNorm.forward(attnOutput + stacked);
        
        // Global Average Pooling (over dim 1 - timeframes)
        var fused = attnOutput.mean(new long[] { 1 }); // (Batch, 64)
        
        // 3. Symbol Embedding
        var symEmbed = _symbolEmbedding.forward(inputs[5].to(torch.int64)).flatten(1);
        
        // 4. Concatenate Base Features
        var toConcat = new List<Tensor>
        {
            fused,
            symEmbed,
            inputs[6], // Trigger
            inputs[7], // Confluence
            inputs[8], // Portfolio
            inputs[9]  // Risk
        };
        
        // 5. Optional Features
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
        
        // 6. Dense Layers
        var x = combined;
        for (int i = 0; i < _hiddenLayers.Count; i++)
        {
            x = functional.relu(_hiddenLayers[i].forward(x));
            x = _dropouts[i].forward(x);
        }
        
        // 7. Output
        return _outputLayer.forward(x);
    }
}

using TorchSharp.Modules;
using static TorchSharp.torch;
using static TorchSharp.torch.nn;

namespace Ougha.Trading.RL.Models;

/// <summary>
/// Actor-Critic Network for PPO.
/// Enhanced architecture: MLP+LayerNorm body, cross-timeframe attention,
/// stateful LSTM temporal memory (persisted across timesteps), separate value body.
/// </summary>
public sealed class ActorCriticModel : Module<Tensor[], (Tensor ActionLogits, Tensor Value, Tensor TpSlParams)>
{
    private readonly Module<Tensor, Tensor> _cnnM1;
    private readonly Module<Tensor, Tensor> _cnnM5;
    private readonly Module<Tensor, Tensor> _cnnM15;
    private readonly Module<Tensor, Tensor> _cnnH1;
    private readonly Module<Tensor, Tensor> _cnnH4;

    private readonly MultiheadAttention _tfAttention;
    private readonly LayerNorm _tfLayerNorm;
    
    private readonly Linear _featureNet;
    private readonly LayerNorm _featureLayerNorm;

    private readonly Sequential _sharedBody;
    
    // LSTM for temporal memory - learns sequential trading patterns
    private readonly LSTM _temporalLstm;
    private readonly LayerNorm _lstmLayerNorm;
    
    // Separate value body for better value estimation (reduces actor-critic interference)
    private readonly Sequential _valueBody;
    
    // LSTM hidden states - persisted across forward passes for temporal memory
    // Key: symbolId, Value: (h_n, c_n) hidden state tuple
    private readonly Dictionary<int, (Tensor h, Tensor c)> _lstmHiddenStates = new();

    private readonly Sequential _actorHead;
    private readonly Sequential _criticHead;
    private readonly Sequential _tpSlHead;

    private const int TimeframeEmbedDim = 768;   // Scaled 3x for 24GB VRAM
    private const int FeatureDim = 128;            // Model pads smaller inputs up to this size
    private const int HiddenDim = 4096;           // Scaled 4x for 24GB VRAM
    private const int ValueHiddenDim = 2048;      // Scaled 4x for 24GB VRAM
    private const int LstmHiddenDim = 2048;       // Scaled 4x for 24GB VRAM

    public ActorCriticModel(string name, int numActions = 4, float dropout = 0.1f) : base(name)
    {
        _cnnM1 = CreateTimeframeEncoder();
        _cnnM5 = CreateTimeframeEncoder();
        _cnnM15 = CreateTimeframeEncoder();
        _cnnH1 = CreateTimeframeEncoder();
        _cnnH4 = CreateTimeframeEncoder();

        _tfAttention = MultiheadAttention(TimeframeEmbedDim, 8, dropout: 0.0, bias: true, add_bias_kv: false, add_zero_attn: false, kdim: null, vdim: null);
        _tfLayerNorm = LayerNorm([TimeframeEmbedDim]);

        _featureNet = Linear(FeatureDim, 256);  // Increased from 128
        _featureLayerNorm = LayerNorm([256]);

        long inputDim = 5 * TimeframeEmbedDim + 256;  // Updated for larger feature net
        
        // Shared body for feature extraction - 5 layers deep for massive capacity
        _sharedBody = Sequential(
            Linear(inputDim, HiddenDim),
            LayerNorm([HiddenDim]),
            ReLU(),
            Dropout(dropout),
            Linear(HiddenDim, HiddenDim),
            LayerNorm([HiddenDim]),
            ReLU(),
            Dropout(dropout),
            Linear(HiddenDim, HiddenDim),
            LayerNorm([HiddenDim]),
            ReLU(),
            Dropout(dropout),
            Linear(HiddenDim, HiddenDim),  // 4th layer
            LayerNorm([HiddenDim]),
            ReLU(),
            Dropout(dropout),
            Linear(HiddenDim, HiddenDim),  // 5th layer
            LayerNorm([HiddenDim]),
            ReLU(),
            Dropout(dropout)
        );
        
        // LSTM for temporal memory - 3 layers for deep temporal patterns
        _temporalLstm = LSTM(HiddenDim, LstmHiddenDim, numLayers: 3, bidirectional: false, dropout: dropout, batchFirst: true);
        _lstmLayerNorm = LayerNorm([LstmHiddenDim]);
        
        // Separate value body - 4 layers for robust value estimation
        _valueBody = Sequential(
            Linear(inputDim, ValueHiddenDim),
            LayerNorm([ValueHiddenDim]),
            ReLU(),
            Dropout(dropout),
            Linear(ValueHiddenDim, ValueHiddenDim),
            LayerNorm([ValueHiddenDim]),
            ReLU(),
            Dropout(dropout),
            Linear(ValueHiddenDim, ValueHiddenDim),
            LayerNorm([ValueHiddenDim]),
            ReLU(),
            Dropout(dropout),
            Linear(ValueHiddenDim, ValueHiddenDim),  // 4th layer
            LayerNorm([ValueHiddenDim]),
            ReLU()
        );
        
        // Note: Cross-symbol attention removed - requires synchronized timestamps which
        // we cannot guarantee during PPO training with shuffled batches

        _actorHead = Sequential(
            Linear(LstmHiddenDim, 256),  // Takes from LSTM output
            LayerNorm([256]),
            ReLU(),
            Dropout(dropout),
            Linear(256, numActions)
        );

        // Critic head takes from separate value body
        _criticHead = Sequential(
            Linear(ValueHiddenDim, 256),  // Increased intermediate
            LayerNorm([256]),
            ReLU(),
            Linear(256, 1)
        );

        _tpSlHead = Sequential(
            Linear(LstmHiddenDim, 256),  // Takes from LSTM output
            LayerNorm([256]),
            ReLU(),
            Linear(256, 2),
            Sigmoid()
        );
        
        RegisterComponents();

        ApplyOrthogonalInit();
    }
    
    private void ApplyOrthogonalInit()
    {
        foreach (var (paramName, param) in named_parameters())
        {
            if (param.dim() >= 2)
            {
                init.orthogonal_(param, gain: Math.Sqrt(2));
            }
            else if (paramName.Contains("bias"))
            {
                init.zeros_(param);
            }
        }

        var actorParams = _actorHead.parameters().ToArray();
        if (actorParams.Length > 0)
        {
            var lastWeight = actorParams[^2];
            if (lastWeight.dim() >= 2)
            {
                init.orthogonal_(lastWeight, gain: 0.01);
            }
        }
    }

    private Module<Tensor, Tensor> CreateTimeframeEncoder()
    {
        return Sequential(
            Conv1d(in_channels: 45, out_channels: 128, kernel_size: 3, padding: 1),  // Doubled from 64
            ReLU(),
            Conv1d(in_channels: 128, out_channels: TimeframeEmbedDim, kernel_size: 3, padding: 1),  // 128→256
            ReLU(),
            AdaptiveAvgPool1d(1),
            Flatten()
        );
    }
    
    /// <summary>
    /// Forward pass with cross-timeframe attention and STATEFUL LSTM temporal memory.
    /// Inputs (packed): [PackedTimeframes, SymbolId, PackedFeatures]
    /// The LSTM hidden states are persisted across forward calls for each symbol,
    /// allowing the network to learn temporal patterns across trading decisions.
    /// </summary>
    public override (Tensor ActionLogits, Tensor Value, Tensor TpSlParams) forward(Tensor[] inputs)
    {
        var packedTf = inputs[0];  // [B, 5, W, F]
        var symbolId = inputs[1];  // [B, 1]
        var packedFeats = inputs[2];
        
        var batchSize = packedTf.shape[0];
        
        // Single permute: [B, 5, W, F] -> [B, 5, F, W] for Conv1d
        var permuted = packedTf.permute(0, 1, 3, 2).contiguous();  // [B, 5, F, W]
        
        // Process each timeframe with its dedicated encoder
        var m1 = _cnnM1.forward(permuted.select(1, 0));   // [B, EmbedDim]
        var m5 = _cnnM5.forward(permuted.select(1, 1));
        var m15 = _cnnM15.forward(permuted.select(1, 2));
        var h1 = _cnnH1.forward(permuted.select(1, 3));
        var h4 = _cnnH4.forward(permuted.select(1, 4));

        var tfStack = stack([m1, m5, m15, h1, h4], dim: 1);  // [B, 5, EmbedDim]
        var tfSeq = tfStack.transpose(0, 1);  // [5, B, EmbedDim] for attention

        var (attended, _) = _tfAttention.forward(tfSeq, tfSeq, tfSeq, key_padding_mask: null, need_weights: false, attn_mask: null);
        attended = attended + tfSeq;

        attended = attended.transpose(0, 1);  // [B, 5, EmbedDim]
        attended = _tfLayerNorm.forward(attended.reshape(-1, TimeframeEmbedDim));
        attended = attended.reshape(batchSize, 5, TimeframeEmbedDim);

        var fusedTf = attended.flatten(1);

        var sym = symbolId.to_type(ScalarType.Float32);
        var feats = cat([sym, packedFeats], dim: 1);
        
        if (feats.shape[1] < FeatureDim)
        {
            var pad = zeros(new[] { feats.shape[0], FeatureDim - feats.shape[1] }, device: feats.device);
            feats = cat([feats, pad], dim: 1);
        }
        
        var featEmbed = _featureNet.forward(feats);
        featEmbed = _featureLayerNorm.forward(featEmbed);
        featEmbed = functional.relu(featEmbed);

        var combined = cat([fusedTf, featEmbed], dim: 1);
        
        // Actor path: shared body -> LSTM -> actor head
        var hidden = _sharedBody.forward(combined);
        
        // LSTM temporal processing with STATEFUL hidden states
        // Each sample in batch may have its own hidden state based on symbolId
        var lstmInput = hidden.unsqueeze(1);  // [B, 1, HiddenDim]
        
        // For inference/single samples, use symbol-specific hidden state
        // For training batches, we process without persistent state (shuffled data)
        if (batchSize == 1)
        {
            // Single sample - use persistent hidden state for this symbol
            var symIdValue = (int)symbolId.cpu().data<long>()[0];
            
            (Tensor h, Tensor c)? priorState = null;
            if (_lstmHiddenStates.TryGetValue(symIdValue, out var cachedState))
            {
                // Move to same device if needed
                priorState = (cachedState.h.to(lstmInput.device), cachedState.c.to(lstmInput.device));
            }
            
            var (lstmOut, h_n, c_n) = _temporalLstm.forward(lstmInput, priorState);
            hidden = lstmOut.squeeze(1);  // [B, LstmHiddenDim]
            
            // Store detached hidden states for next forward pass
            // Detach to prevent gradients flowing across episodes
            _lstmHiddenStates[symIdValue] = (h_n.detach().cpu(), c_n.detach().cpu());
        }
        else
        {
            // Batch processing (training) - no persistent state since batch is shuffled
            var (lstmOut, _, _) = _temporalLstm.forward(lstmInput);
            hidden = lstmOut.squeeze(1);  // [B, LstmHiddenDim]
        }
        
        hidden = _lstmLayerNorm.forward(hidden);
        
        var actionLogits = _actorHead.forward(hidden);
        var tpSl = _tpSlHead.forward(hidden);
        
        // Critic path: separate value body -> critic head (reduces gradient interference)
        var valueHidden = _valueBody.forward(combined);
        var value = _criticHead.forward(valueHidden);
        
        return (actionLogits, value, tpSl);
    }
    
    /// <summary>
    /// Forward pass with explicit LSTM hidden state for sequence-based training.
    /// Use this during training to maintain temporal coherence within sequences.
    /// </summary>
    /// <param name="inputs">Input tensors: [PackedTimeframes, SymbolId, PackedFeatures]</param>
    /// <param name="h">Optional LSTM hidden state from previous timestep</param>
    /// <param name="c">Optional LSTM cell state from previous timestep</param>
    /// <returns>Outputs plus updated hidden state for next timestep</returns>
    public (Tensor ActionLogits, Tensor Value, Tensor TpSlParams, Tensor H, Tensor C) ForwardWithState(
        Tensor[] inputs, 
        Tensor? h = null,
        Tensor? c = null)
    {
        var packedTf = inputs[0];  // [B, 5, W, F]
        var symbolId = inputs[1];  // [B, 1]
        var packedFeats = inputs[2];
        
        var batchSize = packedTf.shape[0];
        
        // Single permute: [B, 5, W, F] -> [B, 5, F, W] for Conv1d
        var permuted = packedTf.permute(0, 1, 3, 2).contiguous();  // [B, 5, F, W]
        
        // Process each timeframe with its dedicated encoder
        var m1 = _cnnM1.forward(permuted.select(1, 0));   // [B, EmbedDim]
        var m5 = _cnnM5.forward(permuted.select(1, 1));
        var m15 = _cnnM15.forward(permuted.select(1, 2));
        var h1 = _cnnH1.forward(permuted.select(1, 3));
        var h4 = _cnnH4.forward(permuted.select(1, 4));

        var tfStack = stack([m1, m5, m15, h1, h4], dim: 1);  // [B, 5, EmbedDim]
        var tfSeq = tfStack.transpose(0, 1);  // [5, B, EmbedDim] for attention

        var (attended, _) = _tfAttention.forward(tfSeq, tfSeq, tfSeq, key_padding_mask: null, need_weights: false, attn_mask: null);
        attended = attended + tfSeq;

        attended = attended.transpose(0, 1);  // [B, 5, EmbedDim]
        attended = _tfLayerNorm.forward(attended.reshape(-1, TimeframeEmbedDim));
        attended = attended.reshape(batchSize, 5, TimeframeEmbedDim);

        var fusedTf = attended.flatten(1);

        var sym = symbolId.to_type(ScalarType.Float32);
        var feats = cat([sym, packedFeats], dim: 1);
        
        if (feats.shape[1] < FeatureDim)
        {
            var pad = zeros(new[] { feats.shape[0], FeatureDim - feats.shape[1] }, device: feats.device);
            feats = cat([feats, pad], dim: 1);
        }
        
        var featEmbed = _featureNet.forward(feats);
        featEmbed = _featureLayerNorm.forward(featEmbed);
        featEmbed = functional.relu(featEmbed);

        var combined = cat([fusedTf, featEmbed], dim: 1);
        
        // Actor path: shared body -> LSTM -> actor head
        var hidden = _sharedBody.forward(combined);
        
        // LSTM temporal processing with EXPLICIT hidden state
        var lstmInput = hidden.unsqueeze(1);  // [B, 1, HiddenDim]
        
        Tensor h_n, c_n;
        if (h is not null && c is not null)
        {
            // Move hidden state to correct device if needed
            var hDevice = h.to(lstmInput.device);
            var cDevice = c.to(lstmInput.device);
            var (lstmOut, newH, newC) = _temporalLstm.forward(lstmInput, (hDevice, cDevice));
            hidden = lstmOut.squeeze(1);  // [B, LstmHiddenDim]
            h_n = newH;
            c_n = newC;
        }
        else
        {
            // No prior state - start fresh
            var (lstmOut, newH, newC) = _temporalLstm.forward(lstmInput);
            hidden = lstmOut.squeeze(1);  // [B, LstmHiddenDim]
            h_n = newH;
            c_n = newC;
        }
        
        hidden = _lstmLayerNorm.forward(hidden);
        
        var actionLogits = _actorHead.forward(hidden);
        var tpSl = _tpSlHead.forward(hidden);
        
        // Critic path: separate value body -> critic head
        var valueHidden = _valueBody.forward(combined);
        var value = _criticHead.forward(valueHidden);
        
        return (actionLogits, value, tpSl, h_n, c_n);
    }

    /// <summary>
    /// Reset the LSTM hidden state for a specific symbol.
    /// Call this at episode boundaries to prevent information leakage across episodes.
    /// </summary>
    public void ResetHiddenState(int symbolId)
    {
        if (_lstmHiddenStates.TryGetValue(symbolId, out var state))
        {
            state.h.Dispose();
            state.c.Dispose();
            _lstmHiddenStates.Remove(symbolId);
        }
    }
    
    /// <summary>
    /// Reset all LSTM hidden states.
    /// Call this when starting a new episode or when syncing networks.
    /// </summary>
    public void ResetAllHiddenStates()
    {
        foreach (var (_, state) in _lstmHiddenStates)
        {
            state.h.Dispose();
            state.c.Dispose();
        }
        _lstmHiddenStates.Clear();
    }
}


using TorchSharp.Modules;
using static TorchSharp.torch;
using static TorchSharp.torch.nn;

namespace Ougha.Trading.RL.Models;

/// <summary>
/// Actor-Critic Network for PPO.
/// Enhanced architecture: MLP+LayerNorm body, cross-timeframe attention,
/// stateful LSTM temporal memory (persisted across timesteps), separate value body.
/// 
/// Supports two modes:
/// - Full mode (~10M params): For final training with good hyperparameters
/// - Debug mode (~600K params): For fast hyperparameter tuning (16x faster)
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

    // Model dimensions - configurable based on debug mode
    private readonly int _timeframeEmbedDim;
    private readonly int _featureDim;
    private readonly int _hiddenDim;
    private readonly int _valueHiddenDim;
    private readonly int _lstmHiddenDim;
    private readonly int _headDim;
    
    /// <summary>
    /// Whether this model is running in debug mode (smaller, faster).
    /// </summary>
    public bool IsDebugMode { get; }

    /// <summary>
    /// Creates an ActorCriticModel.
    /// </summary>
    /// <param name="name">Model name for TorchSharp</param>
    /// <param name="numActions">Number of discrete actions (default: 4 = HOLD, BUY, SELL, CLOSE)</param>
    /// <param name="dropout">Dropout rate for regularization</param>
    /// <param name="debugMode">If true, uses ~16x smaller model for faster hyperparameter tuning</param>
    public ActorCriticModel(string name, int numActions = 4, float dropout = 0.1f, bool debugMode = false) : base(name)
    {
        IsDebugMode = debugMode;
        
        // Configure dimensions based on mode
        if (debugMode)
        {
            // Debug mode: ~600K parameters (16x smaller) - for fast hyperparameter tuning
            _timeframeEmbedDim = 64;    // vs 256
            _featureDim = 128;          // same
            _hiddenDim = 256;           // vs 1024
            _valueHiddenDim = 128;      // vs 512
            _lstmHiddenDim = 128;       // vs 512
            _headDim = 64;              // vs 256
        }
        else
        {
            // Full mode: ~10M parameters - for final training
            _timeframeEmbedDim = 256;
            _featureDim = 128;
            _hiddenDim = 1024;
            _valueHiddenDim = 512;
            _lstmHiddenDim = 512;
            _headDim = 256;
        }
        
        _cnnM1 = CreateTimeframeEncoder();
        _cnnM5 = CreateTimeframeEncoder();
        _cnnM15 = CreateTimeframeEncoder();
        _cnnH1 = CreateTimeframeEncoder();
        _cnnH4 = CreateTimeframeEncoder();

        var attentionHeads = debugMode ? 2 : 4;
        _tfAttention = MultiheadAttention(_timeframeEmbedDim, attentionHeads, dropout: 0.1, bias: true, add_bias_kv: false, add_zero_attn: false, kdim: null, vdim: null);
        _tfLayerNorm = LayerNorm([_timeframeEmbedDim]);

        var featureOutDim = debugMode ? 64 : 256;
        _featureNet = Linear(_featureDim, featureOutDim);
        _featureLayerNorm = LayerNorm([featureOutDim]);

        long inputDim = 5 * _timeframeEmbedDim + featureOutDim;
        
        // Shared body for feature extraction - reduced layers in debug mode
        var bodyLayers = debugMode ? 2 : 3;
        var bodyModules = new List<Module<Tensor, Tensor>>();
        bodyModules.Add(Linear(inputDim, _hiddenDim));
        bodyModules.Add(LayerNorm([_hiddenDim]));
        bodyModules.Add(ReLU());
        bodyModules.Add(Dropout(dropout));
        for (int i = 1; i < bodyLayers; i++)
        {
            bodyModules.Add(Linear(_hiddenDim, _hiddenDim));
            bodyModules.Add(LayerNorm([_hiddenDim]));
            bodyModules.Add(ReLU());
            bodyModules.Add(Dropout(dropout));
        }
        _sharedBody = Sequential(bodyModules);
        
        // LSTM for temporal memory
        var lstmLayers = debugMode ? 1 : 2;
        _temporalLstm = LSTM(_hiddenDim, _lstmHiddenDim, numLayers: lstmLayers, bidirectional: false, dropout: lstmLayers > 1 ? dropout : 0, batchFirst: true);
        _lstmLayerNorm = LayerNorm([_lstmHiddenDim]);
        
        // Separate value body
        var valueModules = new List<Module<Tensor, Tensor>>();
        valueModules.Add(Linear(inputDim, _valueHiddenDim));
        valueModules.Add(LayerNorm([_valueHiddenDim]));
        valueModules.Add(ReLU());
        valueModules.Add(Dropout(dropout));
        for (int i = 1; i < bodyLayers; i++)
        {
            valueModules.Add(Linear(_valueHiddenDim, _valueHiddenDim));
            valueModules.Add(LayerNorm([_valueHiddenDim]));
            valueModules.Add(ReLU());
            if (i < bodyLayers - 1) valueModules.Add(Dropout(dropout));
        }
        _valueBody = Sequential(valueModules);

        _actorHead = Sequential(
            Linear(_lstmHiddenDim, _headDim),
            LayerNorm([_headDim]),
            ReLU(),
            Dropout(dropout),
            Linear(_headDim, numActions)
        );

        // Critic head takes from separate value body
        _criticHead = Sequential(
            Linear(_valueHiddenDim, _headDim),
            LayerNorm([_headDim]),
            ReLU(),
            Linear(_headDim, 1)
        );

        _tpSlHead = Sequential(
            Linear(_lstmHiddenDim, _headDim),
            LayerNorm([_headDim]),
            ReLU(),
            Linear(_headDim, 2),
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
        var cnnHiddenDim = IsDebugMode ? 32 : 64;
        return Sequential(
            Conv1d(in_channels: 45, out_channels: cnnHiddenDim, kernel_size: 3, padding: 1),
            ReLU(),
            Conv1d(in_channels: cnnHiddenDim, out_channels: _timeframeEmbedDim, kernel_size: 3, padding: 1),
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
        attended = _tfLayerNorm.forward(attended.reshape(-1, _timeframeEmbedDim));
        attended = attended.reshape(batchSize, 5, _timeframeEmbedDim);

        var fusedTf = attended.flatten(1);

        var sym = symbolId.to_type(ScalarType.Float32);
        var feats = cat([sym, packedFeats], dim: 1);
        
        if (feats.shape[1] < _featureDim)
        {
            var pad = zeros(new[] { feats.shape[0], _featureDim - feats.shape[1] }, device: feats.device);
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
                // Validate cached tensors are still valid (handle not disposed)
                try
                {
                    // Check if tensors have valid handles by accessing shape
                    _ = cachedState.h.shape;
                    _ = cachedState.c.shape;
                    
                    // Clone and move to device to ensure independent tensors
                    priorState = (cachedState.h.clone().to(lstmInput.device), 
                                  cachedState.c.clone().to(lstmInput.device));
                }
                catch
                {
                    // Cached state is invalid - remove it and proceed without prior state
                    _lstmHiddenStates.Remove(symIdValue);
                    priorState = null;
                }
            }
            
            var (lstmOut, h_n, c_n) = _temporalLstm.forward(lstmInput, priorState);
            hidden = lstmOut.squeeze(1);  // [B, LstmHiddenDim]
            
            // Dispose old cached states if they exist
            if (_lstmHiddenStates.TryGetValue(symIdValue, out var oldState))
            {
                try { oldState.h.Dispose(); } catch { /* ignore */ }
                try { oldState.c.Dispose(); } catch { /* ignore */ }
            }
            
            // Store cloned, detached hidden states for next forward pass
            // Use clone() to create independent copies that won't be invalidated
            _lstmHiddenStates[symIdValue] = (h_n.detach().clone().cpu(), c_n.detach().clone().cpu());
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
        attended = _tfLayerNorm.forward(attended.reshape(-1, _timeframeEmbedDim));
        attended = attended.reshape(batchSize, 5, _timeframeEmbedDim);

        var fusedTf = attended.flatten(1);

        var sym = symbolId.to_type(ScalarType.Float32);
        var feats = cat([sym, packedFeats], dim: 1);
        
        if (feats.shape[1] < _featureDim)
        {
            var pad = zeros(new[] { feats.shape[0], _featureDim - feats.shape[1] }, device: feats.device);
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
    /// Optimized forward pass for sequence-based training.
    /// Processes an entire sequence in a SINGLE forward pass using native LSTM sequence processing.
    /// This is O(1) forward passes instead of O(n) per sequence - major performance improvement.
    /// </summary>
    /// <param name="packedTf">Packed timeframes [SeqLen, 5, W, F]</param>
    /// <param name="symbolIds">Symbol IDs [SeqLen, 1]</param>
    /// <param name="packedFeats">Packed features [SeqLen, FeatureCount]</param>
    /// <returns>Logits, Values, TpSl for entire sequence</returns>
    public (Tensor ActionLogits, Tensor Values, Tensor TpSlParams) ForwardSequenceBatch(
        Tensor packedTf,
        Tensor symbolIds, 
        Tensor packedFeats)
    {
        var seqLen = packedTf.shape[0];
        
        // Process all timesteps through the shared feature processing (CNN + attention)
        // [SeqLen, 5, W, F] -> [SeqLen, 5, F, W] for Conv1d
        var permuted = packedTf.permute(0, 1, 3, 2).contiguous();
        
        // Process each timeframe with its dedicated encoder - works on full batch
        var m1 = _cnnM1.forward(permuted.select(1, 0));   // [SeqLen, EmbedDim]
        var m5 = _cnnM5.forward(permuted.select(1, 1));
        var m15 = _cnnM15.forward(permuted.select(1, 2));
        var h1 = _cnnH1.forward(permuted.select(1, 3));
        var h4 = _cnnH4.forward(permuted.select(1, 4));

        var tfStack = stack([m1, m5, m15, h1, h4], dim: 1);  // [SeqLen, 5, EmbedDim]
        var tfSeq = tfStack.transpose(0, 1);  // [5, SeqLen, EmbedDim] for attention

        var (attended, _) = _tfAttention.forward(tfSeq, tfSeq, tfSeq, key_padding_mask: null, need_weights: false, attn_mask: null);
        attended = attended + tfSeq;

        attended = attended.transpose(0, 1);  // [SeqLen, 5, EmbedDim]
        attended = _tfLayerNorm.forward(attended.reshape(-1, _timeframeEmbedDim));
        attended = attended.reshape(seqLen, 5, _timeframeEmbedDim);

        var fusedTf = attended.flatten(1);  // [SeqLen, 5*EmbedDim]

        var sym = symbolIds.to_type(ScalarType.Float32);
        var feats = cat([sym, packedFeats], dim: 1);
        
        if (feats.shape[1] < _featureDim)
        {
            var pad = zeros(new[] { feats.shape[0], _featureDim - feats.shape[1] }, device: feats.device);
            feats = cat([feats, pad], dim: 1);
        }
        
        var featEmbed = _featureNet.forward(feats);
        featEmbed = _featureLayerNorm.forward(featEmbed);
        featEmbed = functional.relu(featEmbed);

        var combined = cat([fusedTf, featEmbed], dim: 1);  // [SeqLen, InputDim]
        
        // Actor path: shared body -> LSTM -> actor head
        var hidden = _sharedBody.forward(combined);  // [SeqLen, HiddenDim]
        
        // KEY OPTIMIZATION: Use LSTM native sequence processing
        // Reshape for LSTM: [SeqLen, HiddenDim] -> [1, SeqLen, HiddenDim] (batch=1, seq=SeqLen)
        var lstmInput = hidden.unsqueeze(0);  // [1, SeqLen, HiddenDim]
        
        // LSTM processes entire sequence at once - much faster than timestep-by-timestep
        var (lstmOut, _, _) = _temporalLstm.forward(lstmInput);  // [1, SeqLen, LstmHiddenDim]
        
        // Squeeze back to [SeqLen, LstmHiddenDim]
        hidden = lstmOut.squeeze(0);
        hidden = _lstmLayerNorm.forward(hidden);
        
        var actionLogits = _actorHead.forward(hidden);
        var tpSl = _tpSlHead.forward(hidden);
        
        // Critic path: separate value body -> critic head
        var valueHidden = _valueBody.forward(combined);
        var values = _criticHead.forward(valueHidden);
        
        return (actionLogits, values, tpSl);
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


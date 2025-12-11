using System;
using System.Linq;
using TorchSharp;
using TorchSharp.Modules;
using static TorchSharp.torch;
using static TorchSharp.torch.nn;

namespace Ougha.Trading.RL.Models;

/// <summary>
/// Actor-Critic Network for PPO.
/// Improved architecture: MLP+LayerNorm body, cross-timeframe attention.
/// </summary>
public class ActorCriticModel : Module<Tensor[], (Tensor ActionLogits, Tensor Value, Tensor TpSlParams)>
{
    // Feature Extractors
    private readonly Module<Tensor, Tensor> _cnnM1;
    private readonly Module<Tensor, Tensor> _cnnM5;
    private readonly Module<Tensor, Tensor> _cnnM15;
    private readonly Module<Tensor, Tensor> _cnnH1;
    private readonly Module<Tensor, Tensor> _cnnH4;
    
    // Cross-timeframe attention
    private readonly MultiheadAttention _tfAttention;
    private readonly LayerNorm _tfLayerNorm;
    
    private readonly Linear _featureNet;
    private readonly LayerNorm _featureLayerNorm;
    
    // Shared MLP body (replaces LSTM for better single-step processing)
    private readonly Sequential _sharedBody;
    
    // Heads
    private readonly Sequential _actorHead;
    private readonly Sequential _criticHead;
    private readonly Sequential _tpSlHead;
    
    // Feature dimensions
    private const int TimeframeEmbedDim = 64;
    private const int FeatureDim = 120;
    private const int HiddenDim = 512;

    public ActorCriticModel(string name, int numActions = 3, float dropout = 0.1f) : base(name)
    {
        // 1. Timeframe Feature Extractors (1D CNNs)
        _cnnM1 = CreateTimeframeEncoder("M1_Encoder");
        _cnnM5 = CreateTimeframeEncoder("M5_Encoder");
        _cnnM15 = CreateTimeframeEncoder("M15_Encoder");
        _cnnH1 = CreateTimeframeEncoder("H1_Encoder");
        _cnnH4 = CreateTimeframeEncoder("H4_Encoder");
        
        // 2. Cross-Timeframe Attention (NEW)
        // Allows learning relationships between different timeframes
        _tfAttention = MultiheadAttention(TimeframeEmbedDim, 4, dropout: 0.0, bias: true, add_bias_kv: false, add_zero_attn: false, kdim: null, vdim: null);
        _tfLayerNorm = LayerNorm(new long[] { TimeframeEmbedDim });
        
        // 3. Tabular Feature Encoder with normalization
        _featureNet = Linear(FeatureDim, 128);
        _featureLayerNorm = LayerNorm(new long[] { 128 });
        
        // 4. Shared MLP Body (replaces single-step LSTM)
        // MLP is more appropriate for single-step processing
        long inputDim = 5 * TimeframeEmbedDim + 128; // 320 + 128 = 448
        _sharedBody = Sequential(
            Linear(inputDim, HiddenDim),
            LayerNorm(new long[] { HiddenDim }),
            ReLU(),
            Dropout(dropout),
            Linear(HiddenDim, HiddenDim),
            LayerNorm(new long[] { HiddenDim }),
            ReLU(),
            Dropout(dropout)
        );
        
        // 5. Actor Head (Policy) - smaller init gain for stable policy
        _actorHead = Sequential(
            Linear(HiddenDim, 256),
            LayerNorm(new long[] { 256 }),
            ReLU(),
            Dropout(dropout),
            Linear(256, numActions)
        );
        
        // 6. Critic Head (Value)
        _criticHead = Sequential(
            Linear(HiddenDim, 256),
            LayerNorm(new long[] { 256 }),
            ReLU(),
            Dropout(dropout),
            Linear(256, 1)
        );
        
        // 7. Continuous Parameter Head (TP/SL Multipliers)
        _tpSlHead = Sequential(
            Linear(HiddenDim, 128),
            LayerNorm(new long[] { 128 }),
            ReLU(),
            Linear(128, 2),
            Sigmoid()
        );
        
        RegisterComponents();
        
        // Apply orthogonal initialization for better training stability
        ApplyOrthogonalInit();
    }
    
    private void ApplyOrthogonalInit()
    {
        foreach (var (paramName, param) in named_parameters())
        {
            if (param.dim() >= 2)
            {
                // Orthogonal init for weight matrices
                init.orthogonal_(param, gain: Math.Sqrt(2));
            }
            else if (paramName.Contains("bias"))
            {
                // Zero init for biases
                init.zeros_(param);
            }
        }
        
        // Smaller init for actor head output (more stable initial policy)
        var actorParams = _actorHead.parameters().ToArray();
        if (actorParams.Length > 0)
        {
            var lastWeight = actorParams[^2]; // Second to last is weight of final linear
            if (lastWeight.dim() >= 2)
            {
                init.orthogonal_(lastWeight, gain: 0.01);
            }
        }
    }

    private Module<Tensor, Tensor> CreateTimeframeEncoder(string name)
    {
        return Sequential(
            Conv1d(in_channels: 45, out_channels: 32, kernel_size: 3, padding: 1),
            ReLU(),
            Conv1d(in_channels: 32, out_channels: TimeframeEmbedDim, kernel_size: 3, padding: 1),
            ReLU(),
            AdaptiveAvgPool1d(1),
            Flatten()
        );
    }
    
    /// <summary>
    /// Forward pass with cross-timeframe attention.
    /// Inputs (packed): [PackedTimeframes, SymbolId, PackedFeatures]
    /// </summary>
    public override (Tensor ActionLogits, Tensor Value, Tensor TpSlParams) forward(Tensor[] inputs)
    {
        // Unpack the 3 input tensors
        var packedTf = inputs[0];      // [batch, 5, 20, 45]
        var symbolId = inputs[1];      // [batch, 1]
        var packedFeats = inputs[2];   // [batch, 77]
        
        // 1. Process Timeframes
        var m1 = _cnnM1.forward(packedTf.select(1, 0).transpose(1, 2));
        var m5 = _cnnM5.forward(packedTf.select(1, 1).transpose(1, 2));
        var m15 = _cnnM15.forward(packedTf.select(1, 2).transpose(1, 2));
        var h1 = _cnnH1.forward(packedTf.select(1, 3).transpose(1, 2));
        var h4 = _cnnH4.forward(packedTf.select(1, 4).transpose(1, 2));
        
        // 2. Cross-Timeframe Attention (NEW)
        // Stack: [Batch, 5, 64] -> [5, Batch, 64] for attention
        var tfStack = torch.stack(new[] { m1, m5, m15, h1, h4 }, dim: 1);
        var tfSeq = tfStack.transpose(0, 1); // [5, Batch, 64]
        
        // Self-attention across timeframes
        var (attended, _) = _tfAttention.forward(tfSeq, tfSeq, tfSeq, key_padding_mask: null, need_weights: false, attn_mask: null);
        attended = attended + tfSeq; // Residual connection
        
        // Apply layer norm and get back to [Batch, 5, 64]
        attended = attended.transpose(0, 1);
        var batchSize = attended.shape[0];
        attended = _tfLayerNorm.forward(attended.reshape(-1, TimeframeEmbedDim));
        attended = attended.reshape(batchSize, 5, TimeframeEmbedDim);
        
        // Flatten attended timeframes
        var fusedTf = attended.flatten(1); // [Batch, 320]
        
        // 3. Process Features
        var sym = symbolId.to_type(ScalarType.Float32);
        var feats = torch.cat(new[] { sym, packedFeats }, dim: 1);
        
        if (feats.shape[1] < FeatureDim)
        {
            var pad = torch.zeros(new long[] { feats.shape[0], FeatureDim - feats.shape[1] }, device: feats.device);
            feats = torch.cat(new[] { feats, pad }, dim: 1);
        }
        
        var featEmbed = _featureNet.forward(feats);
        featEmbed = _featureLayerNorm.forward(featEmbed);
        featEmbed = functional.relu(featEmbed);
        
        // 4. Combine and process through shared body
        var combined = torch.cat(new[] { fusedTf, featEmbed }, dim: 1);
        var hidden = _sharedBody.forward(combined);
        
        // 5. Heads
        var actionLogits = _actorHead.forward(hidden);
        var value = _criticHead.forward(hidden);
        var tpSl = _tpSlHead.forward(hidden);
        
        return (actionLogits, value, tpSl);
    }
}


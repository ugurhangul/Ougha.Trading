using TorchSharp.Modules;
using static TorchSharp.torch;
using static TorchSharp.torch.nn;

namespace Ougha.Trading.RL.Models;

/// <summary>
/// Actor-Critic Network for PPO.
/// Improved architecture: MLP+LayerNorm body, cross-timeframe attention.
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

    private readonly Sequential _actorHead;
    private readonly Sequential _criticHead;
    private readonly Sequential _tpSlHead;

    private const int TimeframeEmbedDim = 64;
    private const int FeatureDim = 120;
    private const int HiddenDim = 512;

    public ActorCriticModel(string name, int numActions = 3, float dropout = 0.1f) : base(name)
    {
        _cnnM1 = CreateTimeframeEncoder();
        _cnnM5 = CreateTimeframeEncoder();
        _cnnM15 = CreateTimeframeEncoder();
        _cnnH1 = CreateTimeframeEncoder();
        _cnnH4 = CreateTimeframeEncoder();

        _tfAttention = MultiheadAttention(TimeframeEmbedDim, 4, dropout: 0.0, bias: true, add_bias_kv: false, add_zero_attn: false, kdim: null, vdim: null);
        _tfLayerNorm = LayerNorm([TimeframeEmbedDim]);

        _featureNet = Linear(FeatureDim, 128);
        _featureLayerNorm = LayerNorm([128]);

        long inputDim = 5 * TimeframeEmbedDim + 128;
        _sharedBody = Sequential(
            Linear(inputDim, HiddenDim),
            LayerNorm([HiddenDim]),
            ReLU(),
            Dropout(dropout),
            Linear(HiddenDim, HiddenDim),
            LayerNorm([HiddenDim]),
            ReLU(),
            Dropout(dropout)
        );

        _actorHead = Sequential(
            Linear(HiddenDim, 256),
            LayerNorm([256]),
            ReLU(),
            Dropout(dropout),
            Linear(256, numActions)
        );

        _criticHead = Sequential(
            Linear(HiddenDim, 256),
            LayerNorm([256]),
            ReLU(),
            Dropout(dropout),
            Linear(256, 1)
        );

        _tpSlHead = Sequential(
            Linear(HiddenDim, 128),
            LayerNorm([128]),
            ReLU(),
            Linear(128, 2),
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
        var packedTf = inputs[0];
        var symbolId = inputs[1];
        var packedFeats = inputs[2];

        var m1 = _cnnM1.forward(packedTf.select(1, 0).transpose(1, 2));
        var m5 = _cnnM5.forward(packedTf.select(1, 1).transpose(1, 2));
        var m15 = _cnnM15.forward(packedTf.select(1, 2).transpose(1, 2));
        var h1 = _cnnH1.forward(packedTf.select(1, 3).transpose(1, 2));
        var h4 = _cnnH4.forward(packedTf.select(1, 4).transpose(1, 2));

        var tfStack = stack([m1, m5, m15, h1, h4], dim: 1);
        var tfSeq = tfStack.transpose(0, 1);

        var (attended, _) = _tfAttention.forward(tfSeq, tfSeq, tfSeq, key_padding_mask: null, need_weights: false, attn_mask: null);
        attended = attended + tfSeq;

        attended = attended.transpose(0, 1);
        var batchSize = attended.shape[0];
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
        var hidden = _sharedBody.forward(combined);

        var actionLogits = _actorHead.forward(hidden);
        var value = _criticHead.forward(hidden);
        var tpSl = _tpSlHead.forward(hidden);
        
        return (actionLogits, value, tpSl);
    }
}


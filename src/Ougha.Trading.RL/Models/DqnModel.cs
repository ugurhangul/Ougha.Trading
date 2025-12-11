using static TorchSharp.torch;
using static TorchSharp.torch.nn;
using TorchSharp.Modules;

namespace Ougha.Trading.RL.Models;

public sealed class DqnModel : Module<Tensor[], (Tensor QValues, Tensor TpSlMultipliers)>
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

    private const int N_TIMEFRAMES = 5;

    public DqnModel(
        string name,
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
        hiddenSizes ??= [256, 128];

        _lstmEncoders = new ModuleList<LSTM>();
        _inputNorms = new ModuleList<LayerNorm>();
        for (var i = 0; i < N_TIMEFRAMES; i++)
        {
            _inputNorms.Add(LayerNorm([nFeatures]));
            _lstmEncoders.Add(LSTM(nFeatures, lstmUnits, batchFirst: true));
        }

        _attention = MultiheadAttention(lstmUnits, attentionHeads, dropout: 0.0, bias: true, add_bias_kv: false, add_zero_attn: false, kdim: null, vdim: null);
        _attnNorm = LayerNorm([lstmUnits]);

        _symbolEmbedding = Embedding(numSymbols, symbolEmbedDim);

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

        _hiddenLayers = new ModuleList<Linear>();
        _dropouts = new ModuleList<Dropout>();

        var inputDim = totalInputDim;
        foreach (var size in hiddenSizes)
        {
            _hiddenLayers.Add(Linear(inputDim, size));
            _dropouts.Add(Dropout(0.2));
            inputDim = size;
        }

        _actionHead = Linear(inputDim, actionSpace);
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
        for (var i = 0; i < N_TIMEFRAMES; i++)
        {
            var normalized = _inputNorms[i].forward(inputs[i]);
            var (_, hn, _) = _lstmEncoders[i].forward(normalized);
            encodedTfs.Add(hn.squeeze(0));
        }

        var stacked = stack(encodedTfs, dim: 1);

        var stackedPermuted = stacked.permute(1, 0, 2);
        var (attnOutput, _) = _attention.forward(stackedPermuted, stackedPermuted, stackedPermuted, key_padding_mask: null, need_weights: false, attn_mask: null);
        attnOutput = attnOutput.permute(1, 0, 2);

        attnOutput = _attnNorm.forward(attnOutput + stacked);
        var fused = attnOutput.mean([1]);

        var symEmbed = _symbolEmbedding.forward(inputs[5].to(int64)).flatten(1);

        var toConcat = new List<Tensor>
        {
            fused,
            symEmbed,
            inputs[6],
            inputs[7],
            inputs[8],
            inputs[9]
        };

        var processedIdx = 10;

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
            var expo = functional.relu(_exposureEncoder.forward(inputs[processedIdx]));
            toConcat.Add(expo);
        }

        var combined = cat(toConcat, dim: 1);

        var x = combined;
        for (var i = 0; i < _hiddenLayers.Count; i++)
        {
            x = functional.relu(_hiddenLayers[i].forward(x));
            x = _dropouts[i].forward(x);
        }

        var qValues = _actionHead.forward(x);

        var tpSlRaw = functional.sigmoid(_tpSlHead.forward(x));

        return (qValues, tpSlRaw);
    }
}

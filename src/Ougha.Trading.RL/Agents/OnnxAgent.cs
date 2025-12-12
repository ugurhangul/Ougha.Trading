using Microsoft.ML.OnnxRuntime;
using Microsoft.ML.OnnxRuntime.Tensors;

namespace Ougha.Trading.RL.Agents;

/// <summary>
/// ONNX-based agent for inference using a pre-trained multi-input model.
/// Supports multi-timeframe inputs (M1, M5, M15, H1, H4) with 3D tensors,
/// plus additional feature inputs (symbol_id, portfolio, risk, etc.).
/// Supports both CPU and GPU (CUDA) execution providers.
/// </summary>
public class OnnxAgent : IAgent
{
    private readonly InferenceSession _session;
    private readonly Dictionary<string, OnnxInputInfo> _inputInfos = new();
    private bool _disposed;

    private const int DefaultWindowSize = 20;
    private const int DefaultNFeatures = 45;
    private const int NEWS_FEATURE_DIM = 32;
    private const int CORRELATION_FEATURE_DIM = 20;
    private const int PORTFOLIO_EXPOSURE_DIM = 5;

    private static readonly string[] Timeframes = ["M1", "M5", "M15", "H1", "H4"];

    private record OnnxInputInfo(string Name, int[] Dimensions, bool Is3D, Type ElementType);

    /// <summary>
    /// Creates an ONNX agent from a model file.
    /// </summary>
    /// <param name="modelPath">Path to the .onnx model file</param>
    /// <param name="useCuda">Whether to use CUDA GPU acceleration (falls back to CPU if unavailable)</param>
    public OnnxAgent(string modelPath, bool useCuda = true)
    {
        if (!File.Exists(modelPath))
            throw new FileNotFoundException($"ONNX model not found: {modelPath}");

        var sessionOptions = new SessionOptions();
        var cudaEnabled = false;

        if (useCuda)
        {
            cudaEnabled = TryEnableCuda(sessionOptions);
        }

        if (!cudaEnabled)
        {
            Console.WriteLine("[OnnxAgent] Using CPU execution provider");
        }

        sessionOptions.AppendExecutionProvider_CPU();

        _session = new InferenceSession(modelPath, sessionOptions);
        List<string> outputNames = _session.OutputMetadata.Keys.ToList();

        Console.WriteLine($"[OnnxAgent] Loaded model: {Path.GetFileName(modelPath)}");
        Console.WriteLine($"[OnnxAgent] Inputs ({_session.InputMetadata.Count}):");

        foreach (var input in _session.InputMetadata)
        {
            var dims = input.Value.Dimensions;
            var is3D = dims.Length == 3;
            var elementType = GetClrType(input.Value.ElementDataType);
            _inputInfos[input.Key] = new OnnxInputInfo(input.Key, dims, is3D, elementType);
            Console.WriteLine($"  - {input.Key}: [{string.Join(", ", dims)}] type={elementType.Name} {(is3D ? "(3D timeframe)" : "")}");
        }

        Console.WriteLine($"[OnnxAgent] Outputs: {string.Join(", ", outputNames)}");
    }

    /// <summary>
    /// Attempts to enable CUDA execution provider.
    /// </summary>
    private static bool TryEnableCuda(SessionOptions sessionOptions)
    {
        try
        {
            sessionOptions.AppendExecutionProvider_CUDA();
            Console.WriteLine("[OnnxAgent] Using CUDA execution provider");
            return true;
        }
        catch (DllNotFoundException ex)
        {
            Console.WriteLine($"[OnnxAgent] CUDA libraries not found: {ex.Message}");
            Console.WriteLine("[OnnxAgent] Install CUDA Toolkit 12.x or set UseCuda=false in config");
            return false;
        }
        catch (Exception ex)
        {
            Console.WriteLine($"[OnnxAgent] CUDA not available: {ex.Message}");
            return false;
        }
    }

    #region IAgent Implementation

    public int Act(AgentInput state, bool training = false)
    {
        return SelectAction(state);
    }

    public void AddExperience(AgentInput state, int action, float reward, AgentInput? nextState, bool done)
    {
    }

    public void AddExperienceBatch(
        AgentInput[] states,
        int[] actions,
        float[] rewards,
        AgentInput?[] nextStates,
        bool[] dones)
    {
    }

    public (int[] Actions, float[,] TpSlMultipliers) ActBatchWithTpSl(AgentInput[] inputs, bool training = true)
    {
        var actions = SelectPortfolioActions(inputs);
        var tpSl = new float[inputs.Length, 2];
        for(var i=0; i<inputs.Length; i++) { tpSl[i,0] = 1.0f; tpSl[i,1] = 1.0f; } 
        return (actions, tpSl);
    }

    public (int[] Actions, float[,] TpSlMultipliers, float[] LogProbs) ActBatchWithTpSlAndLogProbs(AgentInput[] inputs, bool training = true)
    {
        var (actions, tpSl) = ActBatchWithTpSl(inputs, training);
        return (actions, tpSl, new float[inputs.Length]);
    }

    public void AddExperienceBatchWithLogProbs(
        AgentInput[] states,
        int[] actions,
        float[] rewards,
        AgentInput?[] nextStates,
        bool[] dones,
        float[] logProbs)
    {
    }
    
    public void SyncInferenceNetwork() {
    }
    public void DecayEpsilon() {
    }
    
    public float TrainMultipleBatches(int batches) { return 0f; }

    public float Train()
    {
        return 0f;
    }

    public void Save(string path)
    {
        throw new NotSupportedException("OnnxAgent does not support saving models. Use TorchAgent for training.");
    }

    public void Load(string path)
    {
        throw new NotSupportedException("OnnxAgent loads model via constructor only.");
    }

    public void ResetOnlineLearning()
    {
    }

    public int BufferCount => 0;

    #endregion

    #region Structured Input Methods (New - Recommended)

    /// <summary>
    /// Get Q-values for all actions given structured agent input.
    /// This is the recommended method matching Python's input structure.
    /// </summary>
    public float[] GetQValues(AgentInput input)
    {
        var inputs = CreateInputsFromAgentInput(input);

        using var results = _session.Run(inputs);
        var output = results.First().AsTensor<float>();

        var numActions = output.Dimensions.Length == 1 
            ? output.Dimensions[0] 
            : output.Dimensions[1];

        var qValues = new float[numActions];
        for (var i = 0; i < numActions; i++)
        {
            qValues[i] = output.Dimensions.Length == 1 ? output[i] : output[0, i];
        }

        return qValues;
    }

    /// <summary>
    /// Select an action based on structured agent input.
    /// </summary>
    /// <param name="input">Structured agent input with all features</param>
    /// <returns>Action index (0-7)</returns>
    public int SelectAction(AgentInput input)
    {
        var qValues = GetQValues(input);

        _selectActionCallCount++;
        if (_selectActionCallCount <= 5 || _selectActionCallCount % 1000 == 0)
        {
            Console.WriteLine($"[OnnxAgent] Q-values: [{string.Join(", ", qValues.Select(q => q.ToString("F4")))}] -> Action {FindBestAction(qValues)}");
        }

        return FindBestAction(qValues);
    }

    private int _selectActionCallCount;

    /// <summary>
    /// Select actions for a portfolio of symbols using structured inputs.
    /// </summary>
    /// <param name="inputs">Array of AgentInput, one per symbol</param>
    /// <returns>Array of actions, one per symbol</returns>
    public int[] SelectPortfolioActions(AgentInput[] inputs)
    {
        var actions = new int[inputs.Length];
        
        for (var i = 0; i < inputs.Length; i++)
        {
            actions[i] = SelectAction(inputs[i]);
        }
        
        return actions;
    }

    /// <summary>
    /// Creates ONNX input tensors from structured AgentInput.
    /// Matches Python's ONNXAgentWrapper.act_greedy() input structure.
    /// Uses actual model input names (input_m1, input_symbol_id, etc.)
    /// </summary>
    private List<NamedOnnxValue> CreateInputsFromAgentInput(AgentInput input)
    {
        var inputs = new List<NamedOnnxValue>();

        foreach (var tf in Timeframes)
        {
            var inputName = $"input_{tf.ToLowerInvariant()}";

            if (!_inputInfos.TryGetValue(inputName, out var inputInfo))
                continue;
                
            var seqLen = inputInfo.Dimensions[1] > 0 ? inputInfo.Dimensions[1] : DefaultWindowSize;
            var features = inputInfo.Dimensions[2] > 0 ? inputInfo.Dimensions[2] : DefaultNFeatures;
            
            var tensorData = new float[1 * seqLen * features];

            if (input.TimeframeFeatures.TryGetValue(tf, out var tfFeatures))
            {
                var copyRows = Math.Min(seqLen, tfFeatures.GetLength(0));
                var copyCols = Math.Min(features, tfFeatures.GetLength(1));
                
                for (var row = 0; row < copyRows; row++)
                {
                    for (var col = 0; col < copyCols; col++)
                    {
                        tensorData[row * features + col] = tfFeatures[row, col];
                    }
                }
            }

            SanitizeFloatArray(tensorData);
            
            var tensor = new DenseTensor<float>(tensorData, new[] { 1, seqLen, features });
            inputs.Add(NamedOnnxValue.CreateFromTensor(inputName, tensor));
        }

        if (_inputInfos.TryGetValue("input_symbol_id", out var symInfo))
        {
            var size = symInfo.Dimensions[1] > 0 ? symInfo.Dimensions[1] : 1;
            var symbolTensor = new DenseTensor<int>(new[] { input.SymbolId }, new[] { 1, size });
            inputs.Add(NamedOnnxValue.CreateFromTensor("input_symbol_id", symbolTensor));
        }

        AddFloatTensorIfExists(inputs, "input_trigger_context", input.TriggerContext, 5);

        AddFloatTensorIfExists(inputs, "input_confluence", input.ConfluenceFeatures, 10);

        AddFloatTensorIfExists(inputs, "input_portfolio", input.PortfolioFeatures, 4);

        AddFloatTensorIfExists(inputs, "input_risk_state", input.RiskState, 9);

        AddFloatTensorIfExists(inputs, "input_news", input.NewsFeatures, input.NewsFeatures?.Length ?? NEWS_FEATURE_DIM);

        AddFloatTensorIfExists(inputs, "input_correlation", input.CorrelationFeatures, CORRELATION_FEATURE_DIM);

        AddFloatTensorIfExists(inputs, "input_portfolio_exposure", input.PortfolioExposure, 12);

        return inputs;
    }

    /// <summary>
    /// Helper to add a float tensor input only if the model expects it.
    /// Uses actual model dimensions when available.
    /// </summary>
    private void AddFloatTensorIfExists(List<NamedOnnxValue> inputs, string name, float[]? data, int defaultSize)
    {
        if (!_inputInfos.TryGetValue(name, out var inputInfo))
            return;

        var size = inputInfo.Dimensions[1] > 0 ? inputInfo.Dimensions[1] : defaultSize;
        var tensorData = new float[size];
        
        if (data != null)
        {
            var copyLen = Math.Min(data.Length, size);
            Array.Copy(data, tensorData, copyLen);
        }

        SanitizeFloatArray(tensorData);
        
        var tensor = new DenseTensor<float>(tensorData, new[] { 1, size });
        inputs.Add(NamedOnnxValue.CreateFromTensor(name, tensor));
    }

    /// <summary>
    /// Sanitize a float array by replacing NaN/inf with 0 (matches Python's np.nan_to_num).
    /// </summary>
    private static void SanitizeFloatArray(float[] data)
    {
        for (var i = 0; i < data.Length; i++)
        {
            if (float.IsNaN(data[i]) || float.IsInfinity(data[i]))
            {
                data[i] = 0f;
            }
        }
    }

    /// <summary>
    /// Find action with the highest Q-value (argmax).
    /// Python uses max_action=7 (excludes CLOSE action) - we match this behavior.
    /// </summary>
    private static int FindBestAction(float[] qValues)
    {
        var bestAction = 0;
        var bestValue = qValues[0];
        var maxAction = Math.Min(qValues.Length, 7);

        for (var i = 1; i < maxAction; i++)
        {
            if (qValues[i] > bestValue)
            {
                bestValue = qValues[i];
                bestAction = i;
            }
        }

        return bestAction;
    }

    #endregion

    #region Legacy Methods (Deprecated - for backward compatibility)

    /// <summary>
    /// Get Q-values for all actions given the multi-input state.
    /// Creates properly shaped tensors for each model input.
    /// </summary>
    [Obsolete("Use GetQValues(AgentInput) for proper multi-input support")]
    public float[] GetQValues(float[] state)
    {
        var inputs = CreateInputsFromFlatState(state);

        using var results = _session.Run(inputs);
        var output = results.First().AsTensor<float>();

        var numActions = output.Dimensions.Length == 1 
            ? output.Dimensions[0] 
            : output.Dimensions[1];

        var qValues = new float[numActions];
        for (var i = 0; i < numActions; i++)
        {
            qValues[i] = output.Dimensions.Length == 1 ? output[i] : output[0, i];
        }

        return qValues;
    }

    /// <summary>
    /// Creates input tensors for all model inputs from a flat state array.
    /// This is a legacy method - prefer CreateInputsFromAgentInput for new code.
    /// </summary>
    private List<NamedOnnxValue> CreateInputsFromFlatState(float[] state)
    {
        var inputs = new List<NamedOnnxValue>();
        var stateOffset = 0;

        foreach (var inputInfo in _inputInfos.Values.OrderBy(i => i.Name))
        {
            if (inputInfo.Is3D)
            {
                var dims = inputInfo.Dimensions;
                var batchSize = 1;
                var seqLen = dims[1] > 0 ? dims[1] : DefaultWindowSize;
                var features = dims[2] > 0 ? dims[2] : DefaultNFeatures;
                var totalElements = seqLen * features;

                var tensorData = new float[batchSize * seqLen * features];

                if (stateOffset + totalElements <= state.Length)
                {
                    Array.Copy(state, stateOffset, tensorData, 0, totalElements);
                    stateOffset += totalElements;
                }

                var tensor = new DenseTensor<float>(tensorData, new[] { batchSize, seqLen, features });
                inputs.Add(NamedOnnxValue.CreateFromTensor(inputInfo.Name, tensor));
            }
            else if (inputInfo.Dimensions.Length == 2)
            {
                var batchSize = 1;
                var features = inputInfo.Dimensions[1] > 0 
                    ? inputInfo.Dimensions[1] 
                    : GetDefaultFeatureSize(inputInfo.Name);

                if (inputInfo.ElementType == typeof(int))
                {
                    var tensorData = new int[batchSize * features];
                    var tensor = new DenseTensor<int>(tensorData, new[] { batchSize, features });
                    inputs.Add(NamedOnnxValue.CreateFromTensor(inputInfo.Name, tensor));
                }
                else
                {
                    var tensorData = new float[batchSize * features];

                    if (inputInfo.Name.Contains("symbol", StringComparison.OrdinalIgnoreCase))
                    {
                        tensorData[0] = 0;
                    }
                    else if (stateOffset + features <= state.Length)
                    {
                        Array.Copy(state, stateOffset, tensorData, 0, features);
                        stateOffset += features;
                    }

                    var tensor = new DenseTensor<float>(tensorData, new[] { batchSize, features });
                    inputs.Add(NamedOnnxValue.CreateFromTensor(inputInfo.Name, tensor));
                }
            }
            else if (inputInfo.Dimensions.Length == 1)
            {
                var size = inputInfo.Dimensions[0] > 0 ? inputInfo.Dimensions[0] : 1;
                var tensorData = new float[size];

                if (stateOffset + size <= state.Length)
                {
                    Array.Copy(state, stateOffset, tensorData, 0, size);
                    stateOffset += size;
                }

                var tensor = new DenseTensor<float>(tensorData, new[] { size });
                inputs.Add(NamedOnnxValue.CreateFromTensor(inputInfo.Name, tensor));
            }
        }

        return inputs;
    }

    /// <summary>
    /// Get the default feature size for known input types.
    /// </summary>
    private static int GetDefaultFeatureSize(string inputName)
    {
        return inputName.ToLowerInvariant() switch
        {
            var n when n.Contains("symbol") => 1,
            var n when n.Contains("trigger") => 5,
            var n when n.Contains("confluence") => 10,
            var n when n.Contains("portfolio") => 4,
            var n when n.Contains("risk") => 9,
            var n when n.Contains("news") => NEWS_FEATURE_DIM,
            var n when n.Contains("correlation") => CORRELATION_FEATURE_DIM,
            var n when n.Contains("exposure") => PORTFOLIO_EXPOSURE_DIM,
            _ => 10
        };
    }

    /// <summary>
    /// Convert ONNX tensor element type to CLR type.
    /// </summary>
    private static Type GetClrType(TensorElementType elementType)
    {
        return elementType switch
        {
            TensorElementType.Float => typeof(float),
            TensorElementType.Int32 => typeof(int),
            TensorElementType.Int64 => typeof(long),
            TensorElementType.Double => typeof(double),
            TensorElementType.Float16 => typeof(float),
            TensorElementType.Int8 => typeof(sbyte),
            TensorElementType.Int16 => typeof(short),
            TensorElementType.UInt8 => typeof(byte),
            TensorElementType.UInt16 => typeof(ushort),
            TensorElementType.UInt32 => typeof(uint),
            TensorElementType.UInt64 => typeof(ulong),
            TensorElementType.Bool => typeof(bool),
            _ => typeof(float)
        };
    }

    /// <summary>
    /// Select an action based on the current state using the ONNX model.
    /// </summary>
    /// <param name="state">The state vector (flattened features and portfolio state)</param>
    /// <returns>Action index (0-7)</returns>
    [Obsolete("Use SelectAction(AgentInput) for proper multi-input support")]
    public int SelectAction(float[] state)
    {
#pragma warning disable CS0618
        var qValues = GetQValues(state);
#pragma warning restore CS0618
        return FindBestAction(qValues);
    }

    /// <summary>
    /// Select actions for a portfolio of symbols.
    /// If the model outputs numSymbols * 8 actions, it returns one action per symbol.
    /// Otherwise, applies the single best action to all symbols.
    /// </summary>
    /// <param name="portfolioState">Combined state for all symbols</param>
    /// <param name="numSymbols">Number of symbols in the portfolio</param>
    /// <returns>Array of actions, one per symbol</returns>
    [Obsolete("Use SelectPortfolioActions(AgentInput[]) for proper multi-input support")]
    public int[] SelectPortfolioActions(float[] portfolioState, int numSymbols)
    {
#pragma warning disable CS0618
        var qValues = GetQValues(portfolioState);
#pragma warning restore CS0618

        if (qValues.Length == numSymbols * 8)
        {
            var actions = new int[numSymbols];
            for (var s = 0; s < numSymbols; s++)
            {
                var offset = s * 8;
                var bestAction = 0;
                var bestValue = qValues[offset];

                for (var a = 1; a < 8; a++)
                {
                    if (qValues[offset + a] > bestValue)
                    {
                        bestValue = qValues[offset + a];
                        bestAction = a;
                    }
                }
                actions[s] = bestAction;
            }
            return actions;
        }
        else
        {
            var bestAction = FindBestAction(qValues);

            var actions = new int[numSymbols];
            actions[0] = bestAction;
            return actions;
        }
    }

    #endregion

    public void Dispose()
    {
        if (_disposed) return;
        _session.Dispose();
        _disposed = true;
    }
}


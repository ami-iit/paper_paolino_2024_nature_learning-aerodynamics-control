classdef Concat_To_TransposeLayer1050 < nnet.layer.Layer & nnet.layer.Formattable
    % A custom layer auto-generated while importing an ONNX network.
    
    %#codegen
    %#ok<*PROPLC>
    %#ok<*NBRAK>
    %#ok<*INUSL>
    %#ok<*VARARG>
    
    properties (Learnable)
        hidden_layers_0_0_bi
        hidden_layers_0_0_we
        hidden_layers_1_0_bi
        hidden_layers_1_0_we
        hidden_layers_2_0_bi
        hidden_layers_2_0_we
        hidden_layers_3_0_bi
        hidden_layers_3_0_we
        hidden_layers_4_0_bi
        hidden_layers_4_0_we
        hidden_layers_5_0_bi
        hidden_layers_5_0_we
        hidden_layers_6_0_bi
        hidden_layers_6_0_we
        hidden_layers_7_0_bi
        hidden_layers_7_0_we
        hidden_layers_8_0_bi
        hidden_layers_8_0_we
        input_layer_bias
        input_layer_weight
        output_layer_bias
        output_layer_weight
    end
    
    properties
        ONNXParams         % An ONNXParameters object containing parameters used by this layer.
    end
    
    methods
        function this = Concat_To_TransposeLayer1050(name, onnxParams)
            this.Name = name;
            this.NumInputs = 2;
            this.OutputNames = {'output'};
            this.ONNXParams = onnxParams;
            this.hidden_layers_0_0_bi = onnxParams.Learnables.hidden_layers_0_0_bi;
            this.hidden_layers_0_0_we = onnxParams.Learnables.hidden_layers_0_0_we;
            this.hidden_layers_1_0_bi = onnxParams.Learnables.hidden_layers_1_0_bi;
            this.hidden_layers_1_0_we = onnxParams.Learnables.hidden_layers_1_0_we;
            this.hidden_layers_2_0_bi = onnxParams.Learnables.hidden_layers_2_0_bi;
            this.hidden_layers_2_0_we = onnxParams.Learnables.hidden_layers_2_0_we;
            this.hidden_layers_3_0_bi = onnxParams.Learnables.hidden_layers_3_0_bi;
            this.hidden_layers_3_0_we = onnxParams.Learnables.hidden_layers_3_0_we;
            this.hidden_layers_4_0_bi = onnxParams.Learnables.hidden_layers_4_0_bi;
            this.hidden_layers_4_0_we = onnxParams.Learnables.hidden_layers_4_0_we;
            this.hidden_layers_5_0_bi = onnxParams.Learnables.hidden_layers_5_0_bi;
            this.hidden_layers_5_0_we = onnxParams.Learnables.hidden_layers_5_0_we;
            this.hidden_layers_6_0_bi = onnxParams.Learnables.hidden_layers_6_0_bi;
            this.hidden_layers_6_0_we = onnxParams.Learnables.hidden_layers_6_0_we;
            this.hidden_layers_7_0_bi = onnxParams.Learnables.hidden_layers_7_0_bi;
            this.hidden_layers_7_0_we = onnxParams.Learnables.hidden_layers_7_0_we;
            this.hidden_layers_8_0_bi = onnxParams.Learnables.hidden_layers_8_0_bi;
            this.hidden_layers_8_0_we = onnxParams.Learnables.hidden_layers_8_0_we;
            this.input_layer_bias = onnxParams.Learnables.input_layer_bias;
            this.input_layer_weight = onnxParams.Learnables.input_layer_weight;
            this.output_layer_bias = onnxParams.Learnables.output_layer_bias;
            this.output_layer_weight = onnxParams.Learnables.output_layer_weight;
        end
        
        function [output] = predict(this, windDirection, x_Div_output_0)
            if isdlarray(windDirection)
                windDirection = stripdims(windDirection);
            end
            if isdlarray(x_Div_output_0)
                x_Div_output_0 = stripdims(x_Div_output_0);
            end
            windDirectionNumDims = 2;
            x_Div_output_0NumDims = 2;
            onnxParams = this.ONNXParams;
            onnxParams.Learnables.hidden_layers_0_0_bi = this.hidden_layers_0_0_bi;
            onnxParams.Learnables.hidden_layers_0_0_we = this.hidden_layers_0_0_we;
            onnxParams.Learnables.hidden_layers_1_0_bi = this.hidden_layers_1_0_bi;
            onnxParams.Learnables.hidden_layers_1_0_we = this.hidden_layers_1_0_we;
            onnxParams.Learnables.hidden_layers_2_0_bi = this.hidden_layers_2_0_bi;
            onnxParams.Learnables.hidden_layers_2_0_we = this.hidden_layers_2_0_we;
            onnxParams.Learnables.hidden_layers_3_0_bi = this.hidden_layers_3_0_bi;
            onnxParams.Learnables.hidden_layers_3_0_we = this.hidden_layers_3_0_we;
            onnxParams.Learnables.hidden_layers_4_0_bi = this.hidden_layers_4_0_bi;
            onnxParams.Learnables.hidden_layers_4_0_we = this.hidden_layers_4_0_we;
            onnxParams.Learnables.hidden_layers_5_0_bi = this.hidden_layers_5_0_bi;
            onnxParams.Learnables.hidden_layers_5_0_we = this.hidden_layers_5_0_we;
            onnxParams.Learnables.hidden_layers_6_0_bi = this.hidden_layers_6_0_bi;
            onnxParams.Learnables.hidden_layers_6_0_we = this.hidden_layers_6_0_we;
            onnxParams.Learnables.hidden_layers_7_0_bi = this.hidden_layers_7_0_bi;
            onnxParams.Learnables.hidden_layers_7_0_we = this.hidden_layers_7_0_we;
            onnxParams.Learnables.hidden_layers_8_0_bi = this.hidden_layers_8_0_bi;
            onnxParams.Learnables.hidden_layers_8_0_we = this.hidden_layers_8_0_we;
            onnxParams.Learnables.input_layer_bias = this.input_layer_bias;
            onnxParams.Learnables.input_layer_weight = this.input_layer_weight;
            onnxParams.Learnables.output_layer_bias = this.output_layer_bias;
            onnxParams.Learnables.output_layer_weight = this.output_layer_weight;
            [output, outputNumDims] = Concat_To_TransposeFcn(windDirection, x_Div_output_0, windDirectionNumDims, x_Div_output_0NumDims, onnxParams, 'Training', false, ...
                'InputDataPermutation', {[2 1], [2 1], ['as-is'], ['as-is']}, ...
                'OutputDataPermutation', {[2 1], ['as-is']});
            if any(cellfun(@(A)~isnumeric(A), {output}))
                fprintf('Runtime error in network. The custom layer ''%s'' output a non-numeric value.\n', 'Concat_To_TransposeLayer1050');
                error(message('nnet_cnn_onnx:onnx:BadCustomLayerRuntimeOutput', 'Concat_To_TransposeLayer1050'));
            end
            output = dlarray(single(output), 'CB');
            if ~coder.target('MATLAB')
                output = extractdata(output);
            end
        end
        
        function [output] = forward(this, windDirection, x_Div_output_0)
            if isdlarray(windDirection)
                windDirection = stripdims(windDirection);
            end
            if isdlarray(x_Div_output_0)
                x_Div_output_0 = stripdims(x_Div_output_0);
            end
            windDirectionNumDims = 2;
            x_Div_output_0NumDims = 2;
            onnxParams = this.ONNXParams;
            onnxParams.Learnables.hidden_layers_0_0_bi = this.hidden_layers_0_0_bi;
            onnxParams.Learnables.hidden_layers_0_0_we = this.hidden_layers_0_0_we;
            onnxParams.Learnables.hidden_layers_1_0_bi = this.hidden_layers_1_0_bi;
            onnxParams.Learnables.hidden_layers_1_0_we = this.hidden_layers_1_0_we;
            onnxParams.Learnables.hidden_layers_2_0_bi = this.hidden_layers_2_0_bi;
            onnxParams.Learnables.hidden_layers_2_0_we = this.hidden_layers_2_0_we;
            onnxParams.Learnables.hidden_layers_3_0_bi = this.hidden_layers_3_0_bi;
            onnxParams.Learnables.hidden_layers_3_0_we = this.hidden_layers_3_0_we;
            onnxParams.Learnables.hidden_layers_4_0_bi = this.hidden_layers_4_0_bi;
            onnxParams.Learnables.hidden_layers_4_0_we = this.hidden_layers_4_0_we;
            onnxParams.Learnables.hidden_layers_5_0_bi = this.hidden_layers_5_0_bi;
            onnxParams.Learnables.hidden_layers_5_0_we = this.hidden_layers_5_0_we;
            onnxParams.Learnables.hidden_layers_6_0_bi = this.hidden_layers_6_0_bi;
            onnxParams.Learnables.hidden_layers_6_0_we = this.hidden_layers_6_0_we;
            onnxParams.Learnables.hidden_layers_7_0_bi = this.hidden_layers_7_0_bi;
            onnxParams.Learnables.hidden_layers_7_0_we = this.hidden_layers_7_0_we;
            onnxParams.Learnables.hidden_layers_8_0_bi = this.hidden_layers_8_0_bi;
            onnxParams.Learnables.hidden_layers_8_0_we = this.hidden_layers_8_0_we;
            onnxParams.Learnables.input_layer_bias = this.input_layer_bias;
            onnxParams.Learnables.input_layer_weight = this.input_layer_weight;
            onnxParams.Learnables.output_layer_bias = this.output_layer_bias;
            onnxParams.Learnables.output_layer_weight = this.output_layer_weight;
            [output, outputNumDims] = Concat_To_TransposeFcn(windDirection, x_Div_output_0, windDirectionNumDims, x_Div_output_0NumDims, onnxParams, 'Training', true, ...
                'InputDataPermutation', {[2 1], [2 1], ['as-is'], ['as-is']}, ...
                'OutputDataPermutation', {[2 1], ['as-is']});
            if any(cellfun(@(A)~isnumeric(A), {output}))
                fprintf('Runtime error in network. The custom layer ''%s'' output a non-numeric value.\n', 'Concat_To_TransposeLayer1050');
                error(message('nnet_cnn_onnx:onnx:BadCustomLayerRuntimeOutput', 'Concat_To_TransposeLayer1050'));
            end
            output = dlarray(single(output), 'CB');
            if ~coder.target('MATLAB')
                output = extractdata(output);
            end
        end
    end
end

function [output, outputNumDims, state] = Concat_To_TransposeFcn(windDirection, x_Div_output_0, windDirectionNumDims, x_Div_output_0NumDims, params, varargin)
%CONCAT_TO_TRANSPOSEFCN Function implementing an imported ONNX network.
%
% THIS FILE WAS AUTO-GENERATED BY importONNXFunction.
% ONNX Operator Set Version: 13
%
% Variable names in this function are taken from the original ONNX file.
%
% [OUTPUT] = Concat_To_TransposeFcn(WINDDIRECTION, X_DIV_OUTPUT_0, PARAMS)
%			- Evaluates the imported ONNX network CONCAT_TO_TRANSPOSEFCN with input(s)
%			WINDDIRECTION, X_DIV_OUTPUT_0 and the imported network parameters in PARAMS. Returns
%			network output(s) in OUTPUT.
%
% [OUTPUT, STATE] = Concat_To_TransposeFcn(WINDDIRECTION, X_DIV_OUTPUT_0, PARAMS)
%			- Additionally returns state variables in STATE. When training,
%			use this form and set TRAINING to true.
%
% [__] = Concat_To_TransposeFcn(WINDDIRECTION, X_DIV_OUTPUT_0, PARAMS, 'NAME1', VAL1, 'NAME2', VAL2, ...)
%			- Specifies additional name-value pairs described below:
%
% 'Training'
% 			Boolean indicating whether the network is being evaluated for
%			prediction or training. If TRAINING is true, state variables
%			will be updated.
%
% 'InputDataPermutation'
%			'auto' - Automatically attempt to determine the permutation
%			 between the dimensions of the input data and the dimensions of
%			the ONNX model input. For example, the permutation from HWCN
%			(MATLAB standard) to NCHW (ONNX standard) uses the vector
%			[4 3 1 2]. See the documentation for IMPORTONNXFUNCTION for
%			more information about automatic permutation.
%
%			'none' - Input(s) are passed in the ONNX model format. See 'Inputs'.
%
%			numeric vector - The permutation vector describing the
%			transformation between input data dimensions and the expected
%			ONNX input dimensions.%
%			cell array - If the network has multiple inputs, each cell
%			contains 'auto', 'none', or a numeric vector.
%
% 'OutputDataPermutation'
%			'auto' - Automatically attempt to determine the permutation
%			between the dimensions of the output and a conventional MATLAB
%			dimension ordering. For example, the permutation from NC (ONNX
%			standard) to CN (MATLAB standard) uses the vector [2 1]. See
%			the documentation for IMPORTONNXFUNCTION for more information
%			about automatic permutation.
%
%			'none' - Return output(s) as given by the ONNX model. See 'Outputs'.
%
%			numeric vector - The permutation vector describing the
%			transformation between the ONNX output dimensions and the
%			desired output dimensions.%
%			cell array - If the network has multiple outputs, each cell
%			contains 'auto', 'none' or a numeric vector.
%
% Inputs:
% -------
% WINDDIRECTION, X_DIV_OUTPUT_0
%			- Input(s) to the ONNX network.
%			  The input size(s) expected by the ONNX file are:
%				  WINDDIRECTION:		[3, batch_size]				Type: FLOAT
%				  X_DIV_OUTPUT_0:		[Unknown, Unknown]				Type: FLOAT
%			  By default, the function will try to permute the input(s)
%			  into this dimension ordering. If the default is incorrect,
%			  use the 'InputDataPermutation' argument to control the
%			  permutation.
%
%
% PARAMS	- Network parameters returned by 'importONNXFunction'.
%
%
% Outputs:
% --------
% OUTPUT
%			- Output(s) of the ONNX network.
%			  Without permutation, the size(s) of the outputs are:
%				  OUTPUT:		[39, batch_size]				Type: FLOAT
%			  By default, the function will try to permute the output(s)
%			  from this dimension ordering into a conventional MATLAB
%			  ordering. If the default is incorrect, use the
%			  'OutputDataPermutation' argument to control the permutation.
%
% STATE		- (Optional) State variables. When TRAINING is true, these will
% 			  have been updated from the original values in PARAMS.State.
%
%
%  See also importONNXFunction

% Preprocess the input data and arguments:
[windDirection, x_Div_output_0, Training, outputDataPerms, anyDlarrayInputs] = preprocessInput(windDirection, x_Div_output_0, params, varargin{:});
% Put all variables into a single struct to implement dynamic scoping:
[Vars, NumDims] = packageVariables(params, {'windDirection', 'x_Div_output_0'}, {windDirection, x_Div_output_0}, [windDirectionNumDims x_Div_output_0NumDims]);
% Call the top-level graph function:
[output, outputNumDims, state] = Concat_To_TransposeGraph1023(windDirection, x_Div_output_0, NumDims.windDirection, NumDims.x_Div_output_0, Vars, NumDims, Training, params.State);
% Postprocess the output data
[output] = postprocessOutput(output, outputDataPerms, anyDlarrayInputs, Training, varargin{:});
end

function [output, outputNumDims1049, state] = Concat_To_TransposeGraph1023(windDirection, x_Div_output_0, windDirectionNumDims1047, x_Div_output_0NumDims1048, Vars, NumDims, Training, state)
% Function implementing the graph 'Concat_To_TransposeGraph1023'
% Update Vars and NumDims from the graph's formal input parameters. Note that state variables are already in Vars.
Vars.windDirection = windDirection;
NumDims.windDirection = windDirectionNumDims1047;
Vars.x_Div_output_0 = x_Div_output_0;
NumDims.x_Div_output_0 = x_Div_output_0NumDims1048;

% Execute the operators:
% Concat:
[Vars.x_Concat_output_0, NumDims.x_Concat_output_0] = onnxConcat(0, {Vars.windDirection, Vars.x_Div_output_0}, [NumDims.windDirection, NumDims.x_Div_output_0]);

% Gemm:
[A, B, C, alpha, beta, NumDims.x_input_layer_Gemm_o] = prepareGemmArgs(Vars.x_Concat_output_0, Vars.input_layer_weight, Vars.input_layer_bias, Vars.Gemmalpha1024, Vars.Gemmbeta1025, 1, 1, NumDims.input_layer_bias);
Vars.x_input_layer_Gemm_o = alpha*B*A + beta*C;

% Relu:
Vars.x_Relu_output_0 = relu(Vars.x_input_layer_Gemm_o);
NumDims.x_Relu_output_0 = NumDims.x_input_layer_Gemm_o;

% Gemm:
[A, B, C, alpha, beta, NumDims.x_hidden_layers_0_hi] = prepareGemmArgs(Vars.x_Relu_output_0, Vars.hidden_layers_0_0_we, Vars.hidden_layers_0_0_bi, Vars.Gemmalpha1026, Vars.Gemmbeta1027, 0, 1, NumDims.hidden_layers_0_0_bi);
Vars.x_hidden_layers_0_hi = alpha*B*A + beta*C;

% Relu:
Vars.x_hidden_layers_0__1 = relu(Vars.x_hidden_layers_0_hi);
NumDims.x_hidden_layers_0__1 = NumDims.x_hidden_layers_0_hi;

% Gemm:
[A, B, C, alpha, beta, NumDims.x_hidden_layers_1_hi] = prepareGemmArgs(Vars.x_hidden_layers_0__1, Vars.hidden_layers_1_0_we, Vars.hidden_layers_1_0_bi, Vars.Gemmalpha1028, Vars.Gemmbeta1029, 0, 1, NumDims.hidden_layers_1_0_bi);
Vars.x_hidden_layers_1_hi = alpha*B*A + beta*C;

% Relu:
Vars.x_hidden_layers_1__1 = relu(Vars.x_hidden_layers_1_hi);
NumDims.x_hidden_layers_1__1 = NumDims.x_hidden_layers_1_hi;

% Gemm:
[A, B, C, alpha, beta, NumDims.x_hidden_layers_2_hi] = prepareGemmArgs(Vars.x_hidden_layers_1__1, Vars.hidden_layers_2_0_we, Vars.hidden_layers_2_0_bi, Vars.Gemmalpha1030, Vars.Gemmbeta1031, 0, 1, NumDims.hidden_layers_2_0_bi);
Vars.x_hidden_layers_2_hi = alpha*B*A + beta*C;

% Relu:
Vars.x_hidden_layers_2__1 = relu(Vars.x_hidden_layers_2_hi);
NumDims.x_hidden_layers_2__1 = NumDims.x_hidden_layers_2_hi;

% Gemm:
[A, B, C, alpha, beta, NumDims.x_hidden_layers_3_hi] = prepareGemmArgs(Vars.x_hidden_layers_2__1, Vars.hidden_layers_3_0_we, Vars.hidden_layers_3_0_bi, Vars.Gemmalpha1032, Vars.Gemmbeta1033, 0, 1, NumDims.hidden_layers_3_0_bi);
Vars.x_hidden_layers_3_hi = alpha*B*A + beta*C;

% Relu:
Vars.x_hidden_layers_3__1 = relu(Vars.x_hidden_layers_3_hi);
NumDims.x_hidden_layers_3__1 = NumDims.x_hidden_layers_3_hi;

% Gemm:
[A, B, C, alpha, beta, NumDims.x_hidden_layers_4_hi] = prepareGemmArgs(Vars.x_hidden_layers_3__1, Vars.hidden_layers_4_0_we, Vars.hidden_layers_4_0_bi, Vars.Gemmalpha1034, Vars.Gemmbeta1035, 0, 1, NumDims.hidden_layers_4_0_bi);
Vars.x_hidden_layers_4_hi = alpha*B*A + beta*C;

% Relu:
Vars.x_hidden_layers_4__1 = relu(Vars.x_hidden_layers_4_hi);
NumDims.x_hidden_layers_4__1 = NumDims.x_hidden_layers_4_hi;

% Gemm:
[A, B, C, alpha, beta, NumDims.x_hidden_layers_5_hi] = prepareGemmArgs(Vars.x_hidden_layers_4__1, Vars.hidden_layers_5_0_we, Vars.hidden_layers_5_0_bi, Vars.Gemmalpha1036, Vars.Gemmbeta1037, 0, 1, NumDims.hidden_layers_5_0_bi);
Vars.x_hidden_layers_5_hi = alpha*B*A + beta*C;

% Relu:
Vars.x_hidden_layers_5__1 = relu(Vars.x_hidden_layers_5_hi);
NumDims.x_hidden_layers_5__1 = NumDims.x_hidden_layers_5_hi;

% Gemm:
[A, B, C, alpha, beta, NumDims.x_hidden_layers_6_hi] = prepareGemmArgs(Vars.x_hidden_layers_5__1, Vars.hidden_layers_6_0_we, Vars.hidden_layers_6_0_bi, Vars.Gemmalpha1038, Vars.Gemmbeta1039, 0, 1, NumDims.hidden_layers_6_0_bi);
Vars.x_hidden_layers_6_hi = alpha*B*A + beta*C;

% Relu:
Vars.x_hidden_layers_6__1 = relu(Vars.x_hidden_layers_6_hi);
NumDims.x_hidden_layers_6__1 = NumDims.x_hidden_layers_6_hi;

% Gemm:
[A, B, C, alpha, beta, NumDims.x_hidden_layers_7_hi] = prepareGemmArgs(Vars.x_hidden_layers_6__1, Vars.hidden_layers_7_0_we, Vars.hidden_layers_7_0_bi, Vars.Gemmalpha1040, Vars.Gemmbeta1041, 0, 1, NumDims.hidden_layers_7_0_bi);
Vars.x_hidden_layers_7_hi = alpha*B*A + beta*C;

% Relu:
Vars.x_hidden_layers_7__1 = relu(Vars.x_hidden_layers_7_hi);
NumDims.x_hidden_layers_7__1 = NumDims.x_hidden_layers_7_hi;

% Gemm:
[A, B, C, alpha, beta, NumDims.x_hidden_layers_8_hi] = prepareGemmArgs(Vars.x_hidden_layers_7__1, Vars.hidden_layers_8_0_we, Vars.hidden_layers_8_0_bi, Vars.Gemmalpha1042, Vars.Gemmbeta1043, 0, 1, NumDims.hidden_layers_8_0_bi);
Vars.x_hidden_layers_8_hi = alpha*B*A + beta*C;

% Relu:
Vars.x_hidden_layers_8__1 = relu(Vars.x_hidden_layers_8_hi);
NumDims.x_hidden_layers_8__1 = NumDims.x_hidden_layers_8_hi;

% Gemm:
[A, B, C, alpha, beta, NumDims.x_output_layer_Gemm_] = prepareGemmArgs(Vars.x_hidden_layers_8__1, Vars.output_layer_weight, Vars.output_layer_bias, Vars.Gemmalpha1044, Vars.Gemmbeta1045, 0, 1, NumDims.output_layer_bias);
Vars.x_output_layer_Gemm_ = alpha*B*A + beta*C;

% Transpose:
[perm, NumDims.output] = prepareTransposeArgs(Vars.TransposePerm1046, NumDims.x_output_layer_Gemm_);
if ~isempty(perm)
    Vars.output = permute(Vars.x_output_layer_Gemm_, perm);
end

% Set graph output arguments from Vars and NumDims:
output = Vars.output;
outputNumDims1049 = NumDims.output;
% Set output state from Vars:
state = updateStruct(state, Vars);
end

function [inputDataPerms, outputDataPerms, Training] = parseInputs(windDirection, x_Div_output_0, numDataOutputs, params, varargin)
% Function to validate inputs to Concat_To_TransposeFcn:
p = inputParser;
isValidArrayInput = @(x)isnumeric(x) || isstring(x);
isValidONNXParameters = @(x)isa(x, 'ONNXParameters');
addRequired(p, 'windDirection', isValidArrayInput);
addRequired(p, 'x_Div_output_0', isValidArrayInput);
addRequired(p, 'params', isValidONNXParameters);
addParameter(p, 'InputDataPermutation', 'auto');
addParameter(p, 'OutputDataPermutation', 'auto');
addParameter(p, 'Training', false);
parse(p, windDirection, x_Div_output_0, params, varargin{:});
inputDataPerms = p.Results.InputDataPermutation;
outputDataPerms = p.Results.OutputDataPermutation;
Training = p.Results.Training;
if isnumeric(inputDataPerms)
    inputDataPerms = {inputDataPerms};
end
if isstring(inputDataPerms) && isscalar(inputDataPerms) || ischar(inputDataPerms)
    inputDataPerms = repmat({inputDataPerms},1,2);
end
if isnumeric(outputDataPerms)
    outputDataPerms = {outputDataPerms};
end
if isstring(outputDataPerms) && isscalar(outputDataPerms) || ischar(outputDataPerms)
    outputDataPerms = repmat({outputDataPerms},1,numDataOutputs);
end
end

function [windDirection, x_Div_output_0, Training, outputDataPerms, anyDlarrayInputs] = preprocessInput(windDirection, x_Div_output_0, params, varargin)
% Parse input arguments
[inputDataPerms, outputDataPerms, Training] = parseInputs(windDirection, x_Div_output_0, 1, params, varargin{:});
anyDlarrayInputs = any(cellfun(@(x)isa(x, 'dlarray'), {windDirection, x_Div_output_0}));
% Make the input variables into unlabelled dlarrays:
windDirection = makeUnlabeledDlarray(windDirection);
x_Div_output_0 = makeUnlabeledDlarray(x_Div_output_0);
% Permute inputs if requested:
windDirection = permuteInputVar(windDirection, inputDataPerms{1}, 2);
x_Div_output_0 = permuteInputVar(x_Div_output_0, inputDataPerms{2}, 2);
end

function [output] = postprocessOutput(output, outputDataPerms, anyDlarrayInputs, Training, varargin)
% Set output type:
if ~anyDlarrayInputs && ~Training
    if isdlarray(output)
        output = extractdata(output);
    end
end
% Permute outputs if requested:
output = permuteOutputVar(output, outputDataPerms{1}, 2);
end


%% dlarray functions implementing ONNX operators:

function [Y, numDimsY] = onnxConcat(ONNXAxis, XCell, numDimsXArray)
% Concatentation that treats all empties the same. Necessary because
% dlarray.cat does not allow, for example, cat(1, 1x1, 1x0) because the
% second dimension sizes do not match.
numDimsY = numDimsXArray(1);
XCell(cellfun(@isempty, XCell)) = [];
if isempty(XCell)
    Y = dlarray([]);
else
    if ONNXAxis<0
        ONNXAxis = ONNXAxis + numDimsY;
    end
    DLTAxis = numDimsY - ONNXAxis;
    Y = cat(DLTAxis, XCell{:});
end
end

function [A, B, C, alpha, beta, numDimsY] = prepareGemmArgs(A, B, C, alpha, beta, transA, transB, numDimsC)
% Prepares arguments for implementing the ONNX Gemm operator
if transA
    A = A';
end
if transB
    B = B';
end
if numDimsC < 2
    C = C(:);   % C can be broadcast to [N M]. Make C a col vector ([N 1])
end
numDimsY = 2;
% Y=B*A because we want (AB)'=B'A', and B and A are already transposed.
end

function [perm, numDimsA] = prepareTransposeArgs(ONNXPerm, numDimsA)
% Prepares arguments for implementing the ONNX Transpose operator
if numDimsA <= 1        % Tensors of numDims 0 or 1 are unchanged by ONNX Transpose.
    perm = [];
else
    if isempty(ONNXPerm)        % Empty ONNXPerm means reverse the dimensions.
        perm = numDimsA:-1:1;
    else
        perm = numDimsA-flip(ONNXPerm);
    end
end
end

%% Utility functions:

function s = appendStructs(varargin)
% s = appendStructs(s1, s2,...). Assign all fields in s1, s2,... into s.
if isempty(varargin)
    s = struct;
else
    s = varargin{1};
    for i = 2:numel(varargin)
        fromstr = varargin{i};
        fs = fieldnames(fromstr);
        for j = 1:numel(fs)
            s.(fs{j}) = fromstr.(fs{j});
        end
    end
end
end

function checkInputSize(inputShape, expectedShape, inputName)

if numel(expectedShape)==0
    % The input is a scalar
    if ~isequal(inputShape, [1 1])
        inputSizeStr = makeSizeString(inputShape);
        error(message('nnet_cnn_onnx:onnx:InputNeedsResize',inputName, "[1,1]", inputSizeStr));
    end
elseif numel(expectedShape)==1
    % The input is a vector
    if ~shapeIsColumnVector(inputShape) || ~iSizesMatch({inputShape(1)}, expectedShape)
        expectedShape{2} = 1;
        expectedSizeStr = makeSizeString(expectedShape);
        inputSizeStr = makeSizeString(inputShape);
        error(message('nnet_cnn_onnx:onnx:InputNeedsResize',inputName, expectedSizeStr, inputSizeStr));
    end
else
    % The input has 2 dimensions or more
    
    % The input dimensions have been reversed; flip them back to compare to the
    % expected ONNX shape.
    inputShape = fliplr(inputShape);
    
    % If the expected shape has fewer dims than the input shape, error.
    if numel(expectedShape) < numel(inputShape)
        expectedSizeStr = strjoin(["[", strjoin(string(expectedShape), ","), "]"], "");
        error(message('nnet_cnn_onnx:onnx:InputHasGreaterNDims', inputName, expectedSizeStr));
    end
    
    % Prepad the input shape with trailing ones up to the number of elements in
    % expectedShape
    inputShape = num2cell([ones(1, numel(expectedShape) - length(inputShape)) inputShape]);
    
    % Find the number of variable size dimensions in the expected shape
    numVariableInputs = sum(cellfun(@(x) isa(x, 'char') || isa(x, 'string'), expectedShape));
    
    % Find the number of input dimensions that are not in the expected shape
    % and cannot be represented by a variable dimension
    nonMatchingInputDims = setdiff(string(inputShape), string(expectedShape));
    numNonMatchingInputDims  = numel(nonMatchingInputDims) - numVariableInputs;
    
    expectedSizeStr = makeSizeString(expectedShape);
    inputSizeStr = makeSizeString(inputShape);
    if numNonMatchingInputDims == 0 && ~iSizesMatch(inputShape, expectedShape)
        % The actual and expected input dimensions match, but in
        % a different order. The input needs to be permuted.
        error(message('nnet_cnn_onnx:onnx:InputNeedsPermute',inputName, expectedSizeStr, inputSizeStr));
    elseif numNonMatchingInputDims > 0
        % The actual and expected input sizes do not match.
        error(message('nnet_cnn_onnx:onnx:InputNeedsResize',inputName, expectedSizeStr, inputSizeStr));
    end
end
end

function doesMatch = iSizesMatch(inputShape, expectedShape)
% Check whether the input and expected shapes match, in order.
% Size elements match if (1) the elements are equal, or (2) the expected
% size element is a variable (represented by a character vector or string)
doesMatch = true;
for i=1:numel(inputShape)
    if ~(isequal(inputShape{i},expectedShape{i}) || ischar(expectedShape{i}) || isstring(expectedShape{i}))
        doesMatch = false;
        return
    end
end
end

function sizeStr = makeSizeString(shape)
sizeStr = strjoin(["[", strjoin(string(shape), ","), "]"], "");
end

function isVec = shapeIsColumnVector(shape)
if numel(shape) == 2 && shape(2) == 1
    isVec = true;
else
    isVec = false;
end
end
function X = makeUnlabeledDlarray(X)
% Make numeric X into an unlabelled dlarray
if isa(X, 'dlarray')
    X = stripdims(X);
elseif isnumeric(X)
    if isinteger(X)
        % Make ints double so they can combine with anything without
        % reducing precision
        X = double(X);
    end
    X = dlarray(X);
end
end

function [Vars, NumDims] = packageVariables(params, inputNames, inputValues, inputNumDims)
% inputNames, inputValues are cell arrays. inputRanks is a numeric vector.
Vars = appendStructs(params.Learnables, params.Nonlearnables, params.State);
NumDims = params.NumDimensions;
% Add graph inputs
for i = 1:numel(inputNames)
    Vars.(inputNames{i}) = inputValues{i};
    NumDims.(inputNames{i}) = inputNumDims(i);
end
end

function X = permuteInputVar(X, userDataPerm, onnxNDims)
% Returns reverse-ONNX ordering
if onnxNDims == 0
    return;
elseif onnxNDims == 1 && isvector(X)
    X = X(:);
    return;
elseif isnumeric(userDataPerm)
    % Permute into reverse ONNX ordering
    if numel(userDataPerm) ~= onnxNDims
        error(message('nnet_cnn_onnx:onnx:InputPermutationSize', numel(userDataPerm), onnxNDims));
    end
    perm = fliplr(userDataPerm);
elseif isequal(userDataPerm, 'auto') && onnxNDims == 4
    % Permute MATLAB HWCN to reverse onnx (WHCN)
    perm = [2 1 3 4];
elseif isequal(userDataPerm, 'as-is')
    % Do not permute the input
    perm = 1:ndims(X);
else
    % userDataPerm is either 'none' or 'auto' with no default, which means
    % it's already in onnx ordering, so just make it reverse onnx
    perm = max(2,onnxNDims):-1:1;
end
X = permute(X, perm);
end

function Y = permuteOutputVar(Y, userDataPerm, onnxNDims)
switch onnxNDims
    case 0
        perm = [];
    case 1
        if isnumeric(userDataPerm)
            % Use the user's permutation because Y is a column vector which
            % already matches ONNX.
            perm = userDataPerm;
        elseif isequal(userDataPerm, 'auto')
            % Treat the 1D onnx vector as a 2D column and transpose it
            perm = [2 1];
        else
            % userDataPerm is 'none'. Leave Y alone because it already
            % matches onnx.
            perm = [];
        end
    otherwise
        % ndims >= 2
        if isnumeric(userDataPerm)
            % Use the inverse of the user's permutation. This is not just the
            % flip of the permutation vector.
            perm = onnxNDims + 1 - userDataPerm;
        elseif isequal(userDataPerm, 'auto')
            if onnxNDims == 2
                % Permute reverse ONNX CN to DLT CN (do nothing)
                perm = [];
            elseif onnxNDims == 4
                % Permute reverse onnx (WHCN) to MATLAB HWCN
                perm = [2 1 3 4];
            else
                % User wants the output in ONNX ordering, so just reverse it from
                % reverse onnx
                perm = onnxNDims:-1:1;
            end
        elseif isequal(userDataPerm, 'as-is')
            % Do not permute the input
            perm = 1:ndims(Y);
        else
            % userDataPerm is 'none', so just make it reverse onnx
            perm = onnxNDims:-1:1;
        end
end
if ~isempty(perm)
    Y = permute(Y, perm);
end
end

function s = updateStruct(s, t)
% Set all existing fields in s from fields in t, ignoring extra fields in t.
for name = transpose(fieldnames(s))
    s.(name{1}) = t.(name{1});
end
end

classdef NeuralNetworkConstants < handle
    %% Rules for automatic creation of new layers
    methods (Access = public)
        function const = CREATE_LAYER_WITH_ONE_NODE(~)
            const = 1;
        end
        
        function const = CREATE_LAYER_EQUAL_OUTPUT(~)
            const = 2;
        end
        
        function const = CREATE_LAYER_BY_ARGUMENT(~)
            const = 3;
        end
        
        function const = CREATE_MIRRORED_LAYER(~)
            const = 16;
        end
        
    end
    
    %% Rules for prune nodes in a layer
    methods (Access = public)
        function const = PRUNE_SINGLE_LEAST_CONTRIBUTION_NODES(~)
            const = 14;
        end
        
        function const = PRUNE_MULTIPLE_NODES_WITH_CONTRIBUTION_BELOW_EXPECTED(~)
            const = 15;
        end
    end
    
    %% Activation functions
    methods (Access = public)
        function const = ACTIVATION_FUNCTION_SIGMOID(~)
            const = 4;
        end
        
        function const = ACTIVATION_FUNCTION_TANH(~)
            const = 5;
        end
        
        function const = ACTIVATION_FUNCTION_RELU(~)
            const = 6;
        end
        
        function const = ACTIVATION_FUNCTION_LINEAR(~)
            const = 7;
        end
        
        function const = ACTIVATION_FUNCTION_SOFTMAX(~)
            const = 8;
        end
    end
    
    %% Activation functions and Loss functions (normally used as output activation function)
    methods (Access = public)
        function const = ACTIVATION_LOSS_FUNCTION_SIGMOID_MSE(~)
            const = 9;
        end
        
        function const = ACTIVATION_LOSS_FUNCTION_TANH(~)
            const = 10;
        end
        
        function const = ACTIVATION_LOSS_FUNCTION_RELU(~)
            const = 11;
        end
        
        function const = ACTIVATION_LOSS_FUNCTION_SOFTMAX_CROSS_ENTROPY(~)
            const = 12;
        end
        
        function const = ACTIVATION_LOSS_FUNCTION_LINEAR_CROSS_ENTROPY(~)
            const = 13;
        end
    end
end


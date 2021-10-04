classdef DenoisingAutoEncoder < AutoEncoder
    %DenoisingAutoEncoder
    %   This object mimics the behavior of a Denoising Auto Encoder network,
    %   which is an Auto Encoder that receives noised input data and tries 
    %   to denoise it
    %   This object has elastic habilities, being able to grow and prune
    %   nodes automatically.
    %   TODO: Provide the paper or study material for the Denoising Auto Encoder
    
    methods (Access = public)
%         function self = DenoisingAutoEncoder(nInput, nHiddenNodes)
%             %DenoisingAutoEncoder Construct an instance of this class
%             %   nInput (integer)
%             %       Number of input nodes
%             %   nHiddenNodes (integer)
%             %       Number of nodes at the hidden layer
%             self@AutoEncoder(nInput, nHiddenNodes);
%         end

        function self = DenoisingAutoEncoder(layers)
            %DenoisingAutoEncoder Construct an instance of this class
            self@AutoEncoder(layers);
        end
        
        function train(self, X, noiseRatio, nWeight)
            % train
            %   See train@NeuralNetwork
            %   X (matrix)
            %       Input and output data
            %   noiseRatio (double)
            %       Value between 0.0 and 1.0
            %       It indicates the percentage of noise that will be
            %       applied on the input datapreparing the network for
            %       another kind of data.
            %   nWeight (integer) [optional]
            %       You has the ability to define which weight and bias you
            %       want to update using backpropagation. This method will
            %       update only that weight and bias, even if there is
            %       weights and biases on layers before and after that.
            %       The number of the weight and bias you want to update.
            %       Remember that 1 indicates the weight and bias that get
            %       out of the input layer.
            if nargin == 3
                train@AutoEncoder(self, X, noiseRatio)
            elseif nargin == 4
                train@AutoEncoder(self, X, noiseRatio, nWeight);
            end
        end
        
        function greddyLayerWiseTrain(self, X, nEpochs, noiseRatio)
            %greddyLayerWiseTrain
            %   Performs Greedy Layer Wise train
            %   TODO: Provide the paper or study material for the Greedy
            %   layer Wise train
            %   X (matrix)
            %       Input and output data
            %   nEpochs (integer)
            %       The number of epochs which the greedy layer wise train
            %       will occurs. If you are running a single pass model,
            %       you want this to be equal one.
            %   noiseRatio (double)
            %       Value between 0.0 and 1.0
            %       It indicates the percentage of noise that will be
            %   isTiedWeight (bool) [optional]
            %       On a Tied Weight training, after the train the weights
            %       after the middle layer will be a transpose version of
            %       the weights before the middle layer. The bias is still
            %       kept. This make the network find it hard to train, and
            %       that's is good when we are preparing the network for
            %       another kind of data.
            greddyLayerWiseTrain@AutoEncoder(self, X, nEpochs, noiseRatio);
        end
    end
end


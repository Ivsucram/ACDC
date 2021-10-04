classdef AutoEncoder < NeuralNetwork
    %AutoEncoder
    %   This object mimics the behavior of a Auto Encoder network, which is
    %   a Neural Network that has the output equal to input.
    %   This object has elastic habilities, being able to grow and prune
    %   nodes automatically.
    %   TODO: Provide the paper or study material for the Auto Encoder
     
    properties (Access = protected)
        greedyLayerBias       = [];
        greedyLayerOutputBias;
    end
    methods (Access = public)
        function self = AutoEncoder(layers)
            %   AutoEncoder
            %   layers (array)
            %       This array describes a FeedForward Network structure by
            %       the number of layers on it.
            %       An FFNN with an input layer of 8 nodes, a hidden layer
            %       of 10 nodes and an output layer of 3 nodes would be
            %       described by [8 10 3].
            %       An FFNN with an input layer of 784 nodes, a hidden
            %       layer 1 of 800 nodes, a hidden layer 2 of 400 nodes and
            %       an output layer of 10 nodes would be described as [784 800 400 10]
            
            self@NeuralNetwork(layers);
            self.outputActivationFunctionLossFunction = self.ACTIVATION_LOSS_FUNCTION_SIGMOID_MSE();
        end        
        
        function test(self, X)
            % test
            %   See test@NeuralNetwork
            %   X (matrix)
            %       Input and output data
            test@NeuralNetwork(self, X, X)
        end
        
        function grow(self, layerNo)
            grow@NeuralNetwork(self, layerNo);
            self.growGreedyLayerBias(layerNo);
        end
        
        function prune(self, layerNo, nodeNo)
            prune@NeuralNetwork(self, layerNo, nodeNo);
            self.pruneGreedyLayerBias(layerNo, nodeNo);
        end
        
        function growGreedyLayerBias(self, layerNo)
            b = layerNo; %readability
            if b == (numel(self.layers) - 1)
                self.greedyLayerOutputBias = [self.greedyLayerOutputBias normrnd(0, sqrt(2 / (self.layers(end-1) + 1)))];
            else
                self.greedyLayerBias{b} = [self.greedyLayerBias{b} normrnd(0, sqrt(2 / (self.layers(b) + 1)))];
            end
            
        end
        
        function growLayer(self, option, numberOfNodes)
            if option == self.CREATE_MIRRORED_LAYER()
                nhl = self.nHiddenLayers + 1;
                growLayer@NeuralNetwork(self, self.CREATE_LAYER_BY_ARGUMENT(), numberOfNodes);
                growLayer@NeuralNetwork(self, self.CREATE_LAYER_BY_ARGUMENT(), self.layers(nhl));
            else
                growLayer@NeuralNetwork(self, option, numberOfNodes);
                self.greedyLayerBias{size(self.greedyLayerBias, 2) + 1} = self.greedyLayerOutputBias;
                self.greedyLayerOutputBias = normrnd(0, sqrt(2 / (self.layers(end-1) + 1)));
            end
        end
        
        function pruneGreedyLayerBias(self, layerNo, nodeNo)
            b = layerNo; % readability
            n = nodeNo;   %readability
            if b == (numel(self.layers) - 1)
                self.greedyLayerOutputBias(n) = [];
            else
                self.greedyLayerBias{b}(n) = [];
            end
        end
        
        function greddyLayerWiseTrain(self, X, nEpochs, noiseRatio)
            %greddyLayerWiseTrain
            %   Performs Greedy Layer Wise train
            %   X (matrix)
            %       Input and output data
            %   nEpochs (integer)
            %       The number of epochs which the greedy layer wise train
            %       will occurs. If you are running a single pass model,
            %       you want this to be equal one.
            if nargin == 3
                noiseRatio = 0;
            end
%             disp(self.layers)
            for i = 1 : numel(self.layers) - 1
                self.forwardpass(X);
                trainingX = self.layerValue{i};
                Xnoise = (rand(size(trainingX)) >= noiseRatio) .* trainingX;
                
                if i > self.nHiddenLayers
                    nn = NeuralNetwork([self.layers(i) self.layers(end) self.layers(i)]);
                else
                    nn = NeuralNetwork([self.layers(i) self.layers(i+1) self.layers(i)]);
                end
                nn.outputActivationFunctionLossFunction = self.ACTIVATION_LOSS_FUNCTION_SIGMOID_MSE();
                
                if i > self.nHiddenLayers
                    nn.weight{1}    = self.outputWeight;
                    nn.bias{1}      = self.outputBias;
                    nn.outputWeight = self.outputWeight';
                    if isempty(self.greedyLayerOutputBias)
                        self.greedyLayerOutputBias = normrnd(0, sqrt(2 / (size(self.outputWeight', 2) + 1)),...
                                                             1, size(self.outputWeight', 1));
                        nn.outputBias   = self.greedyLayerOutputBias;
                    else
                        nn.outputBias   = self.greedyLayerOutputBias;
                    end
                else
                    nn.weight{1}    = self.weight{i};
                    nn.bias{1}      = self.bias{i};
                    nn.outputWeight = self.weight{i}';
                    try
                        nn.outputBias   = self.greedyLayerBias{i};
                    catch
                        self.greedyLayerBias{i} = normrnd(0, sqrt(2 / (size(self.weight{i}', 2) + 1)),...
                                                          1, size(self.weight{i}', 1));
                        nn.outputBias   = self.greedyLayerBias{i};
                    end
                end
                
                for j = 1 : nEpochs
                    nn.train(Xnoise, trainingX);
                end
                
                if i > self.nHiddenLayers
                    self.outputWeight = nn.weight{1};
                    self.outputBias   = nn.bias{1};
                else
                    self.weight{i} = nn.weight{1};
                    self.bias{i}   = nn.bias{1};
                end
            end
        end
        
        function loss = updateWeightsByKullbackLeibler(self, Xs, Xt, GAMMA)
            if nargin == 3
                GAMMA = 0.0001;
            end
            loss = updateWeightsByKullbackLeibler@NeuralNetwork(self, Xs, Xs, Xt, Xt, GAMMA);
        end
    end
    methods (Access = protected)
        function BIAS2 = computeBIAS2(~, Ez, y)
            %getBIAS2
            %   The way AutoEncoders calculata its BIAS2 value per layer is
            %   different than normal neural networks. Because we use
            %   sigmoid as our output activation function, and because the
            %   error is too high, we prefer use mean as a way to squish
            %   the bias2
            %   Ez (double, vector or matrix)
            %       Expected outbound value of that layer
            %   y (double, vector or matrix)
            %       A target class
            %
            %   return BIAS2 = The network squared BIAS
            BIAS2 = mean((Ez - y') .^ 2);
        end
        function var = computeVAR(~, Ez, Ez2)
            %getVAR
            %   The way AutoEncoders calculata its VAR value per layer is
            %   different than normal neural networks. Because we use
            %   sigmoid as our output activation function, and because the
            %   error is too high, we prefer use mean as a way to squish
            %   the bias2
            %   Ez (double, vector or matrix)
            %       Expected outbound value of that layer
            %   Ez2 (double, vector or matrix)
            %       Expected outbound squared value of that layer
            %
            %   return VAR = The network VAR (variance)
            var = mean(Ez2 - Ez .^ 2);
        end
    end
end


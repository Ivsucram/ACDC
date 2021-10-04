classdef NeuralNetwork < handle & ElasticNodes & NeuralNetworkConstants
    %NEURALNETWORK It encapsulate a common MLP (Multilayer Perceptron, aka
    %Feedforward network)
    %   This object has the main attributes a Neural Network needs to
    %   operate, along with its main functions/behaviors. Some extra
    %   behaviors were built in order to achieve research goals.
    %
    %   This class features elastic network width by ElasticNodes
    %   inheritance. Network edith adaptation supports automatic generation
    %   of new hidden nodes and prunning of inconsequential nodes. This
    %   mechanism is controlled by the NS (Network Significance) method
    %   which estimates the network generalization power in terms of bias
    %   and variance.
    
    %% Standard Neural Network public properties
    properties (Access = public) 
        layers %Layers of a standard neural network
        layerValue % Layer (Input and Hidden Layer) values
        outputLayerValue % Output layers values
        
        weight % Weights
        bias % Added bias
        momentum % Weight momentum
        biasMomentum
        
        outputWeight % Weights to output layer
        outputBias % Bias to output layer
        outputMomentum % Weight momentum from output layer
        outputBiasMomentum
        
        gradient % Gradients
        outputGradient % Gradients from output layers
        biasGradient;
        outputBiasGradient;
        
        activationFunction % Each real layer activation function
        outputActivationFunctionLossFunction % Each output activation function
        
        learningRate = 0.01; % Learning rate
        momentumRate = 0.95; % Momentum rate
        
        errorValue % Network error
        lossValue % Network Loss
        
        lambda = 0.001;
    end
    
    %% Standard Neural Network protected properties
    properties (Access = protected)
        nHiddenLayers % Number of hidden layers (i.e., not counting input and output layer
        
        inputSize % Size of input layer
        outputSize % Size of output layer
    end
    %% TODO define section name
    properties (Access = public)
        agmm
    end
    properties (Access = protected)
        isAgmmAble = false;
    end
    
    %% Metrics and performance public properties
    properties (Access = public)
        %test metrics
        sigma              % Network's prediction
        misclassifications % Number of misclassifications after test
        classificationRate % Classification rate after test
        residualError      % Residual error after test
        outputedClasses    % Classed outputed during classes
        trueClasses        % True target classes
    end
    
    %% Helpers protected properties
    properties (Access = protected)
        util = Util; % Caller for several util computations
    end
    
    %% Standard Neural Network public methods
    methods (Access = public)
        function self = NeuralNetwork(layers)
            %NeuralNetwork
            %   layers (array)
            %       This array describes a FeedForward Network structure by
            %       the number of layers on it.
            %       An FFNN with an input layer of 8 nodes, a hidden layer
            %       of 10 nodes and an output layer of 3 nodes would be
            %       described by [8 10 3].
            %       An FFNN with an input layer of 784 nodes, a hidden
            %       layer 1 of 800 nodes, a hidden layer 2 of 400 nodes and
            %       an output layer of 10 nodes would be described as [784 800 400 10]
            self@ElasticNodes(numel(layers) - 1);
            
            self.inputSize  = layers(1);
            self.outputSize = layers(end);
            
            self.layers = layers;
            self.nHiddenLayers = length(layers) - 2;
            
            for i = 1 : self.nHiddenLayers
                self.weight{i}       = normrnd(0, sqrt(2 / self.layers(i) + 1), [self.layers(i + 1), self.layers(i)]);
                self.bias{i}         = normrnd(0, sqrt(2 / self.layers(i) + 1), [1,                  self.layers(i + 1)]);
                self.momentum{i}     = zeros(size(self.weight{i}));
                self.biasMomentum{i} = zeros(size(self.bias{i}));
                self.activationFunction(i) = self.ACTIVATION_FUNCTION_SIGMOID();
            end
            self.outputWeight          = normrnd(0, sqrt(2 / self.layers(end) + 1), [self.layers(end), self.layers(end - 1)]);
            self.outputBias            = normrnd(0, sqrt(2 / self.layers(end) + 1), [1,                self.layers(end)]);
            self.outputMomentum        = zeros(size(self.outputWeight));
            self.outputBiasMomentum = zeros(size(self.outputBias));
            self.outputActivationFunctionLossFunction = self.ACTIVATION_LOSS_FUNCTION_SOFTMAX_CROSS_ENTROPY();
        end
                
        function feedforward(self, X, y)
%             % feedforward
            %   Perform the forwarding pass throughout the network and
            %   calculate the network error
            %   X (matrix)
            %       Input matrix
            %   y (matrix)
            %       Target matrix
            self.forwardpass(X)
            self.calculateError(y)
        end

        function forwardpass(self, X)
            % forwardpass
            %   Perform the forwarding pass throughout the network without
            %   calculate the network error, that's why it doesn't need the
            %   target class.
            %   Because of this, we can use this class just to populate the
            %   hidden layers from the source data
            %   X (matrix)
            %       Input matrix
            self.layerValue{1} = X;
            
            for i = 1 : self.nHiddenLayers
                previousLayerValueWithBias = [ones(size(self.layerValue{i}, 1), 1) self.layerValue{i}];
                switch self.activationFunction(i)
                    case self.ACTIVATION_FUNCTION_SIGMOID()
                        self.layerValue{i + 1} = sigmf(previousLayerValueWithBias * [self.bias{i}' self.weight{i}]', [1, 0]);
                    
                    case self.ACTIVATION_FUNCTION_TANH()
                        error('Not implemented');
                    
                    case self.ACTIVATION_FUNCTION_RELU()
                        error('Not implemented yet');
                    
                    case self.ACTIVATION_FUNCTION_LINEAR()
                        error('Not implemented yet');
                    
                    case self.ACTIVATION_FUNCTION_SOFTMAX()
                        error('Not implemented yet');
                end
                
            end
            
            previousLayerValueWithBias = [ones(size(self.layerValue{end}, 1), 1) self.layerValue{end}];
            switch self.outputActivationFunctionLossFunction
                case self.ACTIVATION_LOSS_FUNCTION_SIGMOID_MSE()
                    self.outputLayerValue = sigmf(previousLayerValueWithBias * [self.outputBias' self.outputWeight]', [1, 0]);
                
                case self.ACTIVATION_LOSS_FUNCTION_TANH()
                    error('Not implemented yet');
                
                case self.ACTIVATION_LOSS_FUNCTION_RELU()
                    error('Not implemented yet');
                
                case self.ACTIVATION_LOSS_FUNCTION_SOFTMAX_CROSS_ENTROPY()
                    self.outputLayerValue = previousLayerValueWithBias * [self.outputBias' self.outputWeight]';
                    self.outputLayerValue = exp(self.outputLayerValue - max(self.outputLayerValue, [], 2));
                    self.outputLayerValue = self.outputLayerValue./sum(self.outputLayerValue, 2);
                
                case self.ACTIVATION_LOSS_FUNCTION_LINEAR_CROSS_ENTROPY()
                    error('Not implemented yet');
            end
        end
        
        function backpropagate(self)
            %backpropagate
            % Perform back-propagation thoughout the network.
            % We assume that you already populate the hidden layers and the
            % network error by calling the feedforward method.

            dW = {zeros(1, self.nHiddenLayers + 1)};
            db = {zeros(1, self.nHiddenLayers + 1)};
            for i = self.nHiddenLayers : - 1 : 1
                if i == self.nHiddenLayers
                    % THIS IS THE GRADIENT OF THE LOSS FUNCTION
                    switch self.outputActivationFunctionLossFunction
                        case self.ACTIVATION_LOSS_FUNCTION_SIGMOID_MSE()
                            dW{i + 1} = - self.errorValue .* self.outputLayerValue .* (1 - self.outputLayerValue);
                            db{i + 1} = - sum(self.errorValue, 1)/size(self.errorValue, 1);
                        
                        case self.ACTIVATION_LOSS_FUNCTION_TANH()
                            error('Not implemented');
                        
                        case self.ACTIVATION_LOSS_FUNCTION_RELU()
                            error('Not implemented');
                        
                        case self.ACTIVATION_LOSS_FUNCTION_SOFTMAX_CROSS_ENTROPY()
                            dW{i + 1} = - self.errorValue;
                            db{i + 1} = - sum(self.errorValue, 1)/size(self.errorValue, 1);
                        
                        case self.ACTIVATION_LOSS_FUNCTION_LINEAR_CROSS_ENTROPY()
                            dW{i + 1} = - self.errorValue;
                            db{i + 1} = - sum(self.errorValue, 1)/size(self.errorValue, 1);
                    end
                    
                end
                
                switch char(self.activationFunction(i))
                    case self.ACTIVATION_FUNCTION_SIGMOID()
                        dActivationFunction = self.layerValue{i + 1} .* (1 - self.layerValue{i + 1});
                    case self.ACTIVATION_FUNCTION_TANH()
                        error('Not implemented');
                    
                    case self.ACTIVATION_FUNCTION_RELU()
                        error('Not implemented');
                    
                    case self.ACTIVATION_FUNCTION_LINEAR()
                        dActivationFunction = 1;
                    
                    case self.ACTIVATION_FUNCTION_SOFTMAX()
                        error('Not implemented');
                end
                
                if i == self.nHiddenLayers
                        z     = dW{i + 1} * self.outputWeight;
                        dW{i} = z .* dActivationFunction;
                        db{i} = sum(dW{i}, 1)/size(dW{i}, 1);
                else
                        z     = dW{i + 1} * self.weight{i + 1};
                        dW{i} = z .* dActivationFunction;
                        db{i} = sum(dW{i}, 1)/size(dW{i}, 1);
                end
                
            end
            
            self.outputGradient     = dW{end}' * self.layerValue{end};
            self.outputBiasGradient = db{end};
            for i = 1 : self.nHiddenLayers
                self.gradient{i}     = dW{i}' * self.layerValue{i};
                self.biasGradient{i} = db{i};
            end
        end
        
        function test(self, X, y)
            %test
            %   Test the neural network, getting its output by an ensemble
            %   composed of a selected numbers of outputLayers.
            %   It also has the ability to update the importance weight of
            %   each output layer, if necessary.
            %   X (matrix)
            %       Input matrix
            %   y (matrix)
            %       Target matrix
            
            self.feedforward(X, y);
            
            m = size(y, 1);
            [~, self.trueClasses] = max(y, [], 2);
            
            self.sigma = self.outputLayerValue;
            [rawOutput, outputtedClasses] = max(self.sigma, [], 2);
            self.misclassifications = find(outputtedClasses ~= self.trueClasses);
            self.classificationRate = 1 - numel(self.misclassifications) / m;
            self.residualError = 1 - rawOutput;
            self.outputedClasses = outputtedClasses;
        end
        
        function train(self, X, y, weightNo)
            %train
            %   Train the neural network performing 3 complete stages:
            %       - Feed-forward
            %       - Back-propagation
            %       - Weight updates
            %   X (matrix)
            %       Input matrix
            %   y (matrix)
            %       Target matrix
            %   weightNo (integer) [optional]
            %       You has the ability to define which weight and bias you
            %       want to update using backpropagation. This method will
            %       update only that weight and bias, even if there is
            %       weights and biases on layers before and after that.
            %       The number of the weight and bias you want to update.
            %       Remember that 1 indicates the weight and bias that get
            %       out of the input layer.
            self.feedforward(X,y);
            self.backpropagate();
            
            switch nargin
                case 4
                    self.trainWeight(weightNo);
                case 3
                    for i = self.nHiddenLayers + 1 : -1 : 1
                        self.trainWeight(i);
                    end
            end
        end
        
        function trainWeight(self,weightNo)
            %trainWeight
            %   This methods will only update a set of weights and biases.
            %   Normally you will not call this method directly, but will
            %   the method train as a middle man.
            %   weightNo (integer)
            %       The number of the weight and bias you want to update.
            %       Remember that 1 indicates the weight and bias that get
            %       out of the input layer.
            self.updateWeight(weightNo);
        end
    end
    
    %% Standard Neural Network private methods
    methods (Access = private)
        function updateWeight(self, weightNo)
            %updateWeights
            % Perform weight and bias update into a single weight and bias
            %   weightNo (integer)
            %       Number/Position of the weight/bias you want to update
            w = weightNo; %readability
            if w > self.nHiddenLayers
                dW = self.learningRate .* self.outputGradient;
                db = self.learningRate' .* self.outputBiasGradient;
                if self.momentumRate > 0
                    self.outputMomentum     = self.momentumRate * self.outputMomentum + dW;
                    self.outputBiasMomentum = self.momentumRate * self.outputBiasMomentum + db;
                    dW = self.outputMomentum;
                    db = self.outputBiasMomentum;
                end
                if true == false
                    self.outputWeight = (1 - self.learningRate * self.lambda) * self.outputWeight - dW;
                    self.outputBias   = (1 - self.learningRate * self.lambda) * self.outputBias   - db;
                    self.outputWeight = (1 - self.learningRate * self.lambda) * self.outputWeight - dW;
                    self.outputBias   = (1 - self.learningRate * self.lambda) * self.outputBias   - db;
                else
                    self.outputWeight = self.outputWeight - dW;
                    self.outputBias   = self.outputBias   - db;
                end
                
            else
                dW = self.learningRate .* self.gradient{w};
                db = self.learningRate' .* self.biasGradient{w};
                if self.momentumRate > 0
                    self.momentum{w}     = self.momentumRate * self.momentum{w} + dW;
                    self.biasMomentum{w} = self.momentumRate * self.biasMomentum{w} + db;
                    dW = self.momentum{w};
                    db = self.biasMomentum{w};
                end
                if true == false
                    self.weight{w} = (1 - self.learningRate * self.lambda) * self.weight{w} - dW;
                    self.bias{w}   = (1 - self.learningRate * self.lambda) * self.bias{w}   - db;
                else
                    self.weight{w} = self.weight{w} - dW;
                    self.bias{w}   = self.bias{w}   - db;
                end
                
            end
        end
        
        function calculateError(self, y)
            %calculateError
            %   Calculates the error.
            %   This method probably will be called by the feedforward
            %   method and seldom will be used standalone.
            m = size(y,1);
            
            %TODO: Add the possibility to input which error function we
            %want to use
            switch self.outputActivationFunctionLossFunction
                case self.ACTIVATION_LOSS_FUNCTION_SIGMOID_MSE()
                    self.errorValue = y - self.outputLayerValue;
                    self.lossValue  = 1 / 2 * sum(sum(self.errorValue .^ 2)) / m;
                
                case self.ACTIVATION_LOSS_FUNCTION_TANH()
                    error('Not implemented');
                
                case self.ACTIVATION_LOSS_FUNCTION_RELU()
                    error('Not implemented');
                
                case self.ACTIVATION_LOSS_FUNCTION_SOFTMAX_CROSS_ENTROPY()
                    self.errorValue = y - self.outputLayerValue;
                    self.lossValue  = - sum(sum(y .* log(self.outputLayerValue))) / m;
                
                case self.ACTIVATION_LOSS_FUNCTION_LINEAR_CROSS_ENTROPY()
                    error('Not implemented yet');
            end
        end
    end
    
    %% Standard Neural Network statistical metrics public methods
    methods (Access = public)
        function bias2 = computeNetworkBiasSquare(self, y)
            %computeNetworkBias
            %   Compute the Network Squared Bias in relation to a target
            %
            %   y (vector)
            %       A single target
            %   agmm (object)
            %       AGMM object
            %
            %   Returns
            %       The squared bias of the network related to this target
            dataMean = self.dataMean;
            dataStd  = self.dataStd;
            dataVar  = self.dataVar;
            self.nSamplesFeed = self.nSamplesFeed + 1;
            [~, ~, Ez, ~] = self.computeExpectedValues(self.nHiddenLayers + 1);
            bias2 = self.computeBIAS2(Ez, y);
            self.nSamplesFeed = self.nSamplesFeed - 1;
            self.dataMean = dataMean;
            self.dataStd  = dataStd;
            self.dataVar  = dataVar;
        end
        
        function var = computeNetworkVariance(self)
            %computeNetworkVariance
            %   Compute the Network Variance in relation to a target
            %
            %   agmm (object)
            %       AGMM object
            %
            %   Returns
            %       The squared bias of the network related to this target
            dataMean = self.dataMean;
            dataStd  = self.dataStd;
            dataVar  = self.dataVar;
            self.nSamplesFeed = self.nSamplesFeed + 1;
            [~, ~, Ez, Ez2] = self.computeExpectedValues(self.nHiddenLayers + 1);
            var = self.computeVAR(Ez, Ez2);
            self.nSamplesFeed = self.nSamplesFeed - 1;
            self.dataMean = dataMean;
            self.dataStd  = dataStd;
            self.dataVar  = dataVar;
        end
    end
    %% Standard Neural Networks extra public methods
    methods (Access = public)
        function loss = updateWeightsByKullbackLeibler(self, Xs, ys, Xt, yt, GAMMA)
            %updateWeightsByKullbackLeibler
            %   This method is used on Transfer Learning procedures. The
            %   idea is to approximate the source and target distributions.
            %   If you don't have access to the target domain classes, call
            %   this method from a generative model (AutoEncoder or
            %   DenoisingAutoEncoder)
            %   Xs (matrix)
            %       Source input
            %   ys (matrix)
            %       Source target
            %   Xt (matrix)
            %       Target input
            %   yt (matrix)
            %       Target target
            %   GAMMA (float)
            %       Regularizer coeficient
            if nargin == 5
                GAMMA = 0.0001;
            end
            
            nHL = self.nHiddenLayers + 1; %readability
            
            self.forwardpass(Xs);
            sourceLayerValue       = self.layerValue;
            sourceOutputLayerValue = self.outputLayerValue;
            
            self.forwardpass(Xt);
            targetLayerValue       = self.layerValue;
            targetOutputLayerValue = self.outputLayerValue;
                
            klLoss = 0;
            for i = nHL : -1 : 2
                klLoss = klLoss + self.util.KLDiv(sum(sourceLayerValue{i}), sum(targetLayerValue{i}))...
                       + self.util.KLDiv(sum(targetLayerValue{i}), sum(sourceLayerValue{i}));
            end
            if ~isfinite(klLoss)
                loss = 1000;
            else
                loss = klLoss;
            end
            
                        
            dSource{nHL} = (sourceOutputLayerValue - ys) .* sourceOutputLayerValue .* (1 - sourceOutputLayerValue);
            dTarget{nHL} = (targetOutputLayerValue - yt) .* targetOutputLayerValue .* (1 - targetOutputLayerValue);
            for i = 1 : self.nHiddenLayers
                dSource{i} = sourceLayerValue{i + 1} .* (1 - sourceLayerValue{i + 1});
                dTarget{i} = targetLayerValue{i + 1} .* (1 - targetLayerValue{i + 1});
            end
            
            for i = nHL : -1 : 1    
                if (i == nHL)
                    inboundSourceLayerValue  = sourceLayerValue{end};
                    inboundTargetLayerValue  = targetLayerValue{end};
                else
                    if (i == self.nHiddenLayers)
                        outboundWeight = self.outputWeight;
                    else
                        outboundWeight = self.weight{i + 1};
                    end
                    inboundSourceLayerValue  = sourceLayerValue{i};
                    inboundTargetLayerValue  = targetLayerValue{i};
                end
                
                if (i == nHL)                    
                    dW{i} = (2 * dSource{i}' * inboundSourceLayerValue)...
                          + (2 * dTarget{i}' * inboundTargetLayerValue);
                      
                    b     = sum((2 * dSource{i}) + (2 * dTarget{i}), 1);
                    db{i} = sum(b, 1) / size(b, 1);
                else
                    dW{i} = ((2 * dSource{i + 1} * outboundWeight) .* dSource{i})' * inboundSourceLayerValue...
                          + ((2 * dTarget{i + 1} * outboundWeight) .* dTarget{i})' * inboundTargetLayerValue;
                          
                    b     = ((2 * dSource{i + 1} * outboundWeight) .* dSource{i})...
                          + ((2 * dTarget{i + 1} * outboundWeight) .* dTarget{i});
                          
                    db{i} = sum(b, 1) / size(b, 1);
                end
            end
            for i = nHL : -1 : 1
                if (i == nHL)
                    self.outputWeight =  self.outputWeight - self.learningRate * dW{i};
                    self.outputBias   = self.outputBias   - self.learningRate * db{i};
                else
                    self.weight{i} = self.weight{i} - self.learningRate * dW{i};
                    self.bias{i}   = self.bias{i}   - self.learningRate * db{i};
                end
            end
        end
    end
    
    %% Elastic/Evolving Neural Network public methods
    methods (Access = public)        
        function widthAdaptationStepwise(self, layerNo, y)
            %widthAdaptationStepwise
            %   Performs network width adaptation in a specific layer,
            %   stepwise (it means that it execute one row at a time).
            %   Also, this method assume that you already passe the input
            %   data through the model via forwardpass procedure.
            %   layerNo (integer)
            %       Number of the layer you want to perform width
            %       adaptation. This is normally a hidden layer.
            %   y (double or vector)
            %       Double, if you are performing regression
            %       Vector if you are performing classification
            %       The targer data to be used as validation
            nhl = layerNo; % readability
            
            self.nSamplesFeed = self.nSamplesFeed + 1;
            self.nSamplesLayer(nhl) = self.nSamplesLayer(nhl) + 1;
            
            [Ex, ~, Ey, Ey2] = computeExpectedValues(self, nhl);
            
            bias2 = self.computeBIAS2(Ey, y);
            var   = self.computeVAR(Ey, Ey2);
            
            [self.meanBIAS(nhl), self.varBIAS(nhl), self.stdBIAS(nhl)] ...
                    = self.util.recursiveMeanStd(bias2, self.meanBIAS(nhl), self.varBIAS(nhl), self.nSamplesFeed);
                
            [self.meanVAR(nhl), self.varVAR(nhl), self.stdVAR(nhl)] ...
                    = self.util.recursiveMeanStd(var, self.meanVAR(nhl), self.varVAR(nhl), self.nSamplesFeed);
                
            if self.nSamplesLayer(nhl) <= 1 || self.growable(nhl) == true
                self.minMeanBIAS(nhl) = self.meanBIAS(nhl);
                self.minStdBIAS(nhl)  = self.stdBIAS(nhl);
            else
                self.minMeanBIAS(nhl) = min(self.minMeanBIAS(nhl), self.meanBIAS(nhl));
                self.minStdBIAS(nhl)  = min(self.minStdBIAS(nhl), self.stdBIAS(nhl));
            end
            
            if self.nSamplesLayer(nhl) <= self.inputSize + 1 || self.prunable{nhl}(1) ~= 0
                self.minMeanVAR(nhl) = self.meanVAR(nhl);
                self.minStdVAR(nhl)  = self.stdVAR(nhl);
            else
                self.minMeanVAR(nhl) = min(self.minMeanVAR(nhl), self.meanVAR(nhl));
                self.minStdVAR(nhl)  = min(self.minStdVAR(nhl), self.stdVAR(nhl));
            end
            
            self.BIAS2{nhl} = [self.BIAS2{nhl} self.meanBIAS(nhl)];
            self.VAR{nhl}   = [self.VAR{nhl} self.meanVAR(nhl)];
            
            self.growable(nhl) = self.isGrowable(nhl, bias2);
            if nargin == 3
                self.prunable{nhl} = self.isPrunable(nhl, var, Ex, self.PRUNE_SINGLE_LEAST_CONTRIBUTION_NODES());
            elseif nargin == 4
                self.prunable{nhl} = self.isPrunable(nhl, var, Ex, self.PRUNE_MULTIPLE_NODES_WITH_CONTRIBUTION_BELOW_EXPECTED());
            end
        end 
        
        function grow(self, layerNo)
            %grow
            %   Add 1 new node to a hidden layer. Because of this, it will
            %   add 1 extra weight and bias at the outbound row and 1 extra
            %   weight at the inbound row.
            %   layerNo (integer)
            %       Number of the layer you want to add a node.
            self.layers(layerNo) = self.layers(layerNo) + 1;
            if layerNo > 1
                self.growWeightRow(layerNo - 1)
                self.growBias(layerNo - 1);
            end
            if layerNo < numel(self.layers)
                self.growWeightColumn(layerNo)
            end
        end
        
        function prune(self, layerNo, nodeNo)
            %prune
            %   Remove 1 node from the hidden layer. Because of this, it
            %   will remove 1 weight and bias at the outbound row and 1
            %   weight from the inbound row.
            %   layerNo (integer)
            %       Number of the layer you want to add a node.
            %   nodeNo (integer)
            %       Position of the node to be removed
            self.layers(layerNo) = self.layers(layerNo) - 1;
            if layerNo > 1
                self.pruneWeightRow(layerNo - 1, nodeNo);
                self.pruneBias(layerNo - 1, nodeNo);
            end
            if layerNo < numel(self.layers)
                self.pruneWeightColumn(layerNo, nodeNo);
            end
        end
    end
    
    %% Elastic/Evolving Neural Network protected methods
    methods (Access = protected) 
        function isGrowable = isGrowable(self, layerNo, BIAS2)
            %isGrowable
            %   Evaluate if a specific layer need a node added to have its
            %   network significance parameters stable
            %   layerNo (integer)
            %       Layer which the evaluation will be performed. Usually
            %       it is a hidden layer.
            %   BIAS2 (double)
            %       The squished BIAS2 of that layer at that time
            %
            %   returns a boolean indicating if that layer is ready to
            %   receive a new node or not.
            nhl = layerNo; %readability
            isGrowable = false;
            ALPHA_1 = 1.25;
            ALPHA_2 = 0.75;
            
            current    = (self.meanBIAS(nhl) + self.stdBIAS(nhl));
            biased_min = (self.minMeanBIAS(nhl)...
                       + (ALPHA_1 * exp(-BIAS2) + ALPHA_2)...
                       * self.minStdBIAS(nhl));
            
            if self.nSamplesLayer(nhl) > 1 && current >= biased_min
                isGrowable = true;
            end
        end
        
        function prunableNodes = isPrunable(self, layerNo, VAR, expectedY, option)
            %isPrunable
            %   Evaluate if a specific layer need a node pruned to have its
            %   network significance parameters stable
            %   layerNo (integer)
            %       Layer which the evaluation will be performed. Usually
            %       it is a hidden layer.
            %   VAR (double)
            %       The squished VAR of that layer at that time
            %   expectedY (vector)
            %       See self.getExpectedValues
            %       This value is used to determine the node with minimum
            %       contribution to the network.
            %   option (string)
            %       'least_contribution': In case the pruning rule get
            %       approved, it will return the position for the least
            %       contributing node.
            %       'below_contribution': In case the pruning rule get
            %       approved, it will return an array with the position for
            %       all nodes that have the contribution below a certain
            %       quantity
            %
            %   returns a integer indicating the position of which node
            %   should be removed from that layer. If no node should be
            %   removed, returns zero instead.
            nhl = layerNo; %readability
            prunableNodes = 0;
            ALPHA_1 = 2.5;
            ALPHA_2 = 1.5;
            
            current = (self.meanVAR(nhl) + self.stdVAR(nhl));
            biased_min = (self.minMeanVAR(nhl)...
                       + (ALPHA_1 * exp(-VAR) + ALPHA_2)...
                       * self.minStdVAR(nhl));
            
            if self.growable(nhl) == false ...
                    && self.layers(nhl) > 1 ...
                    && self.nSamplesLayer(nhl) > self.inputSize + 1 ...
                    && current >= biased_min
                
                switch option
                    case self.PRUNE_SINGLE_LEAST_CONTRIBUTION_NODES()
                        [~, prunableNodes] = min(expectedY);
                    case self.PRUNE_MULTIPLE_NODES_WITH_CONTRIBUTION_BELOW_EXPECTED()
                        nodesToPrune = expectedY <= abs(mean(expectedY) - var(expectedY));
                        if sum(nodesToPrune)
                            prunableNodes = find(expectedY <= abs(mean(expectedY) - var(expectedY)));
                        else
                            [~, prunableNodes] = min(expectedY);
                        end
                end
            end
        end
        
        function growWeightRow(self, weightArrayNo)
            %growWeightRow
            %   Add 1 extra weight at the inbound row.
            %   weightArrayNo (integer)
            %       Weight position
            w = weightArrayNo; % readability
            if w > numel(self.weight)
                [n_in, n_out] = size(self.outputWeight);
                n_in = n_in + 1;
                self.outputWeight = [self.outputWeight; normrnd(0, sqrt(2 / (n_in)), [1, n_out])];
                self.outputMomentum = [self.outputMomentum; zeros(1, n_out)];
            else               
                [n_in, n_out] = size(self.weight{w});
                n_in = n_in + 1;
                self.weight{w} = [self.weight{w}; normrnd(0, sqrt(2 / (n_in)), [1, n_out])];
                self.momentum{w} = [self.momentum{w}; zeros(1, n_out)];
            end
        end
        
        function growWeightColumn(self, weightArrayNo)
            %growWeightColumn
            %   Add 1 extra weight at the outbound column.
            %   weightArrayNo (integer)
            %       Weight position
            w = weightArrayNo; % readability
            if w > numel(self.weight)
                [n_out, n_in] = size(self.outputWeight);
                n_in = n_in + 1;
                self.outputWeight = [self.outputWeight normrnd(0, sqrt(2 / (n_in)), [n_out, 1])];
                self.outputMomentum = [self.outputMomentum zeros(n_out, 1)];
            else              
                [n_out, n_in] = size(self.weight{w});
                n_in = n_in + 1;
                self.weight{w} = [self.weight{w} normrnd(0, sqrt(2 / (n_in)), [n_out, 1])];
                self.momentum{w} = [self.momentum{w} zeros(n_out, 1)];
            end
        end
        
        function pruneWeightRow(self, weightNo, nodeNo)
            %pruneWeightRow
            %   Remove 1 weight from the inbound row.
            %   weightNo (integer)
            %       Weight position
            %   nodeNo (integer)
            %       Position of the node to be removed
            w = weightNo; % readability
            n = nodeNo;   %readability
            if w > numel(self.weight)
                self.outputWeight(n, :)   = [];
                self.outputMomentum(n, :) = [];
            else
                self.weight{w}(n, :)   = [];
                self.momentum{w}(n, :) = [];
            end
        end
        
        function pruneWeightColumn(self, weightNo, nodeNo)
            %pruneWeightColumn
            %   Remove 1 weight from the outbound column
            %   weightArrayNo (integer)
            %       Weight position
            %   nodeNo (integer)
            %       Position of the node to be removed
            w = weightNo; % readability
            n = nodeNo;   %readability
            if w > numel(self.weight)
                self.outputWeight(:, n)   = [];
                self.outputMomentum(:, n) = [];
            else
                self.weight{w}(:, n)   = [];
                self.momentum{w}(:, n) = [];
            end
        end
        
        function growBias(self, biasArrayNo)
            %growBias
            %   Add 1 extra bias at the inbound row.
            %   biasArrayNo (integer)
            %       Bias position
            b = biasArrayNo; %readability
            if b > numel(self.weight)
                self.outputBias         = [self.outputBias normrnd(0, sqrt(2 / (self.layers(end) + 1)))];
                self.outputBiasMomentum = [self.outputBiasMomentum 0];
            else
                self.bias{b} = [self.bias{b} normrnd(0, sqrt(2 / (self.layers(b) + 1)))];
                self.biasMomentum{b} = [self.biasMomentum{b} 0];
            end
        end
        
        function pruneBias(self, biasArrayNo, nodeNo)
            %pruneBias
            %   Remove 1 bias from the inbound row.
            %   biasArrayNo (integer)
            %       Bias position
            %   nodeNo (integer)
            %       Position of the node to be removed
            b = biasArrayNo; % readability
            n = nodeNo;   %readability
            if b > numel(self.weight)
                self.outputBias(n) = [];
                self.outputBiasMomentum(n) = [];
            else
                self.bias{b}(n) = [];
                self.biasMomentum{b}(n) = [];
            end
        end
        
        function [Ex, Ex2, Ez, Ez2] = computeExpectedValues(self, nHiddenLayer)
            %computeExpectedValues
            %   Compute statisticals expectations values for a specific
            %   hidden layer
            %
            %   Returns Ex  = Expected value of that layer
            %           Ex2 = Expected squared value of that layer
            %           Ez  = Expected outbound value of that layer
            %           Ez2 = Expected outbound squared value of that layer
            nhl = nHiddenLayer; %readability
            x = self.layerValue{1};
            [self.dataMean, self.dataVar, self.dataStd]...
                = self.util.recursiveMeanStd(x, self.dataMean, self.dataVar, self.nSamplesFeed);
            
            if self.isAgmmAble
                Ex  = 0;
                Ex2 = 0;
                for m = 1 : self.agmm.M()
                    gmm = self.agmm.gmmArray(m);
                    [tempEx, tempEx2] = computeInboundExpectedValues(self, nhl, gmm);
                    Ex  = Ex  + tempEx;
                    Ex2 = Ex2 + tempEx2;
                end
            else
                [Ex, Ex2] = computeInboundExpectedValues(self, nhl);
            end

            [Ez, Ez2] = computeOutboundExpectedValues(self, Ex, Ex2);
        end
        
        function [Ex, Ex2] = computeInboundExpectedValues(self, layerNo, gmm)
            %computeInboundExpectedValues
            %   Compute statisticals expectations values for a specific
            %   hidden layer
            %   nHiddenLayer (integer)
            %       layer to be evaluated
            %
            %   Returns Ex  = Expected value of that layer
            %           Ex2 = Expected squared value of that layer
            nhl = layerNo - 1; %readability
            if nhl == 1
                inference = 1;
                center    = self.dataMean;
                std       = self.dataStd;
                
                if nargin == 3
                    inference = gmm.weight;
                    center    = gmm.center;
                    std       = sqrt(gmm.var);
                end
                
                py = self.util.probit(center, std);
                Ex = inference * sigmf(self.weight{1} * py' + self.bias{1}', [1, 0]);
            else
                [Ex, ~] = self.computeInboundExpectedValues(nhl);
                weight_ = self.outputWeight;
                bias_   = self.outputBias;
                
                if nhl < self.nHiddenLayers + 1
                    weight_ = self.weight{nhl};
                    bias_   = self.bias{nhl};
                end                
                
                Ex = sigmf(weight_ * Ex + bias_', [1, 0]);
            end
            Ex2 = Ex .^ 2;
        end
        
        function [Ez, Ez2] = computeOutboundExpectedValues(self, Ex, Ex2)
            %computeOutboundExpectedValues
            %   Compute statisticals expectations values for a specific
            %   hidden layer
            %   Ey (double, vector or matrix)
            %       Expected value
            %   Ey2 (double, vector or matrix)
            %       Expected squared value
            %
            %   Returns Ez  = Expected outbound value of that layer
            %           Ez2 = Expected outbound squared value of that layer
            Ez = self.outputWeight * Ex + self.outputBias';
            Ez = exp(Ez - max(Ez));
            Ez = Ez ./ sum(Ez);

            Ez2 = self.outputWeight * Ex2 + self.outputBias';
            Ez2 = exp(Ez2 - max(Ez2));
            Ez2 = Ez2 ./ sum(Ez2);
        end
        
        function NS = computeNetworkSignificance(self, Ez, Ez2, y)
            %computeNetworkSignificance
            %   Compute the current Network Significance of the model in
            %   respect to a target
            %   Ez (double, vector or matrix)
            %       Expected outbound value of that layer
            %   Ez2 (double, vector or matrix)
            %       Expected outbound squared value of that layer
            %   y (double, vector or matrix)
            %       A target class
            %
            %   return NS = The network significance
            NS = self.computeBIAS2(Ez, z) + self.computeVAR(Ez, Ez2);
        end
        
        function BIAS2 = computeBIAS2(~, Ez, y)
            %computeBIAS2
            %   Compute the current BIAS2 of the model wrt a target
            %   Ez (double, vector or matrix)
            %       Expected outbound value of that layer
            %   y (double, vector or matrix)
            %       A target class
            %
            %   return BIAS2 = The network squared BIAS
            BIAS2 = norm((Ez - y') .^2 , 'fro');
        end
        
        function VAR = computeVAR(~, Ez, Ez2)
            %computeVAR
            %   Compute the current VAR of the model
            %   Ez (double, vector or matrix)
            %       Expected outbound value of that layer
            %   Ez2 (double, vector or matrix)
            %       Expected outbound squared value of that layer
            %
            %   return VAR = The network VAR (variance)
            VAR = norm(Ez2 - Ez .^ 2, 'fro');
        end        
    end
    %% GIVE A NAME TO THIS SECTION
    methods (Access = public)
        function agmm = runAgmm(self, x, y)
            
            bias2 = self.computeNetworkBiasSquare(y);
            
            self.agmm.run(x, bias2);
            
            agmm = self.agmm;
        end
    end
    
    %% Getters and Setters
    methods (Access = public)
        function setAgmm(self, agmm)
            %setAgmm
            %   You can use this method to set your own AGMM to this
            %   network
            %   agmm (AGMM)
            %       The AGMM you want to set to this network.
            self.isAgmmAble = true;
            self.agmm = agmm;
        end
        
        function agmm = getAgmm(self)
            %getAgmm
            %   Gets the agmm that the network is using. If the network has
            %   an empty agmm or is not using a agmm, it will enable AGMM
            %   and return to you a new AGMM
            if isempty(self.agmm) || self.isAgmmAble == false
                self.enableAgmm();
            end
            agmm = self.agmm;
        end
        
        function enableAgmm(self)
            %enableAgmm
            %   Tell the network that it will use AGMM from now on
            %   It also creates a random AGMM. If you want to use your own
            %   AGMM, make sure to use setAgmm method afterwards
            self.isAgmmAble = true;
            self.agmm = AGMM();
        end
        
        function disableAgmm(self)
            %disableAgmm
            %   Tell the network that it will NOT use AGMM frmo now on.
            %   It deletes the agmm that was attached to this model. If you
            %   want to keep track of that agmm, make sure to load it into
            %   some variable using the getAgmm method.
            self.isAgmmAble = false;
            self.agmm = [];
        end
        
        function nHiddenLayers = getNumberHiddenLayers(self)
            %getNumberHiddenLayers
            %   Return the number of hidden layers in the network
            %
            %   Returns
            %       nHiddenLayers (integer): Number of hidden layers
            nHiddenLayers = self.nHiddenLayers;
        end
    end
end


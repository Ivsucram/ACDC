classdef DataManipulator < handle
    %DataManipulator It manipulates and prepare data
    %   It manipulates and prepare data used to train and test our research
    %   models.
    %   It is already prepared to load and interact with mostly of the data
    %   used in our lab.
    properties (Access = public)
        data = []; % Whole dataset
        nFeatures = 0; % Number of features from the dataset
        nClasses = 0; % Number of classes from the dataset
        
        nFoldElements = 0; % Number of elements per fold
        nMinibatches = 0; % Number of minibatches
        
        source = {}; % Souce data
        target = {}; % Target data
    end
    properties (Access = private)
        X  = {}; % Input data
        y  = {}; % Class data
        Xs = {}; % Source input data
        ys = {}; % Source class data
        Xt = {}; % Target input data
        yt = {}; % Target class data
        
        permutedX = {}; % Permutted Input data
        permutedy = {}; % Permutted Class data
        
        indexPermutation = {}; % Permuttation index (in order to know if it source or target)
        
        dataFolderPath = '';
    end
    
    methods (Access = public)
        function self = DataManipulator(dataFolderPath)
            self.dataFolderPath = dataFolderPath;
        end
        
        function loadSourceCSV(self, dataset)
            self.loadCustomCSV(join([dataset, '_source.csv']))
        end
        
        function loadTargetCSV(self, dataset)
            self.loadCustomCSV(join([dataset, '_target.csv']))
        end
        
        function loadCustomCSV(self, filename)
            self.data = [];
            self.data = csvread(strcat(self.dataFolderPath, filename));
            self.checkDatasetEven();
            self.data = double(self.data);
            self.nFeatures = size(self.data, 2) - 1;
            self.nClasses = 1;
            self.X = self.data(:,1:end-self.nClasses);
            self.y = self.data(:,self.nFeatures+1:end);
            self.nClasses = max(self.y);
            
            y_one_hot = zeros(size(self.y, 1), self.nClasses);
            for i = 1 : self.nClasses
                rows = self.y == i;
                y_one_hot(rows, i) = 1;
            end
            self.y = y_one_hot;
            self.data = [self.X self.y];
        end
        
        function normalize(self)
            %normalize
            %   Normalize every feature between 0 and 1
            fprintf('Normalizing data\n');
            for i = 1 : self.nFeatures
                self.data(:, i) = (self.data(:, i) - min(self.data(:, i), [], 'all'))/max(self.data(:, i), [], 'all');
            end
            
            
            self.X = self.data(:, 1 : self.nFeatures);
            self.y = self.data(:, self.nFeatures + 1 : end);
        end
        
        function splitAsSourceTargetStreams(self, nFoldElements, method, samplingRatio)
            %splitAsSourceTargetStreams
            %   Split the function to simulate a Multistream classification
            %   input domains.
            %   In a Multistream classification problem, we consider that
            %   two different but related processes generate data
            %   continuously from a domain D (in this case, self.data). The
            %   first process operates in a supervised environment, i.e.,
            %   all the data instances that are generated from the first
            %   process are labeled. On the contraty, the second process
            %   generates unlabeled data from the same domain. The stream
            %   of data generated form the above processes are called the
            %   source stream and the target stream.
            %   This functions will return label for the target stream,
            %   which the user should only use for ensemble evaluation
            %   purposes
            %   nFoldElements (integer)
            %       Both source and target data will be splited in chunks
            %       of data containing n elements per chunk/fold.
            %       If you only want one chunk, pass zero or size(data,1)
            %       as argument.
            %   method (string)
            %       What kind of method will be used to generated
            %       distribute the data into source and target. Usually,
            %       Multistream Classification problems distribute the data
            %       using some bias probability.
            %       Options:
            %           'none': Source and Target streams will be splited on
            %           half
            %           'dallas_1: Source and Target streams will be splited
            %           on half using the bias described by paper "An
            %           adaptive framework for multistream classification"
            %           from the CS deparment of the university of Texas at
            %           Dallas
            %           'dallas_2:' Source and Target streams will be
            %           splited on half using the bias described by paper
            %           "FUSION - An online method for multistream
            %           classification" from the university of Texas at
            %           Dallas.
            %   samplingRatio (double)
            %       Value in the interval [0.0,1.0] which describes the
            %       percentage of sampling that would go to Source Stream.
            %       Target will have 1 - n percentagem of data.
            if nFoldElements == 0
                self.nFoldElements = length(self.data);
            else
                self.nFoldElements = nFoldElements;
            end
            
            switch method
                case 'none'
                    self.splitAsSourceTargetStreams_none(self.nFoldElements, samplingRatio);
                case 'dallas_1'
                    self.splitAsSourceTargetStreams_dallas1(self.nFoldElements, samplingRatio);
                case 'dallas_2'
                    self.splitAsSourceTargetStreams_dallas2(self.nFoldElements, samplingRatio);
            end
            
            self.createXsYsXtYt()
        end
        
        function X = getX(self, idx)
            X = self.X(idx,:);
        end
        
        function y = getY(self, idx)
            y = self.y(idx,:);
        end
        
        function Xs = getXs(self, nMinibatch)
            %getXs
            %   Get the input matrix from a specific source data stream.
            %   The source stream will be only created when we are dealing 
            %   with a dataset that was splitted into source and target 
            %   data streams.
            %   nMinibatch (integer)
            %       The minibatch iteration
            Xs = self.Xs{nMinibatch};
        end
        function ys = getYs(self, nMinibatch)
            %getXs
            %   Get the target matrix from a specific source data stream.
            %   The source stream will be only created when we are dealing 
            %   with a dataset that was splitted into source and target 
            %   data streams.
            %   nMinibatch (integer)
            %       The minibatch iteration
            ys = self.ys{nMinibatch};
        end
        function Xt = getXt(self, nMinibatch)
            %getXt
            %   Get the input matrix from a specific target data stream.
            %   The target stream will be only created when we are dealing 
            %   with a dataset that was splitted into source and target 
            %   data streams.
            %   nMinibatch (integer)
            %       The minibatch iteration
            Xt = self.Xt{nMinibatch};
        end
        function yt = getYt(self, nMinibatch)
            %getXs
            %   Get the target matrix from a specific target data stream.
            %   The target stream will be only created when we are dealing 
            %   with a dataset that was splitted into source and target 
            %   data streams.
            %   nMinibatch (integer)
            %       The minibatch iteration
            yt = self.yt{nMinibatch};
        end
    end
    methods (Access = private)
        function splitAsSourceTargetStreams_none(self, elementsPerFold, samplingRatio)
            %splitAsSourceTargetStreams_none
            %   Split the function to simulate a Multistream classification
            %   input domains.
            %
            %   Source and Target streams will be splited on half
            %
            %   nFoldElements (integer)
            %       Both source and target data will be splited in chunks
            %       of data containing n elements per chunk/fold.
            %       If you only want one chunk, pass zero or size(data,1)
            %       as argument.
            %   samplingRatio (double)
            %       Value in the interval [0.0,1.0] which describes the
            %       percentage of sampling that would go to Source Stream.
            %       Target will have 1 - n percentagem of data.
            [rowsNumber, ~] = size(self.data);
            
            self.nFoldElements = elementsPerFold;
            
            j = 0;
            b = 1;
            i = 1;
            source = [];
            while i < size(self.data, 1)
                while j < self.nFoldElements && i < size(self.data, 1)
                    source = [source; self.data(i,:)];
                    j = j + 1;
                    i = i + 1;
                end
                self.source{b} = source;
                self.target{b} = source;
                source = [];
                j = 0;
                b = b + 1;
            end
            
            self.nMinibatches = b - 1;
        end
        function splitAsSourceTargetStreams_dallas1(self, elementsPerFold, samplingRatio)
            %splitAsSourceTargetStreams_dallas1
            %   Split the function to simulate a Multistream classification
            %   input domains.
            %
            %   Source and Target streams will be splited on half using the
            %   bias described by paper "An adaptive framework for 
            %   multistream classification" from the CS deparment of the 
            %   university of Texas at Dallas
            %
            %   nFoldElements (integer)
            %       Both source and target data will be splited in chunks
            %       of data containing n elements per chunk/fold.
            %       If you only want one chunk, pass zero or size(data,1)
            %       as argument.
            %   samplingRatio (double)
            %       Value in the interval [0.0,1.0] which describes the
            %       percentage of sampling that would go to Source Stream.
            %       Target will have 1 - n percentagem of data.
            [rowsNumber, ~] = size(self.data);
    
            numberOfFolds = round(length(self.data)/elementsPerFold);
            chunkSize = round(rowsNumber/numberOfFolds);
            numberOfFoldsRounded = round(rowsNumber/chunkSize);
            self.nFoldElements = min(elementsPerFold, length(self.data)/numberOfFoldsRounded);
            
            if length(self.data)/numberOfFoldsRounded > elementsPerFold
                numberOfFolds = numberOfFolds + 1;
            end
            self.nMinibatches = numberOfFolds;
            ck = self.nFoldElements;
            
            for i = 1:numberOfFolds
                x = [];
                data = [];
                if i > numberOfFoldsRounded
                    x = self.data(ck * (i-1) + 1:end,1:end-self.nClasses);
                    data = self.data(ck * (i-1) + 1:end,1:end);
                else
                    x = self.data(ck * (i-1) + 1:ck * i,1:end-self.nClasses);
                    data = self.data(ck * (i-1) + 1:ck * i,1:end);
                end
                
                x_mean = mean(x);
                probability = exp(-abs(x - x_mean).^2);
                [~,idx] = sort(probability);
                
                m = size(data,1);
                source = data(idx(1:ceil(m*samplingRatio)),1:end);
                target = data(idx(ceil(m*samplingRatio)+1:length(data)),1:end);
                
                self.source{i} = source;
                self.target{i} = target;
            end
        end
        function splitAsSourceTargetStreams_dallas2(self, elementsPerFold, samplingRatio)
            %splitAsSourceTargetStreams_dallas2
            %   Split the function to simulate a Multistream classification
            %   input domains.
            %
            %   Source and Target streams will be splited on half using the
            %   bias described by paper "FUSION - An online method for 
            %   multistream classification" from the university of Texas at
            %   Dallas.
            %
            %   nFoldElements (integer)
            %       Both source and target data will be splited in chunks
            %       of data containing n elements per chunk/fold.
            %       If you only want one chunk, pass zero or size(data,1)
            %       as argument.
            %   samplingRatio (double)
            %       Value in the interval [0.0,1.0] which describes the
            %       percentage of sampling that would go to Source Stream.
            %       Target will have 1 - n percentagem of data.

            [rowsNumber, ~] = size(self.data);
    
            numberOfFolds = round(length(self.data)/elementsPerFold);
            chunkSize = round(rowsNumber/numberOfFolds);
            numberOfFoldsRounded = round(rowsNumber/chunkSize);
            if mod(floor(size(self.data, 1)/numberOfFoldsRounded), 2) == 0
                self.nFoldElements = min(elementsPerFold, floor(size(self.data, 1)/numberOfFoldsRounded));
            else
                self.nFoldElements = min(elementsPerFold, floor(size(self.data, 1)/numberOfFoldsRounded) - 1);
            end
            
            
            if length(self.data)/numberOfFoldsRounded > elementsPerFold
                numberOfFolds = numberOfFolds + 1;
            end
            self.nMinibatches = numberOfFolds;
            ck = self.nFoldElements;
            
            for i = 1 : numberOfFolds
                x = [];
                data = [];
                if i > numberOfFoldsRounded
                    x = self.data(ck * (i-1) + 1:end,1:end-self.nClasses);
                    data = self.data(ck * (i-1) + 1:end,1:end);
                else
                    x = self.data(ck * (i-1) + 1:ck * i,1:end-self.nClasses);
                    data = self.data(ck * (i-1) + 1:ck * i,1:end);
                end
                
                x_mean = mean(x);
                norm_1 = vecnorm((x - x_mean)',1)';
                norm_2 = vecnorm((x - x_mean)',2)';
                numerator   = norm_2;
                denominator = 2 * std(norm_1) ^ 2;
                probability = exp(-numerator/denominator);
                [~,idx] = sort(probability);
                
                m = size(data,1);
                source = data(idx(1 : ceil(m * samplingRatio)), 1 : end);
                target = data(idx(ceil(m * samplingRatio) + 1: size(data, 1)), 1 : end);
                
                self.source{i} = source;
                self.target{i} = target;
            end
        end
        
        function createXsYsXtYt(self)
            %createXsYsXtYt
            %   Split the datastream data into sets of input, output, input
            %   from source, output from source, input from target, output
            %   from target
            %   It also creates a permutted version of this data, in 
            self.X  = {};
            self.y  = {};
            self.Xs = {};
            self.ys = {};
            self.Xt = {};
            self.yt = {};
            self.permutedX = {};
            self.permutedy = {};
            for i = 1 : self.nMinibatches
                self.Xs{i} = self.source{i}(:,1:end-self.nClasses);
                self.ys{i} = self.source{i}(:,self.nFeatures+1:end);
                self.Xt{i} = self.target{i}(:,1:end-self.nClasses);
                self.yt{i} = self.target{i}(:,self.nFeatures+1:end);
                self.X{i}  = [self.Xs{i};self.Xt{i}];
                self.y{i}  = [self.ys{i};self.yt{i}];
                
                x = self.X{i};
                Y = self.y{i};
                
                p  = randperm(size(x, 1));
                self.permutedX{i} = x(p,:);
                self.permutedy{i} = Y(p,:);
                self.indexPermutation{i} = p;
            end
        end
        
        function checkDatasetEven(self)
            %checkDatasetEven
            %   Check if the number of rows in the whole dataset is even,
            %   so we can split in a equal number of elements for source
            %   and stream (when splitting by 0.5 ratio)
            %   If the number is odd, randomly trow a row away.
            if mod(length(self.data),2) ~= 0
                p = ceil(rand() * length(self.data));
                self.data = [self.data(1:p-1,:);self.data(p+1:end,:)];
            end
        end
    end
end


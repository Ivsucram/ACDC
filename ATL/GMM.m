classdef GMM < handle
    %GMM Gaussian Mixture Model
    
    properties (Access = public)
        weight = 1;
        center = 1;
        var = [];
        winCounter = 1;
        inferenceSum = 0;
        surviveCounter = 0;
        yCount;
        
        inference = 0;
        hyperVolume = 0;
    end
    properties (Access = private)
        nFeatures;
    end
    methods (Access = public)
        function self = GMM(x)
            %GMM
            %   x (vector)
            %   	A data-sample (without its target) representing the
            %   	initial GMM center
            self.nFeatures = size(x, 2);
            self.center = x;
            self.var = 0.01 * ones(1, self.nFeatures);
        end
        
        function computeInference(self, x)
            c = self.center;
            
            dist = (x - c) .^ 2 ./ self.var;
            
            [self.inference, maxMahalDistIdx] = min(exp(-0.5 * dist));
            self.hyperVolume = self.var(maxMahalDistIdx);
        end
    end
    
end


classdef AGMM < handle
    properties (Access = public)
        gmmArray = [];
        nSamplesFeed = 0;
        rho = 0.1;
        nFeatures;
    end
    
    methods (Access = public)
        function run(self, x, bias2)
            self.nSamplesFeed = self.nSamplesFeed + 1;
            if size(self.gmmArray, 1) == 0
                self.gmmArray = [self.gmmArray; GMM(x)];
                
                self.nFeatures = size(x, 2);
            else
                self.computeInference(x);
                
                [~, gmmWinnerIdx] = max(self.updateWeights());
                if self.M() > 1
                    self.computeOverlapsDegree(gmmWinnerIdx, 3, 3);
                end
                
                denominator = 1.25 * exp(-bias2) + 0.75 * self.nFeatures;
                numerator   = 4 - 2 * exp( -self.nFeatures / 2);
                threshold = exp(- denominator / numerator);
                if self.gmmArray(gmmWinnerIdx).inference < threshold ...
                        && self.gmmArray(gmmWinnerIdx).hyperVolume > self.rho * (self.computeSumHyperVolume() - self.gmmArray(gmmWinnerIdx).hyperVolume)...
                        && self.nSamplesFeed > 10
                    
                    % Create a new cluster
                    self.createCluster(x);
                    self.gmmArray(end).var = (x - self.gmmArray(gmmWinnerIdx).center) .^ 2;
                else
                    % Update the winning cluster
                    self.updateCluster(x, self.gmmArray(gmmWinnerIdx));
                end
            end
        end
        
        function createCluster(self, x)
            self.gmmArray = [self.gmmArray; GMM(x)];
            
            weightSum = 0;
            for i = 1 : size(self.gmmArray, 1)
                weightSum = weightSum + self.gmmArray(i).weight;
            end
            for i = 1 : size(self.gmmArray, 1)
                self.gmmArray(i).weight = self.gmmArray(i).weight/weightSum;
            end
        end
        
        function updateCluster(~, x, gmm)
            gmm.winCounter = gmm.winCounter + 1;
            gmm.center     = gmm.center +  (x - gmm.center) / gmm.winCounter;
            gmm.var        = gmm.var    + ((x - gmm.center) .^ 2 - gmm.var) / gmm.winCounter;
        end
        
        function deleteCluster(self)
            accu_e = zeros(1, size(self.gmmArray, 1));
            for i = 1 : size(self.gmmArray, 1)
                accu_e(i) = self.gmmArray(i).inferenceSum / self.gmmArray(i).surviveCounter;
            end
            accu_e(isnan(accu_e)) = [];
            deleteList = find(accu_e <= mean(accu_e) - 0.5 * std(accu_e));
            
            if ~isempty(deleteList)
                self.gmmArray(deleteList) = [];
                accu_e(deleteList) = [];
            end
            
            sumWeight = 0;
            for i = 1 : size(self.gmmArray, 1)
                sumWeight = sumWeight + self.gmmArray(i).weight;
            end
            if sumWeight == 0
                [~, maxIdx] = max(accu_e);
                self.gmmArray(maxIdx).weight = self.gmmArray(i).weight + 1;
            end
            
            sumWeight = 0;
            for i = 1 : size(self.gmmArray, 1)
                sumWeight = sumWeight + self.gmmArray(i).weight;
            end
            for i = 1 : size(self.gmmArray, 1)
                self.gmmArray(i).weight = self.gmmArray(i).weight / sumWeight;
            end
        end
        
        function hyperVolume = computeSumHyperVolume(self)
            hyperVolume = 0;
            for i = 1 : size(self.gmmArray, 1)
                hyperVolume = hyperVolume + self.gmmArray(i).hyperVolume;
            end
        end
        
        function computeInference(self, x, y)
            for i = 1 : size(self.gmmArray, 1)
                gmm = self.gmmArray(i);
                
                if nargin == 3
                    gmm.computeInference(x, y);
                else
                    gmm.computeInference(x);
                end
            end
        end
        
        function weights = updateWeights(self)
            denumerator  = zeros(1, size(self.gmmArray, 1));
            probX_J      = zeros(1, size(self.gmmArray, 1));
            probJ        = zeros(1, size(self.gmmArray, 1));
            probX_JprobJ = zeros(1, size(self.gmmArray, 1));
            weights      = zeros(1, size(self.gmmArray, 1));
            
            sumWinCounter = 0;
            maxInference = 0;
            maxInferenceIdx = 1;
            for i = 1 : size(self.gmmArray, 1)
                sumWinCounter = sumWinCounter + self.gmmArray(i).winCounter;
                if self.gmmArray(i).inference > maxInference
                    maxInference = self.gmmArray(i).inference;
                    maxInferenceIdx = i;
                end
            end
            
            for i = 1 : size(self.gmmArray, 1)
                self.gmmArray(i).inferenceSum   = self.gmmArray(i).inferenceSum   + self.gmmArray(i).inference;
                self.gmmArray(i).surviveCounter = self.gmmArray(i).surviveCounter + 1;

                denumerator(i) = sqrt(2 * pi * self.gmmArray(i).hyperVolume);
                probX_J(i) = denumerator(i) .* self.gmmArray(i).inference;
                probJ(i)   = self.gmmArray(i).winCounter / sumWinCounter;
                probX_JprobJ(i) = probX_J(i) * probJ(i);
            end
            
            if sum(probX_JprobJ) == 0
                probX_JprobJ(maxInferenceIdx) = probX_JprobJ(maxInferenceIdx) + 1;
            end
            
            for i = 1 : size(self.gmmArray, 1)
                self.gmmArray(i).weight = probX_JprobJ(i) / sum(probX_JprobJ);
                weights(i) = self.gmmArray(i).weight;
            end
        end
        
        function computeOverlapsDegree(self, gmmWinnerIdx, maximumLimit, minimumLimit)
            if nargin == 2
                maximumLimit = 3;
                minimumLimit = maximumLimit;
            elseif nargin == 3
                minimumLimit = maximumLimit;
            end
            maximumLimit = abs(maximumLimit);
            minimumLimit = abs(minimumLimit);
            
            nGMM = size(self.gmmArray, 1);
            overlap_coefficient = 1/(nGMM-1);
            
            sigmaMaximumWinner = maximumLimit * sqrt(self.gmmArray(gmmWinnerIdx).var);
            sigmaMinimumWinner = minimumLimit * sqrt(self.gmmArray(gmmWinnerIdx).var);
            
            if maximumLimit == minimumLimit
                miu_plus_sigma_winner = self.gmmArray(gmmWinnerIdx).center + sigmaMaximumWinner;
                miu_mins_sigma_winner = self.gmmArray(gmmWinnerIdx).center - sigmaMinimumWinner;
            else
                miu_plus_sigma_winner =   sigmaMinimumWinner + sigmaMaximumWinner;
                miu_mins_sigma_winner = -sigmaMinimumWinner -sigmaMaximumWinner;
            end
            
            miu_plus_sigma    = zeros(nGMM, self.nFeatures);
            miu_mins_sigma    = zeros(nGMM, self.nFeatures);
            overlap_mins_mins = zeros(1, nGMM);
            overlap_mins_plus = zeros(1, nGMM);
            overlap_plus_mins = zeros(1, nGMM);
            overlap_plus_plus = zeros(1, nGMM);
            overlap_score     = zeros(1, nGMM);
            
            for i = 1 : nGMM
                sigmaMaximum = maximumLimit * sqrt(self.gmmArray(i).var);
                sigmaMinimum = minimumLimit * sqrt(self.gmmArray(i).var);
                
                if maximumLimit == minimumLimit
                    miu_plus_sigma(i, :) = self.gmmArray(i).center + sigmaMaximum;
                    miu_mins_sigma(i, :) = self.gmmArray(i).center - sigmaMaximum;
                else
                    miu_plus_sigma(i, :) = sigmaMinimum  + sigmaMaximum;
                    miu_mins_sigma(i, :) = -sigmaMinimum - sigmaMaximum;
                end
                
                overlap_mins_mins(i) = mean(miu_mins_sigma(i,:) - miu_mins_sigma_winner);
                overlap_mins_plus(i) = mean(miu_plus_sigma(i,:) - miu_mins_sigma_winner);
                overlap_plus_mins(i) = mean(miu_mins_sigma(i,:) - miu_plus_sigma_winner);
                overlap_plus_plus(i) = mean(miu_plus_sigma(i,:) - miu_plus_sigma_winner);
                
                condition1 = overlap_mins_mins(i) >= 0 ...
                    && overlap_mins_plus(i) >= 0 ...
                    && overlap_plus_mins(i) <= 0 ...
                    && overlap_plus_plus(i) <= 0;
                condition2 = overlap_mins_mins(i) <= 0 ...
                    && overlap_mins_plus(i) >= 0 ...
                    && overlap_plus_mins(i) <= 0 ...
                    && overlap_plus_plus(i) >= 0;
                condition3 = overlap_mins_mins(i) > 0 ...
                    && overlap_mins_plus(i) > 0 ...
                    && overlap_plus_mins(i) < 0 ...
                    && overlap_plus_plus(i) > 0;
                condition4 = overlap_mins_mins(i) < 0 ...
                    && overlap_mins_plus(i) > 0 ...
                    && overlap_plus_mins(i) < 0 ...
                    && overlap_plus_plus(i) < 0;
                
                if condition1 || condition2
                    % full overlap, the cluster is inside the winning cluster
                    % the score is full score 1/(nGMM-1)
                    overlap_score(i) = overlap_coefficient;
                elseif condition3 || condition4
                    % partial overlap, the score is the full score multiplied
                    % by the overlap degree
                    reward = norm(self.gmmArray(i).center    - self.gmmArray(gmmWinnerIdx).center)...
                                   / norm(self.gmmArray(i).center    + self.gmmArray(gmmWinnerIdx).center)...
                                   + norm(sqrt(self.gmmArray(i).var) - sqrt(self.gmmArray(gmmWinnerIdx).var))...
                                   / norm(sqrt(self.gmmArray(i).var) + sqrt(self.gmmArray(gmmWinnerIdx).var));
                    overlap_score(i) = overlap_coefficient * reward;
                end
            end
            overlap_score(gmmWinnerIdx) = []; % take out the winner score from the array
            self.rho = sum(overlap_score);
            self.rho = min(self.rho, 1);
            self.rho = max(self.rho, 0.1); % Do not let rho = zero
        end
        
        function M = computeNumberOfGmms(self)
            M = size(self.gmmArray, 1);
        end
        
        function M = M(self)
            M = self.computeNumberOfGmms();
        end
    end
end
filename;

dmS = DataManipulator('');
dmS.loadSourceCSV(filename);

dmT = DataManipulator('');
dmT.loadTargetCSV(filename);

crSource             = [];
crTarget             = [];
trainTime            = [];
testTime             = [];
KlLossEvolution      = [];
classificationLoss   = [];
nodeEvolution        = [];
discriminativeLoss   = [];
generativeLossTarget = [];
agmmTargetGenSize    = [];
agmmSourceDiscSize   = [];

nodeEvolutionTarget = [];
nodeEvolutionSource = [];
gmmTargetBatch      = [];
gmmSourceBatch      = [];


nn = NeuralNetwork([dmS.nFeatures 1 dmS.nClasses]);
ae = DenoisingAutoEncoder([nn.layers(1) nn.layers(2) nn.layers(1)]);

% I am building the greedyLayerBias
x = dmS.getX(1);
ae.greddyLayerWiseTrain(x(1, :), 1, 0.1);
% I am building the greedyLayerBias

agmmSourceDisc = AGMM();
agmmTargetGen  = AGMM();

sourceSize = size(dmS.data, 1);
targetSize = size(dmT.data, 1);
sourceIndex = 0;
targetIndex = 0;
i = 0;

originalLearningRate = ae.learningRate;
epochs = 1;
while (sourceIndex + targetIndex) < (sourceSize + targetSize)
    i = i + 1;
    Xs = [];
    ys = [];
    Xt = [];
    yt = [];
    batchCount = 0;
    while batchCount < 1000 && (sourceIndex + targetIndex) <= (sourceSize + targetSize)
        ratio = (sourceSize - sourceIndex) / (sourceSize + targetSize - sourceIndex - targetIndex);
        
        if (rand(1) <= ratio && sourceIndex < sourceSize) || (targetIndex >= targetSize && sourceIndex < sourceSize)
            sourceIndex = sourceIndex + 1;
            Xs = [Xs; dmS.getX(sourceIndex)];
            ys = [ys; dmS.getY(sourceIndex)];
        elseif targetIndex < targetSize
            targetIndex = targetIndex + 1;
            Xt = [Xt; dmT.getX(targetIndex)];
            yt = [yt; dmT.getY(targetIndex)];
        end
        batchCount = batchCount + 1;
    end
    
    %% workaround
    if size(Xs, 1) == 0
        Xs = [Xs, dmS.getX(sourceSize)];
        ys = [ys, dmS.getY(sourceSize)];
    end
    if size(Xt, 1) == 0
        Xt = [Xt, dmT.getX(targetSize)];
        yt = [yt, dmT.getY(targetSize)];
    end
            
    
    %% Evaluation ~ Test Target
    tic
    nn.test(Xt, yt);
    crTarget(end + 1) =  nn.classificationRate;
    classificationLoss(end + 1) = nn.lossValue;
    testTime(end + 1) = toc;
    
    nn.test(Xs(max(Xs, [], 2) ~= 0, :), ys(max(Xs, [], 2) ~= 0, :));
    crSource(end + 1) = nn.classificationRate;
    discriminativeLoss(end + 1) = nn.lossValue;
    
    ae.test(Xt);
    generativeLossTarget(end + 1) = ae.lossValue;
    
    if i > 1
        nodeEvolutionTarget(end + 1) = nodeEvolutionTarget(i - 1);
        nodeEvolutionSource(end + 1) = nodeEvolutionSource(i - 1);
    else
        nodeEvolutionTarget(end + 1) = 0;
        nodeEvolutionSource(end + 1) = 0;
    end
    
    tic
    for epoch = 1 : epochs
        %% Discriminative phase on Source
        nn.setAgmm(agmmSourceDisc);
        for j = 1 : size(Xs, 1)
            x = Xs(j, :);
            y = ys(j, :);
            if max(y) == 0
                continue
            end
            
            lastHiddenLayerNo = numel(nn.layers) - 1;
            
            nn.forwardpass(x);
            if epoch == 1
                agmmSourceDiscSize(end + 1) = nn.runAgmm(x, y).M();
                nn.widthAdaptationStepwise(lastHiddenLayerNo, y);
            else
                nn.nSamplesFeed = nn.nSamplesFeed - 1;
                nn.nSamplesLayer(lastHiddenLayerNo) = nn.nSamplesLayer(lastHiddenLayerNo) - 1;
                nn.widthAdaptationStepwise(lastHiddenLayerNo, y);
                nn.BIAS2{lastHiddenLayerNo}(end) = [];
                nn.VAR{lastHiddenLayerNo}(end) = [];
            end
            
            if nn.growable(lastHiddenLayerNo)
                nodeEvolutionSource(i) = nodeEvolutionSource(i) + nn.getAgmm().M();
                for numberOfGMMs = 1 : nn.getAgmm().M()
                    nn.grow(lastHiddenLayerNo);
                    ae.grow(lastHiddenLayerNo);
                end
            elseif nn.prunable{lastHiddenLayerNo}(1) ~= 0
                for k = size(nn.prunable{lastHiddenLayerNo}, 1) : -1 : 1
                    nodeToPrune = nn.prunable{lastHiddenLayerNo}(k);
                    ae.prune(lastHiddenLayerNo, nodeToPrune);
                    nn.prune(lastHiddenLayerNo, nodeToPrune);
                    nodeEvolutionSource(i) = nodeEvolutionSource(i) - 1;
                end
            end
            nn.train(x, y);
            
        end
        for j = 1 : numel(nn.layers)-2
            ae.weight{j} = nn.weight{j};
            ae.bias{j}   = nn.bias{j};
        end
        agmmSourceDisc = nn.getAgmm();
        %% Generative phase on Target
        ae.setAgmm(agmmTargetGen);
        for j = 1 : size(Xt, 1)
            x = Xt(j, :);
            y = x;
            lastHiddenLayerNo = numel(nn.layers) - 1;
            
            ae.forwardpass(x);
            if epoch == 1
                agmmTargetGenSize(end + 1) = ae.runAgmm(x, y).M();
                ae.widthAdaptationStepwise(lastHiddenLayerNo, y);
            else
                ae.nSamplesFeed = ae.nSamplesFeed - 1;
                ae.nSamplesLayer(lastHiddenLayerNo) = ae.nSamplesLayer(lastHiddenLayerNo) - 1;
                ae.widthAdaptationStepwise(lastHiddenLayerNo, y);
                ae.BIAS2{lastHiddenLayerNo}(end) = [];
                ae.VAR{lastHiddenLayerNo}(end) = [];
            end
            
            if ae.growable(lastHiddenLayerNo)
                nodeEvolutionTarget(i) = nodeEvolutionTarget(i) + ae.getAgmm().M();
                for numberOfGMMs = 1 : ae.getAgmm.M()
                    ae.grow(lastHiddenLayerNo);
                    nn.grow(lastHiddenLayerNo);
                end
            elseif ae.prunable{lastHiddenLayerNo}(1) ~= 0
                for k = size(ae.prunable{lastHiddenLayerNo}, 1) : -1 : 1
                    nodeToPrune = ae.prunable{lastHiddenLayerNo}(k);
                    ae.prune(lastHiddenLayerNo, nodeToPrune);
                    nn.prune(lastHiddenLayerNo, nodeToPrune);
                    nodeEvolutionTarget(i) = nodeEvolutionTarget(i) - 1;
                end
            end
            ae.greddyLayerWiseTrain(x, 1, 0.1);
        end
        for j = 1 : numel(ae.layers)-2
            nn.weight{j} = ae.weight{j};
            nn.bias{j}   = ae.bias{j};
        end
        agmmTargetGen = ae.getAgmm();
        
%         Kullback-Leibler Divergence
        try
            common = min(size(Xs,1), size(Xt,1));
            KlLossEvolution(end + 1) = ae.updateWeightsByKullbackLeibler(Xs(1:common,:), Xt(1:common,:));
        catch
            KlLossEvolution(end + 1) = 0;
        end
        
        for j = 1 : numel(ae.layers)-2
            nn.weight{j} = ae.weight{j};
            nn.bias{j}   = ae.bias{j};
        end
    end
    if agmmSourceDisc.M() > 1
        agmmSourceDisc.deleteCluster();
    end
    if agmmTargetGen.M() > 1
        agmmTargetGen.deleteCluster();
    end
    trainTime(end + 1) = toc;
    gmmTargetBatch(end + 1) = agmmTargetGen.M();
    gmmSourceBatch(end + 1) = agmmSourceDisc.M();
    
    %% Print metrics
    nodeEvolution(i, :) = nn.layers(2 : end - 1);
    
    if i == 2 || mod(i, round((sourceSize + targetSize)/1000/10)) == 0
        fprintf('Minibatch: %d\n', i);
        fprintf('Total of samples: %d Source | %d Target\n', size(Xs,1), size(Xt,1));
        fprintf('Max Mean Min Now Accu Training time: %f %f %f %f %f\n', max(trainTime(1:i)), mean(trainTime(1:i)), min(trainTime(1:i)), trainTime(i), sum(trainTime(1:i)));
        fprintf('Max Mean Min Now Accu Testing time: %f %f %f %f %f\n', max(testTime(1:i)), mean(testTime(1:i)), min(testTime(1:i)), testTime(i), sum(testTime(1:i)));
        fprintf('Max Mean Min Now AGMM Source: %d %f %d %d\n', max(agmmSourceDiscSize), mean(agmmSourceDiscSize), min(agmmSourceDiscSize), agmmSourceDiscSize(end));
        fprintf('Max Mean Min Now AGMM Target: %d %f %d %d\n', max(agmmTargetGenSize), mean(agmmTargetGenSize), min(agmmTargetGenSize), agmmTargetGenSize(end));
        fprintf('Max Mean Min Now CR: %f%% %f%% %f%% %f%%\n', max(crTarget(2:i)) * 100., mean(crTarget(2:i)) * 100., min(crTarget(2:i)) * 100., crTarget(i) * 100.);
        fprintf('Max Mean Min Now Classification Loss: %f %f %f %f\n', max(classificationLoss(2:i)), mean(classificationLoss(2:i)), min(classificationLoss(2:i)), classificationLoss(i));
        fprintf('Max Mean Min Now KL: %f %f %f %f\n', max(KlLossEvolution(2:i)), mean(KlLossEvolution(2:i)), min(KlLossEvolution(2:i)), KlLossEvolution(i));
        fprintf('Max Mean Min Now Nodes: %d %f %d %d\n', max(nodeEvolution(2:i)), mean(nodeEvolution(2:i)), min(nodeEvolution(2:i)), nodeEvolution(i));
        fprintf('Network structure: %s (Discriminative) | %s (Generative)\n', num2str(nn.layers(:).'), num2str(ae.layers(:).'));
        fprintf('\n');
    end
end

fprintf('\n\n')
fprintf('Source CR: %f\n', mean(crSource(2:end)))
fprintf('Target CR: %f\n', mean(crTarget(2:end)))
fprintf('Training time: %f\n', sum(trainTime))


%% ---------------------------- Plotters ----------------------------------
function plotTime(trainTime, testTime)
    figure('Name', 'Processing Time', 'NumberTitle', 'off');
    hold on
    
    ylim([0 max(max(trainTime), max(testTime)) * 1.1]);
    xlim([1 size(trainTime, 2)]);
    
    pTrain = plot(trainTime);
    pTest  = plot(testTime);
    
    if max(trainTime) > 1
        text(find(trainTime == max(trainTime(trainTime > 1)), 1), max(trainTime(trainTime > 1)),...
            strcat('\leftarrow Max Train Time:', {' '}, string(max(trainTime(trainTime > 1)))),...
            'FontSize', 8,...
            'Color', 'black');
        text(find(trainTime == min(trainTime(trainTime > 1)), 1), min(trainTime(trainTime > 1)),...
            strcat('\leftarrow Min Train Time:', {' '}, string(min(trainTime(trainTime > 1)))),...
            'FontSize', 8,...
            'Color', 'black');
    end
    if max(testTime) > 1
        text(find(testTime == max(testTime(testTime > 1)), 1), max(testTime(testTime > 1)),...
            strcat('\leftarrow Max Test Time:', {' '}, string(max(testTime(testTime > 1)))),...
            'FontSize', 8,...
            'Color', 'black');
        text(find(testTime == min(testTime(testTime > 1)), 1), min(testTime(testTime > 1)),...
            strcat('\leftarrow Min Test Time:', {' '}, string(min(testTime(testTime > 1)))),...
            'FontSize', 8,...
            'Color', 'black');
    end
    
    legend([pTrain,...
        pTest], [strcat('Train Time Mean | Accumulative:', {' '}, string(mean(trainTime)), {' | '}, string(sum(trainTime))),...
        strcat('Test Time Mean | Accumulative:',  {' '},  string(mean(testTime)), {' | '}, string(sum(testTime)))]);
    
    
    ylabel('Time in seconds');
    xlabel('Minibatches');
    
    hold off
end

function plotNodeEvolution(nodeEvolution)
    figure('Name', 'Node Evolution', 'NumberTitle', 'off');
    hold on
    
    ylim([0 max(nodeEvolution, [], 'all') * 1.1]);
    xlim([1 size(nodeEvolution, 1)]);
    
    plotArray   = [];
    legendArray = [];
    for i = 1 : size(nodeEvolution, 2)
        p = plot(nodeEvolution(:, i));
        plotArray   = [plotArray, p];
        legendArray = [legendArray, strcat('HL', {' '}, string(i), {' '}, 'mean:', {' '}, string(mean(nodeEvolution(nodeEvolution(:, i) > 0, i))))];
        
        text(find(nodeEvolution(:, i) == max(nodeEvolution(:, i)), 1), max(nodeEvolution(:, i)),...
            strcat('\leftarrow Max nodes HL ', {' '}, string(i), ':', {' '}, string(max(nodeEvolution(:, i)))),...
            'FontSize', 8,...
            'Color', 'black');
        text(find(nodeEvolution(:, i) == min(nodeEvolution(nodeEvolution(:, i) > 0, i)), 1), min(nodeEvolution(nodeEvolution(:, i) > 0, i)),...
            strcat('\leftarrow Min nodes HL ', {' '}, string(i), ':', {' '}, string(min(nodeEvolution(nodeEvolution(:, i) > 0, i)))),...
            'FontSize', 8,...
            'Color', 'black');
    end
    
    ylabel('Number of nodes');
    xlabel('Minibatches');
    
    legend(plotArray, legendArray);
    
    hold off
end

function plotAGMM(agmmSource, agmmTarget)
    figure('Name', 'Number of GMMs on AGMMs', 'NumberTitle', 'off');
    hold on
    
    ylim([0 max(max(agmmTarget), max(agmmSource)) * 1.1]);
    xlim([1 size(agmmSource, 2)]);
    
    pAgmmSource    = plot(agmmSource);
    pAgmmTarget    = plot(agmmTarget);
    
    if max(agmmSource) > 1
        text(find(agmmSource == max(agmmSource(agmmSource > 1)), 1), max(agmmSource(agmmSource > 1)),...
            strcat('\leftarrow Max GMMs Source Discriminative:', {' '}, string(max(agmmSource(agmmSource > 1)))),...
            'FontSize', 8,...
            'Color', 'black');
        text(find(agmmSource == min(agmmSource(agmmSource > 1)), 1), min(agmmSource(agmmSource > 1)),...
            strcat('\leftarrow Min GMMs Source Discriminative:', {' '}, string(min(agmmSource(agmmSource > 1)))),...
            'FontSize', 8,...
            'Color', 'black');
    end
    if max(agmmTarget) > 1
        text(find(agmmTarget == max(agmmTarget(agmmTarget > 1)), 1), max(agmmTarget(agmmTarget > 1)),...
            strcat('\leftarrow Max GMMs Target Generative:', {' '}, string(max(agmmTarget(agmmTarget > 1)))),...
            'FontSize', 8,...
            'Color', 'black');
        text(find(agmmTarget == min(agmmTarget(agmmTarget > 1)), 1), min(agmmTarget(agmmTarget > 1)),...
            strcat('\leftarrow Min GMMs Target Generative:', {' '}, string(min(agmmTarget(agmmTarget > 1)))),...
            'FontSize', 8,...
            'Color', 'black');
    end
    
    legend([pAgmmSource,...
        pAgmmTarget], [strcat('AGMM Source Discriminative Mean:', {' '}, string(mean(agmmSource))),...
        strcat('AGMM Target Generative Mean:', {' '},     string(mean(agmmTarget)))]);
    
    
    ylabel('Number of GMMs');
    xlabel('Samples');
    
    hold off
end

function plotLosses(classificationLoss, discriminativeLoss, generativeTargetLoss, kullbackLeiblerLoss)
    figure('Name', 'Losses', 'NumberTitle', 'off');
    hold on
    
    ylim([0 max(max(kullbackLeiblerLoss), max(max(max(classificationLoss), max(discriminativeLoss)), max(generativeTargetLoss))) * 1.1]);
    xlim([1 size(classificationLoss, 1)]);
    
    pClassificationLoss   = plot(classificationLoss);
    pDiscriminativeLoss   = plot(discriminativeLoss);
    pGenerativeTargetLoss = plot(generativeTargetLoss);
    pKullbackLeiblerLoss  = plot(kullbackLeiblerLoss);
    
    text(find(classificationLoss == max(classificationLoss), 1), max(classificationLoss),...
        strcat('\leftarrow Max Classification Loss:', {' '}, string(max(classificationLoss))),...
        'FontSize', 8,...
        'Color', 'black');
    text(find(classificationLoss == min(classificationLoss), 1), min(classificationLoss),...
        strcat('\leftarrow Min Classification Loss:', {' '}, string(min(classificationLoss))),...
        'FontSize', 8,...
        'Color', 'black');
    
    text(find(discriminativeLoss == max(discriminativeLoss), 1), max(discriminativeLoss),...
        strcat('\leftarrow Max Discriminative Loss:', {' '}, string(max(discriminativeLoss))),...
        'FontSize', 8,...
        'Color', 'black');
    text(find(discriminativeLoss == min(discriminativeLoss), 1), min(discriminativeLoss),...
        strcat('\leftarrow Min Discriminative Loss:', {' '}, string(min(discriminativeLoss))),...
        'FontSize', 8,...
        'Color', 'black');
    
    text(find(generativeTargetLoss == max(generativeTargetLoss), 1), max(generativeTargetLoss),...
        strcat('\leftarrow Max Generative Target Loss:', {' '}, string(max(generativeTargetLoss))),...
        'FontSize', 8,...
        'Color', 'black');
    text(find(generativeTargetLoss == min(generativeTargetLoss), 1), min(generativeTargetLoss),...
        strcat('\leftarrow Min Generative Target Loss:', {' '}, string(min(generativeTargetLoss))),...
        'FontSize', 8,...
        'Color', 'black');
    
    text(find(kullbackLeiblerLoss == max(kullbackLeiblerLoss), 1), max(kullbackLeiblerLoss),...
        strcat('\leftarrow Max KL Div Loss:', {' '}, string(max(kullbackLeiblerLoss))),...
        'FontSize', 8,...
        'Color', 'black');
    text(find(kullbackLeiblerLoss == min(kullbackLeiblerLoss), 1), min(kullbackLeiblerLoss),...
        strcat('\leftarrow Min KL Div Loss:', {' '}, string(min(kullbackLeiblerLoss))),...
        'FontSize', 8,...
        'Color', 'black');
    
    ylabel('Loss Value');
    xlabel('Minibatches');
    
    legend([pClassificationLoss,...
        pDiscriminativeLoss,...
        pGenerativeTargetLoss,...
        pKullbackLeiblerLoss], [strcat('Classification Loss Mean:', {' '}, string(mean(classificationLoss(2:end)))),...
        strcat('Discriminative Loss Mean:', {' '}, string(mean(discriminativeLoss))),...
        strcat('Generative Target Loss Mean:', {' '}, string(mean(generativeTargetLoss))),...
        strcat('Kullback Leibler Divergence Loss Mean:', {' '}, string(mean(kullbackLeiblerLoss)))]);
    
    hold off
end

function plotClassificationRate(source, target, nMinibatches)
    figure('Name', 'Source and Target Classification Rates', 'NumberTitle', 'off');
    hold on
    ylim([0 max(max(source), max(target)) * 1.1]);
    xlim([1 nMinibatches]);
    
    niceBlue    = [0      0.4470 0.7410];
    niceYellow  = [0.8500 0.3250 0.0980];
    
    pSource = plot(source, 'Color', niceYellow, 'LineStyle', ':');
    pTarget = plot(target, 'Color', niceBlue);
    
    text(find(source == max(source), 1), max(source),...
        strcat('\leftarrow Max Source:', {' '}, string(max(source))),...
        'FontSize', 8,...
        'Color', 'black');
    text(find(source == min(source), 1), min(source),...
        strcat('\leftarrow Min Source:', {' '}, string(min(source))),...
        'FontSize', 8,...
        'Color', 'black');
    text(find(target == max(target), 1), max(target),...
        strcat('\leftarrow Max Target:', {' '}, string(max(target))),...
        'FontSize', 8,...
        'Color', 'black');
    text(find(target == min(target), 1), min(target),...
        strcat('\leftarrow Min Target:', {' '}, string(min(target))),...
        'FontSize', 8,...
        'Color', 'black');
    
    ylabel('Classification Rate');
    xlabel('Minibatches');
    
    legend([pSource, pTarget], [strcat('Source Mean:', {' '}, string(mean(source(2:end)))),...
        strcat('Target Mean:', {' '}, string(mean(target(2:end))))]);
    
    hold off
    
end

function plotBIAS2andVAR(BIAS2, VAR)
    sampleLayerCount = zeros(1, size(BIAS2, 2));
    yAxisLim  = 0;
    bias2 = [];
    var   = [];
    for i = 2 : size(BIAS2, 2)
        sampleLayerCount(i) = sampleLayerCount(i - 1) + size(BIAS2{i}, 2);
        for j = 1 : size(BIAS2{i}, 2)
            bias2     = [bias2, BIAS2{i}(j)];
            var       = [var,   VAR{i}(j)];
            yAxisLim  = max(yAxisLim, bias2(end) + var(end));
        end
    end
    clear BIAS2 VAR
    
    figure('Name', 'BIAS2, VAR, and NS', 'NumberTitle', 'off');
    hold on
    
    ylim([0 max(max(bias2), max(var)) * 1.1]);
    xlim([1 size(bias2, 2)]);
    
    p1 = plot(bias2);
    p2 = plot(var);
    p3 = plot(bias2 + var);
    for j = 1: ceil(size(bias2, 2)/4) : size(bias2, 2)
        if ~isnan(bias2(j))
            text(j, bias2(j),...
                strcat('\leftarrow', {' '}, 'BIAS2 =', {' '}, string(bias2(j))),...
                'FontSize', 8);
        end
    end
    text(size(bias2, 2), bias2(end), string(bias2(end)));
    
    for j = 1: ceil(size(var, 2)/4) : size(var, 2)
        if ~isnan(var(j))
            text(j, var(j),...
                strcat('\leftarrow', {' '}, 'VAR =', {' '}, string(var(j))),...
                'FontSize', 8);
        end
    end
    text(size(var, 2), var(end), string(var(end)));
    
    for j = 1: ceil(size(var + bias2, 2)/4) : size(var + bias2, 2)
        if ~isnan(var(j)) && ~isnan(bias2(j)) && ~isnan(var(j) + bias2(j))
            text(j, var(j) + bias2(j),...
                strcat('\leftarrow', {' '}, 'NS =', {' '}, string(var(j) + bias2(j))),...
                'FontSize', 8);
        end
    end
    text(size(var + bias2, 2), var(end) + bias2(end), string(var(end) + bias2(end)));
    
    for i = 2 : size(sampleLayerCount, 2) - 1
        line([sampleLayerCount(i), sampleLayerCount(i)], [-yAxisLim * 2 yAxisLim * 2],...
            'LineStyle', ':',...
            'Color', 'magenta');
    end
    
    ylabel('Value');
    xlabel('Sample');
    
    legend([p1, p2, p3], [strcat('BIAS2 Mean:', {' '}, string(mean(bias2(2:end)))),...
        strcat('VAR Mean:', {' '}, string(mean(var(2:end)))),...
        strcat('NS Mean:', {' '}, string(mean(var(2:end) + bias2(2:end))))]);
    hold off
end

function plotBIAS2andVARGen(BIAS2, VAR)
    sampleLayerCount = zeros(1, size(BIAS2, 2));
    yAxisLim  = 0;
    bias2 = [];
    var   = [];
    for i = 2 : size(BIAS2, 2)
        sampleLayerCount(i) = sampleLayerCount(i - 1) + size(BIAS2{i}, 2);
        for j = 1 : size(BIAS2{i}, 2)
            bias2     = [bias2, BIAS2{i}(j)];
            var       = [var,   VAR{i}(j)];
            yAxisLim  = max(yAxisLim, bias2(end) + var(end));
        end
    end
    clear BIAS2 VAR
    
    figure('Name', 'BIAS2, VAR, and NS Generative', 'NumberTitle', 'off');
    hold on
    
    ylim([0 max(max(bias2), max(var)) * 1.1]);
    xlim([1 size(bias2, 2)]);
    
    p1 = plot(bias2);
    p2 = plot(var);
    p3 = plot(bias2 + var);
    for j = 1: ceil(size(bias2, 2)/4) : size(bias2, 2)
        if ~isnan(bias2(j))
            text(j, bias2(j),...
                strcat('\leftarrow', {' '}, 'BIAS2 =', {' '}, string(bias2(j))),...
                'FontSize', 8);
        end
    end
    text(size(bias2, 2), bias2(end), string(bias2(end)));
    
    for j = 1: ceil(size(var, 2)/4) : size(var, 2)
        if ~isnan(var(j))
            text(j, var(j),...
                strcat('\leftarrow', {' '}, 'VAR =', {' '}, string(var(j))),...
                'FontSize', 8);
        end
    end
    text(size(var, 2), var(end), string(var(end)));
    
    for j = 1: ceil(size(var + bias2, 2)/4) : size(var + bias2, 2)
        if ~isnan(var(j)) && ~isnan(bias2(j)) && ~isnan(var(j) + bias2(j))
            text(j, var(j) + bias2(j),...
                strcat('\leftarrow', {' '}, 'NS =', {' '}, string(var(j) + bias2(j))),...
                'FontSize', 8);
        end
    end
    text(size(var + bias2, 2), var(end) + bias2(end), string(var(end) + bias2(end)));
    
    for i = 2 : size(sampleLayerCount, 2) - 1
        line([sampleLayerCount(i), sampleLayerCount(i)], [-yAxisLim * 2 yAxisLim * 2],...
            'LineStyle', ':',...
            'Color', 'magenta');
    end
    
    ylabel('Value');
    xlabel('Sample');
    
    legend([p1, p2, p3], [strcat('BIAS2 Mean:', {' '}, string(mean(bias2))),...
        strcat('VAR Mean:', {' '}, string(mean(var))),...
        strcat('NS Mean:', {' '}, string(mean(var + bias2)))]);
    hold off
end
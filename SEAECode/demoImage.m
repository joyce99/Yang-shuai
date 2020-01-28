
srcStr = {'Caltech10', 'Caltech10', 'Caltech10', 'amazon', 'amazon', 'amazon', 'webcam', 'webcam', 'webcam', 'dslr', 'dslr', 'dslr'};
tgtStr = {'amazon', 'webcam', 'dslr', 'Caltech10', 'webcam', 'dslr', 'Caltech10', 'amazon', 'dslr', 'Caltech10', 'amazon', 'webcam'};
Result = [];
        
parameter.alpha = 150;
parameter.lambda = 0.001;
parameter.beta = 0.001;
parameter.noise = 0.7;
parameter.k = 10;
parameter.layer = 1;
for iData = 1:12
    disp(num2str(iData));
    src = char(srcStr{iData});
    tgt = char(tgtStr{iData});
    data = strcat(src, '_vs_', tgt);

    benchmark = pwd;
    addpath(genpath(benchmark));

    Datapath1= [benchmark,'/imagedata/',src '_SURF_L10.mat'];
    load(Datapath1);
    Xs = fts ./ repmat(sum(fts, 2), 1, size(fts, 2));
    src_X = Xs';
    src_labels = labels;
    parameter.size = size(src_labels,1);

    Datapath1= [benchmark,'/imagedata/',tgt '_SURF_L10.mat'];
    load(Datapath1);
    Xt = fts ./ repmat(sum(fts, 2), 1, size(fts,2));
    tar_X = Xt';
    tar_labels = labels;

    fprintf('data=%s\n', data);

    total = [src_X,tar_X];
    [allhx, Ws] = mSDA(total, parameter.noise,1);

    [allhx, D_cell, W_cell] = myRepresentationLearningM(allhx,parameter);
     xr=[src_X; allhx(:,1:size(src_X,2))];
    xr=xr';
    bestC = 1./mean(sum(xr.*xr,2));
    model = svmtrain(src_labels,xr,['-q -t 0 -c ',num2str(bestC),' -m 3000']);
    xe=[tar_X; allhx(:,size(src_X,2)+1:end)];
    xe=xe';
    [label,accuracy] = svmpredict(tar_labels,xe,model);

    accuracy(1)
    Result = [Result; accuracy(1)];
    fprintf('\n');
end
Result












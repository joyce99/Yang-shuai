clear all;
srcStr = {'Caltech10', 'Caltech10', 'amazon', 'amazon', 'webcam', 'webcam' };
tgtStr = {'amazon', 'webcam', 'Caltech10', 'webcam', 'Caltech10', 'amazon'};

Result_Final = [];
for iData = 1 : 6
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
    
    [Y_tar_pseudo] = Pseudolable(src_X, tar_X, src_labels, tar_labels);
    Num_Class = length(unique(src_labels));

    lables = unique(src_labels);

    for iter = 1 : 3
        parameter.lambda = 1e-5;
        parameter.beta = 1;
        parameter.noises = 0.6;
        parameter.layers = iter;

        New_src_X = [];
        New_tar_X = [];
        for iter_class = 1: Num_Class
            [Xs_c] = SubData (src_X,src_labels,lables(iter_class));
            [Xt_c] = SubData (tar_X,Y_tar_pseudo,lables(iter_class));
            total  = [Xs_c,Xt_c];

            nbsrc=size(Xs_c,2);
            nbtgt=size(Xt_c,2);
            disp('Computer MMD...')
            parameter.MMD=[(1/nbsrc^2)*ones(nbsrc,nbsrc), -1/(nbsrc*nbtgt)*ones(nbsrc,nbtgt); -1/(nbsrc*nbtgt)*ones(nbtgt,nbsrc), (1/nbtgt^2)*ones(nbtgt,nbtgt)];
            [allhx1, Ws] = mSDAM(double(total>0), parameter);
            New_src_X = [New_src_X,allhx1(:,1:size(Xs_c,2))];
            New_tar_X = [New_tar_X,allhx1(:,1+size(Xs_c,2):end)];
        end

        local_allhx = [New_src_X,New_tar_X];

        nbsrc=size(src_X,2);
        nbtgt=size(tar_X,2);
        disp('Computer MMD...')

        parameter.MMD = MMD(src_X, tar_X, src_labels, tar_labels,Y_tar_pseudo); 
        total = [src_X,tar_X];
        [global_allhx, Ws] = mSDAM(double(total>0), parameter);
        allhx = [local_allhx;global_allhx];
        xr=allhx(:,1:size(src_X,2));
        xr=xr';
        bestC = 1./mean(sum(xr.*xr,2));
        model = svmtrain(src_labels,xr,['-q -t 0 -c ',num2str(bestC),' -m 3000']);
        xe= allhx(:,size(src_X,2)+1:end);
        xe=xe';
        [Y_tar_pseudo,accuracy] = svmpredict(tar_labels,xe,model);
        if iter==3
            accuracy(1)
            Result_Final = [Result_Final;accuracy(1)];
        end
    end
end
Result_Final













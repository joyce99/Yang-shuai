function [Y_tar_pseudo] = Pseudolable(src_X, tar_X, src_labels, tar_labels)
disp('1..');
    xr=src_X';
    bestC = 1./mean(sum(xr.*xr,2));
    model = svmtrain(src_labels,xr,['-q -t 0 -c ',num2str(bestC),' -m 3000']);
    xe=tar_X';
    [Y_tar_pseudo,accuracy] = svmpredict(tar_labels,xe,model);
end
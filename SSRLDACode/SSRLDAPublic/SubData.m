function [X] = SubData (src_X,src_labels,label)
    index = find(src_labels==label);
    X = [];
    for i=1:size(index,1)
        X = [X,src_X(:,i)];
    end
end
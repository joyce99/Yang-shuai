function [allhx, D_cell, W_cell] = myRepresentationLearningM(xx,parameter)
% [U S V ] = svds(xx,20);
% allhx = U'*xx;
% % allhx = xx;
% return;

% xx : dxn input
% ns = size(Xs,1);
% nt = size(Xt,1);
% 
% xx = [Xs;Xt]';
% clear Xs;
% clear Xt;

[d, n]  = size(xx);
% H = eye(n) - ones(n,n)/n;
% xx = xx*H;

% disp('stacking hidden layers...');
prevhx = xx;
allhx = [];
% figure;imagesc(prevhx);title(['rslda' num2str(0)]);
layers = parameter.layer;
D_cell = cell(layers,1);
W_cell = cell(layers,1);
for layer = 1:layers
    disp([' layer:' num2str(layer) ]);
%     [newhx, ~] = myMethod(prevhx, alpha, lambda);
        
%         [newhx, ~] = myMethod_l21_loss( hard_tanh(alpha * prevhx), alpha, lambda);
        [newhx,W, D] = myMethodM(prevhx,parameter);
        D_cell{layer} = D;
        W_cell{layer} = W;
%         [newhx, ~] = myMethod_l21_norm(prevhx, alpha, lambda);
    %     dis = KA_distance(newhx(:,1:ns)', newhx(:,ns+1:end)');
    %     disp([' layer:' num2str(layer) ' dis = ' num2str(dis)]);
%     [U S V] = svds(newhx, 30);
%     allhx = [allhx; newhx; U'*newhx];
      allhx = [allhx; newhx];
%     allhx = [allhx; hard_tanh(newhx*alpha^(layer))];
%     allhx = [allhx; hard_tanh(newhx*alpha*layer)];
%     allhx = [allhx; hard_tanh(newhx*(alpha^(layer)))];
    prevhx = newhx;
    clear newhx;
end
end


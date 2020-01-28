function [hx, W, D] = myMethodM(xx,parameter)
% xx : dxn input

% disp( ' myMethod_l21_loss ')
xx = full(xx);

% c  = var(xx');index = (c>0.000001);xx = xx(index,:); disp(['deleted ' num2str(sum(~index)) ' features ']);
W = [];

[d, n] = size(xx);
t = var(xx');
index = t > 0.000001;
xx = xx(index,:);

H = eye(n) - ones(n,n)/n;
xxc = xx*H;

s = sum(abs(xxc),2);
index = ( s > 1e-9 );
xxc = xxc(index,:);
xx = xx(index,:);

C = xxc*xxc';  %%%%%%
[W, D] = get_W_l21_loss(xxc',diag(C),parameter); % l21 loss
if sum(sum(isnan(W))) > 0
    error('W is wrong' );
end
W = full(W);
hx = W'*xx;
hx = tanh(hx*parameter.alpha);
end

function [W, D] = get_W_l21_loss(X,A,parameter)
% min |XW - X|_{2} + lambda trace(W'*A*W)
[n, d] = size(X);
D = ones(n,1); 
W = update_W(X,A, D,parameter);  
end

function W = update_W(X,DIAG, D,parameter)

C = X'* bsxfun(@times,X,D);
Y = X';
Newsrc_X = Y(:,1:parameter.size);
Newtar_X = Y(:,1+parameter.size:end);
parameter.L = LaplacianMatrix(Newsrc_X,Newtar_X,parameter.k);
% parameter.L = LaplacianMatrixE(Newsrc_X,Newtar_X,parameter.k);% or this
W = inv(C + parameter.lambda*diag(DIAG)+ parameter.beta*X'*parameter.L*X) * C;
end


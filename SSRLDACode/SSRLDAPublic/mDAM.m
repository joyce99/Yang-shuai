function [hx, W] = mDAM(xx,parameter)
% xx : dxn input
% noise: corruption level
% lambda: regularization

% hx: dxn hidden representation
% W: dx(d+1) mapping
disp('mDA...');

[d, n] = size(xx);
% adding bias
xxb = [xx; ones(1, n)];

% scatter matrix S
S = xxb*xxb';

% corruption vector
q = ones(d+1, 1)*(1-parameter.noises);
q(end) = 1;


% Q: (d+1)x(d+1)
Q = S.*(q*q');
Q(1:d+2:end) = q.*diag(S);

% P: dx(d+1)
P = S(1:end-1,:).*repmat(q', d, 1);

MMD = xxb*parameter.MMD*xxb';
MMD = MMD.*(q*q');
MMD(1:d+2:end) = q.*diag(MMD);



disp('Compute MMD...');
% Newsrc_X = xxb(:,1:parameter.size);
% Newtar_X = xxb(:,1+parameter.size:end);
% parameter.L = LaplacianMatrix(Newsrc_X,Newtar_X,10);
% 
% Manifold = xxb*parameter.L*xxb';
% Manifold = Manifold.*(q*q');
% Manifold(1:d+2:end) = q.*diag(Manifold);

% final W = P*Q^-1, dx(d+1);
reg = parameter.lambda*eye(d+1);
% A = diag(S);
% for i=1:size(Q,1)
%     reg(i,i) = lambda*A(i);
% end
% % reg = lambda*diag(S);
reg(end,end) = 0;
W = P/(Q+reg +parameter.beta*MMD);
% W = P/(Q+reg +beta*MMD +parameter.gamma*Manifold);

hx = W*xxb;
hx = tanh(hx);
% hx = sigm(hx);


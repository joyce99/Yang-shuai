function [allhx, Ws] = mSDAM(xx,parameter)

% xx : dxn input
% noise: corruption level
% layers: number of layers to stack
% allhx: (layers*d)xn stacked hidden representations
% lambda = 1e-05;
% % lambda = 0.1;
% beta = 0.1;
disp('mSDA:stacking hidden layers...')
prevhx = xx;
allhx = [];
Ws={};
for layer = 1:parameter.layers
    	disp(['layer:',num2str(parameter.layers)])
	tic
	[newhx, W] = mDAM(prevhx,parameter);
	Ws{layer} = W;
	toc
	allhx = [allhx; newhx];
%     allhx = newhx;
	prevhx = newhx;
end

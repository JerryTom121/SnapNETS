function getEigenScore(graphFile, outFileName)
%assuming we have a list of edges E = [i j weight_i_to_j weight_j_to_i]
  % Every undirected edge appears twice 

G = getGraph(graphFile);
% Infected_id = getLabel(LabelFile);
a=5;
graphFile
eigen = tic;
[u lambda] = eigs(G,1);
u = abs(u);
[v lambda] = eigs(G',1);
v = abs(v);
lambda = abs(lambda);
t_eigen = toc(eigen);
% clear G;
lambda

E = load(graphFile);
a = E(:,1);
b = E(:,2);
beta1 = E(:,3);
beta2 = E(:,4);
clear E;

uv = u.*v;

delLam_vec = -(-lambda*(uv(a)+uv(b)) + v(a).*(((1+beta2)/2).*(lambda*u(a) - beta1.*u(b)) + ((1+beta1)/2).*(lambda*u(b) - beta2.*u(a)))...
    + u(a).*v(b).*beta2 + u(b).*v(a).*beta1)./(v'*u - (uv(a) + uv(b)));

delLam_sol(:,1) = a;
delLam_sol(:,2) = b;
delLam_sol(:,3) = delLam_vec;

delLam_sol = sortrows(delLam_sol,3);
% delLam_sol = remove_candidates(delLam_sol, Infected_id);
outfile = fopen(outFileName,'w');
fprintf(outfile,'%d\t%d\t%.4f\n', delLam_sol');
fclose(outfile);
exit
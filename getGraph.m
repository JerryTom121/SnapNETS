function graph = getGraph( graphFile)
%   getGraph Loads a graph from file into a sparse matrix
%   Assume that graphFile is in weighted edgelist format. The graph is direct ed - [i j weight_i_to_j weight_j_to_i]. Undirected edges appear twice
E = load(graphFile);
n = max(max(E(:,1)), max(E(:,2)));
E = [E;[E(:,2), E(:,1),E(:,4), E(:,3)]];
E(end+1,:)=[n n 0 0];
graph = spconvert(E(:,1:3));
%   graph = graph + graph';
end


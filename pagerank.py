# Lee Jackson
# Network Science CPSC 8480
# Dr. Ilya Safro
# Final Project

# Program Description:
# This program provides an implementation of the PageRank algorithm

# PageRank computes a ranking of the nodes in the graph G based on the structure
# of the incoming links. It was originally designed as an algorithm to rank web
# pages.

import networkx as nx
from networkx import NetworkXError
from scipy.io import mmread


def my_pagerank(graph, alpha=0.85, personalization=None, max_iter=100, tol=1.0e-6, nstart=None, weight='weight',
                dangling=None):
    if len(graph) == 0:
        return {}

    if not graph.is_directed():
        directed_graph = graph.to_directed()
    else:
        directed_graph = graph

    # Create copy in (right) stochastic form
    stochastic_graph = nx.stochastic_graph(directed_graph, weight=weight)
    num_nodes = stochastic_graph.number_of_nodes()

    # Choose fixed starting vector if not given
    if nstart is None:
        x = dict.fromkeys(stochastic_graph, 1.0 / num_nodes)
    else:
        # Normalized nstart vector
        vals_sum = float(sum(nstart.values()))
        x = dict((k, v / vals_sum) for k, v in nstart.items())

    if personalization is None:
        # Assign uniform personalization vector if not given
        p_vec = dict.fromkeys(stochastic_graph, 1.0 / num_nodes)
    else:
        missing = set(graph) - set(personalization)
        if missing:
            raise NetworkXError('Personalization dictionary'
                                'must have a value for every node. '
                                'Missing nodes %s' % missing)
        vals_sum = float(sum(personalization.values()))
        p_vec = dict((k, v / vals_sum) for k, v in personalization.items())

    if dangling is None:
        # Use personalization vector if dangling vector not specified
        dangling_weights = p_vec
    else:
        missing = set(graph) - set(dangling)
        if missing:
            raise NetworkXError('Dangling node dictionary '
                                'must have a value for every node. '
                                'Missing nodes %s' % missing)
        vals_sum = float(sum(dangling.values()))
        dangling_weights = dict((k, v / vals_sum) for k, v in dangling.items())
    dangling_nodes = [n for n in stochastic_graph if stochastic_graph.out_degree(n, weight=weight) == 0.0]

    # power iteration: make up to max_iter iterations
    for _ in range(max_iter):
        x_last = x
        x = dict.fromkeys(x_last.keys(), 0)
        dangle_sum = alpha * sum(x_last[n] for n in dangling_nodes)
        for n in x:

            # this matrix multiply looks odd because it is
            # doing a left multiply x^T = x_last^T*W
            for nbr in stochastic_graph[n]:
                x[nbr] += alpha * x_last[n] * stochastic_graph[n][nbr][weight]
            x[n] += dangle_sum * dangling_weights[n] + (1.0 - alpha) * p_vec[n]
        # check convergence, l1 norm
        err = sum([abs(x[n] - x_last[n]) for n in x])
        if err < num_nodes * tol:
            return x
    raise NetworkXError('pagerank: power iteration failed to converge '
                        'in %d iterations.' % max_iter)


def main():
    # assume choice is not valid
    choice_is_valid = False
    file_name = None
    while not choice_is_valid:
        print('1. Symmetric pattern from Cannes, Lucien Marro, June 1981 (10,010 Nonzeros)')
        print('2. Web matrix (Numerical Computing with MATLAB, Moler, 2004) (2,636 Nonzeros')
        print('3. Test network')
        choice = input('Please choose option (1), (2), or (3) ')

        if choice == '1':
            file_name = 'can_838.mtx'
            choice_is_valid = True
        elif choice == '2':
            file_name = 'Harvard500.mtx'
            choice_is_valid = True
        elif choice == '3':
            file_name = 'test_network.mtx'
            choice_is_valid = True
        else:
            print('Choice is not valid.')

    # read chosen Matrix Market file into a scipy sparse matrix
    a = mmread(file_name)

    # create graph from scipy sparse matrix
    g = nx.from_scipy_sparse_matrix(a, True)

    # call NetworkX's function and my own function
    pr1 = nx.pagerank(g, 0.4)
    pr2 = my_pagerank(g, 0.4)

    # compare the two results
    print('NetworkX pagerank for [0]: ', pr1[0])
    print('  My alg pagerank for [0]: ', pr2[0])

    # find best pagerank
    max_pr_val = max(pr2.values())
    max_pr_keys = [k for k, v in pr2.items() if v == max_pr_val]

    print('Node with highest pagerank (value):', max_pr_keys, '(', max_pr_val, ')')
    print()

    print(pr2)


main()

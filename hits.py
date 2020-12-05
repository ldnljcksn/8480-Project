# Lee Jackson
# Network Science CPSC 8480
# Dr. Ilya Safro
# Final Project

# Program Description:
# This program provides an implementation of the HITS algorithm (Hyperlink-Induced Topic Search)

# Hubs: Nodes that point to many other nodes
# Authorities: Nodes that are pointed to by many nodes

# HITS algorithm is used to determine a list of authorities and hubs from a large set of webpages. A query string is
# searched on internet search engines and a list of webpages that contain the string is returned. From that set, a root
# set of K size (generally 200) is selected from the most relevant pages. Then the root set is expanded and the HITS
# algorithm calculates the hub and authority scores for each page. These scores are returned to the user.

import networkx as nx
from scipy.io import mmread


def my_hits_alg(graph, max_iter=1000, tol=1.0e-6, nstart=None):
	# check if the graph is empty
	if len(graph) == 0:
		return {}, {}

	# if no starting node is given, choose one
	if nstart is None:
		hub_scores = dict.fromkeys(graph, 1.0 / graph.number_of_nodes())
	else:
		hub_scores = nstart

		# normalize starting vector
		normalizer = 1.0 / sum(hub_scores.values())
		for k in hub_scores:
			hub_scores[k] *= normalizer

	for _ in range(max_iter):  # power iteration: make up to max_iter iterations
		hlast = hub_scores
		hub_scores = dict.fromkeys(hlast.keys(), 0)
		authority_scores = dict.fromkeys(hlast.keys(), 0)

		# matrix multiplication to calculate scores
		for n in hub_scores:
			for m in graph[n]:
				authority_scores[m] += hlast[n] * graph[n][m].get('weight', 1)
		for n in hub_scores:
			for m in graph[n]:
				hub_scores[n] += authority_scores[m] * graph[n][m].get('weight', 1)

		# normalize scores
		normalizer = 1.0 / max(hub_scores.values())
		for n in hub_scores:
			hub_scores[n] *= normalizer
		normalizer = 1.0 / max(authority_scores.values())
		for n in authority_scores:
			authority_scores[n] *= normalizer

		# check convergence, l1 norm
		err = sum([abs(hub_scores[n] - hlast[n]) for n in hub_scores])
		if err < tol:
			print('Iterations: ', _)
			break

	# if the algorithm fails to converge within the max number of iterations
	else:
		raise nx.PowerIterationFailedConvergence(max_iter)

	# normalize scores
	normalizer = 1.0 / sum(authority_scores.values())
	for n in authority_scores:
		authority_scores[n] *= normalizer
	normalizer = 1.0 / sum(hub_scores.values())
	for n in hub_scores:
		hub_scores[n] *= normalizer

	return hub_scores, authority_scores


def main():
	# assume choice is not valid
	choice_is_valid = False
	file_name = None

	while not choice_is_valid:
		print('1. Symmetric pattern from Cannes, Lucien Marro, June 1981 (10,010 Nonzeros)')
		print('2. Symmetric connection table from DTNSRDC, Washington (10,426 Nonzeros)')
		choice = input('Please choose option (1) or (2) ')

		if choice == '1':
			file_name = 'can_838.mtx'
			choice_is_valid = True
		elif choice == '2':
			file_name = 'dwt_1242.mtx'
			choice_is_valid = True
		else:
			print('Choice is not valid.')

	# read chosen Matrix Market file into a scipy sparse matrix
	a = mmread(file_name)

	# create graph from scipy sparse matrix
	g = nx.from_scipy_sparse_matrix(a)

	# call NetworkX's function and my own function
	hubs1, authorities1 = nx.hits(g, max_iter=1000)
	hubs2, authorities2 = my_hits_alg(g)

	# compare the two results
	print('NetworkX hub score for [0]: ', hubs1[0])
	print('  My alg hub score for [0]: ', hubs2[0])
	print()
	print('NetworkX auth score for [0]: ', authorities1[0])
	print('  My alg auth score for [0]: ', authorities2[0])
	print()

	# find biggest hub and biggest authority
	max_hub_value = max(hubs2.values())
	max_hub_keys = [k for k, v in hubs2.items() if v == max_hub_value]
	max_auth_value = max(authorities2.values())
	max_auth_keys = [k for k, v in authorities2.items() if v == max_auth_value]

	print('      Biggest hub (value):', max_hub_keys, '(', max_hub_value, ')')
	print('Biggest authority (value):', max_auth_keys, '(', max_auth_value, ')')


main()

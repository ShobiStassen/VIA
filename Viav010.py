import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix, csgraph
import scipy
import igraph as ig
import leidenalg
import time
import hnswlib

def compute_hitting_time(sparse_graph, number_eig=0, x_lazy=0.5, alph_teleport = 0.98, root=0 ):
    #sparse_graph = np.array([[0, 3, 0, 0, 9, 0, 1], [3, 0, 6, 15, 9, 0, 0], [0, 6, 0, 8, 0, 0, 0], [0, 15, 8, 0, 7, 5, 0],
                  #[9, 9, 0, 7, 0, 4, 0], [0, 0, 0, 5, 4, 0, 0], [1, 0, 0, 0, 0, 0, 0]])
    #sparse_graph = np.array([[0,1,1,0,0,0],[1,0,0,0,0,0],[1,0,0,0,1,0],[0,0,0,0,1,1],[0,0,1,1,0,1],[0,0,0,1,1,0]]) #example on page 90
    #sparse_graph = np.array([[0,1,1,1,1],[1,0,1,0,0],[1,1,0,1,1],[1,0,1,0,1],[1,0,1,1,0]]) #example on page 45
    N = sparse_graph.shape[0]
    #print(sparse_graph)
    sparse_graph = scipy.sparse.csr_matrix(sparse_graph)
    print('start compute hitting')
    #print('sparse', sparse_graph)
    A=scipy.sparse.csr_matrix.todense(sparse_graph)

    print('is graph symmetric', (A.transpose()==A).all())
    #print(A)
    #print('laplacian', csgraph.laplacian(A))
    beta_teleport =2*(1-alph_teleport)/(2-alph_teleport)

    lap = csgraph.laplacian(sparse_graph, normed=False)
    A = scipy.sparse.csr_matrix.todense(lap)
    print('is graph symmetric', (A.transpose() == A).all())
    #print('laplacian', A)

    deg = sparse_graph+lap
    deg.data = 1 / np.sqrt(deg.data)  ##inv sqrt of degree matrix
    deg[deg == np.inf] = 0
    #print('degree',deg)
    #norm_lap = lap.dot(deg)
    #norm_lap = deg.dot(norm_lap)
    norm_lap =  csgraph.laplacian(sparse_graph,normed = True)
    A = scipy.sparse.csr_matrix.todense(norm_lap)
    #print("normalized laplacian",A)
    #norm_lap = (norm_lap+norm_lap.transpose())*0.5
    Id = np.zeros((N, N),float)
    np.fill_diagonal(Id, 1)
    #print('ID identity','beta is', beta_teleport)
    #print(Id)
    beta_normlap_test = 2*x_lazy*(1-beta_teleport)*A +beta_teleport*Id
    #print('beta lap test', beta_normlap_test)
    #print('direct inverse of beta lap test')
    #print(np.linalg.inv(beta_normlap_test))
    A=scipy.sparse.csr_matrix.todense(norm_lap)
    #print('is graph symmetric', (A.transpose()==A).all())
    #print("normalized laplacian", A)

    eig_val, eig_vec = np.linalg.eigh(A)  # eig_vec[:,i] is eigenvector for eigenvalue eig_val[i]
    #eig_val, eig_vec = scipy.sparse.linalg.eigsh(norm_lap, k=sparse_graph.shape[0]-1) # eig_vec[:,i] is eigenvector for eigenvalue eig_val[i]
    print('eig val', eig_val.shape, eig_val)
    print('eig vectors shape', eig_vec.shape)

    if number_eig ==0:number_eig = eig_vec.shape[1]
    print('number of eig vec' ,number_eig)
    sum_matrix = np.zeros((N,N),float)
    beta_norm_lap = np.zeros((N,N),float)
    Xu =np.zeros((N,N))
    Xu[:,root]=1
    Id_Xv = np.zeros((N,N),int)
    np.fill_diagonal(Id_Xv,1)
    Xv_Xu =Id_Xv-Xu
    #print('Xv-Xroot', Xv_Xu)

    for i in range(0,number_eig):
        vec_i = eig_vec[:, i]
        factor =beta_teleport+2*eig_val[i]*x_lazy*(1-beta_teleport)
        #print('factor',factor)

        vec_i = np.reshape(vec_i,(-1,1))
        eigen_vec_mult = vec_i.dot(vec_i.T)
        sum_matrix = sum_matrix+(eigen_vec_mult/factor)
        beta_norm_lap = beta_norm_lap+(eigen_vec_mult*factor)
    # print("Greens matrix", sum_matrix)
    # print("beta laplacian", beta_norm_lap)
    # print("product",beta_norm_lap.dot(sum_matrix))
    # print('inverse of greens matrix should be b-lap')
    # print(np.linalg.inv(sum_matrix))
    deg = scipy.sparse.csr_matrix.todense(deg)

    temp = sum_matrix.dot(deg)
    #print(temp.shape)
    temp = deg.dot(temp)*beta_teleport

    hitting_matrix = np.zeros((N,N),float)
    diag_row = np.diagonal(temp)
    for i in range(N):
        hitting_matrix[i,:] = diag_row - temp[i,:]

    print('hitting matrix')
    print(hitting_matrix)
    print('roundtrip matrix')
    print(hitting_matrix+hitting_matrix.T)
    print('node 0 commute times')
    roundtrip_commute_matrix = hitting_matrix+hitting_matrix.T
    print(roundtrip_commute_matrix[0,:])

    temp = Xv_Xu.dot(temp)
    #print(temp.shape, temp)
    final_hitting_times = np.diagonal(temp) ## number_eig x 1 vector of hitting times from root (u) to number_eig of other nodes
    #print('final shape', final.shape)
    roundtrip_times = roundtrip_commute_matrix[root,:]

    return final_hitting_times, roundtrip_times

def local_pruning_hittingtime(adjacency_matrix):
    initial_links_n = len(adjacency_matrix.data)
    print('initial links n', initial_links_n)
    adjacency_matrix = scipy.sparse.csr_matrix.todense(adjacency_matrix)
    print('adjacency')
    #print(type(adjacency_matrix))
    row_list = []
    col_list = []
    weight_list = []
    neighbor_array = adjacency_matrix  # not listed in in any order of proximity

    n_cells = neighbor_array.shape[0]
    rowi = 0

    for i in range(neighbor_array.shape[0]):
        row =np.asarray(neighbor_array[i,:]).flatten()
        to_keep_index = np.where(row > np.mean(row))[0] #we take [1] because row is a 2D matrix, not a 1D matrix like in other cases  # 0*std

        updated_nn_weights = row[to_keep_index]

        for ik in range(len(to_keep_index)):
            row_list.append(rowi)
            col_list.append(to_keep_index[ik])
            dist = updated_nn_weights[ik]
            weight_list.append(dist)

        rowi = rowi + 1
    final_links_n = len(weight_list)
    print('final links n', final_links_n)
    cluster_graph_csr = csr_matrix((np.array(weight_list), (np.array(row_list), np.array(col_list))),
                                   shape=(n_cells, n_cells))
    #print('locally pruned cluster graph')
    #print(cluster_graph_csr)
    trimmed_n= (initial_links_n-final_links_n)/initial_links_n
    print("percentage links trimmed")
    print( "%.2f" % trimmed_n)
    return cluster_graph_csr

def get_sparse_from_igraph(graph, weight_attr=None):
    edges = graph.get_edgelist()
    if weight_attr is None:
        weights = [1] * len(edges)
    else:
        weights = graph.es[weight_attr]
    if not graph.is_directed():
        edges.extend([(v, u) for u, v in edges])
        weights.extend(weights)
    shape = graph.vcount()
    shape = (shape, shape)
    if len(edges) > 0:
        return csr_matrix((weights, zip(*edges)), shape=shape)
    else:
        return csr_matrix(shape)

class PARC:
    def __init__(self, data, true_label=None, dist_std_local=2, jac_std_global='median', keep_all_local_dist='auto',
                 too_big_factor=0.4, small_pop=10, jac_weighted_edges=True, knn=30, n_iter_leiden=5):
        # higher dist_std_local means more edges are kept
        # highter jac_std_global means more edges are kept
        if keep_all_local_dist == 'auto':
            if data.shape[0] > 300000:
                keep_all_local_dist = True  # skips local pruning to increase speed
            else:
                keep_all_local_dist = False

        self.data = data
        self.true_label = true_label
        self.dist_std_local = dist_std_local
        self.jac_std_global = jac_std_global  ##0.15 is also a recommended value performing empirically similar to 'median'
        self.keep_all_local_dist = keep_all_local_dist
        self.too_big_factor = too_big_factor  ##if a cluster exceeds this share of the entire cell population, then the PARC will be run on the large cluster. at 0.4 it does not come into play
        self.small_pop = small_pop  # smallest cluster population to be considered a community
        self.jac_weighted_edges = jac_weighted_edges
        self.knn = knn
        self.n_iter_leiden = n_iter_leiden

    def make_knn_struct(self, too_big=False, big_cluster=None):
        ef = 100
        if too_big == False:
            num_dims = self.data.shape[1]
            n_elements = self.data.shape[0]
            p = hnswlib.Index(space='l2', dim=num_dims)
            p.init_index(max_elements=n_elements, ef_construction=200, M=30)
            p.add_items(self.data)
        if too_big == True:
            num_dims = big_cluster.shape[1]
            n_elements = big_cluster.shape[0]
            p = hnswlib.Index(space='l2', dim=num_dims)
            p.init_index(max_elements=n_elements, ef_construction=200, M=30)
            p.add_items(big_cluster)
        p.set_ef(ef)  # ef should always be > k
        return p



    def make_csrmatrix_noselfloop(self, neighbor_array, distance_array):
        local_pruning_bool = not (self.keep_all_local_dist)
        if local_pruning_bool == True: print('commencing local pruning based on minkowski metric at',
                                             self.dist_std_local, 's.dev above mean')
        row_list = []
        col_list = []
        weight_list = []
        neighbor_array = neighbor_array  # not listed in in any order of proximity
        # print('size neighbor array', neighbor_array.shape)
        num_neigh = neighbor_array.shape[1]
        distance_array = distance_array
        n_neighbors = neighbor_array.shape[1]
        n_cells = neighbor_array.shape[0]
        rowi = 0
        count_0dist = 0
        discard_count = 0

        if local_pruning_bool == True:  # do some local pruning based on distance
            for row in neighbor_array:
                distlist = distance_array[rowi, :]
                to_keep = np.where(distlist < np.mean(distlist) + self.dist_std_local * np.std(distlist))[0]  # 0*std
                updated_nn_ind = row[np.ix_(to_keep)]
                updated_nn_weights = distlist[np.ix_(to_keep)]
                discard_count = discard_count + (num_neigh - len(to_keep))

                for ik in range(len(updated_nn_ind)):
                    if rowi != row[ik]:  # remove self-loops
                        row_list.append(rowi)
                        col_list.append(updated_nn_ind[ik])
                        dist = np.sqrt(updated_nn_weights[ik])
                        if dist == 0:
                            count_0dist = count_0dist + 1
                        weight_list.append(dist)

                rowi = rowi + 1

        if local_pruning_bool == False:  # dont prune based on distance
            row_list.extend(list(np.transpose(np.ones((n_neighbors, n_cells)) * range(0, n_cells)).flatten()))
            col_list = neighbor_array.flatten().tolist()
            weight_list = (1. / (distance_array.flatten() + 0.1)).tolist()
        # if local_pruning_bool == True: print('share of neighbors discarded in local distance pruning %.1f' % (discard_count / neighbor_array.size))

        csr_graph = csr_matrix((np.array(weight_list), (np.array(row_list), np.array(col_list))),
                               shape=(n_cells, n_cells))
        return csr_graph

    def func_mode(self, ll):  # return MODE of list
        # If multiple items are maximal, the function returns the first one encountered.
        return max(set(ll), key=ll.count)

    def run_toobig_subPARC(self, X_data, jac_std_toobig=1,
                           jac_weighted_edges=True):
        n_elements = X_data.shape[0]
        hnsw = self.make_knn_struct(too_big=True,big_cluster=X_data)

        neighbor_array, distance_array = hnsw.knn_query(X_data, k=self.knn)
        # print('shapes of neigh and dist array', neighbor_array.shape, distance_array.shape)
        csr_array = self.make_csrmatrix_noselfloop(neighbor_array, distance_array)
        sources, targets = csr_array.nonzero()
        mask = np.zeros(len(sources), dtype=bool)
        mask |= (csr_array.data > (
                    np.mean(csr_array.data) + np.std(csr_array.data) * 5))  # smaller distance means stronger edge
        # print('sum of mask', sum(mask))
        csr_array.data[mask] = 0
        csr_array.eliminate_zeros()
        sources, targets = csr_array.nonzero()
        edgelist = list(zip(sources.tolist(), targets.tolist()))
        edgelist_copy = edgelist.copy()
        G = ig.Graph(edgelist, edge_attrs={'weight': csr_array.data.tolist()})
        sim_list = G.similarity_jaccard(pairs=edgelist_copy)  # list of jaccard weights
        new_edgelist = []
        sim_list_array = np.asarray(sim_list)
        if jac_std_toobig == 'median':
            threshold = np.median(sim_list)
        else:
            threshold = np.mean(sim_list) - jac_std_toobig * np.std(sim_list)
        strong_locs = np.where(sim_list_array > threshold)[0]
        for ii in strong_locs: new_edgelist.append(edgelist_copy[ii])
        sim_list_new = list(sim_list_array[strong_locs])

        if jac_weighted_edges == True:
            G_sim = ig.Graph(list(new_edgelist), edge_attrs={'weight': sim_list_new})
        else:
            G_sim = ig.Graph(list(new_edgelist))
        G_sim.simplify(combine_edges='sum')
        resolution_parameter = 1
        if jac_weighted_edges == True:
            partition = leidenalg.find_partition(G_sim, leidenalg.ModularityVertexPartition, weights='weight',
                                                 n_iterations=self.n_iter_leiden)
        else:
            partition = leidenalg.find_partition(G_sim, leidenalg.ModularityVertexPartition,
                                                 n_iterations=self.n_iter_leiden)
        # print('Q= %.2f' % partition.quality())
        PARC_labels_leiden = np.asarray(partition.membership)
        PARC_labels_leiden = np.reshape(PARC_labels_leiden, (n_elements, 1))
        small_pop_list = []
        small_cluster_list = []
        small_pop_exist = False
        dummy, PARC_labels_leiden = np.unique(list(PARC_labels_leiden.flatten()), return_inverse=True)
        for cluster in set(PARC_labels_leiden):
            population = len(np.where(PARC_labels_leiden == cluster)[0])
            if population < 10:
                small_pop_exist = True
                small_pop_list.append(list(np.where(PARC_labels_leiden == cluster)[0]))
                small_cluster_list.append(cluster)

        for small_cluster in small_pop_list:
            for single_cell in small_cluster:
                old_neighbors = neighbor_array[single_cell, :]
                group_of_old_neighbors = PARC_labels_leiden[old_neighbors]
                group_of_old_neighbors = list(group_of_old_neighbors.flatten())
                available_neighbours = set(group_of_old_neighbors) - set(small_cluster_list)
                if len(available_neighbours) > 0:
                    available_neighbours_list = [value for value in group_of_old_neighbors if
                                                 value in list(available_neighbours)]
                    best_group = max(available_neighbours_list, key=available_neighbours_list.count)
                    PARC_labels_leiden[single_cell] = best_group


        while small_pop_exist == True:
            small_pop_list = []
            small_pop_exist = False
            for cluster in set(list(PARC_labels_leiden.flatten())):
                population = len(np.where(PARC_labels_leiden == cluster)[0])
                if population < 10:
                    small_pop_exist = True
                    print(cluster, ' has small population of', population, )
                    small_pop_list.append(np.where(PARC_labels_leiden == cluster)[0])
            for small_cluster in small_pop_list:
                for single_cell in small_cluster:
                    old_neighbors = neighbor_array[single_cell, :]
                    group_of_old_neighbors = PARC_labels_leiden[old_neighbors]
                    group_of_old_neighbors = list(group_of_old_neighbors.flatten())
                    best_group = max(set(group_of_old_neighbors), key=group_of_old_neighbors.count)
                    PARC_labels_leiden[single_cell] = best_group

        dummy, PARC_labels_leiden = np.unique(list(PARC_labels_leiden.flatten()), return_inverse=True)
        self.labels = PARC_labels_leiden
        return PARC_labels_leiden
    def recompute_weights(self, sparse_clustergraph, pop_list_raw):
        n = sparse_clustergraph.shape[0]
        sources, targets = sparse_clustergraph.nonzero()
        edgelist = list(zip(sources, targets))
        weights = sparse_clustergraph.data
        new_weights = []
        i=0
        for s,t in edgelist:
            pop_s = pop_list_raw[s]
            pop_t = pop_list_raw[t]
            w = weights[i]
            nw = w/(pop_s+pop_t)#*
            new_weights.append(nw)
            print('old and new', w, nw)
            i = i+1
            scale_factor = max(new_weights)-min(new_weights)
            wmin = min(new_weights)
            wmax = max(new_weights)
        new_weights = [(i-wmin)*1/scale_factor for i in new_weights]

        sparse_clustergraph = csr_matrix((np.array(new_weights), (sources, targets)),
                   shape=(n, n))
        print('new weights', new_weights)
        print(sparse_clustergraph)
        print('reweighted sparse clustergraph')
        print(sparse_clustergraph)
        return sparse_clustergraph


    def run_subPARC(self):
        X_data = self.data
        too_big_factor = self.too_big_factor
        small_pop = self.small_pop
        jac_std_global = self.jac_std_global
        jac_weighted_edges = self.jac_weighted_edges
        n_elements = X_data.shape[0]
        if n_elements < 2000: self.knn = 10

        n_elements = X_data.shape[0]

        # print('number of k-nn is', knn, too_big_factor, 'small pop is', small_pop)

        neighbor_array, distance_array = self.knn_struct.knn_query(X_data, k=self.knn)

        csr_array = self.make_csrmatrix_noselfloop(neighbor_array, distance_array)
        sources, targets = csr_array.nonzero()

        edgelist = list(zip(sources, targets))

        edgelist_copy = edgelist.copy()

        G = ig.Graph(edgelist, edge_attrs={'weight': csr_array.data.tolist()})
        # print('average degree of prejacard graph is %.1f'% (np.mean(G.degree())))
        # print('computing Jaccard metric')
        sim_list = G.similarity_jaccard(pairs=edgelist_copy)

        print('commencing global pruning')

        sim_list_array = np.asarray(sim_list)
        edge_list_copy_array = np.asarray(edgelist_copy)

        if jac_std_global == 'median':
            threshold = np.median(sim_list)
        else:
            threshold = np.mean(sim_list) - jac_std_global * np.std(sim_list)
        strong_locs = np.where(sim_list_array > threshold)[0]
        # print('Share of edges kept after Global Pruning %.2f' % (len(strong_locs) / len(sim_list)), '%')
        new_edgelist = list(edge_list_copy_array[strong_locs])
        sim_list_new = list(sim_list_array[strong_locs])

        G_sim = ig.Graph(list(new_edgelist), edge_attrs={'weight': sim_list_new})
        # print('average degree of graph is %.1f' % (np.mean(G_sim.degree())))
        G_sim.simplify(combine_edges='sum')  # "first"
        # print('average degree of SIMPLE graph is %.1f' % (np.mean(G_sim.degree())))
        print('commencing community detection')
        if jac_weighted_edges == True:
            start_leiden = time.time()
            # print('call leiden on weighted graph for ', self.n_iter_leiden, 'iterations')
            partition = leidenalg.find_partition(G_sim, leidenalg.ModularityVertexPartition, weights='weight',
                                                 n_iterations=self.n_iter_leiden)
            print(time.time() - start_leiden)
        else:
            start_leiden = time.time()
            # print('call leiden on unweighted graph', self.n_iter_leiden, 'iterations')
            partition = leidenalg.find_partition(G_sim, leidenalg.ModularityVertexPartition,
                                                 n_iterations=self.n_iter_leiden)
            print(time.time() - start_leiden)
        time_end_PARC = time.time()
        # print('Q= %.1f' % (partition.quality()))
        PARC_labels_leiden = np.asarray(partition.membership)
        PARC_labels_leiden = np.reshape(PARC_labels_leiden, (n_elements, 1))

        too_big = False

        # print('labels found after Leiden', set(list(PARC_labels_leiden.T)[0])) will have some outlier clusters that need to be added to a cluster if a cluster has members that are KNN

        cluster_i_loc = np.where(PARC_labels_leiden == 0)[
            0]  # the 0th cluster is the largest one. so if cluster 0 is not too big, then the others wont be too big either
        pop_i = len(cluster_i_loc)
        print('largest clustter population', pop_i, too_big_factor, n_elements)
        if pop_i > too_big_factor * n_elements:  # 0.4
            too_big = True
            print('too big is', too_big)
            cluster_big_loc = cluster_i_loc
            list_pop_too_bigs = [pop_i]
            cluster_too_big = 0

        while too_big == True:
            X_data_big = X_data[cluster_big_loc, :]
            print(X_data_big.shape)
            PARC_labels_leiden_big = self.run_toobig_subPARC(X_data_big)
            #print('set of new big labels ', set(PARC_labels_leiden_big.flatten()))
            PARC_labels_leiden_big = PARC_labels_leiden_big + 1000
            #print('set of new big labels +1000 ', set(list(PARC_labels_leiden_big.flatten())))
            pop_list = []
            for item in set(list(PARC_labels_leiden_big.flatten())):
                pop_list.append([item, list(PARC_labels_leiden_big.flatten()).count(item)])

            print('pop of new big labels', pop_list)
            jj = 0
            print('shape PARC_labels_leiden', PARC_labels_leiden.shape)
            for j in cluster_big_loc:
                PARC_labels_leiden[j] = PARC_labels_leiden_big[jj]
                jj = jj + 1
            dummy, PARC_labels_leiden = np.unique(list(PARC_labels_leiden.flatten()), return_inverse=True)
            print('new set of labels ', set(PARC_labels_leiden))
            too_big = False
            set_PARC_labels_leiden = set(PARC_labels_leiden)

            PARC_labels_leiden = np.asarray(PARC_labels_leiden)
            for cluster_ii in set_PARC_labels_leiden:
                cluster_ii_loc = np.where(PARC_labels_leiden == cluster_ii)[0]
                pop_ii = len(cluster_ii_loc)
                not_yet_expanded = pop_ii not in list_pop_too_bigs
                if pop_ii > too_big_factor * n_elements and not_yet_expanded == True:
                    too_big = True
                    print('cluster', cluster_ii, 'is too big and has population', pop_ii)
                    cluster_big_loc = cluster_ii_loc
                    cluster_big = cluster_ii
                    big_pop = pop_ii
            if too_big == True:
                list_pop_too_bigs.append(big_pop)
                print('cluster', cluster_big, 'is too big with population', big_pop, '. It will be expanded')
        dummy, PARC_labels_leiden = np.unique(list(PARC_labels_leiden.flatten()), return_inverse=True)
        small_pop_list = []
        small_cluster_list = []
        small_pop_exist = False

        for cluster in set(PARC_labels_leiden):
            population = len(np.where(PARC_labels_leiden == cluster)[0])

            if population < small_pop:  # 10
                small_pop_exist = True

                small_pop_list.append(list(np.where(PARC_labels_leiden == cluster)[0]))
                small_cluster_list.append(cluster)

        for small_cluster in small_pop_list:

            for single_cell in small_cluster:
                old_neighbors = neighbor_array[single_cell, :]
                group_of_old_neighbors = PARC_labels_leiden[old_neighbors]
                group_of_old_neighbors = list(group_of_old_neighbors.flatten())
                available_neighbours = set(group_of_old_neighbors) - set(small_cluster_list)
                if len(available_neighbours) > 0:
                    available_neighbours_list = [value for value in group_of_old_neighbors if
                                                 value in list(available_neighbours)]
                    best_group = max(available_neighbours_list, key=available_neighbours_list.count)
                    PARC_labels_leiden[single_cell] = best_group

        while small_pop_exist == True:
            small_pop_list = []
            small_pop_exist = False
            for cluster in set(list(PARC_labels_leiden.flatten())):
                population = len(np.where(PARC_labels_leiden == cluster)[0])
                if population < small_pop:
                    small_pop_exist = True
                    print(cluster, ' has small population of', population, )
                    small_pop_list.append(np.where(PARC_labels_leiden == cluster)[0])
            for small_cluster in small_pop_list:
                for single_cell in small_cluster:
                    old_neighbors = neighbor_array[single_cell, :]
                    group_of_old_neighbors = PARC_labels_leiden[old_neighbors]
                    group_of_old_neighbors = list(group_of_old_neighbors.flatten())
                    best_group = max(set(group_of_old_neighbors), key=group_of_old_neighbors.count)
                    PARC_labels_leiden[single_cell] = best_group

        dummy, PARC_labels_leiden = np.unique(list(PARC_labels_leiden.flatten()), return_inverse=True)
        PARC_labels_leiden = list(PARC_labels_leiden.flatten())
        # print('final labels allocation', set(PARC_labels_leiden))
        pop_list = []
        pop_list_raw = []
        for item in range(len(set(PARC_labels_leiden))):
            pop_item=PARC_labels_leiden.count(item)
            pop_list.append((item, pop_item))
            pop_list_raw.append(pop_item)
        print('list of cluster labels and populations', len(pop_list), pop_list)

        self.labels = PARC_labels_leiden  # list
        vc_graph = ig.VertexClustering(G_sim, membership=PARC_labels_leiden)
        vc_graph =  vc_graph.cluster_graph(combine_edges='sum')
        sparse_vf = get_sparse_from_igraph(vc_graph, weight_attr='weight')

        reweighted_sparse = self.recompute_weights(sparse_vf, pop_list_raw)
        layout = vc_graph.layout_fruchterman_reingold()
        print('sparse cluster graph', reweighted_sparse)

        majority_truth_labels = np.empty((n_elements, 1), dtype=object)
        graph_node_label = []
        for cluster_i in range(len(set(PARC_labels_leiden))):
            cluster_i_loc = np.where(np.asarray(PARC_labels_leiden) == cluster_i)[0]
            true_labels = np.asarray(self.true_label)
            majority_truth = self.func_mode(list(true_labels[cluster_i_loc]))
            majority_truth_labels[cluster_i_loc] = 'w' + str(majority_truth) + 'c' + str(cluster_i)
            graph_node_label.append('w' + str(majority_truth) + 'c' + str(cluster_i))
        print('graph node label', graph_node_label)
        majority_truth_labels = list(majority_truth_labels.flatten())
        vc_graph.vs["label"] = graph_node_label
        ig.plot(vc_graph, "/home/shobi/Trajectory/Code/vc_graph_example_alledges.svg", layout=layout,
                edge_width=[e['weight'] for e in vc_graph.es], vertex_label=graph_node_label)

        #DO LOCAL PRUNING before hitting times
        sparse_vf = local_pruning_hittingtime(reweighted_sparse)

        sources, targets = sparse_vf.nonzero()
        edgelist = list(zip(sources.tolist(), targets.tolist())) #of cluster graph
        print('new edgelist', edgelist)
        locallytrimmed_g = ig.Graph(edgelist, edge_attrs={'weight': sparse_vf.data.tolist()})
        locallytrimmed_g = locallytrimmed_g.simplify(combine_edges='sum')
        print('locallytrimmed_g')
        weights_sparse = get_sparse_from_igraph(locallytrimmed_g, weight_attr='weight')
        layout = locallytrimmed_g.layout_fruchterman_reingold()
        locallytrimmed_g.vs["label"] = graph_node_label
        #ig.plot(locallytrimmed_g, "/home/shobi/Trajectory/Code/vc_graph_example_locallytrimmed.svg", layout=layout,
         #        vertex_label=graph_node_label)
        ig.plot(locallytrimmed_g, "/home/shobi/Trajectory/Code/vc_graph_example_locallytrimmed.svg", layout=layout,
               edge_width=[e['weight']*100  for e in locallytrimmed_g.es], vertex_label=graph_node_label)#


        #vc_graph.vs["label"] = vc_graph.vs["name"]

        # from igraph import Plot
        # pl=Plot();
        # pl.add(vc_graph, layout=layout)
        # pl._windows_hacks = True;
        # pl.show()




        # compute hitting times
        root = 55
        hitting_times, roundtrip_times = compute_hitting_time(reweighted_sparse, root=root)
        print('final hitting times:', list(zip(range(len(hitting_times)), hitting_times)))
        print('round trip times:')
        print( list(zip(range(len(hitting_times)), roundtrip_times)))
        hitting_times = np.asarray(hitting_times)

        remove_outliers = hitting_times[hitting_times<np.mean(hitting_times)+np.std(hitting_times)]
        threshold = np.mean(remove_outliers) + 0.3* np.std(remove_outliers)
        print('threshold', threshold)
        th_hitting_times = [x if x < threshold else threshold for x in hitting_times]
        scaled_hitting_times = (th_hitting_times - np.min(th_hitting_times))*100/(threshold)
        print(scaled_hitting_times)
        #threshold = np.mean(scaled_hitting_times)+0.25*np.std(scaled_hitting_times)
        threshold = int(threshold)
        scaled_hitting_times = scaled_hitting_times.astype(int)
        print('scaled hitting times')
        print(scaled_hitting_times)
        pal = ig.drawing.colors.AdvancedGradientPalette(['yellow', 'green','blue'], n=101)

        all_colors = []



        print('100 scaled hitting' ,scaled_hitting_times)
        for i in scaled_hitting_times:
            all_colors.append(pal.get(int(i))[0:3])
        print('extract all colors', zip(scaled_hitting_times,all_colors))


        locallytrimmed_g.vs['hitting_times'] =scaled_hitting_times

        locallytrimmed_g.vs['color']=[pal.get(i)[0:3] for i in scaled_hitting_times]

        print([v for v in locallytrimmed_g.vs['color']])
        ig.plot(locallytrimmed_g, "/home/shobi/Trajectory/Code/vc_graph_example_locallytrimmed_colornode_"+str(root)+".svg", layout=layout, edge_width=[e['weight']*1 for e in locallytrimmed_g.es], vertex_label=graph_node_label)
        #hitting_times = compute_hitting_time(sparse_vf, root=1)
        #print('final hitting times:', list(zip(range(len(hitting_times)), hitting_times)))

        sources, targets = sparse_vf.nonzero()
        mask = np.zeros(len(sources), dtype=bool)
        print('mean and std', np.mean(sparse_vf.data), np.std(sparse_vf.data))
        sparse_vf.data = sparse_vf.data/(np.std(sparse_vf.data))

        #print('after normalization',sparse_vf)
        threshold_global = np.mean(sparse_vf.data)
        mask |= (sparse_vf.data < (threshold_global ))  # smaller Jaccard weight means weaker edge
        print('sum of mask', sum(mask) ,'at threshold of', threshold)
        sparse_vf.data[mask] = 0
        sparse_vf.eliminate_zeros()

        sources, targets = sparse_vf.nonzero()
        edgelist = list(zip(sources.tolist(), targets.tolist()))
        trimmed_g = ig.Graph(edgelist, edge_attrs={'weight': sparse_vf.data.tolist()})
        trimmed_g = trimmed_g.simplify(combine_edges='sum')
        #layout = trimmed_g.layout_fruchterman_reingold()
        trimmed_g.vs["label"] = graph_node_label
        locallytrimmed_g.vs['color'] = [pal.get(i)[0:3] for i in scaled_hitting_times]
        ig.plot(trimmed_g, "/home/shobi/Trajectory/Code/vc_graph_example_globallytrimmed_colornode_"+str(root)+".svg", layout=layout,edge_width = [e['weight'] for e in trimmed_g.es], vertex_label = graph_node_label )

        #print(trimmed_g.vs.indices)

        #trimmed_g.write_svg("/home/shobi/Trajectory/Code/vc_graph_example_trimmed2.svg", layout=layout,edge_stroke_widths = [e['weight'] for e in trimmed_g.es], labels='label' )
        return

    def accuracy(self, onevsall=1):

        true_labels = self.true_label
        Index_dict = {}
        PARC_labels = self.labels
        N = len(PARC_labels)
        n_cancer = list(true_labels).count(onevsall)
        n_pbmc = N - n_cancer

        for k in range(N):
            Index_dict.setdefault(PARC_labels[k], []).append(true_labels[k])
        num_groups = len(Index_dict)
        sorted_keys = list(sorted(Index_dict.keys()))
        error_count = []
        pbmc_labels = []
        thp1_labels = []
        fp, fn, tp, tn, precision, recall, f1_score = 0, 0, 0, 0, 0, 0, 0

        for kk in sorted_keys:
            vals = [t for t in Index_dict[kk]]
            majority_val = self.func_mode(vals)
            if majority_val == onevsall: print('cluster', kk, ' has majority', onevsall, 'with population', len(vals))
            if kk == -1:
                len_unknown = len(vals)
                print('len unknown', len_unknown)
            if (majority_val == onevsall) and (kk != -1):
                thp1_labels.append(kk)
                fp = fp + len([e for e in vals if e != onevsall])
                tp = tp + len([e for e in vals if e == onevsall])
                list_error = [e for e in vals if e != majority_val]
                e_count = len(list_error)
                error_count.append(e_count)
            elif (majority_val != onevsall) and (kk != -1):
                pbmc_labels.append(kk)
                tn = tn + len([e for e in vals if e != onevsall])
                fn = fn + len([e for e in vals if e == onevsall])
                error_count.append(len([e for e in vals if e != majority_val]))

        predict_class_array = np.array(PARC_labels)
        PARC_labels_array = np.array(PARC_labels)
        number_clusters_for_target = len(thp1_labels)
        for cancer_class in thp1_labels:
            predict_class_array[PARC_labels_array == cancer_class] = 1
        for benign_class in pbmc_labels:
            predict_class_array[PARC_labels_array == benign_class] = 0
        predict_class_array.reshape((predict_class_array.shape[0], -1))
        error_rate = sum(error_count) / N
        n_target = tp + fn
        tnr = tn / n_pbmc
        fnr = fn / n_cancer
        tpr = tp / n_cancer
        fpr = fp / n_pbmc

        if tp != 0 or fn != 0: recall = tp / (tp + fn)  # ability to find all positives
        if tp != 0 or fp != 0: precision = tp / (tp + fp)  # ability to not misclassify negatives as positives
        if precision != 0 or recall != 0:
            f1_score = precision * recall * 2 / (precision + recall)

        majority_truth_labels = np.empty((len(true_labels), 1), dtype=object)
        for cluster_i in set(PARC_labels):
            cluster_i_loc = np.where(np.asarray(PARC_labels) == cluster_i)[0]
            true_labels = np.asarray(true_labels)
            majority_truth = self.func_mode(list(true_labels[cluster_i_loc]))
            majority_truth_labels[cluster_i_loc] = majority_truth

        majority_truth_labels = list(majority_truth_labels.flatten())
        accuracy_val = [error_rate, f1_score, tnr, fnr, tpr, fpr, precision,
                        recall, num_groups, n_target]

        return accuracy_val, predict_class_array, majority_truth_labels, number_clusters_for_target

    def run_PARC(self):
        print('input data has shape', self.data.shape[0], '(samples) x', self.data.shape[1], '(features)')
        if self.true_label is None:
            self.true_label = [1] * self.data.shape[0]
        list_roc = []

        time_start_total = time.time()

        time_start_knn = time.time()
        self.knn_struct = self.make_knn_struct()
        time_end_knn_struct = time.time() - time_start_knn
        # Query dataset, k - number of closest elements (returns 2 numpy arrays)
        self.run_subPARC()
        run_time = time.time() - time_start_total
        print('time elapsed {:.1f} seconds'.format(run_time))

        targets = list(set(self.true_label))
        N = len(list(self.true_label))
        self.f1_accumulated = 0
        self.f1_mean = 0
        self.stats_df = pd.DataFrame({'jac_std_global': [self.jac_std_global], 'dist_std_local': [self.dist_std_local],
                                      'runtime(s)': [run_time]})
        self.majority_truth_labels = []
        if len(targets) > 1:
            f1_accumulated = 0
            f1_acc_noweighting = 0
            for onevsall_val in targets:
                print('target is', onevsall_val)
                vals_roc, predict_class_array, majority_truth_labels, numclusters_targetval = self.accuracy(
                    onevsall=onevsall_val)
                f1_current = vals_roc[1]
                print('target', onevsall_val, 'has f1-score of %.2f' % (f1_current * 100))
                f1_accumulated = f1_accumulated + f1_current * (list(self.true_label).count(onevsall_val)) / N
                f1_acc_noweighting = f1_acc_noweighting + f1_current

                list_roc.append(
                    [self.jac_std_global, self.dist_std_local, onevsall_val] + vals_roc + [numclusters_targetval] + [
                        run_time])

            f1_mean = f1_acc_noweighting / len(targets)
            print("f1-score (unweighted) mean %.2f" % (f1_mean * 100), '%')
            print('f1-score weighted (by population) %.2f' % (f1_accumulated * 100), '%')

            df_accuracy = pd.DataFrame(list_roc,
                                       columns=['jac_std_global', 'dist_std_local', 'onevsall-target', 'error rate',
                                                'f1-score', 'tnr', 'fnr',
                                                'tpr', 'fpr', 'precision', 'recall', 'num_groups',
                                                'population of target', 'num clusters', 'clustering runtime'])

            self.f1_accumulated = f1_accumulated
            self.f1_mean = f1_mean
            self.stats_df = df_accuracy
            self.majority_truth_labels = majority_truth_labels
        return


def main():
    import pandas as pd
    import scanpy as sc
    import numpy as np
    from MulticoreTSNE import MulticoreTSNE as TSNE
    import matplotlib.pyplot as plt

    dataset = "Paul15" # GermlineLi

    ## Dataset Germline Li https://zenodo.org/record/1443566#.XZlhEkEzZ5y
    if dataset == "GermlineLi":
        df_expression_ids = pd.read_csv("/home/shobi/Trajectory/Code/Rcode/germline_human_female_weeks_li.csv", 'rt', delimiter=",")
        print(df_expression_ids.shape)
        #print(df_expression_ids[['cell_id',"week","ACTG2","STK31"]])[10:12]
        df_counts = pd.read_csv("/home/shobi/Trajectory/Code/Rcode/germline_human_female_weeks_li_filteredcounts.csv", 'rt', delimiter=",")
        df_ids = pd.read_csv("/home/shobi/Trajectory/Code/Rcode/germline_human_female_weeks_li_labels.csv", 'rt',
                                delimiter=",")
        #print(df_counts.shape, df_counts.head() ,df_ids.shape)
        #X_counts = df_counts.values
        #print(X_counts.shape)
        #varnames = pd.Categorical(list(df_counts.columns))


        adata_counts = sc.AnnData(df_counts,  obs=df_ids)
        print(adata_counts.obs)
        sc.pp.filter_cells(adata_counts,min_counts=1)
        print(adata_counts.n_obs)
        sc.pp.filter_genes(adata_counts, min_counts=1)  # only consider genes with more than 1 count
        print(adata_counts.X.shape)
        sc.pp.normalize_per_cell(  # normalize with total UMI count per cell
            adata_counts, key_n_counts='n_counts_all')
        print(adata_counts.X.shape, len(list(adata_counts.var_names)))


        filter_result = sc.pp.filter_genes_dispersion(  # select highly-variable genes
            adata_counts.X, flavor='cell_ranger', n_top_genes=1000, log=False)
        print(adata_counts.X.shape, len(list(adata_counts.var_names)))#, list(adata_counts.var_names))

        adata_counts = adata_counts[:, filter_result.gene_subset]
        print(adata_counts.X.shape, len(list(adata_counts.var_names)))#,list(adata_counts.var_names))
        # subset the genes
        sc.pp.normalize_per_cell(adata_counts)  # renormalize after filtering
        sc.pp.log1p(adata_counts)  # log transform: adata_counts.X = log(adata_counts.X + 1)
        sc.pp.scale(adata_counts)  # scale to unit variance and shift to zero mean
        sc.tl.pca(adata_counts, svd_solver='arpack', n_comps=20)

        true_label = list(adata_counts.obs['week'])
        sc.pp.neighbors(adata_counts, n_neighbors=10, n_pcs=20)
        sc.tl.draw_graph(adata_counts)
        sc.pl.draw_graph(adata_counts, color='gender_week', legend_loc='right margin', palette='jet')


    ## Dataset Paul15 https://scanpy-tutorials.readthedocs.io/en/latest/paga-paul15.html
    if dataset =='Paul15':
        adata_counts = sc.datasets.paul15()
        sc.pp.recipe_zheng17(adata_counts)
        sc.tl.pca(adata_counts, svd_solver='arpack', n_comps = 20)
        true_label = list(adata_counts.obs['paul15_clusters']) #PAUL


    p1 = PARC(adata_counts.obsm['X_pca'], true_label, jac_std_global=3, knn=10, too_big_factor=0.05)
    p1.run_PARC()
    labels = p1.labels



    embedding = TSNE().fit_transform(adata_counts.obsm['X_pca'])
    num_group = len(set(true_label))
    line=np.linspace(0,1,num_group)

    f, (ax1, ax2) = plt.subplots(1, 2, sharey=True)

    for color,group in zip(line,set(true_label)):
        where = np.where(np.array(true_label)==group)[0]
        ax1.scatter(embedding[where,0],embedding[where,1],label=group, c=plt.cm.jet(color))
    ax1.legend()
    ax1.set_title('true labels')
    num_parc_group = len(set(labels))
    line_parc = np.linspace(0,1,num_parc_group)
    for color, group in zip(line_parc, set(labels)):
        where = np.where(np.array(labels) == group)[0]
        ax2.scatter(embedding[where, 0], embedding[where, 1], label=group, c=plt.cm.jet(color))
    ax2.legend()
    f1_mean = p1.f1_mean*100
    ax2.set_title("parc labels F1 "+ str(int(f1_mean))+ "%" )

    plt.show()

    sc.pp.neighbors(adata_counts, n_neighbors=10, n_pcs=20)
    sc.tl.draw_graph(adata_counts)
    sc.pl.draw_graph(adata_counts, color='gender_week', legend_loc='right margin', palette='jet')






if __name__ == '__main__':
    main()


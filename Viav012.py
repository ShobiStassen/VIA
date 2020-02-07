import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix, csgraph
import scipy
import igraph as ig
import leidenalg
import time
import hnswlib
#jan2020 Righclick->GIT->Repository-> PUSH


def get_absorbing_clusters(cluster_graph):
    absorbing_clusters = 0
    return
def get_biased_weights():
    return
def get_Q_transient_transition(sources, targets, bias_weights, absorbing_clusters):
    return
def get_R_absorbing_transition(sources, targets, bias_weights, absorbing_clusters):
    return

def make_absorbing_markov(cluster_graph, pt):
    #cluster_graph is the vertex_cluster_graph made of sub_clusters in the finer iteration of PARC
    #pt is the pseudotime of each cluster in the graph
    absorbing_clusters = get_absorbing_clusters(cluster_graph)
    n_s = len(absorbing_clusters)
    adjacency_matrix = get_sparse_from_igraph(cluster_graph)
    sources, targets, weights  = zip(adjacency_matrix)
    sources, targets, bias_weights = get_biased_weights(sources, targets, weights, pt)
    Q = get_Q_transient_transition(sources, targets, bias_weights, absorbing_clusters)
    R = get_R_absorbing_transition(sources, targets, bias_weights, absorbing_clusters)
    n_t = Q.shape[0]
    I_t = np.identity(n_t)
    N = np.inv(I_t - Q)
    P_transition_absorbing = np.concatenate() #put together Q, R, Is and 0s
    return Q, R, N, P_transition_absorbing

def expected_num_steps(start_i, N):
    n_t = N.shape[0]
    N_steps = np.dot(N,np.ones(n_t))
    n_steps_i = N_steps[start_i]
    return n_steps_i

def absorption_probability(N, R, absorption_state_j):
    M = np.dot(N,R)
    vec_prob_end_in_j = M[:,absorption_state_j]
    return M, vec_prob_end_in_j


def most_likely_path(P_transition_absorbing_markov, start_i, end_i):
    graph_absorbing_markov =0 #ig()
    shortest_path =graph_absorbing_markov.shortest_path(start_i,end_i)
    print('the shortest path beginning at ', start_i, 'and ending in ', end_i, 'is:')
    return shortest_path

def draw_trajectory_dimred(X_dimred, cluster_labels, super_cluster_labels, super_edgelist, pt_sub,group_colour, x_lazy,alpha_teleport, projected_sc_pt):
    from scipy.interpolate import interp1d
    import matplotlib.pyplot as plt

    x = X_dimred[:,0]
    y = X_dimred[:, 1]

    sc_pseudotime_sub = np.asarray(cluster_labels) #placeholder
    for label_i in set(cluster_labels):
        loc_i = np.where(np.asarray(cluster_labels)==label_i)[0]
        print('ptsub_labeli',pt_sub[label_i],'loc_i', loc_i)
        sc_pseudotime_sub[loc_i]=pt_sub[label_i]
    print('sc_pseudo', sc_pseudotime_sub[0:200])
    df = pd.DataFrame({'x': x, 'y': y, 'cluster': cluster_labels, 'super_cluster': super_cluster_labels, 'pseudotime_sub':sc_pseudotime_sub},
                          columns=['x', 'y', 'cluster', 'super_cluster','pseudotime_sub'])
    df_mean = df.groupby('cluster').mean()
    print('df_mean', df_mean.head())
    df_super_mean = df.groupby('super_cluster').mean()
    print('super mean', df_super_mean.head())
    pt = df_super_mean['pseudotime_sub'].values
    pt_int = [int(i) for i in pt]
    pt_str = [str(i) for i in pt_int]
    print('super_edgelist',super_edgelist)

    for e_i, (start, end) in enumerate(super_edgelist):

        if pt[start] >=pt[end]:
            temp = end
            end = start
            start = temp
        print('edges', e_i, start, end, pt[start], pt[end])
        print('df head', df.head())
        x_i_start = df[df['super_cluster']==start].groupby('cluster').mean()['x'].values
        y_i_start = df[df['super_cluster'] == start].groupby('cluster').mean()['y'].values
        x_i_end = df[df['super_cluster'] == end].groupby('cluster').mean()['x'].values
        y_i_end = df[df['super_cluster'] == end].groupby('cluster').mean()['y'].values
        direction_arrow = 1
        if np.mean(np.asarray(x_i_end)) < np.mean(np.asarray(x_i_start)): direction_arrow = -1

        super_start_x =df[df['super_cluster']==start].mean()['x']
        super_end_x =df[df['super_cluster']==end].mean()['x']
        super_start_y = df[df['super_cluster'] == start].mean()['y']
        super_end_y= df[df['super_cluster'] == end].mean()['y']

        ext_maxx = False
        minx = min(super_start_x, super_end_x)
        maxx = max(super_start_x, super_end_x)

        x_val = np.concatenate([x_i_start, x_i_end])
        y_val = np.concatenate([y_i_start, y_i_end])
        idx_keep = np.where((x_val <= maxx) & (x_val >= minx))[0]
        x_val = x_val[idx_keep]
        y_val = y_val[idx_keep]

        super_mid_x = (super_start_x+super_end_x)/2
        super_mid_y = (super_start_y+super_end_y)/2
        from scipy.spatial import distance

        #x_val = np.concatenate([x_i_start, x_i_end])
        print('abs' ,abs(minx-maxx))
        if abs(minx - maxx) <= 1:
            print('very straight line')
            straight_level = 5
            noise = 0.01
            x_super = np.array(
                [super_start_x, super_end_x, super_start_x, super_end_x, super_start_x + noise, super_end_x + noise,
                 super_start_x - noise, super_end_x - noise, super_mid_x])
            y_super = np.array(
                [super_start_y, super_end_y, super_start_y, super_end_y, super_start_y + noise, super_end_y + noise,
                 super_start_y - noise, super_end_y - noise, super_mid_y])
        else:
            straight_level = 3
            noise = 0.1#0.05
            x_super = np.array(
                [super_start_x, super_end_x, super_start_x, super_end_x, super_start_x + noise, super_end_x + noise,
                 super_start_x - noise, super_end_x - noise])
            y_super = np.array(
            [super_start_y, super_end_y, super_start_y, super_end_y, super_start_y + noise, super_end_y + noise,
             super_start_y - noise, super_end_y - noise])

        #x_super = np.array([super_start_x, super_end_x])
        #y_super = np.array([super_start_y,super_end_y])


        for i in range(straight_level): #DO THE SAME FOR A MIDPOINT TOO TODO
            y_super = np.concatenate([y_super,y_super])
            x_super = np.concatenate([x_super, x_super])
        #noise=np.random.normal(0,0.05,np.size(x_super))
        #x_super = np.concatenate([np.concatenate([x_super,x_super+noise]),x_super])
        #y_super = np.concatenate([np.concatenate([y_super,y_super+noise]),y_super])

        y_super_max = max(y_super)
        y_super_min = min(y_super)

        print('xval', x_val)
        print('yval', y_val)
        list_selected_clus = list(zip(x_val, y_val))
        #idx_keep = np.where((x_val<= maxx) & (x_val>=minx))[0]
        if (len(list_selected_clus)>=1) & (straight_level==5):

            dist = distance.cdist([(super_mid_x, super_mid_y)], list_selected_clus, 'euclidean')
            print('dist', dist)
            if len(list_selected_clus)>=2: k=2
            else: k=1
            midpoint_loc = dist[0].argsort()[:k]#np.where(dist[0]==np.min(dist[0]))[0][0]
            print('midpoint loc', midpoint_loc)
            midpoint_xy = []
            for i in range(k):
                midpoint_xy.append(list_selected_clus[midpoint_loc[i]])

            #midpoint_xy = list_selected_clus[midpoint_loc]
            noise=0.05
            print(midpoint_xy, 'is the midpoint between clus', pt[start],'and ', pt[end])
            if k==1:
                mid_x = np.array([midpoint_xy[0][0], midpoint_xy[0][0] + noise, midpoint_xy[0][0] - noise])#,midpoint_xy[1][0], midpoint_xy[1][0] + noise, midpoint_xy[1][0] - noise])
                mid_y = np.array([midpoint_xy[0][1], midpoint_xy[0][1] + noise, midpoint_xy[0][1] - noise])#,midpoint_xy[1][1], midpoint_xy[1][1] + noise, midpoint_xy[1][1] - noise])
            if k==2:
                mid_x = np.array([midpoint_xy[0][0], midpoint_xy[0][0] + noise, midpoint_xy[0][0] - noise,midpoint_xy[1][0], midpoint_xy[1][0] + noise, midpoint_xy[1][0] - noise])
                mid_y = np.array([midpoint_xy[0][1], midpoint_xy[0][1] + noise, midpoint_xy[0][1] - noise ,midpoint_xy[1][1], midpoint_xy[1][1] + noise, midpoint_xy[1][1] - noise])
            for i in range(3):
                mid_x = np.concatenate([mid_x, mid_x])
                mid_y = np.concatenate([mid_y, mid_y])

            x_super = np.concatenate([x_super, mid_x])
            y_super = np.concatenate([y_super, mid_y])
        x_val = np.concatenate([x_val,x_super])
        y_val = np.concatenate([y_val, y_super])
        z = np.polyfit(x_val, y_val, 2)

        xp = np.linspace(minx,maxx, 500)
        p = np.poly1d(z)
        smooth = p(xp)
        if ext_maxx == False:
            idx_keep = np.where((xp<=(maxx)) &(xp>=(minx)))[0]# minx+3
        else:
            idx_keep = np.where((xp <= (maxx)) & (xp >= (minx)))[0] #maxx-3
        plt.plot(xp[idx_keep],smooth[idx_keep], linewidth = 3, c='black')
        med_loc = np.where(xp==np.median(xp[idx_keep]))[0]
        mean_temp = np.mean(xp[idx_keep])
        closest_val = xp[idx_keep][0]
        closest_loc = idx_keep[0]
        # print('mean_temp is', mean_temp)
        # print('closest val is', closest_val)
        # print('closest loc is', closest_loc)
        for i, xp_val in enumerate(xp[idx_keep]):
            #print('dist1',abs(xp_val - mean_temp))
            #print('dist2', abs(closest_val - mean_temp))
            if abs(xp_val - mean_temp) < abs(closest_val-mean_temp):
                #print('closest val is now', xp_val, 'at', idx_keep[i])
                closest_val = xp_val
                closest_loc = idx_keep[i]

        if direction_arrow ==1: plt.arrow(xp[closest_loc], smooth[closest_loc],xp[closest_loc+1]-xp[closest_loc], smooth[closest_loc+1]-smooth[closest_loc], shape='full', lw=0, length_includes_head = True, head_width = 0.5, color = 'black')#, head_starts_at_zero = direction_arrow )
        else: plt.arrow(xp[closest_loc], smooth[closest_loc],xp[closest_loc-1]-xp[closest_loc], smooth[closest_loc-1]-smooth[closest_loc], shape='full', lw=0, length_includes_head = True, head_width = 0.5, color = 'black')

    #df_mean['cluster'] = df_mean.index()
    x_cluster = df_mean['x']
    y_cluster = df_mean['y']
    print('x-cluster and y-cluster')
    print(x_cluster, y_cluster)
    x_new = np.linspace(x_cluster.min(),x_cluster.max(),500)
    num_parc_group = len(set(cluster_labels))
    line_parc = np.linspace(0, 1, num_parc_group)
    for color, group in zip(line_parc, set(cluster_labels)):
        where = np.where(np.array(cluster_labels) == group)[0]
        plt.scatter(X_dimred[where, 0], X_dimred[where, 1], label=group, c=group_colour[group], alpha=0.5)
    plt.legend()
    # plt.scatter(X_dimred[:,0], X_dimred[:,1], alpha=0.5)
    plt.scatter(x_cluster,y_cluster,c = 'red')
    plt.scatter(df_super_mean['x'], df_super_mean['y'], c='black')
    for i, type in enumerate(pt_str):
        plt.text(df_super_mean['x'][i], df_super_mean['y'][i], type)
    plt.title('lazy:'+str(x_lazy)+' teleport'+str(alpha_teleport))
    plt.show()
    return

def local_pruning_clustergraph(adjacency_matrix, local_pruning_std = 0.5, global_pruning_std =2):
    #larger pruning_std factor means less pruning
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
        to_keep_index = np.where(row > np.mean(row)-local_pruning_std*np.mean(row))[0] #we take [1] because row is a 2D matrix, not a 1D matrix like in other cases  # 0*std

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

    sources, targets = cluster_graph_csr.nonzero()
    mask = np.zeros(len(sources), dtype=bool)
    print('mask is', mask)
    cluster_graph_csr.data = cluster_graph_csr.data / (np.std(cluster_graph_csr.data)) #normalize
    threshold_global = np.mean(cluster_graph_csr.data) - global_pruning_std* np.std(cluster_graph_csr.data)
    mask |= (cluster_graph_csr.data < (threshold_global))  # smaller Jaccard weight means weaker edge

    cluster_graph_csr.data[mask] = 0
    cluster_graph_csr.eliminate_zeros()
    sources, targets = cluster_graph_csr.nonzero()
    edgelist = list(zip(sources, targets))


    #cluster_graph_csr.data = locallytrimmed_sparse_vc.data / (np.std(locallytrimmed_sparse_vc.data))
    edgeweights = cluster_graph_csr.data / (np.std(cluster_graph_csr.data))

    trimmed_n= (initial_links_n-final_links_n)/initial_links_n
    trimmed_n_glob = (initial_links_n - len(edgeweights)) / initial_links_n
    print("percentage links trimmed from local pruning relative to start", trimmed_n)
    print("percentage links trimmed from global pruning relative to start", trimmed_n_glob)
    return edgeweights, edgelist

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
    def __init__(self, data, true_label=None, anndata=None, dist_std_local=2, jac_std_global='median', keep_all_local_dist='auto',
                 too_big_factor=0.4, small_pop=10, jac_weighted_edges=True, knn=30, n_iter_leiden=5, pseudotime= False, root=0, path= '/home/shobi/Trajectory/', seed = 99, super_cluster_labels = False, super_node_degree_list = False, x_lazy=0.95, alpha_teleport = 0.99):
        # higher dist_std_local means more edges are kept
        # highter jac_std_global means more edges are kept
        if keep_all_local_dist == 'auto':
            if data.shape[0] > 300000:
                keep_all_local_dist = True  # skips local pruning to increase speed
            else:
                keep_all_local_dist = False

        self.data = data
        self.true_label = true_label
        self.anndata = anndata
        self.dist_std_local = dist_std_local
        self.jac_std_global = jac_std_global  ##0.15 is also a recommended value performing empirically similar to 'median'
        self.keep_all_local_dist = keep_all_local_dist
        self.too_big_factor = too_big_factor  ##if a cluster exceeds this share of the entire cell population, then the PARC will be run on the large cluster. at 0.4 it does not come into play
        self.small_pop = small_pop  # smallest cluster population to be considered a community
        self.jac_weighted_edges = jac_weighted_edges
        self.knn = knn
        self.n_iter_leiden = n_iter_leiden
        self.pseudotime = pseudotime
        self.root = root
        self.path = path
        self.seed = seed
        self.super_cluster_labels = super_cluster_labels
        self.super_node_degree_list = super_node_degree_list
        self.x_lazy = x_lazy #1-x = probability of staying in same node
        self.alpha_teleport = alpha_teleport #1-alpha is probability of jumping

    def compute_hitting_time(self, sparse_graph, root, x_lazy, alpha_teleport, number_eig=0):
        # 1- alpha is the probabilty of teleporting
        # 1- x_lazy is the probability of staying in current state (be lazy)
        beta_teleport = 2 * (1 - alpha_teleport) / (2 - alpha_teleport)
        N = sparse_graph.shape[0]
        sparse_graph = scipy.sparse.csr_matrix(sparse_graph)
        print('start compute hitting')
        A = scipy.sparse.csr_matrix.todense(sparse_graph)  # A is the adjacency matrix
        print('is graph symmetric', (A.transpose() == A).all())
        lap = csgraph.laplacian(sparse_graph, normed=False)  # compute regular laplacian (normed = False) to infer the degree matrix where D = L+A
        # see example and definition in the SciPy ref https://docs.scipy.org/doc/scipy/reference/generated/scipy.sparse.csgraph.laplacian.html
        A = scipy.sparse.csr_matrix.todense(lap)
        print('is laplacian symmetric', (A.transpose() == A).all())
        deg = sparse_graph + lap  # Recall that L=D-A (modified for weighted where D_ii is sum of edge weights and A_ij is the weight of particular edge)
        deg.data = 1 / np.sqrt(deg.data)  ##inv sqrt of degree matrix
        deg[deg == np.inf] = 0
        norm_lap = csgraph.laplacian(sparse_graph, normed=True)  # returns symmetric normalized D^-.5 xL x D^-.5
        Id = np.zeros((N, N), float)
        np.fill_diagonal(Id, 1)
        norm_lap = scipy.sparse.csr_matrix.todense(norm_lap)

        eig_val, eig_vec = np.linalg.eigh(norm_lap)  # eig_vec[:,i] is eigenvector for eigenvalue eig_val[i]
        print('eig val', eig_val.shape, eig_val)
        if number_eig == 0: number_eig = eig_vec.shape[1]
        print('number of eig vec', number_eig)
        Greens_matrix = np.zeros((N, N), float)
        beta_norm_lap = np.zeros((N, N), float)
        Xu = np.zeros((N, N))
        Xu[:, root] = 1
        Id_Xv = np.zeros((N, N), int)
        np.fill_diagonal(Id_Xv, 1)
        Xv_Xu = Id_Xv - Xu
        start_ = 0
        if alpha_teleport == 1:
            start_ = 1  # if there are no jumps (alph_teleport ==1), then the first term in beta-normalized Green's function will have 0 in denominator (first eigenvalue==0)

        for i in range(start_, number_eig):  # 0 instead of 1th eg
            print(i, 'th eigenvalue is', eig_val[i])
            vec_i = eig_vec[:, i]
            factor = beta_teleport + 2 * eig_val[i] * x_lazy * (1 - beta_teleport)
            print('factor', 1 / factor)

            vec_i = np.reshape(vec_i, (-1, 1))
            eigen_vec_mult = vec_i.dot(vec_i.T)
            Greens_matrix = Greens_matrix + (
                        eigen_vec_mult / factor)  # Greens function is the inverse of the beta-normalized laplacian
            beta_norm_lap = beta_norm_lap + (eigen_vec_mult * factor)  # beta-normalized laplacian

        deg = scipy.sparse.csr_matrix.todense(deg)
        temp = Greens_matrix.dot(deg)
        temp = deg.dot(temp) * beta_teleport
        hitting_matrix = np.zeros((N, N), float)
        diag_row = np.diagonal(temp)
        for i in range(N):
            hitting_matrix[i, :] = diag_row - temp[i, :]

        roundtrip_commute_matrix = hitting_matrix + hitting_matrix.T
        temp = Xv_Xu.dot(temp)
        final_hitting_times = np.diagonal(
            temp)  ## number_eig x 1 vector of hitting times from root (u) to number_eig of other nodes
        roundtrip_times = roundtrip_commute_matrix[root, :]
        return final_hitting_times, roundtrip_times
    def project_hittingtimes_sc(self):
        knn=30
        neighbor_array, distance_array = self.knn_struct.knn_query(self.data, k=knn)
        labels = np.asarray(self)
        sc_pt = labels
        mean_weight = 0
        i=0
        for row in neighbor_array:
            print('row in neighbor array of cells', row)
            neighboring_clus = labels[row]
            print('neighbor clusters labels', neighboring_clus)
            for clus_i in set(list(neighboring_clus)):
                mean_weight = mean_weight + np.sum(neighbor_array==clus_i)/knn
                print('mean weight',mean_weight)
            sc_pt[i]=mean_weight
            i=i+1
        self.single_cell_pt = sc_pt
        return


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
                                                 n_iterations=self.n_iter_leiden, seed=self.seed)
        else:
            partition = leidenalg.find_partition(G_sim, leidenalg.ModularityVertexPartition,
                                                 n_iterations=self.n_iter_leiden, seed=self.seed)
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

        do_while_time = time.time()
        while (small_pop_exist == True) & (time.time()-do_while_time <5):
            small_pop_list = []
            small_pop_exist = False
            for cluster in set(list(PARC_labels_leiden.flatten())):
                population = len(np.where(PARC_labels_leiden == cluster)[0])
                if population < 10:
                    small_pop_exist = True
                    #print(cluster, ' has small population of', population, )
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

        print('finished labels')
        #self.anndata.obs['parc_label'] = self.labels

        #cma1_cluster = self.anndata.obs.groupby('parc_label').mean('Cma1')

        return PARC_labels_leiden
    def recompute_weights(self, clustergraph_ig, pop_list_raw):
        sparse_clustergraph = get_sparse_from_igraph(clustergraph_ig, weight_attr='weight')
        n = sparse_clustergraph.shape[0]
        sources, targets = sparse_clustergraph.nonzero()
        edgelist = list(zip(sources, targets))
        weights = sparse_clustergraph.data
        #print('edgelist of combined clustergraph', edgelist)
        #print('edge weights of combined clustergraph', weights)
        new_weights = []
        i=0
        for s,t in edgelist:
            pop_s = pop_list_raw[s]
            pop_t = pop_list_raw[t]
            w = weights[i]
            nw = w/(pop_s+pop_t)#*
            new_weights.append(nw)
            #print('old and new', w, nw)
            i = i+1
            scale_factor = max(new_weights)-min(new_weights)
            wmin = min(new_weights)
            wmax = max(new_weights)
        new_weights = [(i-wmin)*1/scale_factor for i in new_weights]

        sparse_clustergraph = csr_matrix((np.array(new_weights), (sources, targets)),
                   shape=(n, n))
        # print('new weights', new_weights)
        # print(sparse_clustergraph)
        # print('reweighted sparse clustergraph')
        # print(sparse_clustergraph)
        sources, targets = sparse_clustergraph.nonzero()
        edgelist = list(zip(sources, targets))
        return sparse_clustergraph, edgelist


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
                                                 n_iterations=self.n_iter_leiden, seed=self.seed)
            print(time.time() - start_leiden)
        else:
            start_leiden = time.time()
            # print('call leiden on unweighted graph', self.n_iter_leiden, 'iterations')
            partition = leidenalg.find_partition(G_sim, leidenalg.ModularityVertexPartition,
                                                 n_iterations=self.n_iter_leiden,seed=self.seed)
            print(time.time() - start_leiden)
        time_end_PARC = time.time()
        # print('Q= %.1f' % (partition.quality()))
        PARC_labels_leiden = np.asarray(partition.membership)
        PARC_labels_leiden = np.reshape(PARC_labels_leiden, (n_elements, 1))

        pop_list_1 = []
        for item in set(list(PARC_labels_leiden.flatten())):
            pop_list_1.append([item, list(PARC_labels_leiden.flatten()).count(item)])
        print(pop_list_1)
        too_big = False

        # print('labels found after Leiden', set(list(PARC_labels_leiden.T)[0])) will have some outlier clusters that need to be added to a cluster if a cluster has members that are KNN

        cluster_i_loc = np.where(PARC_labels_leiden == 0)[
            0]  # the 0th cluster is the largest one. so if cluster 0 is not too big, then the others wont be too big either
        pop_i = len(cluster_i_loc)
        print('largest cluster population', pop_i, too_big_factor, n_elements)
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
            print('new set of labels ')
            pop_list_1=[]
            for item in set(list(PARC_labels_leiden.flatten())):
                pop_list_1.append([item, list(PARC_labels_leiden.flatten()).count(item)])
            print(pop_list_1, set(PARC_labels_leiden))
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
        n_clus = len(set(self.labels))

        ##determine majority truth

        majority_truth_labels = np.empty((n_elements, 1), dtype=object)
        graph_node_label = []

        if self.pseudotime == True:

            ## Make cluster-graph (1)
            vc_graph = ig.VertexClustering(G_sim, membership=PARC_labels_leiden)
            #print('vc graph G_sim', vc_graph)


            vc_graph =  vc_graph.cluster_graph(combine_edges='sum')

            print('vc graph G_sim', vc_graph)
            ## Reweight clustergraph (2)

            reweighted_sparse_vc,edgelist = self.recompute_weights(vc_graph, pop_list_raw)
            #reweighted_vc_ig = ig.Graph(edgelist, edge_attrs={'weight': reweighted_sparse_vc.data.tolist()})
            #reweighted_vc_ig = reweighted_vc_ig.simplify(combine_edges='sum')






            # Local pruning on reweighted clustergraph (3)
            edgeweights, edgelist = local_pruning_clustergraph(reweighted_sparse_vc, local_pruning_std=0)
            print('edgelist of reweighted sparse vc', edgelist, edgeweights)

            locallytrimmed_g = ig.Graph(edgelist, edge_attrs={'weight': edgeweights.tolist()})

            locallytrimmed_g = locallytrimmed_g.simplify(combine_edges='sum')
            min_deg = 1000
            super_min_deg = 1000
            found_super_and_sub_root = False
            deg_list = locallytrimmed_g.degree()
            self.node_degree_list = deg_list
            for cluster_i in range(len(set(PARC_labels_leiden))):
                cluster_i_loc = np.where(np.asarray(PARC_labels_leiden) == cluster_i)[0]
                true_labels = np.asarray(self.true_label)
                majority_truth = self.func_mode(list(true_labels[cluster_i_loc]))
                print('cluster', cluster_i, 'has majority', majority_truth, 'with degree list', deg_list)
                if self.super_cluster_labels != False:
                    super_majority_cluster = self.func_mode(list(np.asarray(self.super_cluster_labels)[cluster_i_loc]))
                    super_majority_cluster_loc = np.where(np.asarray(self.super_cluster_labels) == super_majority_cluster)[0]
                    super_majority_truth = self.func_mode(list(true_labels[super_majority_cluster_loc]))
                    super_node_degree = self.super_node_degree_list[super_majority_cluster]
                    if (majority_truth == 'M1') & (super_majority_truth =='M1'):
                        if super_node_degree < super_min_deg:
                        #if deg_list[cluster_i] < min_deg:
                            found_super_and_sub_root = True
                            self.root = cluster_i
                            min_deg = deg_list[cluster_i]
                            super_min_deg = super_node_degree
                            print('new root is', self.root, ' with degree', min_deg, 'and super node degree', super_min_deg)
                majority_truth_labels[cluster_i_loc] = 'w' + str(majority_truth) + 'c' + str(cluster_i)
                graph_node_label.append('w' + str(majority_truth) + 'c' + str(cluster_i))
            if (self.super_cluster_labels == False) | (found_super_and_sub_root == False):
                print('self.super_cluster_labels', self.super_cluster_labels,' foundsuper_cluster_sub and super root', found_super_and_sub_root)
                for cluster_i in range(len(set(PARC_labels_leiden))):
                    cluster_i_loc = np.where(np.asarray(PARC_labels_leiden) == cluster_i)[0]
                    true_labels = np.asarray(self.true_label)
                    majority_truth = self.func_mode(list(true_labels[cluster_i_loc]))
                    print('cluster', cluster_i, 'has majority', majority_truth, 'with degree list', deg_list)
                    if (majority_truth == 'M1'):
                        print('did not find a super and sub cluster with majority M1')
                        if deg_list[cluster_i] < min_deg:
                            self.root = cluster_i
                            min_deg = deg_list[cluster_i]
                            print('new root is', self.root, ' with degree', min_deg)


            print('graph node label', graph_node_label)
            majority_truth_labels = list(majority_truth_labels.flatten())
            print('locallytrimmed_g', locallytrimmed_g)
            locallytrimmed_sparse_vc = get_sparse_from_igraph(locallytrimmed_g, weight_attr='weight')
            layout = locallytrimmed_g.layout_fruchterman_reingold() ##final layout based on locally trimmed
            locallytrimmed_g.vs["label"] = graph_node_label
            locallytrimmed_sparse_vc_copy = locallytrimmed_sparse_vc.copy()

            #globally trimmed link
            sources, targets = locallytrimmed_sparse_vc.nonzero()
            edgelist = list(zip(sources.tolist(), targets.tolist()))
            edgelist = set(tuple(sorted(l)) for l in edgelist) # keep only one of (0,1) and (1,0)
            self.edgelist = edgelist
            print('coarse edgelist', edgelist)

            mask = np.zeros(len(sources), dtype=bool)
            print('mean and std', np.mean(locallytrimmed_sparse_vc.data), np.std(locallytrimmed_sparse_vc.data))
            locallytrimmed_sparse_vc.data = locallytrimmed_sparse_vc.data / (np.std(locallytrimmed_sparse_vc.data))

            # print('after normalization',sparse_vf)
            threshold_global = np.mean(locallytrimmed_sparse_vc.data) -0*np.std(locallytrimmed_sparse_vc.data)
            mask |= (locallytrimmed_sparse_vc.data < (threshold_global))  # smaller Jaccard weight means weaker edge
            print('sum of mask', sum(mask), 'at threshold of', threshold)
            locallytrimmed_sparse_vc.data[mask] = 0
            locallytrimmed_sparse_vc.eliminate_zeros()

            sources, targets = locallytrimmed_sparse_vc.nonzero()
            edgelist = list(zip(sources.tolist(), targets.tolist()))
            globallytrimmed_g = ig.Graph(n=n_clus, edges=edgelist, edge_attrs={'weight': locallytrimmed_sparse_vc.data.tolist()})
            globallytrimmed_g = globallytrimmed_g.simplify(combine_edges='sum')
            #layout = globallytrimmed_g.layout_fruchterman_reingold()
            globallytrimmed_g.vs["label"] = graph_node_label


            # compute hitting times (4)

            root = self.root
            x_lazy = self.x_lazy
            alpha_teleport = self.alpha_teleport
            locallytrimmed_sparse_vc = locallytrimmed_sparse_vc_copy ##hitting times are computed based on the locally trimmed graph without any global pruning
            hitting_times, roundtrip_times = self.compute_hitting_time(locallytrimmed_sparse_vc, root=root, x_lazy=x_lazy, alpha_teleport=alpha_teleport)

            self.hitting_times = hitting_times*1000
            print('round trip times:')
            print( list(zip(range(len(hitting_times)), roundtrip_times)))
            hitting_times = np.asarray(hitting_times)*1000

            print('final hitting times:')
            print(list(zip(range(len(hitting_times)), hitting_times)))
            # plotting with threshold (thresholding the upper limit to make color palette more readable)
            remove_outliers = hitting_times[hitting_times<np.mean(hitting_times)+np.std(hitting_times)]
            threshold = np.mean(remove_outliers) + 1* np.std(remove_outliers)
            print('threshold', threshold)
            th_hitting_times = [x if x < threshold else threshold for x in hitting_times]

            remove_outliers_low = hitting_times[hitting_times < (np.mean(hitting_times) - 0.3*np.std(hitting_times))]
            threshold_low = np.mean(remove_outliers_low) - 0.3 * np.std(remove_outliers_low)
            print('thresh low', threshold_low)
            th_hitting_times = [x if x > threshold_low else threshold_low for x in th_hitting_times]
            #scaled_hitting_times = (th_hitting_times - np.min(th_hitting_times))*100/(threshold)

            scaled_hitting_times = (th_hitting_times - np.min(th_hitting_times))
            scaled_hitting_times = scaled_hitting_times * (1000/np.max(scaled_hitting_times))

            self.scaled_hitting_times = scaled_hitting_times


            #threshold = np.mean(scaled_hitting_times)+0.25*np.std(scaled_hitting_times)
            threshold = int(threshold)
            scaled_hitting_times = scaled_hitting_times.astype(int)
            #print('scaled hitting times')
            print(scaled_hitting_times)
            pal = ig.drawing.colors.AdvancedGradientPalette(['yellow', 'green','blue'], n=1001)

            all_colors = []
            #print('100 scaled hitting' ,scaled_hitting_times)
            for i in scaled_hitting_times:
                all_colors.append(pal.get(int(i))[0:3])
            #print('extract all colors', zip(scaled_hitting_times,all_colors))


            locallytrimmed_g.vs['hitting_times'] =scaled_hitting_times

            locallytrimmed_g.vs['color']=[pal.get(i)[0:3] for i in scaled_hitting_times]
            import matplotlib.colors as colors
            import matplotlib.cm as cm
            self.group_color = [colors.to_hex(v) for v in locallytrimmed_g.vs['color']] #based on ygb scale
            viridis_cmap = cm.get_cmap('viridis_r')

            self.group_color_cmap =[colors.to_hex(v) for v in viridis_cmap(scaled_hitting_times/1000)] #based on ygb scale
            print('group color', self.group_color)
            #ig.plot(locallytrimmed_g, "/home/shobi/Trajectory/Datasets/Toy/Toy_bifurcating/vc_graph_example_locallytrimmed_colornode_"+str(root)+"lazy"+str(lazy_i)+'jac'+str(self.jac_std_global)+".svg", layout=layout, edge_width=[e['weight']*1 for e in locallytrimmed_g.es], vertex_label=graph_node_label)
            svgpath_local =self.path+"vc_graph_locallytrimmed_Root" + str(root) + "lazy" + str(x_lazy) + 'JacG' + str(self.jac_std_global) + 'toobig'+str(int(self.too_big_factor*100))+ ".svg"
            print('svglocal', svgpath_local)
            ig.plot(locallytrimmed_g,svgpath_local, layout=layout,
                    edge_width=[e['weight'] * 1 for e in locallytrimmed_g.es], vertex_label=graph_node_label)
            #hitting_times = compute_hitting_time(sparse_vf, root=1)
            #print('final hitting times:', list(zip(range(len(hitting_times)), hitting_times)))


            globallytrimmed_g.vs['color'] = [pal.get(i)[0:3] for i in scaled_hitting_times]
            ig.plot(globallytrimmed_g,
                    self.path+"/vc_graph_globallytrimmed_Root" + str(
                        root) + "Lazy" + str(x_lazy) + 'JacG' + str(self.jac_std_global) + 'toobig'+str(int(100*self.too_big_factor))+".svg", layout=layout,
                    edge_width=[e['weight'] * .1 for e in globallytrimmed_g.es], vertex_label=graph_node_label, main ='lazy:'+str(x_lazy)+' alpha:'+str(alpha_teleport))

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
        pop_list = []
        for item in set(list(self.true_label)):
            pop_list.append([item, list(self.true_label).count(item)])
        print("population composition", pop_list)
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

    dataset = "Toy3"#"Paul15"#""Toy1" # GermlineLi #Toy1

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
        sc.tl.pca(adata_counts, svd_solver='arpack')
        true_label = list(adata_counts.obs['paul15_clusters']) #PAUL
        #sc.pp.neighbors(adata_counts, n_neighbors=10)
        #sc.tl.draw_graph(adata_counts)
        #sc.pl.draw_graph(adata_counts, color=['paul15_clusters', 'Cma1'], legend_loc='on data')

    if dataset =="Toy3":
        if dataset == "Toy1":
            df_counts = pd.read_csv("/home/shobi/Trajectory/Datasets/Toy1/Toy_bifurcating/toy_bifurcating_n3000.csv",'rt', delimiter=",")
            df_ids = pd.read_csv("/home/shobi/Trajectory/Datasets/Toy1/Toy_bifurcating/toy_bifurcating_n3000_ids.csv", 'rt',  delimiter=",")
        if dataset == "Toy2":
            df_counts = pd.read_csv("/home/shobi/Trajectory/Datasets/Toy2/toy_multifurcating_n6000.csv", 'rt',
                                delimiter=",")
            df_ids = pd.read_csv("/home/shobi/Trajectory/Datasets/Toy2/toy_multifurcating_n6000_ids.csv", 'rt',
                             delimiter=",")
        if dataset == "Toy3":
            df_counts = pd.read_csv("/home/shobi/Trajectory/Datasets/Toy3/toy_multifurcating_M6_n6000d1000.csv", 'rt',
                                    delimiter=",")
            df_ids = pd.read_csv("/home/shobi/Trajectory/Datasets/Toy3/toy_multifurcating_M6_n6000d1000_ids.csv", 'rt',
                                 delimiter=",")
        df_ids['cell_id_num'] = [int(s[1: :]) for s in df_ids['cell_id']]

        print("shape",df_counts.shape, df_ids.shape)
        df_counts = df_counts.drop('Unnamed: 0', 1)

        print(df_ids)
        df_ids = df_ids.sort_values(by=['cell_id_num'] )
        df_ids=df_ids.reset_index(drop=True)
        print('new df_ids')
        print(df_ids)
        true_label = df_ids['group_id']

        adata_counts = sc.AnnData(df_counts, obs=df_ids)
        #sc.pp.recipe_zheng17(adata_counts, n_top_genes=20) not helpful for toy data
        sc.tl.pca(adata_counts, svd_solver='arpack',n_comps=50)

        sc.pp.neighbors(adata_counts, n_neighbors=30, n_pcs=20)#4
        sc.tl.draw_graph(adata_counts)
        sc.pl.draw_graph(adata_counts, color='group_id', legend_loc='on data') #force-directed layout
        start_dfmap = time.time()
        sc.tl.diffmap(adata_counts)
        print('time taken to get diffmap given knn', time.time() - start_dfmap)
        sc.pp.neighbors(adata_counts, n_neighbors=30, use_rep='X_diffmap')#4
        sc.tl.draw_graph(adata_counts)
        sc.pl.draw_graph(adata_counts, color='group_id', legend_loc='on data')
        sc.tl.louvain(adata_counts, resolution=1.0)
        sc.tl.paga(adata_counts, groups='louvain')
        #sc.pl.paga(adata_counts, color=['louvain','group_id'])
        adata_counts.uns['iroot'] = np.flatnonzero(adata_counts.obs['louvain'] == '2')[0]
        sc.tl.dpt(adata_counts)
        sc.pl.paga(adata_counts, color=['louvain', 'group_id', 'dpt_pseudotime'])
        #X = df_counts.values
        print(df_counts)

        from sklearn.decomposition import PCA
        pca = PCA(n_components=20)
        pc = pca.fit_transform(df_counts)
        p0 = PARC(pc, true_label, jac_std_global=2, knn=4, too_big_factor=0.4, pseudotime=True,path = "/home/shobi/Trajectory/Datasets/"+dataset+"/", root = 3) #*.4
        p0.run_PARC()
        super_labels = p0.labels

        super_edges = p0.edgelist
        super_pt = p0.scaled_hitting_times #pseudotime pt
        p1 = PARC(pc, true_label, jac_std_global=1, knn=4, too_big_factor=0.05, path = "/home/shobi/Trajectory/Datasets/"+dataset+"/",pseudotime=True, root =61, super_cluster_labels=super_labels, super_node_degree_list=p0.node_degree_list) #*.4
        p1.run_PARC()
        labels = p1.labels
    #p1 = PARC(adata_counts.obsm['X_pca'], true_label, jac_std_global=1, knn=5, too_big_factor=0.05, anndata= adata_counts, small_pop=2)
    #p1.run_PARC()
    #labels = p1.labels
    print('start tsne')
    n_downsample = 10000
    if len(labels)>n_downsample:
        idx = np.random.randint(len(labels), size=2000)
        embedding = TSNE().fit_transform(adata_counts.obsm['X_pca'][idx,:])
    else:
        embedding = TSNE().fit_transform(adata_counts.obsm['X_pca'])
        idx = np.random.randint(len(labels), size=len(labels))
    print('end tsne')
    draw_trajectory_dimred(embedding, labels, super_labels, super_edges, p1.hitting_times,p1.group_color_cmap, p1.x_lazy, p1.alpha_teleport)
    #embedding = TSNE().fit_transform(pc)
    num_group = len(set(true_label))
    line=np.linspace(0,1,num_group)

    f, (ax1, ax2) = plt.subplots(1, 2, sharey=True)

    for color,group in zip(line,set(true_label)):
        if len(labels) > n_downsample:
            where = np.where(np.array(true_label)[idx]==group)[0]
        else: where = np.where(np.array(true_label) == group)[0]
        ax1.scatter(embedding[where,0],embedding[where,1],label=group, c=plt.cm.jet(color))
    ax1.legend()
    ax1.set_title('true labels')
    num_parc_group = len(set(labels))
    line_parc = np.linspace(0,1,num_parc_group)
    for color, group in zip(line_parc, set(labels)):
        if len(labels)>n_downsample: where = np.where(np.array(labels)[idx] == group)[0]
        else: where = np.where(np.array(labels) == group)[0]
        print('color', int(p1.scaled_hitting_times[group]))
        ax2.scatter(embedding[where, 0], embedding[where, 1], label=group, c=p1.group_color_cmap[group])
    #for color, group in zip(line_parc, set(labels)):
    #    where = np.where(np.array(labels) == group)[0]
    #    ax2.scatter(embedding[where, 0], embedding[where, 1], label=group, c=plt.cm.jet(color))
    ax2.legend()
    f1_mean = p1.f1_mean*100
    ax2.set_title("parc labels F1 "+ str(int(f1_mean))+ "%" )

    plt.show()

    sc.pp.neighbors(adata_counts, n_neighbors=10, n_pcs=20)
    sc.tl.draw_graph(adata_counts)
    sc.pl.draw_graph(adata_counts, color='gender_week', legend_loc='right margin', palette='jet')






if __name__ == '__main__':
    main()


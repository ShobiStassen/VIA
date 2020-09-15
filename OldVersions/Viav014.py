import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix, csgraph
import scipy
import igraph as ig
import leidenalg
import time
import hnswlib
import matplotlib.pyplot as plt
import math


# version before translating chinese on Feb13
# jan2020 Righclick->GIT->Repository-> PUSH

def simulate_multinomial(vmultinomial):
    r = np.random.uniform(0.0, 1.0)
    CS = np.cumsum(vmultinomial)
    CS = np.insert(CS, 0, 0)
    m = (np.where(CS < r))[0]
    nextState = m[len(m) - 1]

    return nextState


def get_absorbing_clusters(cluster_graph):
    absorbing_clusters = 0
    return


def get_biased_weights(edgelist, weights, pt):
    print('weights', type(weights), weights)
    # small nu means less biasing (0.5 is quite mild)
    # larger nu (in our case 1/nu) means more aggressive biasing https://en.wikipedia.org/wiki/Generalised_logistic_function
    print(len(edgelist), len(weights))
    bias_weight = []
    b = 1  # 0.5
    K = 1
    c = 0
    C = 1
    nu = 1
    high_weights_th = np.mean(weights)
    high_pt_th = np.percentile(np.asarray(pt), 80)
    loc_high_weights = np.where(weights > high_weights_th)[0]
    loc_high_pt = np.where(np.asarray(pt) > high_pt_th)[0]
    print('weight  hi th', high_weights_th)
    print('loc hi pt', loc_high_pt)
    print('loc hi weight', loc_high_weights)
    print('edges of high weight', [edgelist[i] for i in loc_high_weights])
    edgelist_hi = [edgelist[i] for i in loc_high_weights]

    for i in loc_high_weights:
        print('loc of high weight along edgeweight', i)
        start = edgelist[i][0]
        end = edgelist[i][1]
        print('start and end node', start, end)
        if (start in loc_high_pt) | (end in loc_high_pt):
            print("found a high pt high weight node", (start, end), pt[start], pt[end])
            weights[i] = 0.5 * np.mean(weights)

    upper_lim = np.percentile(weights, 90)  # 80
    lower_lim = np.percentile(weights, 10)  # 20
    weights = [i if i <= upper_lim else upper_lim for i in weights]
    weights = [i if i >= lower_lim else lower_lim for i in weights]
    for i, (start, end) in enumerate(edgelist):
        print('i, start, end', i, start, end)
        Pt_a = pt[start]
        Pt_b = pt[end]
        P_ab = weights[i]
        t_ab = Pt_b - Pt_a
        Bias_ab = K / ((C + math.exp(-b * (t_ab - c)))) ** nu
        new_weight = (Bias_ab * P_ab)
        bias_weight.append(new_weight)
        print('tab', t_ab, 'pab', P_ab, 'biased_pab', new_weight)
    print('original weights', list(enumerate(zip(edgelist, weights))))
    print('bias weights', list(enumerate(zip(edgelist, bias_weight))))
    print('length bias weights', len(bias_weight))
    # bias_weight=np.asarray(bias_weight)
    # bias_weight = (bias_weight-np.min(bias_weight)+0.1)/(np.max(bias_weight)-np.min(bias_weight)+0.1)
    return list(bias_weight)


def get_Q_transient_transition(sources, targets, bias_weights, absorbing_clusters):
    return


def get_R_absorbing_transition(sources, targets, bias_weights, absorbing_clusters):
    return


def make_absorbing_markov(cluster_graph, pt):
    # cluster_graph is the vertex_cluster_graph made of sub_clusters in the finer iteration of PARC
    # pt is the pseudotime of each cluster in the graph
    absorbing_clusters = get_absorbing_clusters(cluster_graph)
    n_s = len(absorbing_clusters)
    adjacency_matrix = get_sparse_from_igraph(cluster_graph)

    sources, targets, weights = zip(adjacency_matrix)
    sources, targets, bias_weights = get_biased_weights(sources, targets, weights, pt)
    Q = get_Q_transient_transition(sources, targets, bias_weights, absorbing_clusters)
    R = get_R_absorbing_transition(sources, targets, bias_weights, absorbing_clusters)
    n_t = Q.shape[0]
    I_t = np.identity(n_t)
    N = np.inv(I_t - Q)
    P_transition_absorbing = np.concatenate()  # put together Q, R, Is and 0s
    return Q, R, N, P_transition_absorbing


def expected_num_steps(start_i, N):
    n_t = N.shape[0]
    N_steps = np.dot(N, np.ones(n_t))
    n_steps_i = N_steps[start_i]
    return n_steps_i


def absorption_probability(N, R, absorption_state_j):
    M = np.dot(N, R)
    vec_prob_end_in_j = M[:, absorption_state_j]
    return M, vec_prob_end_in_j


def most_likely_path(P_transition_absorbing_markov, start_i, end_i):
    graph_absorbing_markov = 0  # ig() log weight them
    shortest_path = graph_absorbing_markov.shortest_path(start_i, end_i)
    print('the shortest path beginning at ', start_i, 'and ending in ', end_i, 'is:')
    return shortest_path


def draw_trajectory_dimred(X_dimred, cluster_labels, super_cluster_labels, super_edgelist, x_lazy, alpha_teleport,
                           projected_sc_pt, true_label, knn, title_str="hitting times"):
    from scipy.interpolate import interp1d

    x = X_dimred[:, 0]
    y = X_dimred[:, 1]

    # for label_i in set(cluster_labels):
    # loc_i = np.where(np.asarray(cluster_labels)==label_i)[0]
    # print('ptsub_labeli',pt_sub[label_i],'loc_i', loc_i)
    # sc_pseudotime_sub[loc_i]=pt_sub[label_i]
    # print('sc_pseudo', sc_pseudotime_sub[0:200])
    df = pd.DataFrame({'x': x, 'y': y, 'cluster': cluster_labels, 'super_cluster': super_cluster_labels,
                       'projected_sc_pt': projected_sc_pt},
                      columns=['x', 'y', 'cluster', 'super_cluster', 'projected_sc_pt'])
    df_mean = df.groupby('cluster').mean()
    # print('df_mean', df_mean.head())
    df_super_mean = df.groupby('super_cluster').mean()
    # print('super mean', df_super_mean.head())
    # pt = df_super_mean['pseudotime_sub'].values
    pt = df_super_mean['projected_sc_pt'].values
    pt_int = [int(i) for i in pt]
    pt_str = [str(i) for i in pt_int]
    pt_sub = [str(int(i)) for i in df_mean['projected_sc_pt'].values]
    print('pt sub', pt_sub[0:20])
    # f, (ax1, ax2, ax3) = plt.subplots(1, 3, sharey=True)
    f, (ax1, ax2) = plt.subplots(1, 2, sharey=True)
    num_parc_group = len(set(true_label))
    line = np.linspace(0, 1, num_parc_group)
    for color, group in zip(line, set(true_label)):
        where = np.where(np.array(true_label) == group)[0]
        ax1.scatter(X_dimred[where, 0], X_dimred[where, 1], label=group, c=plt.cm.jet(color))
    ax1.legend()
    ax1.set_title('true labels')
    for e_i, (start, end) in enumerate(super_edgelist):

        if pt[start] >= pt[end]:
            temp = end
            end = start
            start = temp
        print('edges', e_i, start, end, pt[start], pt[end])
        print('df head', df.head())
        x_i_start = df[df['super_cluster'] == start].groupby('cluster').mean()['x'].values
        y_i_start = df[df['super_cluster'] == start].groupby('cluster').mean()['y'].values
        x_i_end = df[df['super_cluster'] == end].groupby('cluster').mean()['x'].values
        y_i_end = df[df['super_cluster'] == end].groupby('cluster').mean()['y'].values
        direction_arrow = 1
        # if np.mean(np.asarray(x_i_end)) < np.mean(np.asarray(x_i_start)): direction_arrow = -1

        super_start_x = df[df['super_cluster'] == start].mean()['x']
        super_end_x = df[df['super_cluster'] == end].mean()['x']
        super_start_y = df[df['super_cluster'] == start].mean()['y']
        super_end_y = df[df['super_cluster'] == end].mean()['y']

        if super_start_x > super_end_x: direction_arrow = -1
        ext_maxx = False
        minx = min(super_start_x, super_end_x)
        maxx = max(super_start_x, super_end_x)

        x_val = np.concatenate([x_i_start, x_i_end])
        y_val = np.concatenate([y_i_start, y_i_end])
        idx_keep = np.where((x_val <= maxx) & (x_val >= minx))[0]
        x_val = x_val[idx_keep]
        y_val = y_val[idx_keep]

        super_mid_x = (super_start_x + super_end_x) / 2
        super_mid_y = (super_start_y + super_end_y) / 2
        from scipy.spatial import distance

        # x_val = np.concatenate([x_i_start, x_i_end])
        print('abs', abs(minx - maxx))
        very_straight = False
        if abs(minx - maxx) <= 1:
            very_straight = True
            straight_level = 10
            noise = 0.01
            x_super = np.array(
                [super_start_x, super_end_x, super_start_x, super_end_x, super_start_x + noise, super_end_x + noise,
                 super_start_x - noise, super_end_x - noise, super_mid_x])
            y_super = np.array(
                [super_start_y, super_end_y, super_start_y, super_end_y, super_start_y + noise, super_end_y + noise,
                 super_start_y - noise, super_end_y - noise, super_mid_y])
        else:
            straight_level = 3
            noise = 0.1  # 0.05
            x_super = np.array(
                [super_start_x, super_end_x, super_start_x, super_end_x, super_start_x + noise, super_end_x + noise,
                 super_start_x - noise, super_end_x - noise])
            y_super = np.array(
                [super_start_y, super_end_y, super_start_y, super_end_y, super_start_y + noise, super_end_y + noise,
                 super_start_y - noise, super_end_y - noise])

        # x_super = np.array([super_start_x, super_end_x])
        # y_super = np.array([super_start_y,super_end_y])

        for i in range(straight_level):  # DO THE SAME FOR A MIDPOINT TOO
            y_super = np.concatenate([y_super, y_super])
            x_super = np.concatenate([x_super, x_super])
        # noise=np.random.normal(0,0.05,np.size(x_super))
        # x_super = np.concatenate([np.concatenate([x_super,x_super+noise]),x_super])
        # y_super = np.concatenate([np.concatenate([y_super,y_super+noise]),y_super])

        y_super_max = max(y_super)
        y_super_min = min(y_super)

        print('xval', x_val, 'start/end', start, end)
        print('yval', y_val, 'start/end', start, end)
        list_selected_clus = list(zip(x_val, y_val))
        # idx_keep = np.where((x_val<= maxx) & (x_val>=minx))[0]
        if (len(list_selected_clus) >= 1) & (very_straight == True):

            dist = distance.cdist([(super_mid_x, super_mid_y)], list_selected_clus, 'euclidean')
            print('dist', dist)
            if len(list_selected_clus) >= 2:
                k = 2
            else:
                k = 1
            midpoint_loc = dist[0].argsort()[:k]  # np.where(dist[0]==np.min(dist[0]))[0][0]
            print('midpoint loc', midpoint_loc)
            midpoint_xy = []
            for i in range(k):
                midpoint_xy.append(list_selected_clus[midpoint_loc[i]])

            # midpoint_xy = list_selected_clus[midpoint_loc]
            noise = 0.05
            print(midpoint_xy, 'is the midpoint between clus', pt[start], 'and ', pt[end])
            if k == 1:
                mid_x = np.array([midpoint_xy[0][0], midpoint_xy[0][0] + noise, midpoint_xy[0][
                    0] - noise])  # ,midpoint_xy[1][0], midpoint_xy[1][0] + noise, midpoint_xy[1][0] - noise])
                mid_y = np.array([midpoint_xy[0][1], midpoint_xy[0][1] + noise, midpoint_xy[0][
                    1] - noise])  # ,midpoint_xy[1][1], midpoint_xy[1][1] + noise, midpoint_xy[1][1] - noise])
            if k == 2:
                mid_x = np.array(
                    [midpoint_xy[0][0], midpoint_xy[0][0] + noise, midpoint_xy[0][0] - noise, midpoint_xy[1][0],
                     midpoint_xy[1][0] + noise, midpoint_xy[1][0] - noise])
                mid_y = np.array(
                    [midpoint_xy[0][1], midpoint_xy[0][1] + noise, midpoint_xy[0][1] - noise, midpoint_xy[1][1],
                     midpoint_xy[1][1] + noise, midpoint_xy[1][1] - noise])
            for i in range(3):
                mid_x = np.concatenate([mid_x, mid_x])
                mid_y = np.concatenate([mid_y, mid_y])

            x_super = np.concatenate([x_super, mid_x])
            y_super = np.concatenate([y_super, mid_y])
        x_val = np.concatenate([x_val, x_super])
        y_val = np.concatenate([y_val, y_super])
        z = np.polyfit(x_val, y_val, 2)

        xp = np.linspace(minx, maxx, 500)
        p = np.poly1d(z)
        smooth = p(xp)
        if ext_maxx == False:
            idx_keep = np.where((xp <= (maxx)) & (xp >= (minx)))[0]  # minx+3
        else:
            idx_keep = np.where((xp <= (maxx)) & (xp >= (minx)))[0]  # maxx-3
        ax2.plot(xp[idx_keep], smooth[idx_keep], linewidth=3, c='black')
        print('just drew this edge', start, end, 'This is the', e_i, 'th edge')
        # ax3.plot(xp[idx_keep], smooth[idx_keep], linewidth=3, c='black')
        med_loc = np.where(xp == np.median(xp[idx_keep]))[0]
        mean_temp = np.mean(xp[idx_keep])
        closest_val = xp[idx_keep][0]
        closest_loc = idx_keep[0]
        # print('mean_temp is', mean_temp)
        # print('closest val is', closest_val)
        # print('closest loc is', closest_loc)
        for i, xp_val in enumerate(xp[idx_keep]):
            # print('dist1',abs(xp_val - mean_temp))
            # print('dist2', abs(closest_val - mean_temp))
            if abs(xp_val - mean_temp) < abs(closest_val - mean_temp):
                # print('closest val is now', xp_val, 'at', idx_keep[i])
                closest_val = xp_val
                closest_loc = idx_keep[i]
        step = 1
        if direction_arrow == 1:
            ax2.arrow(xp[closest_loc], smooth[closest_loc], xp[closest_loc + step] - xp[closest_loc],
                      smooth[closest_loc + step] - smooth[closest_loc], shape='full', lw=0, length_includes_head=True,
                      head_width=1, color='black')  # , head_starts_at_zero = direction_arrow )
            # ax3.arrow(xp[closest_loc], smooth[closest_loc], xp[closest_loc + step] - xp[closest_loc],smooth[closest_loc + step] - smooth[closest_loc], shape='full', lw=0, length_includes_head=True,head_width=0.6, color='black')  # , head_starts_at_zero = direction_arrow )
        else:
            ax2.arrow(xp[closest_loc], smooth[closest_loc], xp[closest_loc - step] - xp[closest_loc],
                      smooth[closest_loc - step] - smooth[closest_loc], shape='full', lw=0, length_includes_head=True,
                      head_width=1, color='black')
            # ax3.arrow(xp[closest_loc], smooth[closest_loc], xp[closest_loc - step] - xp[closest_loc],smooth[closest_loc - step] - smooth[closest_loc], shape='full', lw=0, length_includes_head=True,head_width=0.6, color='black')

    # df_mean['cluster'] = df_mean.index()
    x_cluster = df_mean['x']
    y_cluster = df_mean['y']

    x_new = np.linspace(x_cluster.min(), x_cluster.max(), 500)
    num_parc_group = len(set(cluster_labels))
    line_parc = np.linspace(0, 1, num_parc_group)
    # color cells based on sub-=cluster rather than single cell
    # for color, group in zip(line_parc, set(cluster_labels)):
    #    where = np.where(np.array(cluster_labels) == group)[0]
    #    ax2.scatter(X_dimred[where, 0], X_dimred[where, 1], label=group, c=group_colour[group], alpha=0.5)
    # plt.legend()
    # plt.scatter(X_dimred[:,0], X_dimred[:,1], alpha=0.5)
    ax2.scatter(x_cluster, y_cluster, c='red')

    for i, type in enumerate(pt_str):
        ax2.text(df_super_mean['x'][i], df_super_mean['y'][i], type, weight='bold')

    for i in range(len(x_cluster)):
        ax2.text(x_cluster[i], y_cluster[i], pt_sub[i] + 'c' + str(i))
    ax2.set_title('lazy:' + str(x_lazy) + ' teleport' + str(alpha_teleport) + 'super_knn:' + str(knn))
    # ax2.set_title('super_knn:' + str(knn) )
    ax2.scatter(X_dimred[:, 0], X_dimred[:, 1], c=projected_sc_pt, cmap='viridis_r', alpha=0.5)
    ax2.scatter(df_super_mean['x'], df_super_mean['y'], c='black', s=40)
    plt.title(title_str)
    plt.show()
    return


def local_pruning_clustergraph(adjacency_matrix, local_pruning_std=0.0, global_pruning_std=2, too_big_factor=0.05):
    # larger pruning_std factor means less pruning
    sources, targets = adjacency_matrix.nonzero()
    original_edgelist = list(zip(sources, targets))

    initial_links_n = len(adjacency_matrix.data)
    print('initial links n', initial_links_n)
    adjacency_matrix = scipy.sparse.csr_matrix.todense(adjacency_matrix)
    print('adjacency')
    # print(type(adjacency_matrix))
    row_list = []
    col_list = []
    weight_list = []
    neighbor_array = adjacency_matrix  # not listed in in any order of proximity

    n_cells = neighbor_array.shape[0]
    rowi = 0

    for i in range(neighbor_array.shape[0]):
        row = np.asarray(neighbor_array[i, :]).flatten()
        # th=np.mean(row[row>0])-local_pruning_std*np.mean(row[row>0])
        if too_big_factor > 0.1:
            th = 0
            to_keep_index = np.where(row > th)[0]
        else:
            th = np.percentile(row[row > 0], 15)
            to_keep_index = np.where(row >= th)[0]

        # to_keep_index = np.where(row > np.percentile(row, 50))[0]  # we take [1] because row is a 2D matrix, not a 1D matrix like in other cases  # 0*std
        # print('to_keep_index', to_keep_index)
        updated_nn_weights = list(row[to_keep_index])
        # if len(to_keep_index)==1:
        #     nn = to_keep_index[0]
        #     maxedge_of_nn = np.where(neighbor_array[nn,:] == np.max(neighbor_array[nn,:]))
        #     print('maxedge',maxedge_of_nn, maxedge_of_nn[1][0])
        #     maxedge_of_nn = maxedge_of_nn[1][0]
        #     #print(neighbor_array[maxedge_of_nn])
        #     if i!=maxedge_of_nn:
        #         extra_row.append(i)
        #         extra_col.append(maxedge_of_nn)
        #         extra_weight.append(neighbor_array[nn,maxedge_of_nn]*.005)
        # print('updated nn weights', updated_nn_weights)
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
    # #print('mask is', mask)
    # cluster_graph_csr.data = cluster_graph_csr.data / (np.std(cluster_graph_csr.data)) #normalize
    # threshold_global = np.mean(cluster_graph_csr.data) - global_pruning_std* np.std(cluster_graph_csr.data)
    # mask |= (cluster_graph_csr.data < (threshold_global))  # smaller Jaccard weight means weaker edge
    #
    # cluster_graph_csr.data[mask] = 0
    # cluster_graph_csr.eliminate_zeros()
    # sources, targets = cluster_graph_csr.nonzero()
    edgelist = list(zip(sources, targets))
    print('original edges', original_edgelist)
    print('added edges', edgelist)

    # cluster_graph_csr.data = locallytrimmed_sparse_vc.data / (np.std(locallytrimmed_sparse_vc.data))
    edgeweights = cluster_graph_csr.data / (np.std(cluster_graph_csr.data))

    trimmed_n = (initial_links_n - final_links_n) * 100 / initial_links_n
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
    def __init__(self, data, true_label=None, anndata=None, dist_std_local=2, jac_std_global='median',
                 keep_all_local_dist='auto',
                 too_big_factor=0.4, small_pop=10, jac_weighted_edges=True, knn=30, n_iter_leiden=5, pseudotime=False,
                 root=0, path='/home/shobi/Trajectory/', seed=99, super_cluster_labels=False,
                 super_node_degree_list=False, x_lazy=0.95, alpha_teleport=0.99, root_str="root_cluster"):
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
        self.x_lazy = x_lazy  # 1-x = probability of staying in same node
        self.alpha_teleport = alpha_teleport  # 1-alpha is probability of jumping
        self.root_str = root_str

    def path_length_onbias(self, bias_edgelist, bias_edgeweights):
        bias_edgeweights = 1 / np.asarray(bias_edgeweights)
        bias_g = ig.Graph(list(bias_edgelist), edge_attrs={'weight': list(bias_edgeweights)})
        paths = bias_g.shortest_paths_dijkstra(source=self.root, weights='weight')
        print(type(paths))
        return paths[0]

    def compute_hitting_time(self, sparse_graph, root, x_lazy, alpha_teleport, number_eig=0):
        # 1- alpha is the probabilty of teleporting
        # 1- x_lazy is the probability of staying in current state (be lazy)
        beta_teleport = 2 * (1 - alpha_teleport) / (2 - alpha_teleport)
        N = sparse_graph.shape[0]
        print('adjacency in compute hitting', sparse_graph)
        # sparse_graph = scipy.sparse.csr_matrix(sparse_graph)
        print('start compute hitting')
        A = scipy.sparse.csr_matrix.todense(sparse_graph)  # A is the adjacency matrix
        print('is graph symmetric', (A.transpose() == A).all())
        lap = csgraph.laplacian(sparse_graph,
                                normed=False)  # compute regular laplacian (normed = False) to infer the degree matrix where D = L+A
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

        eig_val, eig_vec = np.linalg.eig(
            norm_lap)  # eig_vec[:,i] is eigenvector for eigenvalue eig_val[i] not eigh as this is only for symmetric. the eig vecs are not in decsending order
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
            # print(i, 'th eigenvalue is', eig_val[i])
            vec_i = eig_vec[:, i]
            factor = beta_teleport + 2 * eig_val[i] * x_lazy * (1 - beta_teleport)
            # print('factor', 1 / factor)

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
        return abs(final_hitting_times), roundtrip_times

    def directed_laplacian(self, A_bias):
        n_states = A_bias.shape[0]
        # degree: diagonal array
        # bias_A: adjacency matrix with biased edges (dense array)
        # L_eul is the symmetric eulerian laplacian
        # S is the diagonal matrix whose entries are the stationary probabilities (pagerank) of L_dir

        deg_val = A_bias.sum(axis=1)
        P_bias = A_bias / A_bias.sum(axis=1).reshape((n_states, 1))
        s = self.pagerank_compute(P_bias)
        S_diag = np.diag(s)
        D_diag = np.diag(deg_val)
        D_diag_recip = np.diag(1 / deg_val)
        print('stationary diagonal', S_diag)
        print('degree diagonal', D_diag)
        L_dir = D_diag - A_bias.T
        print("L_eul_sym is symmetric", (L_dir == L_dir.T).all())
        L_eul = L_dir.dot(D_diag_recip)
        print("L_eul_sym is symmetric", (L_eul == L_eul.T).all())
        L_eul = L_eul.dot(S_diag)
        L_eul_sym = 0.5 * (L_eul + L_eul.T)
        print("L_eul_sym is symmetric", (L_eul_sym == L_eul_sym.T).all())
        return L_eul

    def pagerank_compute(self, P_bias, max_iterations=100):
        x_lazy = self.x_lazy  # 1-x is prob lazy
        alpha_teleport = self.alpha_teleport
        # bias_P is the transition probability matrix
        n = P_bias.shape[0]
        P_bias = x_lazy * P_bias + (1 - x_lazy) * np.identity(n)
        P_bias = alpha_teleport * P_bias + ((1 - alpha_teleport) * (1 / n) * (np.ones((n, n)) - np.identity(n)))
        # transition matrix for the lazy, teleporting directed walk
        p0 = 1.0 / float(n)
        # p0=np.zeros((n,1))
        # p0[self.root,0] = 1#np.ones((n,1))*p0
        p0 = np.ones((n, 1)) * p0
        p0 = p0.T  # random uniform initial stationary distribution

        for iteration in range(max_iterations):
            old = p0.copy()
            p0 = p0.dot(P_bias)
            delta = p0 - old
            delta = math.sqrt(delta.dot(delta.T))

        p0 = p0[0] / np.sum(p0[0])
        print('p0', [('c' + str(i), pp0) for i, pp0 in enumerate(p0)])
        upperlim = np.percentile(p0, 90)
        lowerlim = np.percentile(p0, 10)

        # upper_val = p0[p0 >upperlim]
        # upperlim = np.mean(upper_val)
        print('upper lim', upperlim)
        if self.too_big_factor < 0.3:
            p0 = np.array([d if d <= upperlim else upperlim for d in p0])
            p0 = p0 / np.sum(p0)
            print('revised stationary', [(i, pp0) for i, pp0 in enumerate(p0)])
        return p0

    def directed_laplacian2(self, A_bias):
        print('old Abias', A_bias)
        x_lazy = self.x_lazy  # 1-x is prob lazy
        alpha_teleport = self.alpha_teleport

        deg_val = A_bias.sum(axis=1)

        P_bias = A_bias / A_bias.sum(axis=1)
        s = self.pagerank_compute(P_bias)
        n = A_bias.shape[0]
        for i in range(n):
            lazy_edge = deg_val[i] * (1 - x_lazy) / x_lazy
            A_bias[i, i] = lazy_edge
        deg_val = A_bias.sum(axis=1)
        for i in range(n):
            # print((deg_val[i]*(1-alpha_teleport)/(alpha_teleport))/n)
            teleport_edge = (np.ones((1, n)) - np.identity(n)[i, :]) * (deg_val[i] * (1 - alpha_teleport)) / (
                        alpha_teleport * n)
            print('shape teleport edge', teleport_edge.shape, teleport_edge)
            print('old edge', np.sum(A_bias[i, :]))
            A_bias[i, :] = A_bias[i, :] + teleport_edge
            print('new edge', np.sum(A_bias[i, :]))
        print('new A bias ', A_bias.shape, A_bias)

        deg_val = A_bias.sum(axis=1)

        print('new Abias', A_bias)
        # A_bias_new = self.x_lazy*A_bias + (1-self.x_lazy)*np.identity(n)
        # A_bias= self.alpha_teleport*A_bias_new + ((1-self.alpha_teleport)*(1/n)*np.ones((n,n)))
        # degree: diagonal array
        # bias_A: adjacency matrix with biased edges (dense array)
        # L_eul is the symmetric eulerian laplacian
        # S is the diagonal matrix whose entries are the stationary probabilities (pagerank) of L_dir

        S_diag = np.diag(s)
        D_diag = np.diag(deg_val)
        D_diag_recip = np.diag(1 / deg_val)
        L_dir = D_diag - A_bias.T
        L = L_dir.dot(D_diag_recip)  # I-W
        L_pseudo = np.linalg.pinv(L)
        print(L_pseudo.shape)
        S_diag_inv = np.diag(1 / s)
        hitting_array = np.zeros((n, 1))
        one_root = np.zeros((n, 1))
        one_root[self.root, 0] = 1
        for i in range(n):
            one_i = np.zeros((n, 1))
            one_i[i, 0] = 1
            one_vec = one_root - one_i
            second_term = L_pseudo.dot(one_vec)
            first_term = (np.ones((n, 1)) - one_i * S_diag_inv[i, i]).T
            hitting_array[i] = first_term.dot(second_term)
        return hitting_array / 1000, []

    def simulate_markov(self, A):
        if A.shape[0] > 42:
            print('A in markov', A[43, :], A[42, :], A)
        else:
            print('simulate A', A)
        n_states = A.shape[0]
        P = A / A.sum(axis=1).reshape((n_states, 1))
        print('row normed P', P, P.sum(axis=1))
        x_lazy = self.x_lazy  # 1-x is prob lazy
        alpha_teleport = self.alpha_teleport
        # bias_P is the transition probability matrix

        # P = x_lazy * P + (1 - x_lazy) * np.identity(n_states)
        # print(P, P.sum(axis=1))
        # P = alpha_teleport * P + ((1 - alpha_teleport) * (1 / n_states) * (np.ones((n_states, n_states))))
        # print('check prob of each row sum to one', P, P.sum(axis=1))

        currentState = self.root
        root = self.root
        print('root is', root)
        state = np.zeros((1, n_states))
        state[0, currentState] = 1
        state_root = state.copy()
        print('state', state)
        stateHist = state
        dfStateHist = pd.DataFrame(state)
        distr_hist = np.zeros([1, n_states])
        num_sim = 100
        hitting_array = np.ones((P.shape[0], 1)) * 1000
        hitting_array_temp = np.zeros((P.shape[0], 1)) * 1000
        n_steps = max(100, 2 * n_states)
        hitting_array_final = np.zeros((1, n_states))
        for i in range(num_sim):
            # print(i, 'th simulation in Markov')
            if i % 20 == 0: print(i, 'th simulation in Markov')
            state = state_root
            currentState = root
            stateHist = state
            for x in range(n_steps):
                currentRow = np.ma.masked_values((P[currentState]), 0.0)
                nextState = simulate_multinomial(currentRow)
                # print('next state', nextState)
                # Keep track of state changes
                # stateChangeHist[currentState,nextState]+=1
                # Keep track of the state vector itself
                state = np.zeros((1, n_states))
                state[0, nextState] = 1.0

                currentState = nextState

                # Keep track of state history
                stateHist = np.append(stateHist, state, axis=0)
                # calculate the actual distribution over the n_states so far
                totals = np.sum(stateHist, axis=0)
                gt = np.sum(totals)
                distrib = totals / gt
                distrib = np.reshape(distrib, (1, n_states))
                distr_hist = np.append(distr_hist, distrib, axis=0)
            for state_i in range(P.shape[0]):
                # print('first reach state', state_i, 'at step', np.where(stateHist[:, state_i] == 1)[0][0])
                first_time_at_statei = np.where(stateHist[:, state_i] == 1)[0]
                if len(first_time_at_statei) == 0:
                    # print('did not reach state', state_i,'setting dummy path length')
                    hitting_array_temp[state_i, 0] = n_steps + 1
                else:
                    hitting_array_temp[state_i, 0] = first_time_at_statei[0]
            # hitting_array_temp[hitting_array_temp==(n_steps+1)] = np.mean(hitting_array_temp[hitting_array_temp!=n_steps+1])

            hitting_array = np.append(hitting_array, hitting_array_temp, axis=1)
            # print('hitting temp', hitting_array_temp)
            if i % 20 == 0: print('hitting temp', hitting_array_temp)
        hitting_array = hitting_array[:, 1:]

        for i in range(n_states):
            rowtemp = hitting_array[i, :]
            if np.sum(rowtemp != (n_steps + 1)) != 0:
                hitting_array_final[0, i] = np.mean(rowtemp[rowtemp != (n_steps + 1)])
            else:
                hitting_array_final[0, i] = (n_steps + 1)

        # hitting_array=np.mean(hitting_array, axis=1)
        print('hitting from sim markov', hitting_array_final)
        return hitting_array_final[0]

    def directed_laplacian3(self, A_bias):
        print('old Abias', A_bias)
        x_lazy = self.x_lazy  # 1-x is prob lazy
        alpha_teleport = self.alpha_teleport
        P_bias = A_bias / A_bias.sum(axis=1)
        s = self.pagerank_compute(P_bias)
        deg_val = A_bias.sum(axis=1)

        n = A_bias.shape[0]
        for i in range(n):
            lazy_edge = deg_val[i] * (1 - x_lazy) / x_lazy
            A_bias[i, i] = lazy_edge
        deg_val = A_bias.sum(axis=1)
        for i in range(n):
            # print((deg_val[i]*(1-alpha_teleport)/(alpha_teleport))/n)
            teleport_edge = (np.ones((1, n)) - np.identity(n)[i, :]) * (deg_val[i] * (1 - alpha_teleport)) / (
                        alpha_teleport * n)
            print('shape teleport edge', teleport_edge.shape, teleport_edge)
            print('old edge', np.sum(A_bias[i, :]))
            A_bias[i, :] = A_bias[i, :] + teleport_edge
            print('new edge', np.sum(A_bias[i, :]))
        print('new A bias ', A_bias.shape, A_bias)

        deg_val = A_bias.sum(axis=1)
        D_diag_recip = np.diag(1 / deg_val)
        s_sqrt = np.sqrt(s)
        S_diag = np.diag(s_sqrt)
        S_diag_inv = np.diag((1 / s_sqrt))

        L_digraph = np.identity(n) - (D_diag_recip.dot(A_bias.T))
        L_digraph = L_digraph.dot(S_diag_inv)
        L_digraph = S_diag.dot(L_digraph)
        L_pseudo = np.linalg.pinv(L_digraph)
        root = self.root
        hitting_array = np.zeros((n, 1))
        eroot = np.zeros((n, 1))
        eroot[root, 0] = 1
        for j in range(n):
            ej = np.zeros((n, 1))
            ej[j, 0] = 1
            # first_term = L_pseudo[j,j]*(1/math.sqrt(s[j]))
            second_term = L_pseudo.dot(ej / s_sqrt[j])
            first_term = (ej / s_sqrt[j] - eroot / s_sqrt[root]).T
            # second_term=L_pseudo[root,j]*(S_diag_inv[root,root])
            # hitting_array[j] = (first_term -second_term)*(S_diag_inv[j,j])
            hitting_array[j] = first_term.dot(second_term)
        hitting_back = np.zeros((n, 1))
        for j in range(n):
            ej = np.zeros((n, 1))
            ej[j, 0] = 1
            # first_term = L_pseudo[j,j]*(1/math.sqrt(s[j]))
            second_term = L_pseudo.dot(eroot)  # /s_sqrt[root])
            first_term = (eroot / s_sqrt[root] - ej / s_sqrt[j]).T

            # second_term=L_pseudo[root,j]*(S_diag_inv[root,root])
            # hitting_array[j] = (first_term -second_term)*(S_diag_inv[j,j])
            hitting_back[j] = first_term.dot(second_term)
        commute = hitting_array + hitting_back
        # print("hitting array", hitting_array)
        # return hitting_array / 10, []
        return hitting_array / 10, []

    def compute_hitting_time_onbias(self, laplacian, inv_sqr_deg, root, x_lazy, alpha_teleport, number_eig=0):
        # 1- alpha is the probabilty of teleporting
        # 1- x_lazy is the probability of staying in current state (be lazy)
        beta_teleport = 2 * (1 - alpha_teleport) / (2 - alpha_teleport)
        N = laplacian.shape[0]
        print('is laplacian of biased symmetric', (laplacian.transpose() == laplacian).all())
        Id = np.zeros((N, N), float)
        np.fill_diagonal(Id, 1)
        # norm_lap = scipy.sparse.csr_matrix.todense(laplacian)

        eig_val, eig_vec = np.linalg.eig(
            laplacian)  # eig_vec[:,i] is eigenvector for eigenvalue eig_val[i] not eigh as this is only for symmetric. the eig vecs are not in decsending order
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
            # print(i, 'th eigenvalue is', eig_val[i])
            vec_i = eig_vec[:, i]
            factor = beta_teleport + 2 * eig_val[i] * x_lazy * (1 - beta_teleport)
            # print('factor', 1 / factor)

            vec_i = np.reshape(vec_i, (-1, 1))
            eigen_vec_mult = vec_i.dot(vec_i.T)
            Greens_matrix = Greens_matrix + (
                    eigen_vec_mult / factor)  # Greens function is the inverse of the beta-normalized laplacian
            beta_norm_lap = beta_norm_lap + (eigen_vec_mult * factor)  # beta-normalized laplacian

        temp = Greens_matrix.dot(inv_sqr_deg)
        temp = inv_sqr_deg.dot(temp) * beta_teleport
        hitting_matrix = np.zeros((N, N), float)
        diag_row = np.diagonal(temp)
        for i in range(N):
            hitting_matrix[i, :] = diag_row - temp[i, :]

        roundtrip_commute_matrix = hitting_matrix + hitting_matrix.T
        temp = Xv_Xu.dot(temp)
        final_hitting_times = np.diagonal(
            temp)  ## number_eig x 1 vector of hitting times from root (u) to number_eig of other nodes
        roundtrip_times = roundtrip_commute_matrix[root, :]
        return abs(final_hitting_times), roundtrip_times

    def project_hittingtimes_sc(self, pt):
        knn = 30
        neighbor_array, distance_array = self.knn_struct.knn_query(self.data, k=knn)
        print('shape of neighbor', neighbor_array.shape)
        labels = np.asarray(self.labels)
        sc_pt = np.zeros((len(self.labels),))

        print('hitting times used to project onto sc', pt)
        print('set labels', set(labels))
        i = 0
        for row in neighbor_array:
            mean_weight = 0
            # print('row in neighbor array of cells', row, labels.shape)
            neighboring_clus = labels[row]
            # print('neighbor clusters labels', neighboring_clus)
            for clus_i in set(list(neighboring_clus)):
                hitting_time_clus_i = pt[clus_i]
                num_clus_i = np.sum(neighboring_clus == clus_i)
                # print('hitting and num_clus for Clusi', hitting_time_clus_i, num_clus_i)
                mean_weight = mean_weight + hitting_time_clus_i * num_clus_i / knn
                # print('mean weight',mean_weight)
            sc_pt[i] = mean_weight
            i = i + 1
        print('single cell sc_pt', sc_pt[200:250])
        return sc_pt

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
        hnsw = self.make_knn_struct(too_big=True, big_cluster=X_data)

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
            if population < 5:  # <10
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
        while (small_pop_exist == True) & (time.time() - do_while_time < 5):
            small_pop_list = []
            small_pop_exist = False
            for cluster in set(list(PARC_labels_leiden.flatten())):
                population = len(np.where(PARC_labels_leiden == cluster)[0])
                if population < 10:
                    small_pop_exist = True
                    # print(cluster, ' has small population of', population, )
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
        # self.anndata.obs['parc_label'] = self.labels

        # cma1_cluster = self.anndata.obs.groupby('parc_label').mean('Cma1')

        return PARC_labels_leiden

    def recompute_weights(self, clustergraph_ig, pop_list_raw):
        sparse_clustergraph = get_sparse_from_igraph(clustergraph_ig, weight_attr='weight')
        n = sparse_clustergraph.shape[0]
        sources, targets = sparse_clustergraph.nonzero()
        edgelist = list(zip(sources, targets))
        weights = sparse_clustergraph.data
        # print('edgelist of combined clustergraph', edgelist)
        # print('edge weights of combined clustergraph', weights)
        new_weights = []
        i = 0
        for s, t in edgelist:
            pop_s = pop_list_raw[s]
            pop_t = pop_list_raw[t]
            w = weights[i]
            nw = w / (pop_s + pop_t)  # *
            new_weights.append(nw)
            # print('old and new', w, nw)
            i = i + 1
            scale_factor = max(new_weights) - min(new_weights)
            wmin = min(new_weights)
            wmax = max(new_weights)
        new_weights = [(i - wmin) * 1 / scale_factor for i in new_weights]

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
        root_str = self.root_str
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
                                                 n_iterations=self.n_iter_leiden, seed=self.seed)
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
            # print('set of new big labels ', set(PARC_labels_leiden_big.flatten()))
            PARC_labels_leiden_big = PARC_labels_leiden_big + 1000
            # print('set of new big labels +1000 ', set(list(PARC_labels_leiden_big.flatten())))
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
            pop_list_1 = []
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
            pop_item = PARC_labels_leiden.count(item)
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
            # print('vc graph G_sim', vc_graph)

            vc_graph = vc_graph.cluster_graph(combine_edges='sum')

            print('vc graph G_sim', vc_graph)
            ## Reweight clustergraph (2)

            reweighted_sparse_vc, edgelist = self.recompute_weights(vc_graph, pop_list_raw)

            print('len old edge list', edgelist)
            edgeweights, edgelist = local_pruning_clustergraph(reweighted_sparse_vc, local_pruning_std=0.0,
                                                               too_big_factor=self.too_big_factor)  # 0.5

            print('len new edge list', edgelist)

            locallytrimmed_g = ig.Graph(edgelist, edge_attrs={'weight': edgeweights.tolist()})
            print('locally trimmed_g', locallytrimmed_g)
            locallytrimmed_g = locallytrimmed_g.simplify(combine_edges='sum')
            print('locally trimmed_g', locallytrimmed_g)

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
                    super_majority_cluster_loc = \
                    np.where(np.asarray(self.super_cluster_labels) == super_majority_cluster)[0]
                    super_majority_truth = self.func_mode(list(true_labels[super_majority_cluster_loc]))
                    super_node_degree = self.super_node_degree_list[super_majority_cluster]

                    if (majority_truth == root_str) & (super_majority_truth == root_str):
                        if super_node_degree < super_min_deg:
                            # if deg_list[cluster_i] < min_deg:
                            found_super_and_sub_root = True
                            self.root = cluster_i
                            min_deg = deg_list[cluster_i]
                            super_min_deg = super_node_degree
                            print('new root is', self.root, ' with degree', min_deg, 'and super node degree',
                                  super_min_deg)
                majority_truth_labels[cluster_i_loc] = 'w' + str(majority_truth) + 'c' + str(cluster_i)
                graph_node_label.append(str(majority_truth) + 'c' + str(cluster_i))
            if (self.super_cluster_labels == False) | (found_super_and_sub_root == False):
                print('self.super_cluster_labels', self.super_cluster_labels, ' foundsuper_cluster_sub and super root',
                      found_super_and_sub_root)
                for cluster_i in range(len(set(PARC_labels_leiden))):
                    cluster_i_loc = np.where(np.asarray(PARC_labels_leiden) == cluster_i)[0]
                    true_labels = np.asarray(self.true_label)
                    majority_truth = self.func_mode(list(true_labels[cluster_i_loc]))
                    print('cluster', cluster_i, 'has majority', majority_truth, 'with degree list', deg_list)
                    if (majority_truth == root_str):
                        print('did not find a super and sub cluster with majority ', root_str)
                        if deg_list[cluster_i] < min_deg:
                            self.root = cluster_i
                            min_deg = deg_list[cluster_i]
                            print('new root is', self.root, ' with degree', min_deg)

            print('graph node label', graph_node_label)
            majority_truth_labels = list(majority_truth_labels.flatten())
            print('locallytrimmed_g', locallytrimmed_g)
            locallytrimmed_sparse_vc = get_sparse_from_igraph(locallytrimmed_g, weight_attr='weight')
            layout = locallytrimmed_g.layout_fruchterman_reingold()  ##final layout based on locally trimmed

            locallytrimmed_g.vs["label"] = graph_node_label
            locallytrimmed_sparse_vc_copy = locallytrimmed_sparse_vc.copy()
            dense_locallytrimmed = scipy.sparse.csr_matrix.todense(locallytrimmed_sparse_vc_copy)
            degree_array = np.sum(dense_locallytrimmed, axis=1)
            print('degree array', degree_array)
            # globally trimmed link
            sources, targets = locallytrimmed_sparse_vc.nonzero()
            edgelist_simple = list(zip(sources.tolist(), targets.tolist()))
            edgelist_unique = set(tuple(sorted(l)) for l in edgelist_simple)  # keep only one of (0,1) and (1,0)
            self.edgelist_unique = edgelist_unique
            self.edgelist = edgelist

            # mask = np.zeros(len(sources), dtype=bool)

            # locallytrimmed_sparse_vc.data = locallytrimmed_sparse_vc.data / (np.std(locallytrimmed_sparse_vc.data))

            # threshold_global = np.mean(locallytrimmed_sparse_vc.data) -0*np.std(locallytrimmed_sparse_vc.data)
            # mask |= (locallytrimmed_sparse_vc.data < (threshold_global))  # smaller Jaccard weight means weaker edge
            # print('sum of mask', sum(mask), 'at threshold of', threshold)
            # locallytrimmed_sparse_vc.data[mask] = 0
            # locallytrimmed_sparse_vc.eliminate_zeros()

            # sources, targets = locallytrimmed_sparse_vc.nonzero()
            # edgelist = list(zip(sources.tolist(), targets.tolist()))
            # globallytrimmed_g = ig.Graph(n=n_clus, edges=edgelist, edge_attrs={'weight': locallytrimmed_sparse_vc.data.tolist()})
            # globallytrimmed_g = globallytrimmed_g.simplify(combine_edges='sum')
            # layout = globallytrimmed_g.layout_fruchterman_reingold()
            # globallytrimmed_g.vs["label"] = graph_node_label

            # compute hitting times (4)

            root = self.root
            x_lazy = self.x_lazy
            alpha_teleport = self.alpha_teleport
            locallytrimmed_sparse_vc = locallytrimmed_sparse_vc_copy  ##hitting times are computed based on the locally trimmed graph without any global pruning

            hitting_times, roundtrip_times = self.compute_hitting_time(locallytrimmed_sparse_vc, root=root,
                                                                       x_lazy=x_lazy, alpha_teleport=alpha_teleport)

            self.hitting_times = hitting_times * 1000
            biased_edgeweights = get_biased_weights(edgelist, edgeweights, self.hitting_times)

            row = np.asarray([r[0] for r in edgelist])
            print('row', row)
            col = np.asarray([r[1] for r in edgelist])
            from scipy.sparse import csr_matrix
            biased_sparse = csr_matrix((biased_edgeweights, (row, col)))

            # biased_g = ig.Graph(edgelist, edge_attrs={'biasweight': biased_edgeweights})
            # sparse_biased_g = get_sparse_from_igraph(biased_g, weight_attr='biasweight')
            # adjacency_matrix = np.asarray(scipy.sparse.csr_matrix.todense(biased_sparse))
            print('bias sparse', biased_sparse)

            adjacency_matrix = np.zeros((n_clus, n_clus))
            print('initialize A', adjacency_matrix)
            for i, (start, end) in enumerate(edgelist):
                adjacency_matrix[start, end] = biased_edgeweights[i]
            print('finish construct adjacency_matrix', adjacency_matrix)
            '''

            lowerlim = np.percentile(biased_sparse.data, 1)
            lower_val = biased_sparse.data[biased_sparse.data<lowerlim]
            lowerlim = np.mean(lower_val)
            print('lower lim', lowerlim)
            if self.too_big_factor<0.3: biased_sparse.data = np.array([d if d > lowerlim else lowerlim for d in biased_sparse.data])
            '''

            deg_val = adjacency_matrix.sum(axis=1)
            print('deg val', type(deg_val), deg_val)
            d = np.diag(deg_val)
            d_sqrt_inv = np.diag(1 / np.sqrt(deg_val))

            L_eul = self.directed_laplacian(adjacency_matrix)
            # print('adjacency', adjacency_matrix.shape, adjacency_matrix, adjacency_matrix.sum(axis=1))
            # row_norm_adjacency_matrix= adjacency_matrix /adjacency_matrix.sum(axis=1)
            # print('row normalized', row_norm_adjacency_matrix, type(row_norm_adjacency_matrix))
            # row_norm_adjacency_matrix = np.asarray(row_norm_adjacency_matrix)
            # degree_array = np.tile(np.asarray(degree_array),row_norm_adjacency_matrix.shape[0])
            #
            #
            #
            # print('shapes', degree_array.shape, row_norm_adjacency_matrix.shape)
            # print('row norm',row_norm_adjacency_matrix)
            # print('degree', degree_array)
            # print('times the degree matrix', row_norm_adjacency_matrix*degree_array)
            # d=np.diag(degree_array[:,0])
            # d_sqrt_inv = np.diag(1/np.sqrt(degree_array[:,0]))
            # print('1/d_sqrt', d_sqrt_inv)
            # print('diag',d, np.sqrt(d))
            # symm= np.dot(row_norm_adjacency_matrix,np.sqrt(d))
            # symm = np.dot( d_sqrt_inv,symm)

            symm = np.dot(adjacency_matrix, d_sqrt_inv)
            symm = np.dot(d_sqrt_inv, symm)
            N = symm.shape[0]
            Id = np.zeros((N, N), float)
            np.fill_diagonal(Id, 1)
            norm_lap = Id - symm
            print('symm final', symm)
            eig_val, eig_vec = np.linalg.eig(symm)
            print('eigs of bias adjacency', eig_val)
            from scipy import sparse
            # hitting_times_2, roundtrip_times_2 = self.compute_hitting_time(sparse.csr_matrix(row_norm_adjacency_matrix*degree_array), root=root, x_lazy=x_lazy, alpha_teleport=alpha_teleport)
            hitting_times_2, roundtrip_times_2 = self.compute_hitting_time_onbias(norm_lap, d_sqrt_inv, root=root,
                                                                                  x_lazy=x_lazy,
                                                                                  alpha_teleport=alpha_teleport)
            # self.compute_hitting_time_onbias(L_eul, d_sqrt_inv, root=root, x_lazy=x_lazy, alpha_teleport=alpha_teleport)
            # hitting_times_2, roundtrip_times_2 = self.directed_laplacian2(adjacency_matrix)
            hitting_times_2 = self.simulate_markov(adjacency_matrix)
            self.biased_hitting_times = hitting_times_2 * 1000

            hitting_times = np.asarray(hitting_times) * 1000

            print('final original hitting times:')
            print(list(zip(range(len(hitting_times)), hitting_times)))

            print('final bias hitting times:')
            print(list(zip(range(len(hitting_times_2)), hitting_times_2 * 1000)))

            # plotting with threshold (thresholding the upper limit to make color palette more readable)
            print('percentile', np.percentile(hitting_times, 90))
            remove_outliers = hitting_times[hitting_times < np.percentile(hitting_times,
                                                                          90)]  # hitting_times[hitting_times<np.mean(hitting_times)+np.std(hitting_times)]
            # print('mean hitting times', np.mean(hitting_times))
            threshold = np.percentile(remove_outliers, 90)  # np.mean(remove_outliers) + 1* np.std(remove_outliers)
            # print('threshold', threshold)
            th_hitting_times = [x if x < threshold else threshold for x in hitting_times]

            remove_outliers_low = hitting_times[hitting_times < (np.mean(hitting_times) - 0.3 * np.std(hitting_times))]
            threshold_low = np.mean(remove_outliers_low) - 0.3 * np.std(remove_outliers_low)
            threshold_low = np.percentile(remove_outliers_low, 15)
            print('thresh low', threshold_low)
            th_hitting_times = [x if x > threshold_low else threshold_low for x in th_hitting_times]
            # scaled_hitting_times = (th_hitting_times - np.min(th_hitting_times))*100/(threshold)

            scaled_hitting_times = (th_hitting_times - np.min(th_hitting_times))
            scaled_hitting_times = scaled_hitting_times * (1000 / np.max(scaled_hitting_times))

            self.scaled_hitting_times = scaled_hitting_times
            self.single_cell_pt = self.project_hittingtimes_sc(hitting_times)
            self.single_cell_pt_bias = self.project_hittingtimes_sc(self.biased_hitting_times)
            dijkstra_hitting_times = self.path_length_onbias(edgelist, biased_edgeweights)
            self.single_cell_pt_dijkstra_bias = self.project_hittingtimes_sc(dijkstra_hitting_times)

            # threshold = np.mean(scaled_hitting_times)+0.25*np.std(scaled_hitting_times)
            threshold = int(threshold)
            scaled_hitting_times = scaled_hitting_times.astype(int)
            # print('scaled hitting times')
            # print(scaled_hitting_times)
            pal = ig.drawing.colors.AdvancedGradientPalette(['yellow', 'green', 'blue'], n=1001)

            all_colors = []
            print('100 scaled hitting', scaled_hitting_times)
            for i in scaled_hitting_times:
                all_colors.append(pal.get(int(i))[0:3])
            # print('extract all colors', zip(scaled_hitting_times,all_colors))

            locallytrimmed_g.vs['hitting_times'] = scaled_hitting_times

            locallytrimmed_g.vs['color'] = [pal.get(i)[0:3] for i in scaled_hitting_times]
            import matplotlib.colors as colors
            import matplotlib.cm as cm
            self.group_color = [colors.to_hex(v) for v in locallytrimmed_g.vs['color']]  # based on ygb scale
            viridis_cmap = cm.get_cmap('viridis_r')

            self.group_color_cmap = [colors.to_hex(v) for v in
                                     viridis_cmap(scaled_hitting_times / 1000)]  # based on ygb scale
            # print('group color', self.group_color)
            # ig.plot(locallytrimmed_g, "/home/shobi/Trajectory/Datasets/Toy/Toy_bifurcating/vc_graph_example_locallytrimmed_colornode_"+str(root)+"lazy"+str(lazy_i)+'jac'+str(self.jac_std_global)+".svg", layout=layout, edge_width=[e['weight']*1 for e in locallytrimmed_g.es], vertex_label=graph_node_label)
            svgpath_local = self.path + "vc_graph_locallytrimmed_Root" + str(root) + "lazy" + str(
                x_lazy) + 'JacG' + str(self.jac_std_global) + 'toobig' + str(
                int(self.too_big_factor * 100)) + "M" + str(len(set(true_labels))) + ".svg"
            # print('svglocal', svgpath_local)

            ig.plot(locallytrimmed_g, svgpath_local, layout=layout,
                    edge_width=[e['weight'] * 1 for e in locallytrimmed_g.es], vertex_label=graph_node_label)
            # hitting_times = compute_hitting_time(sparse_vf, root=1)
            # print('final hitting times:', list(zip(range(len(hitting_times)), hitting_times)))

            # globallytrimmed_g.vs['color'] = [pal.get(i)[0:3] for i in scaled_hitting_times]
            # ig.plot(globallytrimmed_g,                    self.path+"/vc_graph_globallytrimmed_Root" + str(   root) + "Lazy" + str(x_lazy) + 'JacG' + str(self.jac_std_global) + 'toobig'+str(int(100*self.too_big_factor))+".svg", layout=layout,edge_width=[e['weight'] * .1 for e in globallytrimmed_g.es], vertex_label=graph_node_label, main ='lazy:'+str(x_lazy)+' alpha:'+str(alpha_teleport))
        self.graph_node_label = graph_node_label
        self.edgeweight = [e['weight'] * 1 for e in locallytrimmed_g.es]
        self.graph_node_pos = layout.coords
        self.draw_piechart_graph()
        self.draw_piechart_graph(bias=True)

        return

    def draw_piechart_graph(self, bias=False):
        f, (ax, ax2) = plt.subplots(1, 2, sharey=True)
        arrow_head_w = 0.3
        edgeweight_scale = 0.7
        # fig, ax = plt.subplots()
        node_pos = self.graph_node_pos
        edgelist = self.edgelist_unique
        edgeweight = self.edgeweight
        graph_node_label = self.graph_node_label
        pt = self.scaled_hitting_times
        if bias: pt = self.biased_hitting_times
        import matplotlib.lines as lines
        node_pos = np.asarray(node_pos)
        n_groups = node_pos.shape[0]
        n_truegroups = len(set(self.true_label))
        group_pop = np.zeros([n_groups, 1])
        group_frac = pd.DataFrame(np.zeros([n_groups, n_truegroups]), columns=list(set(self.true_label)))

        for group_i in set(self.labels):
            loc_i = np.where(self.labels == group_i)[0]
            group_pop[group_i] = np.sum(loc_i) / 1000 + 1
            true_label_in_group_i = list(np.asarray(self.true_label)[[loc_i]])
            for ii in set(true_label_in_group_i):
                group_frac[ii][group_i] = true_label_in_group_i.count(ii)
        group_frac = group_frac.div(group_frac.sum(axis=1), axis=0)

        line_true = np.linspace(0, 1, n_truegroups)
        color_true_list = [plt.cm.jet(color) for color in line_true]

        sct = ax.scatter(
            node_pos[:, 0], node_pos[:, 1],
            c='white', edgecolors='face', s=group_pop, cmap='jet')

        for e_i, (start, end) in enumerate(edgelist):
            if pt[start] > pt[end]:
                temp = start
                start = end
                end = temp

            ax.add_line(lines.Line2D([node_pos[start, 0], node_pos[end, 0]], [node_pos[start, 1], node_pos[end, 1]],
                                     color='black', lw=edgeweight[e_i] * edgeweight_scale, alpha=0.5))
            z = np.polyfit([node_pos[start, 0], node_pos[end, 0]], [node_pos[start, 1], node_pos[end, 1]], 1)
            minx = np.min(np.array([node_pos[start, 0], node_pos[end, 0]]))

            if (node_pos[start, 0] < node_pos[end, 0]):
                direction_arrow = 1
            else:
                direction_arrow = -1

            maxx = np.max(np.array([node_pos[start, 0], node_pos[end, 0]]))

            xp = np.linspace(minx, maxx, 500)
            p = np.poly1d(z)
            smooth = p(xp)
            step = 1
            if direction_arrow == 1:

                ax.arrow(xp[250], smooth[250], xp[250 + step] - xp[250], smooth[250 + step] - smooth[250], shape='full',
                         lw=0,
                         length_includes_head=True, head_width=arrow_head_w,
                         color='black')
                # ax.plot(xp, smooth, linewidth=edgeweight[e_i], c='pink')
            else:
                ax.arrow(xp[250], smooth[250], xp[250 - step] - xp[250],
                         smooth[250 - step] - smooth[250], shape='full', lw=0,
                         length_includes_head=True, head_width=arrow_head_w, color='black')
        trans = ax.transData.transform
        bbox = ax.get_position().get_points()
        ax_x_min = bbox[0, 0]
        ax_x_max = bbox[1, 0]
        ax_y_min = bbox[0, 1]
        ax_y_max = bbox[1, 1]
        ax_len_x = ax_x_max - ax_x_min
        ax_len_y = ax_y_max - ax_y_min
        trans2 = ax.transAxes.inverted().transform
        pie_axs = []
        pie_size_ar = ((group_pop - np.min(group_pop)) / (np.max(group_pop) - np.min(group_pop)) + 0.5) / 10
        # print('pie size', pie_size_ar)

        for node_i in range(n_groups):
            pie_size = pie_size_ar[node_i][0]

            x1, y1 = trans(node_pos[node_i])  # data coordinates

            xa, ya = trans2((x1, y1))  # axis coordinates

            xa = ax_x_min + (xa - pie_size / 2) * ax_len_x
            ya = ax_y_min + (ya - pie_size / 2) * ax_len_y
            # clip, the fruchterman layout sometimes places below figure
            # if ya < 0: ya = 0
            # if xa < 0: xa = 0
            rect = [xa, ya, pie_size * ax_len_x, pie_size * ax_len_y]
            frac = group_frac.iloc[node_i].values
            pie_axs.append(plt.axes(rect, frameon=False))
            pie_axs[node_i].pie(frac, wedgeprops={'linewidth': 0.0}, colors=color_true_list)
            pie_axs[node_i].set_xticks([])
            pie_axs[node_i].set_yticks([])
            pie_axs[node_i].set_aspect('equal')
            pie_axs[node_i].text(0.5, 0.5, graph_node_label[node_i])

        patches, texts = pie_axs[node_i].pie(frac, wedgeprops={'linewidth': 0.0}, colors=color_true_list)
        labels = list(set(self.true_label))
        plt.legend(patches, labels, loc=(-5, -5))

        for e_i, (start, end) in enumerate(edgelist):
            if pt[start] > pt[end]:
                temp = start
                start = end
                end = temp

            ax2.add_line(lines.Line2D([node_pos[start, 0], node_pos[end, 0]], [node_pos[start, 1], node_pos[end, 1]],
                                      color='black', lw=edgeweight[e_i] * edgeweight_scale, alpha=0.5))
            z = np.polyfit([node_pos[start, 0], node_pos[end, 0]], [node_pos[start, 1], node_pos[end, 1]], 1)
            minx = np.min(np.array([node_pos[start, 0], node_pos[end, 0]]))

            if (node_pos[start, 0] < node_pos[end, 0]):
                direction_arrow = 1
            else:
                direction_arrow = -1

            maxx = np.max(np.array([node_pos[start, 0], node_pos[end, 0]]))

            xp = np.linspace(minx, maxx, 500)
            p = np.poly1d(z)
            smooth = p(xp)
            step = 1
            if direction_arrow == 1:

                ax2.arrow(xp[250], smooth[250], xp[250 + step] - xp[250], smooth[250 + step] - smooth[250],
                          shape='full', lw=0,
                          length_includes_head=True, head_width=arrow_head_w,
                          color='black')
                # ax.plot(xp, smooth, linewidth=edgeweight[e_i], c='pink')
            else:
                ax2.arrow(xp[250], smooth[250], xp[250 - step] - xp[250],
                          smooth[250 - step] - smooth[250], shape='full', lw=0,
                          length_includes_head=True, head_width=arrow_head_w, color='black')
        ax2.scatter(node_pos[:, 0], node_pos[:, 1], s=group_pop * 1.5, c=pt, cmap='viridis_r', alpha=1, zorder=3)
        if bias == True:
            title_pt = "Pseudotime on biased edges"
        else:
            title_pt = "Pseudotime on random walk"
        ax2.set_title(title_pt)
        ax.set_title('Reference Group Membership')
        plt.show()

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
        # print("population composition", pop_list)
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

    dataset = "Toy3"  # ""Toy1" # GermlineLi #Toy1

    ## Dataset Germline Li https://zenodo.org/record/1443566#.XZlhEkEzZ5y
    if dataset == "GermlineLine":
        df_expression_ids = pd.read_csv("/home/shobi/Trajectory/Code/Rcode/germline_human_female_weeks_li.csv", 'rt',
                                        delimiter=",")
        print(df_expression_ids.shape)
        # print(df_expression_ids[['cell_id',"week","ACTG2","STK31"]])[10:12]
        df_counts = pd.read_csv("/home/shobi/Trajectory/Code/Rcode/germline_human_female_weeks_li_filteredcounts.csv",
                                'rt', delimiter=",")
        df_ids = pd.read_csv("/home/shobi/Trajectory/Code/Rcode/germline_human_female_weeks_li_labels.csv", 'rt',
                             delimiter=",")
        # print(df_counts.shape, df_counts.head() ,df_ids.shape)
        # X_counts = df_counts.values
        # print(X_counts.shape)
        # varnames = pd.Categorical(list(df_counts.columns))

        adata_counts = sc.AnnData(df_counts, obs=df_ids)
        print(adata_counts.obs)
        sc.pp.filter_cells(adata_counts, min_counts=1)
        print(adata_counts.n_obs)
        sc.pp.filter_genes(adata_counts, min_counts=1)  # only consider genes with more than 1 count
        print(adata_counts.X.shape)
        sc.pp.normalize_per_cell(  # normalize with total UMI count per cell
            adata_counts, key_n_counts='n_counts_all')
        print(adata_counts.X.shape, len(list(adata_counts.var_names)))

        filter_result = sc.pp.filter_genes_dispersion(  # select highly-variable genes
            adata_counts.X, flavor='cell_ranger', n_top_genes=1000, log=False)
        print(adata_counts.X.shape, len(list(adata_counts.var_names)))  # , list(adata_counts.var_names))

        adata_counts = adata_counts[:, filter_result.gene_subset]
        print(adata_counts.X.shape, len(list(adata_counts.var_names)))  # ,list(adata_counts.var_names))
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
    if dataset == 'Paul15':
        root_str = "8Mk"
        adata_counts = sc.datasets.paul15()
        sc.pp.recipe_zheng17(adata_counts)
        sc.tl.pca(adata_counts, svd_solver='arpack')
        true_label = list(adata_counts.obs['paul15_clusters'])  # PAUL
        adata_counts.obs['group_id'] = true_label
        # sc.pp.neighbors(adata_counts, n_neighbors=10)
        # sc.tl.draw_graph(adata_counts)
        # sc.pl.draw_graph(adata_counts, color=['paul15_clusters', 'Cma1'], legend_loc='on data')

    if dataset.startswith('Toy'):
        root_str = "M1"
        if dataset == "Toy1":
            df_counts = pd.read_csv("/home/shobi/Trajectory/Datasets/Toy1/Toy_bifurcating/toy_bifurcating_n3000.csv",
                                    'rt', delimiter=",")
            df_ids = pd.read_csv("/home/shobi/Trajectory/Datasets/Toy1/Toy_bifurcating/toy_bifurcating_n3000_ids.csv",
                                 'rt', delimiter=",")
        if dataset == "Toy2":
            df_counts = pd.read_csv("/home/shobi/Trajectory/Datasets/Toy2/toy_multifurcating_n1000.csv", 'rt',
                                    delimiter=",")
            df_ids = pd.read_csv("/home/shobi/Trajectory/Datasets/Toy2/toy_multifurcating_n1000_ids.csv", 'rt',
                                 delimiter=",")
        if dataset == "Toy3":
            df_counts = pd.read_csv("/home/shobi/Trajectory/Datasets/Toy3/toy_multifurcating_M8_n1000d1000.csv", 'rt',
                                    delimiter=",")
            df_ids = pd.read_csv("/home/shobi/Trajectory/Datasets/Toy3/toy_multifurcating_M8_n1000d1000_ids.csv", 'rt',
                                 delimiter=",")
        if dataset == "ToyCyclic":
            df_counts = pd.read_csv("/home/shobi/Trajectory/Datasets/ToyCyclic/ToyCyclic_M5_n3000d1000.csv", 'rt',
                                    delimiter=",")
            df_ids = pd.read_csv("/home/shobi/Trajectory/Datasets/ToyCyclic/ToyCyclic_M5_n3000d1000_ids.csv", 'rt',
                                 delimiter=",")

        df_ids['cell_id_num'] = [int(s[1::]) for s in df_ids['cell_id']]
        print("shape", df_counts.shape, df_ids.shape)
        df_counts = df_counts.drop('Unnamed: 0', 1)
        df_ids = df_ids.sort_values(by=['cell_id_num'])
        df_ids = df_ids.reset_index(drop=True)
        true_label = df_ids['group_id']
        adata_counts = sc.AnnData(df_counts, obs=df_ids)
        # sc.pp.recipe_zheng17(adata_counts, n_top_genes=20) not helpful for toy data

    sc.tl.pca(adata_counts, svd_solver='arpack', n_comps=20)
    '''
    sc.pp.neighbors(adata_counts, n_neighbors=5, n_pcs=20)#4
    sc.tl.draw_graph(adata_counts)
    sc.pl.draw_graph(adata_counts, color='group_id', legend_loc='on data') #force-directed layout
    start_dfmap = time.time()
    sc.tl.diffmap(adata_counts)
    print('time taken to get diffmap given knn', time.time() - start_dfmap)
    sc.pp.neighbors(adata_counts, n_neighbors=5, use_rep='X_diffmap')#4
    sc.tl.draw_graph(adata_counts)
    sc.pl.draw_graph(adata_counts, color='group_id', legend_loc='on data')
    sc.tl.louvain(adata_counts, resolution=1.0)
    sc.tl.paga(adata_counts, groups='louvain')
    #sc.pl.paga(adata_counts, color=['louvain','group_id'])
    adata_counts.uns['iroot'] = np.flatnonzero(adata_counts.obs['louvain'] == '2')[0]
    sc.tl.dpt(adata_counts)
    sc.pl.paga(adata_counts, color=['louvain', 'group_id', 'dpt_pseudotime'])
    #X = df_counts.values

    '''
    from sklearn.decomposition import PCA
    # pca = PCA(n_components=20)
    # pc = pca.fit_transform(df_counts)
    p0 = PARC(adata_counts.obsm['X_pca'][:, 0:50], true_label, jac_std_global=2, knn=30, too_big_factor=0.4,
              pseudotime=True, path="/home/shobi/Trajectory/Datasets/" + dataset + "/", root=3,
              root_str=root_str)  # *.4
    p0.run_PARC()
    super_labels = p0.labels

    super_edges = p0.edgelist
    super_pt = p0.scaled_hitting_times  # pseudotime pt
    p1 = PARC(adata_counts.obsm['X_pca'][:, 0:50], true_label, jac_std_global=1, knn=30, too_big_factor=0.05,
              path="/home/shobi/Trajectory/Datasets/" + dataset + "/", pseudotime=True, root=61,
              super_cluster_labels=super_labels, super_node_degree_list=p0.node_degree_list, root_str=root_str)  # *.4
    p1.run_PARC()
    labels = p1.labels
    # p1 = PARC(adata_counts.obsm['X_pca'], true_label, jac_std_global=1, knn=5, too_big_factor=0.05, anndata= adata_counts, small_pop=2)
    # p1.run_PARC()
    # labels = p1.labels
    print('start tsne')
    n_downsample = 10000
    if len(labels) > n_downsample:
        idx = np.random.randint(len(labels), size=2000)
        embedding = TSNE().fit_transform(adata_counts.obsm['X_pca'][idx, :])
    else:
        embedding = TSNE().fit_transform(adata_counts.obsm['X_pca'])
        idx = np.random.randint(len(labels), size=len(labels))
    print('end tsne')
    draw_trajectory_dimred(embedding, labels, super_labels, super_edges, p1.x_lazy, p1.alpha_teleport,
                           p1.single_cell_pt, true_label, knn=p0.knn, title_str='hitting times: Random walk original')

    draw_trajectory_dimred(embedding, labels, super_labels, super_edges,
                           p1.x_lazy, p1.alpha_teleport, p1.single_cell_pt_bias, true_label, knn=p0.knn,
                           title_str='hitting times: Random Walk on biased edges')

    draw_trajectory_dimred(embedding, labels, super_labels, super_edges,
                           p1.x_lazy, p1.alpha_teleport, p1.single_cell_pt_dijkstra_bias, true_label, knn=p0.knn,
                           title_str='hitting times: Dijkstra on biased edges')
    # embedding = TSNE().fit_transform(pc)
    num_group = len(set(true_label))
    line = np.linspace(0, 1, num_group)

    f, (ax1, ax2, ax3) = plt.subplots(1, 3, sharey=True)

    for color, group in zip(line, set(true_label)):
        if len(labels) > n_downsample:
            where = np.where(np.array(true_label)[idx] == group)[0]
        else:
            where = np.where(np.array(true_label) == group)[0]
        ax1.scatter(embedding[where, 0], embedding[where, 1], label=group, c=plt.cm.jet(color))
    ax1.legend()
    ax1.set_title('true labels')

    num_parc_group = len(set(labels))
    line_parc = np.linspace(0, 1, num_parc_group)
    for color, group in zip(line_parc, set(labels)):
        if len(labels) > n_downsample:
            where = np.where(np.array(labels)[idx] == group)[0]
        else:
            where = np.where(np.array(labels) == group)[0]
        print('color', int(p1.scaled_hitting_times[group]))
        ax2.scatter(embedding[where, 0], embedding[where, 1], label=group, c=p1.group_color_cmap[group])
    # ax2.legend()
    # for color, group in zip(line_parc, set(labels)):
    #    where = np.where(np.array(labels) == group)[0]
    #    ax2.scatter(embedding[where, 0], embedding[where, 1], label=group, c=plt.cm.jet(color))

    f1_mean = p1.f1_mean * 100
    ax2.set_title("parc labels F1 " + str(int(f1_mean)) + "%")
    ax3.scatter(embedding[:, 0], embedding[:, 1], c=p1.single_cell_pt, cmap='viridis_r')
    plt.show()

    sc.pp.neighbors(adata_counts, n_neighbors=10, n_pcs=20)
    sc.tl.draw_graph(adata_counts)
    sc.pl.draw_graph(adata_counts, color='gender_week', legend_loc='right margin', palette='jet')


if __name__ == '__main__':
    main()


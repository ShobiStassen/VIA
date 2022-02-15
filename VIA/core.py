import numpy as np
import pandas as pd
import scipy
from scipy import sparse
from scipy.sparse import csr_matrix, csgraph, find
from scipy.sparse.csgraph import minimum_spanning_tree, connected_components
import igraph as ig
import leidenalg
import time
from datetime import datetime
import hnswlib
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import matplotlib.cm as cm
from matplotlib.path import get_path_collection_extents
import math
import multiprocessing
import pygam as pg
from termcolor import colored
from collections import Counter
from velocity_utils import *
from sklearn.preprocessing import normalize
import math
###work in progress core

def prob_reaching_terminal_state1(terminal_state,  A, root,  n_simulations, q,
                                  cumstateChangeHist, cumstateChangeHist_all, seed):
    # this function is defined outside the VIA class to enable smooth parallel processing in Windows
    np.random.seed(seed)
    n = A.shape[0]

    A /= np.max(A)
    for i, r in enumerate(A):
        if np.all(r == 0):
            A[i, i] = 1
    P = A / A.sum(axis=1).reshape((n, 1))

    n_steps, count_reach_terminal_state = 2*n, 0
    for _ in range(n_simulations):
        cur_state = root
        change_hist = np.zeros((n, n))
        change_hist[root, root] = 1

        x, terminal_state_found = 0, False
        while x < n_steps and not terminal_state_found:
            next_state = np.random.choice(range(P.shape[0]), p=P[cur_state])
            if next_state == terminal_state:
                terminal_state_found = True

            change_hist[cur_state, next_state] += 1
            cur_state = next_state
            x += 1

        if terminal_state_found:
            cumstateChangeHist += np.any(change_hist > 0, axis=0)
            count_reach_terminal_state += 1
        cumstateChangeHist_all += np.any(change_hist > 0, axis=0)

    cumstateChangeHist_all[cumstateChangeHist_all == 0] = 1
    q.append([cumstateChangeHist, cumstateChangeHist_all])


def simulate_markov_sub(A, num_sim, hitting_array, q, root):
    n_states = A.shape[0]
    P = A / A.sum(axis=1).reshape((n_states, 1))
    # hitting_array = np.ones((P.shape[0], 1)) * 1000
    hitting_array_temp = np.zeros((P.shape[0], 1)).astype('float64')
    n_steps = int(2 * n_states)
    hitting_array_final = np.zeros((1, n_states))
    currentState = root

    state = np.zeros((1, n_states))
    state[0, currentState] = 1
    state_root = state.copy()
    for i in range(num_sim):
        dist_list = []
        state = state_root
        currentState = root
        stateHist = state
        for x in range(n_steps):
            nextState = np.random.choice(range(P.shape[0]), p=P[currentState])
            dist = A[currentState, nextState]
            dist_list.append(1. / (1 + math.exp(dist - 1)))

            state = np.zeros((1, n_states))
            state[0, nextState] = 1.0
            currentState = nextState
            # Keep track of state history
            stateHist = np.append(stateHist, state, axis=0)

        for state_i in range(P.shape[0]):
            first_time_at_statei = np.where(stateHist[:, state_i] == 1)[0]
            if len(first_time_at_statei) == 0:
                hitting_array_temp[state_i, 0] = n_steps + 1
            else:
                total_dist = 0
                for ff in range(first_time_at_statei[0]):
                    total_dist = dist_list[ff] + total_dist

                hitting_array_temp[state_i, 0] = total_dist  # first_time_at_statei[0]
        hitting_array = np.append(hitting_array, hitting_array_temp, axis=1)

    hitting_array = hitting_array[:, 1:]
    q.append(hitting_array)


def getbb(sc, ax):
    """
    Function to return a list of bounding boxes in data coordinates for a scatter plot.
    Directly taken from https://stackoverflow.com/questions/55005272/
    """
    ax.figure.canvas.draw()  # need to draw before the transforms are set.
    transform = sc.get_transform()
    transOffset = sc.get_offset_transform()
    offsets = sc._offsets
    paths = sc.get_paths()
    transforms = sc.get_transforms()

    if not transform.is_affine:
        paths = [transform.transform_path_non_affine(p) for p in paths]
        transform = transform.get_affine()
    if not transOffset.is_affine:
        offsets = transOffset.transform_non_affine(offsets)
        transOffset = transOffset.get_affine()

    if isinstance(offsets, np.ma.MaskedArray):
        offsets = offsets.filled(np.nan)

    bboxes = []

    if len(paths) and len(offsets):
        if len(paths) < len(offsets):
            # for usual scatters you have one path, but several offsets
            paths = [paths[0]] * len(offsets)
        if len(transforms) < len(offsets):
            # often you may have a single scatter size, but several offsets
            transforms = [transforms[0]] * len(offsets)

        for p, o, t in zip(paths, offsets, transforms):
            result = get_path_collection_extents(
                transform.frozen(), [p], [t], [o], transOffset.frozen()
            )
            # bboxes.append(result.inverse_transformed(ax.transData))
            bboxes.append(result.transformed(ax.transData.inverted()))
    return bboxes


def plot_sc_pb(ax, fig, embedding, prob, ti, cmap_name='plasma', scatter_size=None):
    from mpl_toolkits.axes_grid1 import make_axes_locatable
    prob = np.sqrt(prob)  # scale values to improve visualization of colors
    cmap = matplotlib.cm.get_cmap(cmap_name)
    norm = matplotlib.colors.Normalize(vmin=0, vmax=np.max(prob))
    if scatter_size is None:
        size_point = 10 if embedding.shape[0] > 10000 else 30
    else: size_point = scatter_size
    # changing the alpha transparency parameter for plotting points
    im =ax.scatter(embedding[:, 0], embedding[:, 1], c=prob, s=0.01, cmap=cmap_name,    edgecolors = 'none')
    ax.set_title('Lineage: ' + str(ti))
    divider = make_axes_locatable(ax)
    cax = divider.append_axes('right', size='5%', pad=0.05)
    fig.colorbar(im, cax=cax, orientation='vertical', label='lineage likelihood')

    c = cmap(norm(prob)).reshape(-1, 4)
    loc_c = np.where(prob <= 0.3)[0]
    ax.scatter(embedding[loc_c, 0], embedding[loc_c, 1], c=prob[loc_c], s=size_point, edgecolors='none',alpha=0.2, cmap=cmap_name)
    c[loc_c, 3] = 0.2
    loc_c = np.where((prob > 0.3) & (prob <= 0.5))[0]
    c[loc_c, 3] = 0.5
    ax.scatter(embedding[loc_c, 0], embedding[loc_c, 1], c=prob[loc_c], s=size_point, edgecolors='none', alpha=0.5, cmap=cmap_name)
    loc_c = np.where((prob > 0.5) & (prob <= 0.7))[0]
    c[loc_c, 3] = 0.8
    ax.scatter(embedding[loc_c, 0], embedding[loc_c, 1], c=prob[loc_c], s=size_point, edgecolors='none', alpha=0.8, cmap=cmap_name)
    loc_c = np.where((prob > 0.7))[0]
    c[loc_c, 3] = 0.8
    ax.scatter(embedding[loc_c, 0], embedding[loc_c, 1], c=prob[loc_c], s=size_point, edgecolors='none', alpha=0.8, cmap=cmap_name)




def get_loc_terminal_states(via0, X_input):
    # we need the location of terminal states from first iteration (Via0) to pass onto the second iterations of Via (Via1)
    # this will allow identification of the terminal-cluster in fine-grained Via1 that best captures the terminal state from coarse Via0
    tsi_list = []  # find the single-cell which is nearest to the average-location of a terminal cluster in PCA space (
    for tsi in via0.terminal_clusters:
        loc_i = np.where(np.asarray(via0.labels) == tsi)[0]
        val_pt = [via0.single_cell_pt_markov[i] for i in loc_i]
        th_pt = np.percentile(val_pt, 50)  # 50
        loc_i = [loc_i[i] for i in range(len(val_pt)) if val_pt[i] >= th_pt]
        temp = np.mean(X_input[loc_i], axis=0)
        labelsq, distances = via0.knn_struct.knn_query(temp, k=1)
        tsi_list.append(labelsq[0][0])
    return tsi_list


def sc_loc_ofsuperCluster_PCAspace(p0, p1, idx):
    # ci_list first finds location in unsampled PCA space of the location of the super-cluster or sub-terminal-cluster and root
    # Returns location (index) of cell nearest to the ci_list in the downsampled space
    print("dict of terminal state pairs, Super: sub: ", p1.dict_terminal_super_sub_pairs)
    p0_labels = np.asarray(p0.labels)
    p1_labels = np.asarray(p1.labels)
    p1_sc_markov_pt = p1.single_cell_pt_markov
    ci_list = []
    for ci in range(len(list(set(p0.labels)))):
        if ci in p1.revised_super_terminal_clusters:  # p0.terminal_clusters:
            loc_i = np.where(p1_labels == p1.dict_terminal_super_sub_pairs[ci])[0]
            # loc_i = np.where(p0_labels == ci)[0]
            # val_pt = [p1.single_cell_pt_markov[i] for i in loc_i]
            val_pt = [p1_sc_markov_pt[i] for i in loc_i]
            th_pt = np.percentile(val_pt, 0)  # 80
            loc_i = [loc_i[i] for i in range(len(val_pt)) if val_pt[i] >= th_pt]
            temp = np.mean(p0.data[loc_i], axis=0)
            labelsq, distances = p0.knn_struct.knn_query(temp, k=1)
            ci_list.append(labelsq[0][0])

        elif (ci in p0.root) & (len(p0.root) == 1):
            loc_root = np.where(np.asarray(p0.root) == ci)[0][0]

            p1_root_label = p1.root[loc_root]
            loc_i = np.where(np.asarray(p1_labels) == p1_root_label)[0]

            # loc_i = np.where(p0.labels == ci)[0]
            val_pt = [p1_sc_markov_pt[i] for i in loc_i]
            th_pt = np.percentile(val_pt, 20)  # 50
            loc_i = [loc_i[i] for i in range(len(val_pt)) if val_pt[i] <= th_pt]
            temp = np.mean(p0.data[loc_i], axis=0)
            labelsq, distances = p0.knn_struct.knn_query(temp, k=1)
            ci_list.append(labelsq[0][0])
        else:
            loc_i = np.where(p0_labels == ci)[0]
            temp = np.mean(p0.data[loc_i], axis=0)
            labelsq, distances = p0.knn_struct.knn_query(temp, k=1)
            ci_list.append(labelsq[0][0])

        X_ds = p0.data[idx]
        p_ds = hnswlib.Index(space='l2', dim=p0.data.shape[1])
        p_ds.init_index(max_elements=X_ds.shape[0], ef_construction=200, M=16)
        p_ds.add_items(X_ds)
        p_ds.set_ef(50)

        new_superclust_index_ds = {}
        for en_item, item in enumerate(ci_list):
            labelsq, distances = p_ds.knn_query(p0.data[item, :], k=1)
            # new_superclust_index_ds.append(labelsq[0][0])
            new_superclust_index_ds.update({en_item: labelsq[0][0]})
    # print('new_superclust_index_ds',new_superclust_index_ds)
    return new_superclust_index_ds

def make_knn_embeddedspace(embedding):
    # knn struct built in the embedded space to be used for drawing the lineage trajectories onto the 2D plot
    knn = hnswlib.Index(space='l2', dim=embedding.shape[1])
    knn.init_index(max_elements=embedding.shape[0], ef_construction=200, M=16)
    knn.add_items(embedding)
    knn.set_ef(50)
    return knn

def draw_clustergraph(via_coarse, type_data='gene', gene_exp='', gene_list='', arrow_head_w=0.4,
                      edgeweight_scale=1.5, cmap=None):
    '''
    #draws the clustergraph for cluster level gene or pseudotime values
    # type_pt can be 'pt' pseudotime or 'gene' for gene expression
    # ax1 is the pseudotime graph
    '''
    n = len(gene_list)

    fig, axs = plt.subplots(1, n)
    pt = via_coarse.markov_hitting_times
    if cmap is None: cmap = 'coolwarm' if type_data == 'gene' else 'viridis_r'

    node_pos = via_coarse.graph_node_pos
    edgelist = list(via_coarse.edgelist_maxout)
    edgeweight = via_coarse.edgeweights_maxout

    node_pos = np.asarray(node_pos)

    import matplotlib.lines as lines
    from mpl_toolkits.axes_grid1 import make_axes_locatable

    n_groups = len(set(via_coarse.labels))  # node_pos.shape[0]
    n_truegroups = len(set(via_coarse.true_label))
    group_pop = np.zeros([n_groups, 1])
    via_coarse.cluster_population_dict = {}
    for group_i in set(via_coarse.labels):
        loc_i = np.where(via_coarse.labels == group_i)[0]

        group_pop[group_i] = len(loc_i)  # np.sum(loc_i) / 1000 + 1
        via_coarse.cluster_population_dict[group_i] = len(loc_i)

    for i in range(n):
        ax_i = axs[i]
        gene_i = gene_list[i]

        for e_i, (start, end) in enumerate(edgelist):
            if pt[start] > pt[end]:
                start, end = end, start

            ax_i.add_line(
                lines.Line2D([node_pos[start, 0], node_pos[end, 0]], [node_pos[start, 1], node_pos[end, 1]],
                             color='black', lw=edgeweight[e_i] * edgeweight_scale, alpha=0.5))
            z = np.polyfit([node_pos[start, 0], node_pos[end, 0]], [node_pos[start, 1], node_pos[end, 1]], 1)
            minx = np.min(np.array([node_pos[start, 0], node_pos[end, 0]]))

            direction = 1 if node_pos[start, 0] < node_pos[end, 0] else -1
            maxx = np.max([node_pos[start, 0], node_pos[end, 0]])
            xp = np.linspace(minx, maxx, 500)
            p = np.poly1d(z)
            smooth = p(xp)
            step = 1

            ax_i.arrow(xp[250], smooth[250], xp[250 + direction * step] - xp[250],
                       smooth[250 + direction * step] - smooth[250],
                       shape='full', lw=0, length_includes_head=True, head_width=arrow_head_w, color='grey')

        c_edge, l_width = [], []
        for ei, pti in enumerate(pt):
            if ei in via_coarse.terminal_clusters:
                c_edge.append('red')
                l_width.append(1.5)
            else:
                c_edge.append('gray')
                l_width.append(0.0)

        group_pop_scale = .5 * group_pop * 1000 / max(group_pop)
        pos = ax_i.scatter(node_pos[:, 0], node_pos[:, 1], s=group_pop_scale, c=gene_exp[gene_i].values, cmap=cmap,
                           edgecolors=c_edge, alpha=1, zorder=3, linewidth=l_width)
        for ii in range(node_pos.shape[0]):
            ax_i.text(node_pos[ii, 0] + 0.5, node_pos[ii, 1] + 0.5, str(round(gene_exp[gene_i].values[ii], 1)),
                      color='black', zorder=4, fontsize=4)
        divider = make_axes_locatable(ax_i)
        cax = divider.append_axes('right', size='10%', pad=0.05)
        fig.colorbar(pos, cax=cax, orientation='vertical')
        ax_i.set_title(gene_i)
        ax_i.grid(False)
        ax_i.set_xticks([])
        ax_i.set_yticks([])
        ax_i.axis('off')
    fig.patch.set_visible(False)

def via_streamplot(via_coarse, embedding , density_grid=0.5, arrow_size=.7, arrow_color = 'k',
arrow_style="-|>",  max_length=4, linewidth=1,min_mass = 1, cutoff_perc = None,scatter_size=500, scatter_alpha=0.5,marker_edgewidth=0.1, density_stream = 2, smooth_transition=1, smooth_grid=0.5, color_scheme = 'annotation', add_outline_clusters=False, cluster_outline_edgewidth = 0.001,gp_color = 'white', bg_color='black' , dpi=300 , title='Streamplot'):

    """
   Construct vector streamplot on the embedding to show a fine-grained view of inferred directions in the trajectory

   Parameters
   ----------
   X_emb: np.ndarray of shape (n_samples, 2)
       umap or other 2-d embedding on which to project the directionality of cells

   scatter_size: int, default = 500

   linewidth: width of arrows in streamplot, defalt = 1

   marker_edgewidth: width of outline arround each scatter point, default = 0.1

   color_scheme: str, default = 'annotation' corresponds to self.true_labels. Other options are 'time' (uses single-cell pseudotime) and 'cluster' (via cluster graph)

   Returns
   -------
   streamplot matplotlib.pyplot instance of fine-grained trajectories drawn on top of scatter plot
   """

    import matplotlib.patheffects as PathEffects


    X_emb, V_emb = via_coarse.velocity_embedding(embedding, smooth_transition)

    V_emb *=20

    X_grid, V_grid = compute_velocity_on_grid(
        X_emb=X_emb,
        V_emb=V_emb,
        density=density_grid,
        smooth=smooth_grid,
        min_mass=min_mass,
        autoscale=False,
        adjust_for_stream=True,
        cutoff_perc=cutoff_perc )

    lengths = np.sqrt((V_grid ** 2).sum(0))

    linewidth = 1 if linewidth is None else linewidth
    #linewidth *= 2 * lengths / np.percentile(lengths[~np.isnan(lengths)],90)
    linewidth *= 2 * lengths / lengths[~np.isnan(lengths)].max()

    #linewidth=0.5
    fig, ax = plt.subplots(dpi=dpi)
    ax.grid(False)
    ax.streamplot(X_grid[0], X_grid[1], V_grid[0], V_grid[1], color=arrow_color, arrowsize=arrow_size, arrowstyle=arrow_style, zorder = 3, linewidth=linewidth, density = density_stream, maxlength=max_length)

    #num_cluster = len(set(super_cluster_labels))

    if add_outline_clusters:
        # add black outline to outer cells and a white inner rim
        #adapted from scanpy (scVelo utils also adapts this from scanpy)
        gp_size = (2 * (scatter_size * cluster_outline_edgewidth *.1) + 0.1*scatter_size) ** 2

        bg_size = (2 * (scatter_size * cluster_outline_edgewidth)+ math.sqrt(gp_size)) ** 2

        ax.scatter(X_emb[:, 0], X_emb[:, 1], s=bg_size, marker=".", c=bg_color, zorder=-2)
        ax.scatter(X_emb[:, 0], X_emb[:, 1], s=gp_size, marker=".", c=gp_color, zorder=-1)
    line = np.linspace(0, 1, len(set(via_coarse.true_label)))
    if color_scheme == 'time':
        ax.scatter(X_emb[:,0],X_emb[:,1], c=via_coarse.single_cell_pt_markov,alpha=scatter_alpha,  zorder = 0, s=scatter_size, linewidths=marker_edgewidth, cmap = 'viridis_r')
    if color_scheme == 'annotation':
        for color, group in zip(line, sorted(set(via_coarse.true_label))):
            where = np.where(np.array(via_coarse.true_label) == group)[0]
            ax.scatter(X_emb[where, 0], X_emb[where, 1], label=group,
                        c=np.asarray(plt.cm.rainbow(color)).reshape(-1, 4),
                        alpha=scatter_alpha,  zorder = 0, s=scatter_size, linewidths=marker_edgewidth)

            x_mean = X_emb[where, 0].mean()
            y_mean = X_emb[where, 1].mean()
            ax.text(x_mean, y_mean, str(group), fontsize=5, zorder=4, path_effects = [PathEffects.withStroke(linewidth=1, foreground='w')], weight = 'bold')

    if color_scheme == 'cluster':
        for color, group in zip(line, sorted(set(via_coarse.labels))):
            where = np.where(np.array(via_coarse.labels) == group)[0]
            ax.scatter(X_emb[where, 0], X_emb[where, 1], label=group,
                        c=np.asarray(plt.cm.rainbow(color)).reshape(-1, 4),
                        alpha=scatter_alpha,  zorder = 0, s=scatter_size, linewidths=marker_edgewidth)
            x_mean = X_emb[where, 0].mean()
            y_mean = X_emb[where, 1].mean()
            ax.text(x_mean, y_mean, str(group), fontsize=5, zorder=4, path_effects = [PathEffects.withStroke(linewidth=1, foreground='w')], weight = 'bold')

    fig.patch.set_visible(False)
    ax.axis('off')
    ax.set_title(title)
def draw_sc_lineage_probability_solo(via_coarse, via_fine, embedding, idx=None, cmap_name='plasma', dpi=150):
    # embedding is the full or downsampled 2D representation of the full dataset.
    # idx is the list of indices of the full dataset for which the embedding is available. if the dataset is very large the the user only has the visual embedding for a subsample of the data, then these idx can be carried forward
    # idx is the selected indices of the downsampled samples used in the visualization
    # G is the igraph knn (low K) used for shortest path in high dim space. no idx needed as it's made on full sample
    # knn_hnsw is the knn made in the embedded space used for query to find the nearest point in the downsampled embedding
    #   that corresponds to the single cells in the full graph
    if idx is None: idx = np.arange(0, via_coarse.nsamples)
    G = via_coarse.full_graph_shortpath
    knn_hnsw = make_knn_embeddedspace(embedding)
    y_root = []
    x_root = []
    root1_list = []
    p1_sc_bp = via_fine.single_cell_bp[idx, :]
    p1_labels = np.asarray(via_fine.labels)[idx]
    p1_cc = via_fine.connected_comp_labels
    p1_sc_pt_markov = list(np.asarray(via_fine.single_cell_pt_markov)[idx])
    X_data = via_fine.data

    X_ds = X_data[idx, :]
    p_ds = hnswlib.Index(space='l2', dim=X_ds.shape[1])
    p_ds.init_index(max_elements=X_ds.shape[0], ef_construction=200, M=16)
    p_ds.add_items(X_ds)
    p_ds.set_ef(50)
    num_cluster = len(set(via_fine.labels))
    G_orange = ig.Graph(n=num_cluster, edges=via_fine.edgelist_maxout, edge_attrs={'weight':via_fine.edgeweights_maxout})
    for ii, r_i in enumerate(via_fine.root):
        loc_i = np.where(p1_labels == via_fine.root[ii])[0]
        x = [embedding[xi, 0] for xi in loc_i]
        y = [embedding[yi, 1] for yi in loc_i]

        labels_root, distances_root = knn_hnsw.knn_query(np.array([np.mean(x), np.mean(y)]), k=1)
        x_root.append(embedding[labels_root, 0][0])
        y_root.append(embedding[labels_root, 1][0])

        labelsroot1, distances1 = via_fine.knn_struct.knn_query(X_ds[labels_root[0][0], :], k=1)
        root1_list.append(labelsroot1[0][0])
        for fst_i in via_fine.terminal_clusters:
            path_orange = G_orange.get_shortest_paths(via_fine.root[ii], to=fst_i)[0]
            #if the roots is in the same component as the terminal cluster, then print the path to output
            if len(path_orange)>0:
                print(colored( f"{datetime.now()}\tCluster path on clustergraph starting from Root Cluster {via_fine.root[ii]} to Terminal Cluster {fst_i} : follows {path_orange} ", "blue"))


    # single-cell branch probability evolution probability
    for i, ti in enumerate(via_fine.terminal_clusters):
        fig, ax = plt.subplots(dpi=dpi)
        plot_sc_pb(ax, fig, embedding, p1_sc_bp[:, i], ti=ti, cmap_name=cmap_name)

        loc_i = np.where(p1_labels == ti)[0]
        val_pt = [p1_sc_pt_markov[i] for i in loc_i]
        th_pt = np.percentile(val_pt, 50)  # 50
        loc_i = [loc_i[i] for i in range(len(val_pt)) if val_pt[i] >= th_pt]
        x = [embedding[xi, 0] for xi in
             loc_i]  # location of sc nearest to average location of terminal clus in the EMBEDDED space
        y = [embedding[yi, 1] for yi in loc_i]
        labels, distances = knn_hnsw.knn_query(np.array([np.mean(x), np.mean(y)]),
                                               k=1)  # knn_hnsw is knn of embedded space
        x_sc = embedding[labels[0], 0]  # terminal sc location in the embedded space
        y_sc = embedding[labels[0], 1]
        start_time = time.time()

        labelsq1, distances1 =via_fine.knn_struct.knn_query(X_ds[labels[0][0], :],
                                                       k=1)  # find the nearest neighbor in the PCA-space full graph

        path = G.get_shortest_paths(root1_list[p1_cc[ti]], to=labelsq1[0][0])  # weights='weight')
        # G is the knn of all sc points

        path_idx = []  # find the single-cell which is nearest to the average-location of a terminal cluster
        # get the nearest-neighbor in this downsampled PCA-space graph. These will make the new path-way points
        path = path[0]

        # clusters of path
        cluster_path = []
        for cell_ in path:
            cluster_path.append(via_fine.labels[cell_])

        # print(colored('cluster_path', 'green'), colored('terminal state: ', 'blue'), ti, cluster_path)
        revised_cluster_path = []
        revised_sc_path = []
        for enum_i, clus in enumerate(cluster_path):
            num_instances_clus = cluster_path.count(clus)
            if (clus == cluster_path[0]) | (clus == cluster_path[-1]):
                revised_cluster_path.append(clus)
                revised_sc_path.append(path[enum_i])
            else:
                if num_instances_clus > 1:  # typically intermediate stages spend a few transitions at the sc level within a cluster
                    if clus not in revised_cluster_path: revised_cluster_path.append(clus)  # cluster
                    revised_sc_path.append(path[enum_i])  # index of single cell
        print(f"{datetime.now()}\tCluster level path on sc-knnGraph from Root Cluster {via_fine.root[p1_cc[ti]]} to Terminal Cluster {ti} along path: {revised_cluster_path}")

        fig.patch.set_visible(False)
        ax.axis('off')

def draw_sc_lineage_probability(via_coarse, via_fine, embedding, idx=None, cmap_name='plasma', dpi=150, scatter_size =None):

    # embedding is the full or downsampled 2D representation of the full dataset.
    # idx is the list of indices of the full dataset for which the embedding is available. if the dataset is very large the the user only has the visual embedding for a subsample of the data, then these idx can be carried forward
    # idx is the selected indices of the downsampled samples used in the visualization
    # G is the igraph knn (low K) used for shortest path in high dim space. no idx needed as it's made on full sample
    # knn_hnsw is the knn made in the embedded space used for query to find the nearest point in the downsampled embedding
    #   that corresponds to the single cells in the full graph
    if idx is None: idx = np.arange(0, via_coarse.nsamples)
    G = via_coarse.full_graph_shortpath
    knn_hnsw = make_knn_embeddedspace(embedding)
    y_root = []
    x_root = []
    root1_list = []
    p1_sc_bp = via_fine.single_cell_bp[idx, :]
    p1_labels = np.asarray(via_fine.labels)[idx]
    p1_cc = via_fine.connected_comp_labels
    p1_sc_pt_markov = list(np.asarray(via_fine.single_cell_pt_markov)[idx])
    X_data = via_fine.data

    X_ds = X_data[idx, :]
    p_ds = hnswlib.Index(space='l2', dim=X_ds.shape[1])
    p_ds.init_index(max_elements=X_ds.shape[0], ef_construction=200, M=16)
    p_ds.add_items(X_ds)
    p_ds.set_ef(50)
    num_cluster = len(set(via_fine.labels))
    G_orange = ig.Graph(n=num_cluster, edges=via_fine.edgelist_maxout, edge_attrs={'weight':via_fine.edgeweights_maxout})
    for ii, r_i in enumerate(via_fine.root):
        loc_i = np.where(p1_labels == via_fine.root[ii])[0]
        x = [embedding[xi, 0] for xi in loc_i]
        y = [embedding[yi, 1] for yi in loc_i]

        labels_root, distances_root = knn_hnsw.knn_query(np.array([np.mean(x), np.mean(y)]), k=1)
        x_root.append(embedding[labels_root, 0][0])
        y_root.append(embedding[labels_root, 1][0])

        labelsroot1, distances1 = via_fine.knn_struct.knn_query(X_ds[labels_root[0][0], :], k=1)
        root1_list.append(labelsroot1[0][0])
        for fst_i in via_fine.terminal_clusters:
            path_orange = G_orange.get_shortest_paths(via_fine.root[ii], to=fst_i)[0]
            #if the roots is in the same component as the terminal cluster, then print the path to output
            if len(path_orange)>0: print(colored('Cluster path on clustergraph starting from Root Cluster', 'blue'), via_fine.root[ii], colored('to Terminal Cluster','blue'), fst_i, colored(':','blue'), path_orange)

    # single-cell branch probability evolution probability
    n_terminal_clusters = len(via_fine.terminal_clusters)
    fig_nrows, mod = divmod(n_terminal_clusters, 4)
    if mod ==0: fig_nrows=fig_nrows
    if mod != 0: fig_nrows+=1
    fig_ncols = 4
    fig, axs = plt.subplots(fig_nrows,fig_ncols,dpi=dpi)
    ti = 0 # counter for terminal cluster
    #for i, ti in enumerate(via_fine.terminal clusters):
    print()
    for r in range(fig_nrows):
        for c in range(fig_ncols):
            if ti < n_terminal_clusters:

                plot_sc_pb(axs[r,c], fig, embedding, p1_sc_bp[:, ti], ti= via_fine.terminal_clusters[ti], cmap_name=cmap_name, scatter_size=scatter_size)
                ti+=1
                loc_i = np.where(p1_labels == ti)[0]
                val_pt = [p1_sc_pt_markov[i] for i in loc_i]
                th_pt = np.percentile(val_pt, 50)  # 50
                loc_i = [loc_i[i] for i in range(len(val_pt)) if val_pt[i] >= th_pt]
                x = [embedding[xi, 0] for xi in
                     loc_i]  # location of sc nearest to average location of terminal clus in the EMBEDDED space
                y = [embedding[yi, 1] for yi in loc_i]
                labels, distances = knn_hnsw.knn_query(np.array([np.mean(x), np.mean(y)]),
                                                       k=1)  # knn_hnsw is knn of embedded space
                x_sc = embedding[labels[0], 0]  # terminal sc location in the embedded space
                y_sc = embedding[labels[0], 1]

                labelsq1, distances1 =via_fine.knn_struct.knn_query(X_ds[labels[0][0], :],
                                                               k=1)  # find the nearest neighbor in the PCA-space full graph

                path = G.get_shortest_paths(root1_list[p1_cc[ti]], to=labelsq1[0][0])  # weights='weight')
                # G is the knn of all sc points

                path_idx = []  # find the single-cell which is nearest to the average-location of a terminal cluster
                # get the nearest-neighbor in this downsampled PCA-space graph. These will make the new path-way points
                path = path[0]

                # clusters of path
                cluster_path = []
                for cell_ in path:
                    cluster_path.append(via_fine.labels[cell_])

                revised_cluster_path = []
                revised_sc_path = []
                for enum_i, clus in enumerate(cluster_path):
                    num_instances_clus = cluster_path.count(clus)
                    if (clus == cluster_path[0]) | (clus == cluster_path[-1]):
                        revised_cluster_path.append(clus)
                        revised_sc_path.append(path[enum_i])
                    else:
                        if num_instances_clus > 1:  # typically intermediate stages spend a few transitions at the sc level within a cluster
                            if clus not in revised_cluster_path: revised_cluster_path.append(clus)  # cluster
                            revised_sc_path.append(path[enum_i])  # index of single cell
                print(f"{datetime.now()}\tCluster level path on sc-knnGraph from Root Cluster {via_fine.root[p1_cc[ti]]} to Terminal Cluster {ti} along path: {revised_cluster_path}")
            fig.patch.set_visible(False)
            axs[r,c].axis('off')

def get_biased_weights(edges, weights, pt, round=1):
    # small nu means less biasing (0.5 is quite mild)
    # larger nu (in our case 1/nu) means more aggressive biasing https://en.wikipedia.org/wiki/Generalised_logistic_function

    # using the pseudotime calculated from lazy-jumping walk. Otherwise using the refined MCMC Psuedotimes before
    # calculating lineage likelihood paths
    b = 1 if round == 1 else 20

    weights_thr, pct_thr = weights.mean(), np.percentile(pt, 80)
    loc_high_pt = set(np.where(pt > pct_thr)[0])
    for i in np.where(weights > weights_thr)[0]:
        start, end = edges[i]
        if start in loc_high_pt or end in loc_high_pt:
            weights[i] = 0.5 * weights.mean()
    weights.clip(np.percentile(weights, 10), np.percentile(weights, 90))

    bias_weight, K, c, C, nu = [], 1, 0, 1, 1
    for (s, t), w in zip(edges, weights):
        t_ab = pt[s] - pt[t]
        bias_weight.append(w * K / (C + math.exp(b * (t_ab + c))) ** nu)

    return bias_weight


def expected_num_steps(start_i, N):
    return np.dot(N, np.ones(N.shape[0]))[start_i]


def absorption_probability(N, R, absorption_state_j):
    M = np.dot(N, R)
    return M, M[:, absorption_state_j]


def draw_trajectory_gams(via_coarse,via_fine, embedding, idx=None,
                         title_str="Pseudotime", draw_all_curves=True, arrow_width_scale_factor=15,
                         scatter_size=50, scatter_alpha=0.5,
                         linewidth=1.5, marker_edgewidth=1, cmap_pseudotime = 'viridis_r',dpi=150):
    from mpl_toolkits.axes_grid1 import make_axes_locatable

    if idx is None: idx = np.arange(0, via_coarse.nsamples)
    cluster_labels = list(np.asarray(via_fine.labels)[idx])
    super_cluster_labels = list(np.asarray(via_coarse.labels)[idx])
    super_edgelist = via_coarse.edgelist
    true_label = list(np.asarray(via_fine.true_label)[idx])
    knn = via_fine.knn
    ncomp = via_fine.ncomp
    if len(via_fine.revised_super_terminal_clusters)>0:
        final_super_terminal = via_fine.revised_super_terminal_clusters
    else: final_super_terminal = via_fine.terminal_clusters

    sub_terminal_clusters = via_fine.terminal_clusters


    sc_pt_markov = list(np.asarray(via_fine.single_cell_pt_markov[idx]))
    super_root = via_coarse.root[0]



    sc_supercluster_nn = sc_loc_ofsuperCluster_PCAspace(via_coarse, via_fine, np.arange(0, len(cluster_labels)))
    # draw_all_curves. True draws all the curves in the piegraph, False simplifies the number of edges
    # arrow_width_scale_factor: size of the arrow head
    X_dimred = embedding * 1. / np.max(embedding, axis=0)
    x = X_dimred[:, 0]
    y = X_dimred[:, 1]
    max_x = np.percentile(x, 90)
    noise0 = max_x / 1000

    df = pd.DataFrame({'x': x, 'y': y, 'cluster': cluster_labels, 'super_cluster': super_cluster_labels,
                       'projected_sc_pt': sc_pt_markov},
                      columns=['x', 'y', 'cluster', 'super_cluster', 'projected_sc_pt'])
    df_mean = df.groupby('cluster', as_index=False).mean()
    sub_cluster_isin_supercluster = df_mean[['cluster', 'super_cluster']]

    sub_cluster_isin_supercluster = sub_cluster_isin_supercluster.sort_values(by='cluster')
    sub_cluster_isin_supercluster['int_supercluster'] = sub_cluster_isin_supercluster['super_cluster'].round(0).astype(
        int)

    df_super_mean = df.groupby('super_cluster', as_index=False).mean()
    pt = df_super_mean['projected_sc_pt'].values

    f, (ax1, ax2) = plt.subplots(1, 2, sharey=True, figsize=[20, 10],dpi=dpi)
    num_true_group = len(set(true_label))
    num_cluster = len(set(super_cluster_labels))
    line = np.linspace(0, 1, num_true_group)
    for color, group in zip(line, sorted(set(true_label))):
        where = np.where(np.array(true_label) == group)[0]
        ax1.scatter(X_dimred[where, 0], X_dimred[where, 1], label=group, c=np.asarray(plt.cm.rainbow(color)).reshape(-1, 4),
                    alpha=scatter_alpha, s=scatter_size, linewidths=marker_edgewidth*.1)  # 10 # 0.5 and 4
    ax1.legend(fontsize=6, frameon = False)
    ax1.set_title('True Labels: ncomps:' + str(ncomp) + '. knn:' + str(knn))

    G_orange = ig.Graph(n=num_cluster, edges=super_edgelist)
    ll_ = []  # this can be activated if you intend to simplify the curves
    for fst_i in final_super_terminal:
        #print('draw traj gams:', G_orange.get_shortest_paths(super_root, to=fst_i))

        path_orange = G_orange.get_shortest_paths(super_root, to=fst_i)[0]
        len_path_orange = len(path_orange)
        for enum_edge, edge_fst in enumerate(path_orange):
            if enum_edge < (len_path_orange - 1):
                ll_.append((edge_fst, path_orange[enum_edge + 1]))

    edges_to_draw = super_edgelist if draw_all_curves else list(set(ll_))
    for e_i, (start, end) in enumerate(edges_to_draw):
        if pt[start] >= pt[end]:
            start, end = end, start

        x_i_start = df[df['super_cluster'] == start]['x'].values
        y_i_start = df[df['super_cluster'] == start]['y'].values
        x_i_end = df[df['super_cluster'] == end]['x'].values
        y_i_end = df[df['super_cluster'] == end]['y'].values


        super_start_x = X_dimred[sc_supercluster_nn[start], 0]
        super_end_x = X_dimred[sc_supercluster_nn[end], 0]
        super_start_y = X_dimred[sc_supercluster_nn[start], 1]
        super_end_y = X_dimred[sc_supercluster_nn[end], 1]
        direction_arrow = -1 if super_start_x > super_end_x else 1
        ext_maxx = False
        minx = min(super_start_x, super_end_x)
        maxx = max(super_start_x, super_end_x)

        miny = min(super_start_y, super_end_y)
        maxy = max(super_start_y, super_end_y)

        x_val = np.concatenate([x_i_start, x_i_end])
        y_val = np.concatenate([y_i_start, y_i_end])

        idx_keep = np.where((x_val <= maxx) & (x_val >= minx))[0]
        idy_keep = np.where((y_val <= maxy) & (y_val >= miny))[0]

        idx_keep = np.intersect1d(idy_keep, idx_keep)

        x_val = x_val[idx_keep]
        y_val = y_val[idx_keep]

        super_mid_x = (super_start_x + super_end_x) / 2
        super_mid_y = (super_start_y + super_end_y) / 2
        from scipy.spatial import distance

        very_straight = False
        straight_level = 3
        noise = noise0
        x_super = np.array(
            [super_start_x, super_end_x, super_start_x, super_end_x, super_start_x, super_end_x, super_start_x,
             super_end_x, super_start_x + noise, super_end_x + noise,
             super_start_x - noise, super_end_x - noise])
        y_super = np.array(
            [super_start_y, super_end_y, super_start_y, super_end_y, super_start_y, super_end_y, super_start_y,
             super_end_y, super_start_y + noise, super_end_y + noise,
             super_start_y - noise, super_end_y - noise])

        if abs(minx - maxx) <= 1:
            very_straight = True
            straight_level = 10
            x_super = np.append(x_super, super_mid_x)
            y_super = np.append(y_super, super_mid_y)

        for i in range(straight_level):  # DO THE SAME FOR A MIDPOINT TOO
            y_super = np.concatenate([y_super, y_super])
            x_super = np.concatenate([x_super, x_super])

        list_selected_clus = list(zip(x_val, y_val))

        if len(list_selected_clus) >= 1 & very_straight:
            dist = distance.cdist([(super_mid_x, super_mid_y)], list_selected_clus, 'euclidean')
            k = min(2, len(list_selected_clus))
            midpoint_loc = dist[0].argsort()[:k]

            midpoint_xy = []
            for i in range(k):
                midpoint_xy.append(list_selected_clus[midpoint_loc[i]])

            noise = noise0 * 2

            if k == 1:
                mid_x = np.array([midpoint_xy[0][0], midpoint_xy[0][0] + noise, midpoint_xy[0][0] - noise])
                mid_y = np.array([midpoint_xy[0][1], midpoint_xy[0][1] + noise, midpoint_xy[0][1] - noise])
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

        x_val = x_val.reshape((len(x_val), -1))
        y_val = y_val.reshape((len(y_val), -1))
        xp = np.linspace(minx, maxx, 500)

        gam50 = pg.LinearGAM(n_splines=4, spline_order=3, lam=10).gridsearch(x_val, y_val)
        XX = gam50.generate_X_grid(term=0, n=500)
        preds = gam50.predict(XX)

        idx_keep = np.where((xp <= (maxx)) & (xp >= (minx)))[0]
        ax2.plot(XX, preds, linewidth=linewidth, c='#323538')  # 3.5#1.5


        mean_temp = np.mean(xp[idx_keep])
        closest_val = xp[idx_keep][0]
        closest_loc = idx_keep[0]

        for i, xp_val in enumerate(xp[idx_keep]):
            if abs(xp_val - mean_temp) < abs(closest_val - mean_temp):
                closest_val = xp_val
                closest_loc = idx_keep[i]
        step = 1

        head_width = noise * arrow_width_scale_factor  # arrow_width needs to be adjusted sometimes # 40#30  ##0.2 #0.05 for mESC #0.00001 (#for 2MORGAN and others) # 0.5#1
        if direction_arrow == 1:
            ax2.arrow(xp[closest_loc], preds[closest_loc], xp[closest_loc + step] - xp[closest_loc],
                      preds[closest_loc + step] - preds[closest_loc], shape='full', lw=0, length_includes_head=False,
                      head_width=head_width, color='#323538')

        else:
            ax2.arrow(xp[closest_loc], preds[closest_loc], xp[closest_loc - step] - xp[closest_loc],
                      preds[closest_loc - step] - preds[closest_loc], shape='full', lw=0, length_includes_head=False,
                      head_width=head_width, color='#323538')

    c_edge = []
    width_edge = []
    pen_color = []
    super_cluster_label = []
    terminal_count_ = 0
    dot_size = []

    for i in sc_supercluster_nn:
        if i in final_super_terminal:
            print('super cluster', i, 'is a super terminal with sub_terminal cluster',
                  sub_terminal_clusters[terminal_count_])
            width_edge.append(2)
            c_edge.append('yellow')  # ('yellow')
            pen_color.append('black')
            # super_cluster_label.append('TS' + str(i))  # +'('+str(i)+')')
            super_cluster_label.append('TS' + str(sub_terminal_clusters[terminal_count_]))  # +'('+str(i)+')')
            dot_size.append(60)  # 60
            terminal_count_ = terminal_count_ + 1
        else:
            width_edge.append(0)
            c_edge.append('black')
            pen_color.append('red')
            super_cluster_label.append(str(' '))  # i or ' '
            dot_size.append(00)  # 20

    ax2.set_title(title_str)

    im2 =ax2.scatter(X_dimred[:, 0], X_dimred[:, 1], c=sc_pt_markov, cmap=cmap_pseudotime,  s=0.01)
    divider = make_axes_locatable(ax2)
    cax = divider.append_axes('right', size='5%', pad=0.05)
    f.colorbar(im2, cax=cax, orientation='vertical', label='pseudotime') #to avoid lines drawn on the colorbar we need an image instance without alpha variable
    ax2.scatter(X_dimred[:, 0], X_dimred[:, 1], c=sc_pt_markov, cmap=cmap_pseudotime, alpha=scatter_alpha,
                s=scatter_size, linewidths=marker_edgewidth*.1)
    count_ = 0
    loci = [sc_supercluster_nn[key] for key in sc_supercluster_nn]
    for i, c, w, pc, dsz, lab in zip(loci, c_edge, width_edge, pen_color, dot_size,
                                     super_cluster_label):  # sc_supercluster_nn
        ax2.scatter(X_dimred[i, 0], X_dimred[i, 1], c='black', s=dsz, edgecolors=c, linewidth=w)
        ax2.annotate(str(lab), xy=(X_dimred[i, 0], X_dimred[i, 1]))
        count_ = count_ + 1

    ax1.grid(False)
    ax2.grid(False)
    f.patch.set_visible(False)
    ax1.axis('off')
    ax2.axis('off')


def csr_mst(adjacency):
    # return minimum spanning tree from adjacency matrix (csr)
    Tcsr = adjacency.copy()
    Tcsr.data *= -1
    Tcsr.data -= np.min(Tcsr.data) - 1
    Tcsr = minimum_spanning_tree(Tcsr)
    return (Tcsr + Tcsr.T) * .5


def connect_all_components(MSTcsr, cluster_graph_csr, adjacency):
    # connect forest of MSTs (csr)
    n, labels = connected_components(csgraph=cluster_graph_csr, directed=False, return_labels=True)
    while n > 1:
        sub_td = MSTcsr[labels == 0, :][:, labels != 0]
        locxy = scipy.sparse.find(MSTcsr == np.min(sub_td.data))
        for i in range(len(locxy[0])):
            if (labels[locxy[0][i]] == 0) & (labels[locxy[1][i]] != 0):
                x, y = locxy[0][i], locxy[1][i]

        cluster_graph_csr[x, y] = adjacency[x, y]
        n, labels = connected_components(csgraph=cluster_graph_csr, directed=False, return_labels=True)
    return cluster_graph_csr


def pruning_clustergraph(adjacency, global_pruning_std=1, max_outgoing=30, preserve_disconnected=True,
                         preserve_disconnected_after_pruning=False):
    # neighbors in the adjacency matrix (neighbor-matrix) are not listed in in any order of proximity
    # larger pruning_std factor means less pruning
    # the mst is only used to reconnect components that become disconnect due to pruning
    # print('global pruning std', global_pruning_std, 'max outoing', max_outgoing)
    from scipy.sparse.csgraph import minimum_spanning_tree

    Tcsr = csr_mst(adjacency)
    initial_links_n = len(adjacency.data)

    n_comp, comp_labels = connected_components(csgraph=adjacency, directed=False, return_labels=True)
    print(f"{datetime.now()}\tGraph has {n_comp} connected components before pruning")
    adjacency = scipy.sparse.csr_matrix.todense(adjacency)
    row_list = []
    col_list = []
    weight_list = []

    rowi = 0

    for i in range(adjacency.shape[0]):
        row = np.asarray(adjacency[i, :]).flatten()
        n_nonz = min(np.sum(row > 0), max_outgoing)
        to_keep_index = np.argsort(row)[::-1][0:n_nonz]  # np.where(row>np.mean(row))[0]#
        # print('to keep', to_keep_index)
        updated_nn_weights = list(row[to_keep_index])
        for ik in range(len(to_keep_index)):
            row_list.append(rowi)
            col_list.append(to_keep_index[ik])
            dist = updated_nn_weights[ik]
            weight_list.append(dist)
        rowi = rowi + 1
    final_links_n = len(weight_list)

    cluster_graph_csr = csr_matrix((weight_list, (row_list, col_list)), shape=adjacency.shape)

    sources, targets = cluster_graph_csr.nonzero()
    mask = np.zeros(len(sources), dtype=bool)

    cluster_graph_csr.data = cluster_graph_csr.data / (np.std(cluster_graph_csr.data))  # normalize
    threshold_global = np.mean(cluster_graph_csr.data) - global_pruning_std * np.std(cluster_graph_csr.data)
    mask |= (cluster_graph_csr.data < threshold_global)  # smaller Jaccard weight means weaker edge

    cluster_graph_csr.data[mask] = 0
    cluster_graph_csr.eliminate_zeros()


    prev_n_comp, prev_comp_labels = n_comp, comp_labels
    n_comp, comp_labels = connected_components(csgraph=cluster_graph_csr, directed=False, return_labels=True)
    n_comp_preserve = n_comp if preserve_disconnected_after_pruning else prev_n_comp

    # preserve initial disconnected components
    if preserve_disconnected and n_comp > prev_n_comp:
        Td = Tcsr.todense()
        Td[Td == 0] = 999.999
        n_comp_ = n_comp
        while n_comp_ > n_comp_preserve:
            for i in range(n_comp_preserve):
                loc_x = np.where(prev_comp_labels == i)[0]
                len_i = len(set(comp_labels[loc_x]))

                while len_i > 1:
                    s = list(set(comp_labels[loc_x]))
                    loc_notxx = np.intersect1d(loc_x, np.where((comp_labels != s[0]))[0])
                    loc_xx = np.intersect1d(loc_x, np.where((comp_labels == s[0]))[0])
                    sub_td = Td[loc_xx, :][:, loc_notxx]
                    locxy = np.where(Td == np.min(sub_td))
                    for i in range(len(locxy[0])):
                        if comp_labels[locxy[0][i]] != comp_labels[locxy[1][i]]:
                            x, y = locxy[0][i], locxy[1][i]

                    cluster_graph_csr[x, y] = adjacency[x, y]
                    n_comp_, comp_labels = connected_components(csgraph=cluster_graph_csr, directed=False, return_labels=True)
                    loc_x = np.where(prev_comp_labels == i)[0]
                    len_i = len(set(comp_labels[loc_x]))
        print(f"{datetime.now()}\tGraph has {n_comp_} connected components after reconnecting")

    elif not preserve_disconnected and n_comp > 1:
        cluster_graph_csr = connect_all_components(Tcsr, cluster_graph_csr, adjacency)
        n_comp, comp_labels = connected_components(csgraph=cluster_graph_csr, directed=False, return_labels=True)

    edges = list(zip(*cluster_graph_csr.nonzero()))
    weights = cluster_graph_csr.data / (np.std(cluster_graph_csr.data))

    trimmed_n = (initial_links_n - final_links_n) * 100. / initial_links_n
    trimmed_n_glob = (initial_links_n - len(weights)) * 100. / initial_links_n
    if global_pruning_std < 0.5:
        print(f"{datetime.now()}\t{round(trimmed_n, 1)}% links trimmed from local pruning relative to start")
        print(f"{datetime.now()}\t{round(trimmed_n_glob, 1)}% links trimmed from global pruning relative to start")
    return weights, edges, comp_labels


def get_sparse_from_igraph(graph: ig.Graph, weight_attr=None):
    edges = graph.get_edgelist()
    weights = graph.es[weight_attr] if weight_attr else [1] * len(edges)

    if not graph.is_directed():
        edges.extend([(v, u) for u, v in edges])
        weights.extend(weights)

    shape = graph.vcount()
    shape = (shape, shape)
    if len(edges) > 0:
        return csr_matrix((weights, zip(*edges)), shape=shape)
    return csr_matrix(shape)


def recompute_weights(graph: ig.Graph, label_counts: Counter):
    graph = get_sparse_from_igraph(graph, weight_attr='weight')
    weights, scale_factor, w_min = [], 1., 0
    for s, t, w in zip(*[*graph.nonzero(), graph.data]):
        ns, nt = label_counts[s], label_counts[t]
        nw = w * (ns + nt) / (1. * ns * nt)
        weights.append(nw)

        scale_factor = max(weights) - min(weights)
        w_min = min(weights)
        #if w_min > nw: w_min = nw

    weights = [(w + w_min) / scale_factor for w in weights]
    return csr_matrix((weights, graph.nonzero()), shape=graph.shape)


class VIA:
    def __init__(self, data, true_label=None, anndata=None, dist_std_local=2, jac_std_global='median',
                 keep_all_local_dist='auto',
                 too_big_factor=0.4, resolution_parameter=1.0, partition_type="ModularityVP", small_pop=10,
                 jac_weighted_edges=True, knn=30, n_iter_leiden=5, random_seed=42,
                 num_threads=-1, distance='l2', time_smallpop=15,
                 super_cluster_labels=False,
                 super_node_degree_list=False, super_terminal_cells=False, x_lazy=0.95, alpha_teleport=0.99,
                 root_user="root_cluster", preserve_disconnected=True, dataset="humanCD34", super_terminal_clusters=[],
                 is_coarse=True, csr_full_graph='', csr_array_locally_pruned='', ig_full_graph='',
                 full_neighbor_array='', full_distance_array='', embedding=None, df_annot=None,
                 preserve_disconnected_after_pruning=False,
                 secondary_annotations=None, pseudotime_threshold_TS=30, cluster_graph_pruning_std=0.15,
                 visual_cluster_graph_pruning=0.15, neighboring_terminal_states_threshold=3, num_mcmc_simulations=1300,
                 piegraph_arrow_head_width=0.1,
                 piegraph_edgeweight_scalingfactor=1.5, max_visual_outgoing_edges=2, via_coarse=None):

        self.data = data
        self.nsamples, self.ncomp = data.shape
        if true_label is not None:
            self.true_label = true_label
        else: self.true_label = [1] * self.nsamples

        self.knn_struct = None
        self.labels = None
        self.connected_comp_labels = None
        self.edgelist = None
        self.edgelist_unique = None

        # higher dist_std_local means more edges are kept
        # highter jac_std_global means more edges are kept
        if keep_all_local_dist == 'auto':
            # If large dataset skip local pruning to increase speed
            keep_all_local_dist = data.shape[0] > 50000

        if resolution_parameter != 1:
            partition_type = "RBVP"  # Reichardt and Bornholdt’s Potts model. Note that this is the same as ModularityVertexPartition when setting 𝛾 = 1 and normalising by 2m

        self.anndata = anndata
        self.dist_std_local = dist_std_local
        self.jac_std_global = jac_std_global  ##0.15 is also a recommended value performing empirically similar to 'median'
        self.keep_all_local_dist = keep_all_local_dist
        self.too_big_factor = too_big_factor  ##if a cluster exceeds this share of the entire cell population, then the PARC will be run on the large cluster. at 0.4 it does not come into play
        self.resolution_parameter = resolution_parameter
        self.partition_type = partition_type
        self.small_pop = small_pop  # smallest cluster population to be considered a community
        self.jac_weighted_edges = jac_weighted_edges
        self.knn = knn
        self.n_iter_leiden = n_iter_leiden
        self.random_seed = random_seed  # enable reproducible Leiden clustering
        self.num_threads = num_threads  # number of threads used in KNN search/construction
        self.distance = distance  # Euclidean distance 'l2' by default; other options 'ip' and 'cosine'
        self.time_smallpop = time_smallpop
        if via_coarse is None:
            self.super_cluster_labels = super_cluster_labels
            self.super_node_degree_list = super_node_degree_list
            self.super_terminal_clusters =super_terminal_clusters
            self.full_neighbor_array = full_neighbor_array
            self.full_distance_array = full_distance_array
            self.ig_full_graph = ig_full_graph
            self.csr_array_locally_pruned = csr_array_locally_pruned
            self.csr_full_graph = csr_full_graph
            self.super_terminal_cells = super_terminal_cells
        else:
            self.super_cluster_labels=via_coarse.labels
            self.super_node_degree_list = via_coarse.node_degree_list
            self.super_terminal_clusters = via_coarse.terminal_clusters
            self.full_neighbor_array = via_coarse.full_neighbor_array
            self.full_distance_array = via_coarse.full_distance_array
            self.ig_full_graph = via_coarse.ig_full_graph
            self.csr_array_locally_pruned = via_coarse.csr_array_locally_pruned
            self.csr_full_graph = via_coarse.csr_full_graph
            self.super_terminal_cells = get_loc_terminal_states(via0=via_coarse, X_input=data)

        self.x_lazy = x_lazy  # 1-x = probability of staying in same node
        self.alpha_teleport = alpha_teleport  # 1-alpha is probability of jumping
        self.root_user = root_user
        self.preserve_disconnected = preserve_disconnected
        self.dataset = dataset
        self.is_coarse = is_coarse #set to True for first round of VIA. if one chooses to run a second iteration of VIA that uses the terminal states from the first round, then set this to False for second iteration
        self.embedding = embedding
        self.df_annot = df_annot
        self.preserve_disconnected_after_pruning = preserve_disconnected_after_pruning
        self.secondary_annotations = secondary_annotations
        self.pseudotime_threshold_TS = pseudotime_threshold_TS
        self.cluster_graph_pruning_std = cluster_graph_pruning_std
        self.visual_cluster_graph_pruning = visual_cluster_graph_pruning  # higher value means more edges retained. This is applied to the clustergraph before visulizing.
        self.neighboring_terminal_states_threshold = neighboring_terminal_states_threshold  # number of neighbors of a terminal state has before it is eliminated as a TS
        self.num_mcmc_simulations = num_mcmc_simulations  # number of mcmc simulations in second state of pseudotime computation
        self.piegraph_arrow_head_width = piegraph_arrow_head_width
        self.piegraph_edgeweight_scalingfactor = piegraph_edgeweight_scalingfactor
        self.max_visual_outgoing_edges = max_visual_outgoing_edges  # higher value means more edges retained. This is applied to the clustergraph and is a strong threshold for number of edges shown

    def knngraph_visual(self, data_visual, knn_umap=15, downsampled=False):
        k_umap = knn_umap
        # neighbors in array are not listed in in any order of proximity
        if not downsampled:
            self.knn_struct.set_ef(k_umap + 1)
            neighbor_array, distance_array = self.knn_struct.knn_query(self.data, k=k_umap)
        else:
            knn_struct_umap = self.construct_knn(data_visual)
            knn_struct_umap.set_ef(k_umap + 1)
            neighbor_array, distance_array = knn_struct_umap.knn_query(data_visual, k=k_umap)
        row_list = []
        n_neighbors = neighbor_array.shape[1]
        n_cells = neighbor_array.shape[0]
        print('ncells and neighs', n_cells, n_neighbors)

        dummy = np.transpose(np.ones((n_neighbors, n_cells)) * range(0, n_cells)).flatten()
        print('dummy size', dummy.size)
        row_list.extend(list(np.transpose(np.ones((n_neighbors, n_cells)) * range(0, n_cells)).flatten()))

        row_min = np.min(distance_array, axis=1)
        row_sigma = np.std(distance_array, axis=1)

        distance_array = (distance_array - row_min[:, np.newaxis]) / row_sigma[:, np.newaxis]

        col_list = neighbor_array.flatten().tolist()
        distance_array = -1 * np.sqrt(distance_array.flatten())

        weight_list = np.exp(distance_array)

        threshold = np.mean(weight_list) + 2 * np.std(weight_list)

        weight_list[weight_list >= threshold] = threshold

        weight_list = weight_list.tolist()
        print('weight list', len(weight_list))

        graph = csr_matrix((np.array(weight_list), (np.array(row_list), np.array(col_list))),shape=(n_cells, n_cells))
        prod_matrix = graph.multiply(graph.T)
        return graph.T + graph - prod_matrix

    def run_umap_hnsw(self, X_input, graph, n_components=2, alpha: float = 1.0, negative_sample_rate: int = 5,
                      gamma: float = 1.0, spread=1.0, min_dist=0.1, init_pos='spectral', random_state=1, ):

        from umap.umap_ import find_ab_params, simplicial_set_embedding
        import matplotlib.pyplot as plt

        a, b = find_ab_params(spread, min_dist)
        print('a,b, spread, dist', a, b, spread, min_dist)
        t0 = time.time()
        X_umap = simplicial_set_embedding(data=X_input, graph=graph, n_components=n_components, initial_alpha=alpha,
                                          a=a, b=b, n_epochs=0, metric_kwds={}, gamma=gamma,
                                          negative_sample_rate=negative_sample_rate, init=init_pos,
                                          random_state=np.random.RandomState(random_state), metric='euclidean',
                                          verbose=1, densmap=False, output_dens=False, densmap_kwds={})
        return X_umap

    def get_terminal_clusters(self, A, markov_pt, root_ai):
        n_ = A.shape[0]  # number of states in the graph component

        if n_ <= 10: n_outlier_std = 3
        if (n_ <= 40) & (n_ > 10): n_outlier_std = 2

        if n_ >= 40: n_outlier_std = 2  # 1

        pop_list = []

        # print('get terminal', set(self.labels), np.where(self.labels == 0))
        for i in list(set(self.labels)):
            pop_list.append(len(np.where(self.labels == i)[0]))
        # we weight the out-degree based on the population of clusters to avoid allowing small clusters to become the terminals based on population alone
        A_new = A.copy()
        for i in range(A.shape[0]):
            for j in range(A.shape[0]):
                A_new[i, j] = A[i, j] * (pop_list[i] + pop_list[j]) / (pop_list[i] * pop_list[j])

        # make an igraph graph to compute the closeness
        g_dis = ig.Graph.Adjacency(
            (A_new > 0).tolist())  # need to manually add the weights as igraph treates A>0 as boolean
        g_dis.es['weights'] = 1 / A_new[A_new.nonzero()]  # we want "distances" not weights for closeness and betweeness

        betweenness_score = g_dis.betweenness(weights='weights')
        betweenness_score_array = np.asarray(betweenness_score)
        betweenness_score_takeout_outlier = betweenness_score_array[betweenness_score_array < (
                np.mean(betweenness_score_array) + n_outlier_std * np.std(betweenness_score_array))]
        betweenness_list = [i for i, score in enumerate(betweenness_score) if score < (
                np.mean(betweenness_score_takeout_outlier) - 0 * np.std(betweenness_score_takeout_outlier))]

        closeness_score = g_dis.closeness(mode='ALL', cutoff=None, weights='weights', normalized=True)
        closeness_score_array = np.asarray(closeness_score)
        closeness_score_takeout_outlier = closeness_score_array[
            closeness_score_array < (np.mean(closeness_score_array) + n_outlier_std * np.std(closeness_score_array))]
        closeness_list = [i for i, score in enumerate(closeness_score) if
                          score < (np.mean(closeness_score_takeout_outlier) - 0 * np.std(
                              closeness_score_takeout_outlier))]

        out_deg = A_new.sum(axis=1)
        out_deg = np.asarray(out_deg)

        outdegree_score_takeout_outlier = out_deg[out_deg < (np.mean(out_deg) + n_outlier_std * np.std(out_deg))]
        loc_deg = [i for i, score in enumerate(out_deg) if
                   score < (np.mean(outdegree_score_takeout_outlier) - 0 * np.std(outdegree_score_takeout_outlier))]
        print(colored(f"{datetime.now()}\tIdentifying terminal clusters corresponding to unique lineages...", "blue"))
        print(f"{datetime.now()}\tCloseness:{closeness_list}")
        print(f"{datetime.now()}\tBetweenness:{betweenness_list}")
        print(f"{datetime.now()}\tOut Degree:{loc_deg}")

        markov_pt = np.asarray(markov_pt)
        pct = 10 if n_ <= 40 else 30
        loc_pt = np.where(markov_pt >= np.percentile(markov_pt, pct))[0]

        terminal_clusters_1 = list(set(closeness_list) & set(betweenness_list))
        terminal_clusters_2 = list(set(closeness_list) & set(loc_deg))
        terminal_clusters_3 = list(set(betweenness_list) & set(loc_deg))
        terminal_clusters = list(set(terminal_clusters_1) | set(terminal_clusters_2))
        terminal_clusters = list(set(terminal_clusters) | set(terminal_clusters_3))
        terminal_clusters = list(set(terminal_clusters) & set(loc_pt))

        terminal_org = terminal_clusters.copy()

        for terminal_i in terminal_org:
            if terminal_i in terminal_clusters:
                removed_terminal_i = False
            else:
                removed_terminal_i = True
            # print('terminal state', terminal_i)
            count_nn = 0
            ts_neigh = []
            neigh_terminal = np.where(A[:, terminal_i] > 0)[0]
            if neigh_terminal.size > 0:
                for item in neigh_terminal:
                    if item in terminal_clusters:
                        ts_neigh.append(item)
                        count_nn = count_nn + 1

                    if n_ >= 10:
                        if item == root_ai:  # if the terminal state is a neighbor of
                            if terminal_i in terminal_clusters:
                                terminal_clusters.remove(terminal_i)
                                print('we removed cluster', terminal_i, 'from the shortlist of terminal states ')
                                removed_terminal_i = True
                if count_nn >= self.neighboring_terminal_states_threshold:  # 2
                    if removed_terminal_i == False:
                        temp_remove = terminal_i
                        temp_time = markov_pt[terminal_i]
                        for to_remove_i in ts_neigh:
                            if markov_pt[to_remove_i] < temp_time:
                                temp_remove = to_remove_i
                                temp_time = markov_pt[to_remove_i]
                        terminal_clusters.remove(temp_remove)
                        print('TS', terminal_i, 'had', self.neighboring_terminal_states_threshold,
                              'or more neighboring terminal states, namely', ts_neigh, ' and so we removed,',
                              temp_remove)
        if len(terminal_clusters) == 0: terminal_clusters = loc_deg
        # print('terminal_clusters', terminal_clusters)
        return terminal_clusters

    def compute_hitting_time(self, sparse_graph, root, x_lazy, alpha_teleport, number_eig=0):
        # 1- alpha is the probability of teleporting
        # 1- x_lazy is the probability of staying in current state (be lazy)
        beta_teleport = 2 * (1 - alpha_teleport) / (2 - alpha_teleport)
        N = sparse_graph.shape[0]

        Lsym = csgraph.laplacian(sparse_graph, normed=True)
        eigval, eigvec = scipy.sparse.linalg.eigsh(Lsym, number_eig or Lsym.shape[0]-1)

        Greens, beta_norm_lap = np.zeros((N, N), float), np.zeros((N, N), float)
        Xu = np.zeros((N, N))
        Xu[:, root] = 1
        Xv_Xu = np.eye(N) - Xu

        for i in range(1 if alpha_teleport == 1 else 0, len(eigval)):
            vv = np.outer(eigvec[:, i], eigvec[:, i])
            factor = beta_teleport + 2 * eigval[i] * x_lazy * (1 - beta_teleport)

            Greens += vv / factor
            beta_norm_lap += vv * factor

        D = np.diag(np.nan_to_num(np.array(sparse_graph.sum(axis=1)).reshape(-1) ** -.5, posinf=0))
        t = D @ Greens @ D * beta_teleport

        hitting_matrix = np.diagonal(t)[np.newaxis, :] - t
        # Calculate only diagonal elements of Xv_Xu @ t
        return np.abs((Xv_Xu * t.T).sum(-1)), (hitting_matrix + hitting_matrix.T)[root]

    def simulate_branch_probability(self, terminal_state, all_terminal_states, A, root, pt, num_sim=500):

        n_states = A.shape[0]

        ncpu = multiprocessing.cpu_count()
        if (ncpu == 1) | (ncpu == 2):
            n_jobs = 1
        elif ncpu > 2:
            n_jobs = min(ncpu - 1, 5)
        # print('njobs', n_jobs)
        num_sim_pp = int(num_sim / n_jobs)  # num of simulations per process
        jobs = []

        manager = multiprocessing.Manager()

        q = manager.list()
        seed_list = list(range(n_jobs))
        for i in range(n_jobs):
            cumstateChangeHist = np.zeros((1, n_states))
            cumstateChangeHist_all = np.zeros((1, n_states))
            process = multiprocessing.Process(target=prob_reaching_terminal_state1, args=(
                terminal_state, A, root,  num_sim_pp, q, cumstateChangeHist,
                cumstateChangeHist_all,
                seed_list[i]))
            jobs.append(process)

        for j in jobs:
            j.start()

        for j in jobs:
            j.join()

        cumhistory_vec = q[0][0]
        cumhistory_vec_all = q[0][1]

        count_reached = cumhistory_vec_all[0, terminal_state]

        for i in range(1, len(q)):
            cumhistory_vec = cumhistory_vec + q[i][0]
            cumhistory_vec_all = cumhistory_vec_all + q[i][1]

            count_reached = count_reached + q[i][1][0, terminal_state]

        print('From root', root, ' to Terminal state', terminal_state, 'is found', int(count_reached), ' times.')

        cumhistory_vec_all[cumhistory_vec_all == 0] = 1
        prob_ = cumhistory_vec / cumhistory_vec_all

        np.set_printoptions(precision=3)

        if count_reached == 0:
            prob_[:, terminal_state] = 0
            print('never reached state', terminal_state)
        else:
            loc_1 = np.where(prob_ == 1)

            loc_1 = loc_1[1]

            prob_[0, loc_1] = 0
            # print('zerod out prob', prob_)
            temp_ = np.max(prob_)
            if temp_ == 0: temp_ = 1
            prob_ = prob_ / min(1, 1.1 * temp_)
        prob_[0, loc_1] = 1

        return list(prob_)[0]

    def simulate_markov(self, A, root):

        n_states = A.shape[0]
        P = A / A.sum(axis=1).reshape((n_states, 1))
        # print('row normed P',P.shape, P, P.sum(axis=1))
        x_lazy = self.x_lazy  # 1-x is prob lazy
        alpha_teleport = self.alpha_teleport
        # bias_P is the transition probability matrix

        P = x_lazy * P + (1 - x_lazy) * np.identity(n_states)
        # print(P, P.sum(axis=1))
        P = alpha_teleport * P + ((1 - alpha_teleport) * (1 / n_states) * (np.ones((n_states, n_states))))
        # print('check prob of each row sum to one', P.sum(axis=1))

        currentState = root
        state = np.zeros((1, n_states))
        state[0, currentState] = 1
        num_sim = self.num_mcmc_simulations  # 1300  # 1000

        ncpu = multiprocessing.cpu_count()
        if (ncpu == 1) | (ncpu == 2):
            n_jobs = 1
        elif ncpu > 2:
            n_jobs = min(ncpu - 1, 5)
        # print('njobs', n_jobs)
        num_sim_pp = int(num_sim / n_jobs)  # num of simulations per process

        n_steps = int(2 * n_states)

        jobs = []

        manager = multiprocessing.Manager()

        q = manager.list()
        for i in range(n_jobs):
            hitting_array = np.ones((P.shape[0], 1)) * 1000
            process = multiprocessing.Process(target=simulate_markov_sub,
                                              args=(P, num_sim_pp, hitting_array, q, root))
            jobs.append(process)

        for j in jobs:
            j.start()

        for j in jobs:
            j.join()

        print('ended all multiprocesses, will retrieve and reshape')
        hitting_array = q[0]
        for qi in q[1:]:
            hitting_array = np.append(hitting_array, qi, axis=1)  # .get(), axis=1)
        # print('finished getting from queue', hitting_array.shape)
        hitting_array_final = np.zeros((1, n_states))
        no_times_state_reached_array = np.zeros((1, n_states))

        for i in range(n_states):
            rowtemp = hitting_array[i, :]
            no_times_state_reached_array[0, i] = np.sum(rowtemp != (n_steps + 1))

        for i in range(n_states):
            rowtemp = hitting_array[i, :]
            no_times_state_reached = np.sum(rowtemp != (n_steps + 1))
            if no_times_state_reached != 0:
                perc = np.percentile(rowtemp[rowtemp != n_steps + 1], 15) + 0.001  # 15 for Human and Toy
                # print('state ', i,' has perc' ,perc)
                hitting_array_final[0, i] = np.mean(rowtemp[rowtemp <= perc])
            else:
                hitting_array_final[0, i] = (n_steps + 1)

        return hitting_array_final[0]

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
        print('eig val', eig_val.shape)
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

        for i in range(start_, number_eig):  # 0 instead of 1st eg
            # print(i, 'th eigenvalue is', eig_val[i])
            vec_i = eig_vec[:, i]
            factor = beta_teleport + 2 * eig_val[i] * x_lazy * (1 - beta_teleport)

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

    def project_branch_probability_sc(self, bp_array_clus, pt):
        n_clus = len(list(set(self.labels)))
        n_cells = self.data.shape[0]

        knn_sc = 3 if self.data.shape[0] > 1000 else 10
        neighbors, _ = self.knn_struct.knn_query(self.data, k=knn_sc)

        rows, cols, weights = [], [], []
        for i, row in enumerate(neighbors):
            neighboring_clus = self.labels[row]
            for c in set(list(neighboring_clus)):
                rows.append(i)
                cols.append(c)
                weights.append(np.sum(neighboring_clus == c) / knn_sc)

        weights = csr_matrix((weights, (rows, cols)), shape=(n_cells, n_clus))
        bp_array_sc = weights.dot(bp_array_clus)
        bp_array_sc /= np.max(bp_array_sc, axis=0)  # divide cell by max value in that column

        for i, label_ts in enumerate(list(self.terminal_clusters)):
            loc_i = np.where(self.labels == label_ts)[0]
            loc_noti = np.where(self.labels != label_ts)[0]
            if np.max(bp_array_sc[loc_noti, i]) > 0.8: bp_array_sc[loc_i, i] = 1.2
        pt = np.asarray(pt)
        pt = np.reshape(pt, (n_clus, 1))
        pt_sc = weights.dot(pt)
        pt_sc /= np.amax(pt_sc)

        return bp_array_sc, pt_sc.flatten()

    def sc_transition_matrix(self,smooth_transition,perc = 10):
        #n_clus = len(list(set(self.labels)))
        #n_cells = self.data.shape[0]

        #global pruning of the locally pruned sc knn graph

        T = self.csr_array_locally_pruned.copy()
        '''
        thr_global = np.mean(T.data) - 3*np.std(T.data)
        T.data[T.data < thr_global] = 0
        T.eliminate_zeros()
        '''
        size_T = T.size
        T.data = T.data.clip(np.percentile(T.data, 10), np.percentile(T.data, 80))
        find_T = find(T)
        bias_weight, K, c, C, nu, b = [], 1, 0, 1, 1, 1
        #print('transition before biasing')
        #print(T)
        for i in range(size_T):

            #start = find_T[0][i]
            #end = find_T[1][i]
            weight = find_T[2][i]
            t_dif = self.single_cell_pt_markov[find_T[0][i]] - self.single_cell_pt_markov[find_T[1][i]]

            bias_weight.append(weight * K / ((C + math.exp(b * (t_dif + c))) ** nu))
        T = csr_matrix((bias_weight/np.mean(np.array(bias_weight)), (np.array(find_T[0]), np.array(find_T[1]))), shape=T.shape)
        #T = T.multiply(csr_matrix(1.0 / np.abs(T).sum(1)))

        T.setdiag(0)
        T.eliminate_zeros()
        T = np.expm1(T)

        #thr_perc = np.percentile(T.data, 10)
        #T.data[T.data < thr_perc] = 0
        #T.eliminate_zeros()
        #print('T rows')
        #print(find(T)[0].shape)
        #print(np.unique(find(T)[0]).size)
        T = normalize(T, norm='l1', axis=1) **smooth_transition

        T = T.multiply(csr_matrix(1.0 / np.abs(T).sum(1)))  # rows sum to one

        return T

    def velocity_embedding(self, X_emb, smooth_transition):
        #T transition matrix at single cell level
        n_obs = X_emb.shape[0]
        V_emb = np.zeros(X_emb.shape)
        T = self.sc_transition_matrix(smooth_transition)

        #the change in embedding distance when moving from cell i to its neighbors is given by dx
        for i in range(n_obs):
            indices = T[i].indices
            dX = X_emb[indices] - X_emb[i, None]  # shape (n_neighbors, 2)
            dX /= l2_norm(dX)[:, None]
            #dX /= np.sqrt(dX.multiply(dX).sum(axis=1).A1)[:, None]
            dX[np.isnan(dX)] = 0  # zero diff in a steady-state
            #neighbor edge weights are used to weight the overall dX or velocity from cell i.
            probs =  T[i].data
            if probs.size ==0: print(probs, i)
            V_emb[i] = probs.dot(dX) - probs.mean() * dX.sum(0)
        V_emb /= 3 * quiver_autoscale(X_emb, V_emb)

        return X_emb, V_emb


    def construct_knn(self, data: np.ndarray, too_big: bool = False) -> hnswlib.Index:
        """
        Construct K-NN graph for given data

        Parameters
        ----------
        data: np.ndarray of shape (n_samples, n_features)
            Data matrix over which to construct knn graph

        too_big: bool, default = False

        Returns
        -------
        Initialized instance of hnswlib.Index to be used over given data
        """
        if self.knn > 100:
            print(colored(f'Passed number of neighbors exceeds max value. Setting number of neighbors too 100', 'blue'))
        k = min(100, self.knn + 1)

        nsamples, dim = data.shape
        ef_const, M = 200, 30
        if not too_big:
            if nsamples < 10000:
                k = ef_const = min(nsamples - 10, 500)
            if nsamples <= 50000 and dim > 30:
                M = 48  # good for scRNA-seq where dimensionality is high

        p = hnswlib.Index(space=self.distance, dim=dim)
        p.set_num_threads(self.num_threads)
        p.init_index(max_elements=nsamples, ef_construction=ef_const, M=M)
        p.add_items(data)
        p.set_ef(k)
        return p

    def make_csrmatrix_noselfloop(self, neighbors: np.ndarray, distances: np.ndarray,
                                  auto_: bool = True, distance_factor=.01) -> csr_matrix:
        """
        Create sparse matrix from weighted knn graph

        Parameters
        ----------
        neighbors: np.ndarray of shape (n_samples, n_neighbors)
            Indicating neighbors of each sample. neighbors[i,j] means that sample j is a neighbor of sample i

        distances: np.ndarray of shape (n_samples, n_neighbors)
            Distances between neighboring samples corresponding `neighbors`

        auto_: bool, default=True
            If `False` and `self.keep_all_local_dist = False` perform local pruning (according to self.dist_std_local)
            and remove self-loops

        distance_factor: float, default=0.01
            Factor used in calculation of edge weights. mean(sqrt(distances))^2 / (sqrt(distances) + distance_factor)

        Returns
        -------
            sparse matrix representing the locally pruned weighted knn graph
        """
        distances = np.sqrt(distances.astype(np.float64))

        if auto_ and not self.keep_all_local_dist:
            # Local pruning based on neighbor being too far
            msk = distances <= (np.mean(distances, axis=1) + self.dist_std_local * np.std(distances, axis=1))[:, np.newaxis]
            # Remove self-loops
            msk &= (neighbors != np.arange(neighbors.shape[0])[:, np.newaxis])
        else:
            msk = np.full_like(distances, True, dtype=np.bool_)

        # Inverting the distances outputs values in range [0-1]. This also causes many ``good'' neighbors ending up
        # having a weight near zero (misleading as non-neighbors have a weight of zero). Therefore we scale by the
        # mean distance.
        weights = (np.mean(distances[msk]) ** 2) / (distances[msk] + distance_factor) #larger weight is a stronger edge
        #weights = 1 / (distances[msk] + distance_factor)
        rows = np.array([np.repeat(i, len(x)) for i, x in enumerate(neighbors)])[msk]
        cols = neighbors[msk]

        return csr_matrix((weights, (rows, cols)), shape=(len(neighbors), len(neighbors)), dtype=np.float64)

    def func_mode(self, ll):
        # return MODE of list
        # If multiple items are maximal, the function returns the first one encountered.
        return max(set(ll), key=ll.count)

    def make_JSON(self, folderpath='/home/shobi/JavaCode/basicgraph/', filename='VIA_JSON.js'):
        import networkx as nx

        from networkx.readwrite import json_graph
        from collections import defaultdict
        edgelist = self.edgelist_maxout
        weightlist = self.edgeweights_maxout
        min_w = min(weightlist)
        max_w = max(weightlist)
        weightlist = [(10 * (i - min_w) / (max_w - min_w)) + 1 for i in weightlist]
        cluster_population_dict = self.cluster_population_dict
        pop_max = cluster_population_dict[max(cluster_population_dict, key=cluster_population_dict.get)]
        pop_min = cluster_population_dict[min(cluster_population_dict, key=cluster_population_dict.get)]
        print(cluster_population_dict, pop_min, pop_max)
        pt_max = max(self.scaled_hitting_times)
        pt_min = min(self.scaled_hitting_times)
        scaled_hitting_times = [10 * (i - pt_min) / (pt_max - pt_min) for i in self.scaled_hitting_times]
        node_majority_truth_labels = []
        for ci, cluster_i in enumerate(sorted(list(set(self.labels)))):
            cluster_i_loc = np.where(np.asarray(self.labels) == cluster_i)[0]
            majority_truth = str(self.func_mode(list(np.asarray(self.true_label)[cluster_i_loc])))
            node_majority_truth_labels.append(majority_truth)

        majority_truth_labels_dict = dict(enumerate(node_majority_truth_labels))
        temp = defaultdict(lambda: len(temp))

        df_edges = pd.DataFrame(edgelist, columns=['source', 'target'])
        df_edges['weight'] = weightlist
        df_edges['distance'] = [150 / i for i in weightlist]
        G = nx.DiGraph()  # directed graph
        for key in majority_truth_labels_dict:
            print('node', key, majority_truth_labels_dict[key], round(scaled_hitting_times[key],1), cluster_population_dict[key],
                  temp[majority_truth_labels_dict[key]])
            # val denotes size in d3 by default
            G.add_node(key, group=majority_truth_labels_dict[key], pseudotime=scaled_hitting_times[key],
                       val=(10 * (cluster_population_dict[key] - pop_min) / (pop_max - pop_min)) + 1,
                       group_num=temp[majority_truth_labels_dict[key]])
        for enum_i, i in enumerate(edgelist):
            print('edge', i, weightlist[enum_i], cluster_population_dict[i[0]], 150 / weightlist[enum_i])
            if (scaled_hitting_times[i[0]] < scaled_hitting_times[i[1]]):
                source_node = i[0]
                target_node = i[1]
            else:
                source_node = i[1]
                target_node = i[0]
            # val edge controls number of emitted particles
            G.add_edge(u_of_edge=source_node, v_of_edge=target_node, weight=weightlist[enum_i],
                       val=(5 * (cluster_population_dict[source_node] - pop_min) / (pop_max - pop_min)) + 1,
                       distance=150 / weightlist[enum_i])

        # Visualize the network:
        nx.draw_networkx(G)
        plt.show()
        import json

        j = json_graph.node_link_data(G)

        js = json.dumps(j, ensure_ascii=False, indent=2)
        with open(folderpath + filename, "w") as file:
            file.write(js)
        return

    def run_toobig_subPARC(self, X_data, jac_std_toobig=1,
                           jac_weighted_edges=True):
        n_elements = X_data.shape[0]
        hnsw = self.construct_knn(X_data, too_big=True)
        if self.knn >= 0.8 * n_elements:
            k = int(0.5 * n_elements)
        else:
            k = self.knn
        neighbor_array, distance_array = hnsw.knn_query(X_data, k=k)
        csr_array = self.make_csrmatrix_noselfloop(neighbor_array, distance_array)
        sources, targets = csr_array.nonzero()
        mask = np.zeros(len(sources), dtype=bool)
        mask |= (csr_array.data > (
                np.mean(csr_array.data) + np.std(csr_array.data) * 5))  # smaller distance means stronger edge
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
            G_sim = ig.Graph(n=n_elements, edges=list(new_edgelist), edge_attrs={'weight': sim_list_new})
        else:
            G_sim = ig.Graph(n=n_elements, edges=list(new_edgelist))
        G_sim.simplify(combine_edges='sum')
        if jac_weighted_edges == True:
            if self.partition_type == 'ModularityVP':
                partition = leidenalg.find_partition(G_sim, leidenalg.ModularityVertexPartition, weights='weight',
                                                     n_iterations=self.n_iter_leiden, seed=self.random_seed)
                # print('partition type MVP')
            else:
                partition = leidenalg.find_partition(G_sim, leidenalg.RBConfigurationVertexPartition, weights='weight',
                                                     n_iterations=self.n_iter_leiden, seed=self.random_seed,
                                                     resolution_parameter=self.resolution_parameter)

        else:
            if self.partition_type == 'ModularityVP':
                # print('partition type MVP')
                partition = leidenalg.find_partition(G_sim, leidenalg.ModularityVertexPartition,
                                                     n_iterations=self.n_iter_leiden, seed=self.random_seed)
            else:
                print('partition type RBC')
                partition = leidenalg.find_partition(G_sim, leidenalg.RBConfigurationVertexPartition,
                                                     n_iterations=self.n_iter_leiden, seed=self.random_seed,
                                                     resolution_parameter=self.resolution_parameter)
        PARC_labels_leiden = np.asarray(partition.membership)
        PARC_labels_leiden = np.reshape(PARC_labels_leiden, (n_elements, 1))
        small_pop_list = []
        small_cluster_list = []
        small_pop_exist = False
        dummy, PARC_labels_leiden = np.unique(list(PARC_labels_leiden.flatten()), return_inverse=True)
        for cluster in set(PARC_labels_leiden):
            population = len(np.where(PARC_labels_leiden == cluster)[0])
            if population < 5:
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

    def find_root_group(self, graph_dense, PARC_labels_leiden, root_user, true_labels, super_cluster_labels_sub,
                        super_node_degree_list):
        # PARC_labels_leiden is the subset belonging to the component of the graph being considered. graph_dense is a component of the full graph
        majority_truth_labels = np.empty((len(PARC_labels_leiden), 1), dtype=object)
        graph_node_label = []
        min_deg = 1000
        super_min_deg = 1000
        found_super_and_sub_root = False
        found_any_root = False
        true_labels = np.asarray(true_labels)

        deg_list = graph_dense.sum(axis=1).reshape((1, -1)).tolist()[0]

        for ci, cluster_i in enumerate(sorted(list(set(PARC_labels_leiden)))):
            cluster_i_loc = np.where(np.asarray(PARC_labels_leiden) == cluster_i)[0]
            majority_truth = str(self.func_mode(list(true_labels[cluster_i_loc])))
            if self.super_cluster_labels != False:
                super_majority_cluster = self.func_mode(list(np.asarray(super_cluster_labels_sub)[cluster_i_loc]))
                super_majority_cluster_loc = np.where(np.asarray(super_cluster_labels_sub) == super_majority_cluster)[0]
                super_majority_truth = self.func_mode(list(true_labels[super_majority_cluster_loc]))
                super_node_degree = super_node_degree_list[super_majority_cluster]
                if (str(root_user) in majority_truth) & ( str(root_user) in str(super_majority_truth)):  # sub == with 'in'
                    if super_node_degree < super_min_deg:
                        found_super_and_sub_root = True
                        root = cluster_i
                        found_any_root = True
                        min_deg = deg_list[ci]
                        super_min_deg = super_node_degree
                        print('new root is', root, ' with degree %.2f' % min_deg,
                              'and super node degree %.2f' % super_min_deg)
            majority_truth_labels[cluster_i_loc] = str(majority_truth) + 'c' + str(cluster_i)

            graph_node_label.append(str(majority_truth) + 'c' + str(cluster_i))
        if (self.super_cluster_labels == False) | (found_super_and_sub_root == False):
            # print('self.super_cluster_labels', super_cluster_labels_sub, ' foundsuper_cluster_sub and super root',found_super_and_sub_root)
            for ic, cluster_i in enumerate(sorted(list(set(PARC_labels_leiden)))):
                cluster_i_loc = np.where(np.asarray(PARC_labels_leiden) == cluster_i)[0]
                # print('cluster', cluster_i, 'set true labels', set(true_labels))
                true_labels = np.asarray(true_labels)

                majority_truth = str(self.func_mode(list(true_labels[cluster_i_loc])))

                # print('cluster', cluster_i, 'has majority', majority_truth, 'with degree list', deg_list)

                if (str(root_user) in str(majority_truth)):  # 'in' not ==
                    if deg_list[ic] < min_deg:
                        root = cluster_i
                        found_any_root = True
                        min_deg = deg_list[ic]
                        print('new root is', root, ' with degree %.2f' % min_deg, majority_truth)
        # print('len graph node label', graph_node_label)
        if found_any_root == False:
            print('setting arbitrary root', cluster_i)
            root = cluster_i
        return graph_node_label, majority_truth_labels, deg_list, root


    def find_root(self, graph_dense, PARC_labels_leiden, root_user, true_labels):
        # root-user is the singlecell index given by the user when running VIA
        majority_truth_labels = np.empty((len(PARC_labels_leiden), 1), dtype=object)
        graph_node_label = []
        true_labels = np.asarray(true_labels)

        deg_list = graph_dense.sum(axis=1).reshape((1, -1)).tolist()[0]

        for ci, cluster_i in enumerate(sorted(list(set(PARC_labels_leiden)))):
            # print('cluster i', cluster_i)
            cluster_i_loc = np.where(np.asarray(PARC_labels_leiden) == cluster_i)[0]

            majority_truth = self.func_mode(list(true_labels[cluster_i_loc]))

            majority_truth_labels[cluster_i_loc] = str(majority_truth) + 'c' + str(cluster_i)

            graph_node_label.append(str(majority_truth) + 'c' + str(cluster_i))
        root = PARC_labels_leiden[root_user]
        return graph_node_label, majority_truth_labels, deg_list, root

    def find_root_2Morgan(self, graph_dense, PARC_labels_leiden, root_idx, true_labels):
        # single cell index given corresponding to user defined root cell
        majority_truth_labels = np.empty((len(PARC_labels_leiden), 1), dtype=object)
        graph_node_label = []
        true_labels = np.asarray(true_labels)

        deg_list = graph_dense.sum(axis=1).reshape((1, -1)).tolist()[0]
        secondary_annotations = np.asarray(self.secondary_annotations)
        for ci, cluster_i in enumerate(sorted(list(set(PARC_labels_leiden)))):
            cluster_i_loc = np.where(np.asarray(PARC_labels_leiden) == cluster_i)[0]

            majority_truth = self.func_mode(list(true_labels[cluster_i_loc]))
            majority_truth_secondary = str(self.func_mode(list(secondary_annotations[cluster_i_loc])))
            majority_truth_labels[cluster_i_loc] = str(majority_truth) + 'c' + str(cluster_i)

            #graph_node_label.append(str(majority_truth)[0:5] + 'C' + str(cluster_i))
            graph_node_label.append(str(majority_truth)[0:5] + 'C' + str(cluster_i) + str(majority_truth_secondary))
        root = PARC_labels_leiden[root_idx]
        return graph_node_label, majority_truth_labels, deg_list, root


    def full_graph_paths(self, data, n_components_original=1):
        # make igraph object of very low-K KNN using the knn_struct PCA-dimension space made in PARC.
        # This is later used by find_shortest_path for sc_bp visual
        # neighbor array is not listed in in any order of proximity
        print('number of components in the original full graph', n_components_original)
        print('for downstream visualization purposes we are also constructing a low knn-graph ')
        first, k0, n_comp = True, 3, n_components_original+1
        while (n_components_original == 1 and n_comp > 1) or \
                (n_components_original > 1 and k0 <= 5 and n_comp > n_components_original):
            neighbors, distances = self.knn_struct.knn_query(data, k=k0)
            csr_array = self.make_csrmatrix_noselfloop(neighbors, distances, auto_=False)
            n_comp, comp_labels = connected_components(csr_array, return_labels=True)
            if first:
                first = False
            else:
                k0 += 1

        print('size neighbor array in low-KNN in pca-space for visualization', neighbors.shape)
        n_cells, n_neighbors = neighbors.shape
        rows = np.transpose(np.ones((n_neighbors, n_cells)) * range(0, n_cells)).flatten()
        cols = neighbors.flatten()
        csr_full_graph = csr_matrix((distances.flatten(), (rows, cols)), shape=(n_cells, n_cells))

        return ig.Graph(list(zip(*csr_full_graph.nonzero()))).simplify(combine_edges='sum')

    def get_gene_expression(self, gene_exp, title_gene="", spline_order=4, cmap='jet', verbose=False):
        df_gene = pd.DataFrame()
        fig_0, ax = plt.subplots(dpi=300)
        sc_pt = self.single_cell_pt_markov
        sc_bp_original = self.single_cell_bp
        n_terminal_states = sc_bp_original.shape[1]

        palette = cm.get_cmap(cmap, n_terminal_states)
        cmap_ = palette(range(n_terminal_states))
        corr_max = 0
        for i in range(n_terminal_states):  # [0]:

            sc_bp = sc_bp_original.copy()
            if len(np.where(sc_bp[:, i] > 0.8)[0]) > 0:  # check in case this terminal state i cannot be reached (sc_bp is all 0)

                loc_i = np.where(sc_bp[:, i] > 0.95)[0]  # 0.8
                val_pt = [sc_pt[pt_i] for pt_i in loc_i]  # TODO,  replace with array to speed up

                max_val_pt = max(val_pt)

                loc_i_bp = np.where(sc_bp[:, i] > 0.000)[0]  # 0.000
                loc_i_sc = np.where(np.asarray(sc_pt) <= max_val_pt)[0]

                loc_ = np.intersect1d(loc_i_bp, loc_i_sc)

                gam_in = np.asarray(sc_pt)[loc_]
                gam_in = gam_in / max(gam_in)

                x = gam_in.reshape(-1, 1)
                y = np.asarray(gene_exp)[loc_].reshape(-1, 1)

                weights = np.asarray(sc_bp[:, i])[loc_].reshape(-1, 1)

                if len(loc_) > 1:
                    geneGAM = pg.LinearGAM(n_splines=6, spline_order=spline_order, lam=10).fit(x, y, weights=weights)
                    xval = np.linspace(min(sc_pt), 1, 100 * 2)
                    yg = geneGAM.predict(X=xval)
                else:
                    print('loc_ has length zero')

                df_temp = pd.DataFrame()
                df_temp['pt'] = xval
                df_temp['gene'] = yg
                corr = df_temp['pt'].corr(df_temp['gene'], method='pearson')
                corr_kend = df_temp['pt'].corr(df_temp['gene'], method='kendall')
                if corr < corr_max: corr_max = corr

                col_title_gene = title_gene + '_' + str(i)
                ts_ = self.terminal_clusters[i]
                do_plot = True

                if do_plot:
                    col_title_pt = f"pt_{self.terminal_clusters[i]}_{title_gene}"

                    df_gene[col_title_pt] = xval
                    df_gene[col_title_gene] = yg

                    label_str_pears = 'Lineage:' + str(self.terminal_clusters[i]) + ' ' + str(int(corr * 100)) + '%'
                    if verbose==True:
                        print('gene corr pear', title_gene, 'of lineage',  self.terminal_clusters[i], '%.0f' % (corr * 100) + '%')
                        print('gene corr kend', title_gene, 'of lineage',self.terminal_clusters[i], '%.0f' % (corr_kend * 100) + '%')
                    ax.plot(xval, yg, color=cmap_[i], linewidth=3.5, zorder=3, label=label_str_pears)
            ax.spines['right'].set_visible(False)
            ax.spines['top'].set_visible(False)
            plt.legend(fontsize=6,frameon=False)
            str_title = 'Trend:' + title_gene# + ' ' + '%.0f' % (corr_max * 100) + '%'
            plt.title(str_title)
        return

    def get_gene_expression_multi(self, ax, gene_exp, title_gene=""):
        sc_pt = self.single_cell_pt_markov
        sc_bp_original = self.single_cell_bp
        n_terminal_states = sc_bp_original.shape[1]

        jet = cm.get_cmap('jet', n_terminal_states)
        cmap_ = jet(range(n_terminal_states))

        for i in [0]:  # range(n_terminal_states):
            sc_bp = sc_bp_original.copy()
            if len(np.where(sc_bp[:, i] > 0.9)[
                       0]) > 0:  # check in case this terminal state i cannot be reached (sc_bp is all 0)

                loc_i = np.where(sc_bp[:, i] > 0.9)[0]
                val_pt = [sc_pt[pt_i] for pt_i in loc_i]  # TODO,  replace with array to speed up

                max_val_pt = max(val_pt)

                loc_i_bp = np.where(sc_bp[:, i] > 0.000)[0]  # 0.001
                loc_i_sc = np.where(np.asarray(sc_pt) <= max_val_pt)[0]

                loc_ = np.intersect1d(loc_i_bp, loc_i_sc)

                gam_in = np.asarray(sc_pt)[loc_]
                x = gam_in.reshape(-1, 1)
                y = np.asarray(gene_exp)[loc_].reshape(-1, 1)

                weights = np.asarray(sc_bp[:, i])[loc_].reshape(-1, 1)

                if len(loc_) > 1:
                    geneGAM = pg.LinearGAM(n_splines=10, spline_order=4, lam=10).fit(x, y, weights=weights)
                    xval = np.linspace(min(sc_pt), max_val_pt, 100 * 2)
                    yg = geneGAM.predict(X=xval)
                else:
                    print('loc_ has length zero')

                ax.plot(xval, yg, color='navy', linewidth=3.5, zorder=3, label=f"TS:{self.terminal_clusters[i]}")

            ax.set_title(title_gene)

    def do_impute(self, df_gene, magic_steps=3, gene_list=[]):
        # ad_gene is an ann data object from scanpy
        # normalize across columns to get Transition matrix.
        transition_full_graph = normalize(self.csr_full_graph, norm='l1', axis=1) ** magic_steps

        #print('shape of transition matrix raised to power', magic_steps, transition_full_graph.shape)
        subset = df_gene[gene_list].values
        return pd.DataFrame(transition_full_graph.dot(subset), index=df_gene.index, columns=gene_list)

    def run_subPARC(self):
        print(f"{datetime.now()}\tGlobal pruning of weighted knn graph")
        # Construct graph or obtain from previous run
        if self.is_coarse:
            neighbors, distances = self.knn_struct.knn_query(self.data, k=self.knn)
            adjacency = self.make_csrmatrix_noselfloop(neighbors, distances) #this function has local pruning
        else:
            neighbors, distances = self.full_neighbor_array, self.full_distance_array
            adjacency = self.csr_array_locally_pruned

        edges = np.array(list(zip(*adjacency.nonzero())))
        sim = np.array(ig.Graph(n=self.nsamples, edges=edges.tolist(), edge_attrs={'weight': adjacency.data})\
                       .similarity_jaccard(pairs=edges))
        tot = len(sim)

        # Prune edges off graph on global level
        threshold = np.median(sim) if self.jac_std_global == 'median' else sim.mean() - self.jac_std_global * sim.std()
        strong_locs = np.asarray(np.where(sim > threshold)[0])
        #strong_locs = np.where(sim > threshold)[0]
        pruned_similarities = ig.Graph(n=self.nsamples, edges=list(edges[strong_locs]),
                                       edge_attrs={'weight': list(sim[strong_locs])}).simplify(combine_edges='sum') #used for clustering
        #pruned_similarities = ig.Graph(n=self.nsamples, edges=edges[strong_locs],    edge_attrs={'weight': sim[strong_locs]})\    .simplify(combine_edges='sum')
        print(f"{datetime.now()}\tFinished global pruning. Kept {round(100 * len(strong_locs) / tot, 2)} of edges. ")

        if self.is_coarse:
            # Construct full graph with no pruning - used for cluster graph edges, not listed in any order of proximity
            csr_full_graph = self.make_csrmatrix_noselfloop(neighbors, distances, auto_=False, distance_factor=0.05) #no local pruning: auto_ set to false
            n_original_comp, n_original_comp_labels = connected_components(csr_full_graph, directed=False)

            #edges = list(zip(*adjacency.nonzero()))
            edges = list(zip(*csr_full_graph.nonzero()))
            sim = ig.Graph(edges, edge_attrs={'weight': csr_full_graph.data}).similarity_jaccard(pairs=edges)
            #ig_fullgraph = ig.Graph(edges, edge_attrs={'weight': sim}).simplify(combine_edges='sum')
            self.ig_full_graph = ig.Graph(edges, edge_attrs={'weight': sim}).simplify(combine_edges='sum') # for VIA we prune the vertex cluster graph *after* making the clustergraph

            self.csr_array_locally_pruned = adjacency #used for clustering
            #self.ig_full_graph = ig_fullgraph
            self.csr_full_graph = csr_full_graph
            self.full_neighbor_array = neighbors
            self.full_distance_array = distances

            # knn graph used for making trajectory drawing on the visualization
            self.full_graph_shortpath = self.full_graph_paths(self.data, n_original_comp)
            neighbors = self.full_neighbor_array


        print(colored(f"{datetime.now()}\tCommencing community detection", 'blue'))
        weights = 'weight' if self.jac_weighted_edges else None
        type = leidenalg.ModularityVertexPartition if self.partition_type == 'ModularityVP' else leidenalg.RBConfigurationVertexPartition
        partition = leidenalg.find_partition(pruned_similarities, partition_type=type, weights=weights,
                                             n_iterations=self.n_iter_leiden, seed=self.random_seed)
        labels = np.array(partition.membership)

        print(f"{datetime.now()}\tFinished running Leiden algorithm. Found {len(set(labels))} clusters.")

        # Searching for clusters that are too big and split them
        too_big_clusters = [k for k, v in Counter(labels).items() if v > self.too_big_factor * self.nsamples]
        if len(too_big_clusters):
            print(colored(f"{datetime.now()}\tFound {len(too_big_clusters)} clusters that are too big", "blue"))

        time0_big = time.time()
        count_big_pop = len(too_big_clusters)
        num_times_expanded = 0
        # TODO - add max running time condition
        while len(too_big_clusters)>0 & (not ((time.time() - time0_big > 200) & (num_times_expanded >= count_big_pop))):
        #while len(too_big_clusters) & (not((time.time() - time0_big >200) & (num_times_expanded >= count_big_pop))):
            print('inside handle too big')
            cluster = too_big_clusters.pop(0)
            idx = labels == cluster
            print(f"{datetime.now()}\tCluster {cluster} contains "
                  f"{idx.sum()}>{round(self.too_big_factor * self.nsamples)} samples and is too big")

            data = self.data[idx]
            membership = max(labels) + 1 + np.array(self.run_toobig_subPARC(data))
            num_times_expanded +=1

            if len(set(membership)) > 1:
                labels[idx] = membership
                too_big_clusters.extend(
                    [k for k, v in Counter(membership).items() if v > self.too_big_factor * self.nsamples])
            else:
                print(f"{datetime.now()}\t\tCould not expand cluster {cluster}")

        # Search for clusters that are too small (like singletons) and merge them to non-small clusters based on neighbors' majority vote
        #first we make a quick pass through all clusters to remove very small outliers by merging with a larger cluster
        #print('before final small cluster handling we have',len(set(labels)), 'communities')

        too_small_clusters = {k for k, v in Counter(labels).items() if v < self.small_pop}
        print(f"{datetime.now()}\tMerging {len(set(too_small_clusters))} very small clusters (<{self.small_pop})")
        idx = np.where(np.isin(labels, list(too_small_clusters)))[0]
        neighbours_labels = labels[neighbors[idx]]
        for i, nl in zip(*[idx, neighbours_labels]):
            # Retrieve the first non small cluster, with highest number of neighbours
            label = next((label for label, n in Counter(nl).most_common() if label not in too_small_clusters), None)
            # label = next((label for label, n in Counter(nl).most_common()), None)
            if label is not None:  # recall 0 is a valid label value
                labels[i] = label


            #too_small_clusters = {k for k, v in Counter(labels).items() if v < self.small_pop}
        too_small_clusters = {k for k, v in Counter(labels).items() if v < self.small_pop}
        #in this pass we allow clusters to be merged even if they are not a Large Cluster.. as multiple smaller ones might come together to form an acceptably large cluster
        do_while_time = time.time()
        while len(too_small_clusters) & (time.time() - do_while_time < 15):
            # membership of neighbours of samples in small clusters
            idx = np.where(np.isin(labels, list(too_small_clusters)))[0]
            neighbours_labels = labels[neighbors[idx]]
            for i, nl in zip(*[idx, neighbours_labels]):
                # Retrieve the first non small cluster, with highest number of neighbours
                # label = next((label for label, n in Counter(nl).most_common() if label not in too_small_clusters), None)
                label = next((label for label, n in Counter(nl).most_common()), None)
                if label is not None:  # recall 0 is a valid label value
                    labels[i] = label
            # Update set of too small clusters, stopping if converged
            too_small_clusters = {k for k, v in Counter(labels).items() if v < self.small_pop}
        # Reset labels to begin from zero and with no missing numbers
        self.labels = labels = np.unique(labels, return_inverse=True)[1]

        print(f"{datetime.now()}\tFinished detecting communities. Found", len(set(self.labels)), 'communities')

        # end community detection
        # do kmeans instead
        '''
        from sklearn.cluster import KMeans
        kmeans = KMeans(n_clusters=20, random_state=1).fit(X_data)
        self.labels = kmeans.labels_
        n_clus = len(set(self.labels))
        self.labels = kmeans.labels_.flatten().tolist()

        pop_list = []
        pop_list_raw = []
        for item in range(len(set(PARC_labels_leiden))):
            pop_item = PARC_labels_leiden.count(item)
            pop_list.append((item, pop_item))
            pop_list_raw.append(pop_item)
        '''

        # Make cluster-graph
        print(f"{datetime.now()}\tMaking cluster graph. Global cluster graph pruning level: {self.cluster_graph_pruning_std}")
        graph = ig.VertexClustering(self.ig_full_graph, membership=self.labels).cluster_graph(combine_edges='sum')

        graph = recompute_weights(graph, Counter(labels)) #this is the cluster graph we want to use to weight velocity. before global pruning.
        edgeweights, edges, comp_labels = pruning_clustergraph(graph,
            global_pruning_std=self.cluster_graph_pruning_std,
            preserve_disconnected=self.preserve_disconnected,
            preserve_disconnected_after_pruning=self.preserve_disconnected_after_pruning)
        self.connected_comp_labels = comp_labels

        locallytrimmed_g = ig.Graph(edges, edge_attrs={'weight': edgeweights}).simplify(combine_edges='sum')
        locallytrimmed_sparse_vc = get_sparse_from_igraph(locallytrimmed_g, weight_attr='weight')
        weights_for_layout = np.asarray(locallytrimmed_sparse_vc.data)

        weights_for_layout= np.clip(weights_for_layout, np.percentile(weights_for_layout, 10), np.percentile(weights_for_layout, 90)) #want to clip the weights used to get the layout
        weights_for_layout = list(weights_for_layout)

        g_layout = ig.Graph(list(zip(*locallytrimmed_sparse_vc.nonzero())), edge_attrs={'weight': weights_for_layout})
        #layout = locallytrimmed_g.layout_fruchterman_reingold(weights='weight') #uses non-clipped weights but this can skew layout due to one or two outlier edges
        layout = g_layout.layout_fruchterman_reingold(weights='weight')

        # globally trimmed link
        self.edgelist_unique = set(tuple(sorted(l)) for l in zip(*locallytrimmed_sparse_vc.nonzero()))  # keep only one of (0,1) and (1,0)
        self.edgelist = edges

        # number of components
        n_components, labels_cc = connected_components(csgraph=locallytrimmed_sparse_vc, directed=False, return_labels=True)

        df_graph = pd.DataFrame(locallytrimmed_sparse_vc.todense())

        df_graph['cc'] = labels_cc
        df_graph['pt'] = float('NaN')
        df_graph['majority_truth'] = 'maj truth'
        df_graph['graph_node_label'] = 'node label'

        PARC_labels_leiden = self.labels
        set_parc_labels = list(set(PARC_labels_leiden))
        set_parc_labels.sort()

        #root_user = self.root_user
        tsi_list = []
        df_graph['markov_pt'] = float('NaN')
        terminal_clus = []
        node_deg_list = []
        super_terminal_clus_revised = []
        pd_columnnames_terminal = []
        dict_terminal_super_sub_pairs = {}
        self.root = []
        large_components = []
        for comp_i in range(n_components):
            loc_compi = np.where(labels_cc == comp_i)[0]
            if len(loc_compi) > 1:
                large_components.append(comp_i)
        for comp_i in large_components:  # range(n_components):
            loc_compi = np.where(labels_cc == comp_i)[0]

            a_i = df_graph.iloc[loc_compi][loc_compi].values
            a_i = csr_matrix(a_i, (a_i.shape[0], a_i.shape[0]))
            cluster_labels_subi = [x for x in loc_compi]
            # print('cluster_labels_subi', cluster_labels_subi)
            sc_labels_subi = [PARC_labels_leiden[i] for i in range(len(PARC_labels_leiden)) if
                              (PARC_labels_leiden[i] in cluster_labels_subi)]

            sc_truelabels_subi = [self.true_label[i] for i in range(len(PARC_labels_leiden)) if
                                  (PARC_labels_leiden[i] in cluster_labels_subi)]

            # TODO - remove this code and dataset specific `find_root` methods to external file
            if self.dataset in ['toy','faced','mESC','iPSC','group']:#((self.dataset == 'toy') | (self.dataset == 'faced')):

                for ri in self.root_user:
                    if ri in sc_truelabels_subi: root_user_ = ri
                if self.super_cluster_labels:# != False:
                    # find which sub-cluster has the super-cluster root

                    super_labels_subi = [self.super_cluster_labels[i] for i in range(len(PARC_labels_leiden)) if
                                         (PARC_labels_leiden[i] in cluster_labels_subi)]

                    graph_node_label, majority_truth_labels, node_deg_list_i, root_i = self.find_root_group(a_i,
                                                                                                            sc_labels_subi,
                                                                                                            root_user_,
                                                                                                            sc_truelabels_subi,
                                                                                                            super_labels_subi,
                                                                                                            self.super_node_degree_list)
                else:

                    graph_node_label, majority_truth_labels, node_deg_list_i, root_i = self.find_root_group(a_i,
                                                                                                            sc_labels_subi,
                                                                                                            root_user_,
                                                                                                            sc_truelabels_subi,
                                                                                                            [], [])

            elif self.dataset in ['humanCD34' 'bcell','EB']:#(self.dataset == 'humanCD34'):  # | (self.dataset == '2M'):
                for ri in self.root_user:
                    if PARC_labels_leiden[ri] in cluster_labels_subi: root_user_ = ri
                graph_node_label, majority_truth_labels, node_deg_list_i, root_i = self.find_root(a_i,
                                                                                                            sc_labels_subi,
                                                                                                            root_user_,
                                                                                                            sc_truelabels_subi)

            elif (self.dataset == '2M'):
                for ri in self.root_user:
                    if PARC_labels_leiden[ri] in cluster_labels_subi: root_user_ = ri
                graph_node_label, majority_truth_labels, node_deg_list_i, root_i = self.find_root_2Morgan(a_i,
                                                                                                          sc_labels_subi,
                                                                                                          root_user_,
                                                                                                          sc_truelabels_subi)

            else:
                if comp_i > len(self.root_user) - 1:
                    root_generic = 0
                else:
                    for ri in self.root_user:
                        if PARC_labels_leiden[ri] in cluster_labels_subi:
                            root_user_ = ri
                            print(f"{datetime.now()}\tThe root index, {ri} provided by the user belongs to cluster number {PARC_labels_leiden[ri]}                                  and corresponds to cell type {self.true_label[ri]}")
                graph_node_label, majority_truth_labels, node_deg_list_i, root_i = self.find_root(a_i,
                                                                                                        PARC_labels_leiden,
                                                                                                        root_user_,
                                                                                                        self.true_label)

            self.root.append(root_i)
            self.majority_truth_labels = majority_truth_labels  # single cell level "Majority Truth of that Cluster + Clusterlabel"

            for item in node_deg_list_i:
                node_deg_list.append(item)


            new_root_index_found = False
            for ii, llabel in enumerate(cluster_labels_subi):
                if root_i == llabel:
                    new_root_index = ii
                    new_root_index_found = True

            if not new_root_index_found:
                print('cannot find the new root index')
                new_root_index = 0

            print(f"{datetime.now()}\tComputing lazy-teleporting expected hitting times")
            hitting_times, roundtrip_times = self.compute_hitting_time(a_i, new_root_index, self.x_lazy, self.alpha_teleport)
            # rescale hitting times
            very_high = np.mean(hitting_times) + 1.5 * np.std(hitting_times)
            without_very_high_pt = [iii for iii in hitting_times if iii < very_high]
            new_very_high = np.mean(without_very_high_pt) + np.std(without_very_high_pt)
            # print('very high, and new very high', very_high, new_very_high)
            new_hitting_times = [x if x < very_high else very_high for x in hitting_times]
            hitting_times = np.asarray(new_hitting_times)
            scaling_fac = 10 / max(hitting_times)
            hitting_times = hitting_times * scaling_fac
            s_ai, t_ai = a_i.nonzero()
            edgelist_ai = list(zip(s_ai, t_ai))
            edgeweights_ai = a_i.data
            # print('edgelist ai', edgelist_ai)
            # print('edgeweight ai', edgeweights_ai)
            biased_edgeweights_ai = get_biased_weights(edgelist_ai, edgeweights_ai, hitting_times)

            # biased_sparse = csr_matrix((biased_edgeweights, (row, col)))
            adjacency_matrix_ai = np.zeros((a_i.shape[0], a_i.shape[0]))

            for i, (start, end) in enumerate(edgelist_ai):
                adjacency_matrix_ai[start, end] = biased_edgeweights_ai[i]

            markov_hitting_times_ai = self.simulate_markov(adjacency_matrix_ai,
                                                           new_root_index)  # +adjacency_matrix.T))

            # for eee, ttt in enumerate(markov_hitting_times_ai):print('cluster ', eee, ' had markov time', ttt)

            very_high = np.mean(markov_hitting_times_ai) + 1.5 * np.std(markov_hitting_times_ai)  # 1.5
            very_high = min(very_high, max(markov_hitting_times_ai))
            without_very_high_pt = [iii for iii in markov_hitting_times_ai if iii < very_high]
            new_very_high = min(np.mean(without_very_high_pt) + np.std(without_very_high_pt), very_high)

            new_markov_hitting_times_ai = [x if x < very_high else very_high for x in markov_hitting_times_ai]
            # for eee, ttt in enumerate(new_markov_hitting_times_ai):      print('cluster ', eee, ' had markov time', ttt)

            markov_hitting_times_ai = np.asarray(new_markov_hitting_times_ai)
            scaling_fac = 10 / max(markov_hitting_times_ai)
            markov_hitting_times_ai = markov_hitting_times_ai * scaling_fac
            # for eee, ttt in enumerate(markov_hitting_times_ai):print('cluster ', eee, ' had markov time', ttt)

            # print('markov hitting times', [(i, j) for i, j in enumerate(markov_hitting_times_ai)])
            # print('hitting times', [(i, j) for i, j in enumerate(hitting_times)])
            markov_hitting_times_ai = (markov_hitting_times_ai)  # + hitting_times)*.5 #consensus
            adjacency_matrix_csr_ai = sparse.csr_matrix(adjacency_matrix_ai)
            (sources, targets) = adjacency_matrix_csr_ai.nonzero()
            edgelist_ai = list(zip(sources, targets))
            weights_ai = adjacency_matrix_csr_ai.data
            bias_weights_2_ai = get_biased_weights(edgelist_ai, weights_ai, markov_hitting_times_ai, round=2)
            adjacency_matrix2_ai = np.zeros((adjacency_matrix_ai.shape[0], adjacency_matrix_ai.shape[0]))

            for i, (start, end) in enumerate(edgelist_ai):
                adjacency_matrix2_ai[start, end] = bias_weights_2_ai[i]

            if self.super_terminal_cells == False: # when is_coarse = True, there is no list of terminal clusters/cells that are passed into VIA based on a previous iteration.
                # print('new_root_index', new_root_index, ' before get terminal')
                terminal_clus_ai = self.get_terminal_clusters(adjacency_matrix2_ai, markov_hitting_times_ai, new_root_index)
                temp_terminal_clus_ai = []
                for i in terminal_clus_ai:
                    if markov_hitting_times_ai[i] > np.percentile(np.asarray(markov_hitting_times_ai),
                                                                  self.pseudotime_threshold_TS):
                        terminal_clus.append(cluster_labels_subi[i])
                        temp_terminal_clus_ai.append(i)
                terminal_clus_ai = temp_terminal_clus_ai

            elif len(self.super_terminal_clusters) > 0:  # in the case where is_coarse = False, VIA is instantiated with a list of super_terminal_cells which belong to super_terminal clusters. We need to use these to select which of the clusters in the fine-grained graph capture the original set of super-clusters
                #print('super_terminal_clusters', self.super_terminal_clusters)
                sub_terminal_clus_temp_ = []
                # print('cluster_labels_subi', cluster_labels_subi)
                terminal_clus_ai = []
                super_terminal_clusters_i = [stc_i for stc_i in self.super_terminal_clusters if
                                             stc_i in cluster_labels_subi]

                for i in self.super_terminal_clusters:

                    sub_terminal_clus_temp_loc = np.where(np.asarray(self.super_cluster_labels) == i)[0]

                    true_majority_i = [xx for xx in np.asarray(self.true_label)[sub_terminal_clus_temp_loc]]
                    # print(true_majority_i[0], 'true_majority_i', 'of cluster', i)
                    # 0:1 for single connected structure #when using Toy as true label has T1 or T2

                    temp_set = set(list(np.asarray(self.labels)[
                                            sub_terminal_clus_temp_loc]))  # the clusters in second iteration that make up the super clusters in first iteration


                    temp_set = [t_s for t_s in temp_set if t_s in cluster_labels_subi]

                    temp_max_pt = 0
                    most_likely_sub_terminal = False
                    count_frequency_super_in_sub = 0
                    # If you have disconnected components and corresponding labels to identify which cell belongs to which components, then use the Toy 'T1_M1' format
                    CHECK_BOOL = False

                    #if (self.dataset == 'toy'): if (root_user_[0:2] in true_majority_i[0]) | (root_user_[0:1] == 'M'): CHECK_BOOL = True
                    if self.dataset =='group':
                        if root_user_ in true_majority_i:
                            CHECK_BOOL = True

                    if root_i in cluster_labels_subi:
                        CHECK_BOOL= True

                    # Find the sub-terminal cluster in second iteration of VIA that best corresponds to the super-terminal cluster  (i)from iteration 1
                    if (CHECK_BOOL==True) & (len(temp_set)>0):# | (                            self.dataset != 'toy'):  # 0:1 for single connected structure #when using Toy as true label has T1 or T2

                        for j in temp_set:
                            loc_j_in_sub_ai = np.where(loc_compi == j)[0]


                            super_cluster_composition_loc = np.where(np.asarray(self.labels) == j)[0]
                            super_cluster_composition = self.func_mode(
                                list(np.asarray(self.super_cluster_labels)[super_cluster_composition_loc]))

                            if (markov_hitting_times_ai[loc_j_in_sub_ai] > temp_max_pt) & (
                                    super_cluster_composition == i):
                                temp_max_pt = markov_hitting_times_ai[loc_j_in_sub_ai]

                                most_likely_sub_terminal = j
                        if most_likely_sub_terminal == False:
                            print('no sub cluster has majority made of super-cluster ', i)
                            for j in temp_set:
                                super_cluster_composition_loc = np.where(np.asarray(self.labels) == j)[0]
                                count_frequency_super_in_sub_temp = list(
                                    np.asarray(self.super_cluster_labels)[super_cluster_composition_loc]).count(i)
                                count_frequency_super_in_sub_temp_ratio = count_frequency_super_in_sub_temp / len(
                                    super_cluster_composition_loc)
                                if (markov_hitting_times_ai[loc_j_in_sub_ai] > np.percentile(
                                        np.asarray(markov_hitting_times_ai), 30)) & (  # 30
                                        count_frequency_super_in_sub_temp_ratio > count_frequency_super_in_sub):
                                    count_frequency_super_in_sub = count_frequency_super_in_sub_temp

                                    most_likely_sub_terminal = j

                        sub_terminal_clus_temp_.append(most_likely_sub_terminal)

                        if (markov_hitting_times_ai[loc_j_in_sub_ai] > np.percentile(
                                np.asarray(markov_hitting_times_ai), self.pseudotime_threshold_TS)):  # 30

                            dict_terminal_super_sub_pairs.update({i: most_likely_sub_terminal})
                            super_terminal_clus_revised.append(i)
                            terminal_clus.append(most_likely_sub_terminal)
                            terminal_clus_ai.append(
                                np.where(np.asarray(cluster_labels_subi) == most_likely_sub_terminal)[0][0])  # =i

                            print('the sub terminal cluster that best captures the super terminal', i, 'is', most_likely_sub_terminal)
                        else:
                            print('the sub terminal cluster that best captures the super terminal', i, 'is',
                                  most_likely_sub_terminal, 'but the pseudotime is too low')

            else: #this would only happen in a second iteration of VIA. if super-terminal cells was provided but no super_terminal_clusters where provided, then we use these to identify the terminal clusters in the current fine-grained VIA iteration
                print('super terminal cells', self.super_terminal_cells)

                temp = [self.labels[ti] for ti in self.super_terminal_cells if
                        self.labels[ti] in cluster_labels_subi]
                terminal_clus_ai = []
                for i in temp:
                    terminal_clus_ai.append(np.where(np.asarray(cluster_labels_subi) == i)[0][0])
                    terminal_clus.append(i)
                    dict_terminal_super_sub_pairs.update({i: most_likely_sub_terminal})

            print( f"{datetime.now()}\tTerminal clusters corresponding to unique lineages in this component are {terminal_clus_ai} "   )

            #print('final terminal clus', terminal_clus)
            for target_terminal in terminal_clus_ai:
                prob_ai = self.simulate_branch_probability(target_terminal, terminal_clus_ai,
                                                           adjacency_matrix2_ai,
                                                           new_root_index, pt=markov_hitting_times_ai,
                                                           num_sim=500)
                df_graph['terminal_clus' + str(cluster_labels_subi[target_terminal])] = 0.0000000

                pd_columnnames_terminal.append('terminal_clus' + str(cluster_labels_subi[target_terminal]))

                for k, prob_ii in enumerate(prob_ai):
                    df_graph.at[cluster_labels_subi[k], 'terminal_clus' + str(
                        cluster_labels_subi[target_terminal])] = prob_ii
            bp_array = df_graph[pd_columnnames_terminal].values
            bp_array[np.isnan(bp_array)] = 1e-8
            bp_array = bp_array / bp_array.sum(axis=1)[:, None]
            bp_array[np.isnan(bp_array)] = 1e-8

            for ei, ii in enumerate(loc_compi):
                df_graph.at[ii, 'pt'] = hitting_times[ei]
                df_graph.at[ii, 'graph_node_label'] = graph_node_label[ei]
                df_graph.at[ii, 'majority_truth'] = graph_node_label[ei]
                df_graph.at[ii, 'markov_pt'] = markov_hitting_times_ai[ei]

            locallytrimmed_g.vs["label"] = df_graph['graph_node_label'].values
            hitting_times = df_graph['pt'].values

        if len(super_terminal_clus_revised) > 0:
            self.revised_super_terminal_clusters = super_terminal_clus_revised
        else:
            self.revised_super_terminal_clusters = self.super_terminal_clusters
        self.hitting_times = hitting_times
        self.markov_hitting_times = df_graph['markov_pt'].values  # hitting_times#
        self.terminal_clusters = terminal_clus
        print(colored(f"{datetime.now()}\tTerminal clusters corresponding to unique lineages are {self.terminal_clusters} ", "blue"))
        self.node_degree_list = node_deg_list
        print(colored(f"{datetime.now()}\tBegin projection of pseudotime and lineage likelihood", "blue"))
        self.single_cell_bp, self.single_cell_pt_markov = self.project_branch_probability_sc(bp_array, df_graph['markov_pt'].values)
        #print('testing sc_transition')
        #self.sc_transition_matrix()

        self.dict_terminal_super_sub_pairs = dict_terminal_super_sub_pairs
        hitting_times = self.markov_hitting_times

        bias_weights_2_all = get_biased_weights(self.edgelist, edgeweights, self.markov_hitting_times, round=2)

        n_clus = len(set(self.labels))
        temp_csr = csr_matrix((bias_weights_2_all, tuple(zip(*self.edgelist))), shape=(n_clus, n_clus))

        #visual_g = ig.Graph(self.edgelist, edge_attrs={'weight': bias_weights_2_all}).simplify(combine_edges='sum')
        #layout = visual_g.layout_fruchterman_reingold(weights='weight')
        #self.layout =layout
        # simplifying structure of edges used on the visual layout
        edgeweights_maxout_2, edgelist_maxout_2, comp_labels_2 = pruning_clustergraph(temp_csr,
                                 global_pruning_std=self.visual_cluster_graph_pruning,
                                 max_outgoing=self.max_visual_outgoing_edges,
                                 preserve_disconnected=self.preserve_disconnected)

        temp_csr = csr_matrix((np.array(edgeweights_maxout_2), tuple(zip(*edgelist_maxout_2))), shape=(n_clus, n_clus))
        temp_csr = temp_csr.transpose().todense() + temp_csr.todense()
        temp_csr = np.tril(temp_csr, -1)  # elements along the main diagonal and above are set to zero
        temp_csr = csr_matrix(temp_csr)
        edgeweights_maxout_2 = temp_csr.data
        scale_factor = max(edgeweights_maxout_2) - min(edgeweights_maxout_2)
        edgeweights_maxout_2 = [((wi + .1) * 2.5 / scale_factor) + 0.1 for wi in edgeweights_maxout_2]

        sources, targets = temp_csr.nonzero()
        edgelist_maxout_2 = list(zip(sources.tolist(), targets.tolist()))
        self.edgelist_maxout = edgelist_maxout_2
        self.edgeweights_maxout = edgeweights_maxout_2

        remove_outliers = hitting_times
        threshold = np.percentile(remove_outliers, 95)  # np.mean(remove_outliers) + 1* np.std(remove_outliers)
        th_hitting_times = [x if x < threshold else threshold for x in hitting_times]
        remove_outliers_low = hitting_times[hitting_times < (np.mean(hitting_times) - 0.3 * np.std(hitting_times))]
        threshold_low = 0 if remove_outliers_low.size else np.percentile(remove_outliers_low, 5)
        th_hitting_times = [x if x > threshold_low else threshold_low for x in th_hitting_times]

        scaled_hitting_times = (th_hitting_times - np.min(th_hitting_times))
        npmax = np.max(scaled_hitting_times) or 1
        scaled_hitting_times = scaled_hitting_times * (1000 / npmax)

        self.scaled_hitting_times = scaled_hitting_times
        scaled_hitting_times = scaled_hitting_times.astype(int)
        pal = ig.drawing.colors.AdvancedGradientPalette(['yellow', 'green', 'blue'], n=1001)

        all_colors = []
        for i in scaled_hitting_times:
            all_colors.append(pal.get(int(i))[0:3])

        locallytrimmed_g.vs['hitting_times'] = scaled_hitting_times
        locallytrimmed_g.vs['color'] = [pal.get(i)[0:3] for i in scaled_hitting_times]

        self.group_color = [colors.to_hex(v) for v in locallytrimmed_g.vs['color']]  # based on ygb scale
        viridis_cmap = cm.get_cmap('viridis_r')

        self.group_color_cmap = [colors.to_hex(v) for v in
                                 viridis_cmap(scaled_hitting_times / 1000)]  # based on ygb scale

        self.graph_node_label = df_graph['graph_node_label'].values
        self.edgeweight = [e['weight'] * 1 for e in locallytrimmed_g.es]

        self.graph_node_pos = layout.coords


        self.draw_piechart_graph()
        self.labels = list(self.labels)


        import statistics
        from statistics import mode
        for tsi in self.terminal_clusters:
            loc_i = np.where(np.asarray(self.labels) == tsi)[0]
            val_pt = [self.single_cell_pt_markov[i] for i in loc_i]
            if self.dataset == '2M':
                major_traj = [self.df_annot.loc[i, ['Main_trajectory']].values[0] for i in loc_i]
                major_cell_type = [self.df_annot.loc[i, ['Main_cell_type']].values[0] for i in loc_i]
                print(tsi, 'has major traj and cell type', mode(major_traj), mode(major_cell_type))
            th_pt = np.percentile(val_pt, 50)  # 50
            loc_i = [loc_i[i] for i in range(len(val_pt)) if val_pt[i] >= th_pt]
            temp = np.mean(self.data[loc_i], axis=0)
            labelsq, distances = self.knn_struct.knn_query(temp, k=1)

            tsi_list.append(labelsq[0][0])

        if self.embedding != None:
            plt.scatter(self.embedding[:, 0], self.embedding[:, 1], c=self.single_cell_pt_markov,
                        cmap='viridis_r', s=3, alpha=0.5)
            plt.scatter(self.embedding[self.root_user[0], 0], self.embedding[self.root_user[0], 1], c='orange', s=20)
            plt.title('root:' + str(self.root_user[0]) + 'knn' + str(self.knn) + 'Ncomp' + str(self.ncomp))
            for i in tsi_list:
                # print(i, ' has traj and cell type', self.df_annot.loc[i, ['Main_trajectory', 'Main_cell_type']])
                plt.text(self.embedding[i, 0], self.embedding[i, 1], str(i))
                plt.scatter(self.embedding[i, 0], self.embedding[i, 1], c='red', s=10)
            plt.show()
        return

    def draw_piechart_graph(self, type_data='pt', gene_exp='', title='', cmap=None, ax_text=True,dpi=150):
        # type_data can be 'pt' pseudotime or 'gene' for gene expression

        import matplotlib.lines as lines
        from mpl_toolkits.axes_grid1 import make_axes_locatable
        f, ((ax, ax1)) = plt.subplots(1, 2, sharey=True, dpi=dpi)
        arrow_head_w = self.piegraph_arrow_head_width  # 0.4
        edgeweight_scale = self.piegraph_edgeweight_scalingfactor  # 1.5

        node_pos = self.graph_node_pos
        edgelist = list(self.edgelist_maxout)
        edgeweight = self.edgeweights_maxout

        node_pos = np.asarray(node_pos)
        if cmap is None: cmap = 'coolwarm' if type_data == 'gene' else 'viridis_r'
        graph_node_label = self.graph_node_label
        if type_data == 'pt':
            pt = self.scaled_hitting_times  # these are the final MCMC refined pt then slightly scaled
            title_ax1 = "Pseudotime"

        if type_data == 'gene':
            pt = gene_exp
            title_ax1 = title

        n_groups = len(set(self.labels))
        n_truegroups = len(set(self.true_label))
        group_pop = np.zeros([n_groups, 1])
        group_frac = pd.DataFrame(np.zeros([n_groups, n_truegroups]), columns=list(set(self.true_label)))
        self.cluster_population_dict = {}
        for group_i in set(self.labels):
            loc_i = np.where(self.labels == group_i)[0]

            group_pop[group_i] = len(loc_i)  # np.sum(loc_i) / 1000 + 1
            self.cluster_population_dict[group_i] = len(loc_i)
            true_label_in_group_i = list(np.asarray(self.true_label)[loc_i])
            for ii in set(true_label_in_group_i):
                group_frac[ii][group_i] = true_label_in_group_i.count(ii)

        line_true = np.linspace(0, 1, n_truegroups)
        color_true_list = [plt.cm.rainbow(color) for color in line_true]

        sct = ax.scatter(node_pos[:, 0], node_pos[:, 1],
                         c='white', edgecolors='face', s=group_pop, cmap='jet')

        bboxes = getbb(sct, ax)
        for e_i, (start, end) in enumerate(edgelist):
            if pt[start] > pt[end]:
                start, end = end, start

            ax.add_line(lines.Line2D([node_pos[start, 0], node_pos[end, 0]], [node_pos[start, 1], node_pos[end, 1]],
                                     color='black', lw=edgeweight[e_i] * edgeweight_scale, alpha=0.6))
            z = np.polyfit([node_pos[start, 0], node_pos[end, 0]], [node_pos[start, 1], node_pos[end, 1]], 1)
            minx = np.min(np.array([node_pos[start, 0], node_pos[end, 0]]))

            direction = 1 if node_pos[start, 0] < node_pos[end, 0] else -1
            maxx = np.max(np.array([node_pos[start, 0], node_pos[end, 0]]))

            xp = np.linspace(minx, maxx, 500)
            p = np.poly1d(z)
            smooth = p(xp)
            step = 1

            ax.arrow(xp[250], smooth[250], xp[250 + direction * step] - xp[250],
                     smooth[250+ direction * step] - smooth[250], shape='full',
                     lw=0, length_includes_head=True, head_width=arrow_head_w,color='grey')

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
        # print('pie_size_ar', pie_size_ar)

        for node_i in range(n_groups):

            cluster_i_loc = np.where(np.asarray(self.labels) == node_i)[0]
            majority_true = self.func_mode(list(np.asarray(self.true_label)[cluster_i_loc]))
            pie_size = pie_size_ar[node_i][0]

            x1, y1 = trans(node_pos[node_i])  # data coordinates
            xa, ya = trans2((x1, y1))  # axis coordinates

            xa = ax_x_min + (xa - pie_size / 2) * ax_len_x
            ya = ax_y_min + (ya - pie_size / 2) * ax_len_y
            # clip, the fruchterman layout sometimes places below figure
            if ya < 0: ya = 0
            if xa < 0: xa = 0
            rect = [xa, ya, pie_size * ax_len_x, pie_size * ax_len_y]
            frac = np.asarray([ff for ff in group_frac.iloc[node_i].values])

            pie_axs.append(plt.axes(rect, frameon=False))
            pie_axs[node_i].pie(frac, wedgeprops={'linewidth': 0.0}, colors=color_true_list)
            pie_axs[node_i].set_xticks([])
            pie_axs[node_i].set_yticks([])
            pie_axs[node_i].set_aspect('equal')
            #pie_axs[node_i].text(0.5, 0.5, graph_node_label[node_i])
            pie_axs[node_i].text(0.5, 0.5, majority_true)

        patches, texts = pie_axs[node_i].pie(frac, wedgeprops={'linewidth': 0.0}, colors=color_true_list)
        labels = list(set(self.true_label))
        plt.legend(patches, labels, loc=(-5, -5), fontsize=6, frameon = False)

        ti = 'Cluster Composition. K=' + str(self.knn) + '. ncomp = ' + str(self.ncomp)  # "+ is_sub
        ax.set_title(ti)
        ax.grid(False)
        ax.set_xticks([])
        ax.set_yticks([])

        title_list = [title_ax1]  # , "PT on undirected original graph"]
        for i, ax_i in enumerate([ax1]):
            pt = self.markov_hitting_times if type_data == 'pt' else gene_exp

            for e_i, (start, end) in enumerate(edgelist):
                if pt[start] > pt[end]:
                    start, end = end, start

                ax_i.add_line(
                    lines.Line2D([node_pos[start, 0], node_pos[end, 0]], [node_pos[start, 1], node_pos[end, 1]],
                                 color='black', lw=edgeweight[e_i] * edgeweight_scale, alpha=0.6))
                z = np.polyfit([node_pos[start, 0], node_pos[end, 0]], [node_pos[start, 1], node_pos[end, 1]], 1)
                minx = np.min(np.array([node_pos[start, 0], node_pos[end, 0]]))

                direction_arrow = 1 if node_pos[start, 0] < node_pos[end, 0] else -1
                maxx = np.max(np.array([node_pos[start, 0], node_pos[end, 0]]))

                xp = np.linspace(minx, maxx, 500)
                p = np.poly1d(z)
                smooth = p(xp)
                step = 1
                ax_i.arrow(xp[250], smooth[250], xp[250 + direction_arrow*step] - xp[250], smooth[250 + direction_arrow*step] - smooth[250],shape='full', lw=0,length_includes_head=True, head_width=arrow_head_w,color='grey')

            c_edge, l_width = [], []
            for ei, pti in enumerate(pt):
                if ei in self.terminal_clusters:
                    c_edge.append('red')
                    l_width.append(1.5)
                else:
                    c_edge.append('gray')
                    l_width.append(0.0)

            gp_scaling = 1000 / max(group_pop)  # 500 / max(group_pop)
            # print(gp_scaling, 'gp_scaling')
            group_pop_scale = group_pop * gp_scaling * 0.5

            im1=ax_i.scatter(node_pos[:, 0], node_pos[:, 1], s=group_pop_scale, c=pt, cmap=cmap, edgecolors=c_edge,
                         alpha=1, zorder=3, linewidth=l_width)
            if ax_text:
                x_max_range = np.amax(node_pos[:, 0])/100
                y_max_range = np.amax(node_pos[:, 1])/100

                for ii in range(node_pos.shape[0]):
                    ax_i.text(node_pos[ii, 0] + max(x_max_range,y_max_range), node_pos[ii, 1] + min(x_max_range,y_max_range), 'C' + str(ii) + 'pop' + str(int(group_pop[ii][0])),
                          color='black', zorder=4)

            title_pt = title_list[i]
            ax_i.set_title(title_pt)
            ax_i.grid(False)
            ax_i.set_xticks([])
            ax_i.set_yticks([])

        divider = make_axes_locatable(ax1)
        cax = divider.append_axes('right', size='5%', pad=0.05)
        if type_data=='pt':  f.colorbar(im1, cax=cax, orientation='vertical', label = 'pseudotime')
        else:  f.colorbar(im1, cax=cax, orientation='vertical', label = 'Gene expression')
        f.patch.set_visible(False)

        ax1.axis('off')
        ax.axis('off')
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
            # if majority_val == onevsall: print('cluster', kk, ' has majority', onevsall, 'with population', len(vals))
            if kk == -1:
                len_unknown = len(vals)
                # print('len unknown', len_unknown)
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

    def run_VIA(self):
        print(f'{datetime.now()}\tRunning VIA over input data of {self.data.shape[0]} (samples) x {self.data.shape[1]} (features)')

        self.knn_struct = self.construct_knn(self.data)
        st = time.time()
        self.run_subPARC()
        run_time = time.time() - st
        print('time elapsed {:.1f} seconds'.format(run_time))

        targets = set(self.true_label)
        N = len(self.true_label)
        self.f1_accumulated = 0
        self.f1_mean = 0
        self.stats_df = pd.DataFrame({'jac_std_global': [self.jac_std_global], 'dist_std_local': [self.dist_std_local],
                                      'runtime(s)': [run_time]})
        # self.majority_truth_labels = []
        list_roc = []
        if len(targets) > 1:
            f1_accumulated, f1_acc_noweighting = 0, 0
            for onevsall_val in targets:
                # print('target is', onevsall_val)
                vals_roc, predict_class_array, majority_truth_labels, numclusters_targetval = \
                    self.accuracy(onevsall=onevsall_val)
                f1_current = vals_roc[1]
                f1_accumulated = f1_accumulated + f1_current * (list(self.true_label).count(onevsall_val)) / N
                f1_acc_noweighting = f1_acc_noweighting + f1_current

                list_roc.append([self.jac_std_global, self.dist_std_local, onevsall_val] +
                                vals_roc + [numclusters_targetval] + [run_time])

            f1_mean = f1_acc_noweighting / len(targets)

            df_accuracy = pd.DataFrame(list_roc,
                                       columns=['jac_std_global', 'dist_std_local', 'onevsall-target', 'error rate',
                                                'f1-score', 'tnr', 'fnr',
                                                'tpr', 'fpr', 'precision', 'recall', 'num_groups',
                                                'population of target', 'num clusters', 'clustering runtime'])

            self.f1_accumulated = f1_accumulated
            self.f1_mean = f1_mean
            self.stats_df = df_accuracy
            # self.majority_truth_labels = majority_truth_labels


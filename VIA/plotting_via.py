#based on : https://github.com/theislab/scvelo/blob/1805ab4a72d3f34496f0ef246500a159f619d3a2/scvelo/plotting/velocity_embedding_grid.py#L27
import numpy as np
from sklearn.neighbors import NearestNeighbors
from datetime import datetime
from scipy.stats import norm as normal
from typing import Union
from scipy.spatial import distance
from scipy.sparse import csr_matrix, csgraph, find
import math
import pandas as pd
import numpy as np
from numpy import ndarray
from scipy.sparse import issparse, spmatrix
import hnswlib
import time
import matplotlib
import igraph as ig
import matplotlib.pyplot as plt
from pyVIA.utils_via import *

def draw_sc_lineage_probability(via_coarse, via_fine, embedding, idx=None, cmap_name='plasma', dpi=150, scatter_size =None):
    '''
    embedding is the full or downsampled 2D representation of the full dataset.
    idx is the list of indices of the full dataset for which the embedding is available. if the dataset is very large the the user only has the visual embedding for a subsample of the data, then these idx can be carried forward
    idx is the selected indices of the downsampled samples used in the visualization
    G is the igraph knn (low K) used for shortest path in high dim space. no idx needed as it's made on full sample
    knn_hnsw is the knn made in the embedded space used for query to find the nearest point in the downsampled embedding
    that corresponds to the single cells in the full graph

    :param via_coarse:
    :param via_fine:
    :param embedding:
    :param idx:
    :param cmap_name:
    :param dpi:
    :param scatter_size:
    :return:
    '''

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
            if len(path_orange)>0: print(f'{datetime.now()}\tCluster path on clustergraph starting from Root Cluster {via_fine.root[ii]}to Terminal Cluster {fst_i}')

    # single-cell branch probability evolution probability
    n_terminal_clusters = len(via_fine.terminal_clusters)
    fig_nrows, mod = divmod(n_terminal_clusters, 4)
    if mod ==0: fig_nrows=fig_nrows
    if mod != 0:        fig_nrows+=1

    fig_ncols = 4
    fig, axs = plt.subplots(fig_nrows,fig_ncols,dpi=dpi)
    ti = 0 # counter for terminal cluster
    #for i, ti in enumerate(via_fine.terminal clusters):
    for r in range(fig_nrows):
        for c in range(fig_ncols):
            if ti < n_terminal_clusters:
                if fig_nrows ==1: plot_sc_pb(axs[c], fig, embedding, p1_sc_bp[:, ti], ti= via_fine.terminal_clusters[ti], cmap_name=cmap_name, scatter_size=scatter_size)
                else: plot_sc_pb(axs[r,c], fig, embedding, p1_sc_bp[:, ti], ti= via_fine.terminal_clusters[ti], cmap_name=cmap_name, scatter_size=scatter_size)
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
            if fig_nrows==1: axs[c].axis('off')
            else: axs[r,c].axis('off')


def draw_clustergraph(via_coarse, type_data='gene', gene_exp='', gene_list='', arrow_head=0.1,
                      edgeweight_scale=1.5, cmap=None, label_=True):
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
        '''
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
        '''
        c_edge, l_width = [], []
        for ei, pti in enumerate(pt):
            if ei in via_coarse.terminal_clusters:
                c_edge.append('red')
                l_width.append(1.5)
            else:
                c_edge.append('gray')
                l_width.append(0.0)
        ax_i = plot_edge_bundle(ax_i, via_coarse.hammer_bundle, layout=via_coarse.graph_node_pos, CSM=via_coarse.CSM,
                                velocity_weight=via_coarse.velo_weight, pt=pt, headwidth_bundle=arrow_head, alpha_bundle=0.4, linewidth_bundle=edgeweight_scale)
        group_pop_scale = .5 * group_pop * 1000 / max(group_pop)
        pos = ax_i.scatter(node_pos[:, 0], node_pos[:, 1], s=group_pop_scale, c=gene_exp[gene_i].values, cmap=cmap,
                           edgecolors=c_edge, alpha=1, zorder=3, linewidth=l_width)
        if label_==True:
            for ii in range(node_pos.shape[0]):
                ax_i.text(node_pos[ii, 0] + 0.1, node_pos[ii, 1] + 0.1, 'C'+str(ii)+' '+str(round(gene_exp[gene_i].values[ii], 1)),
                          color='black', zorder=4, fontsize=6)
        divider = make_axes_locatable(ax_i)
        cax = divider.append_axes('right', size='10%', pad=0.05)

        cbar=fig.colorbar(pos, cax=cax, orientation='vertical')
        cbar.ax.tick_params(labelsize=8)
        ax_i.set_title(gene_i)
        ax_i.grid(False)
        ax_i.set_xticks([])
        ax_i.set_yticks([])
        ax_i.axis('off')
    fig.patch.set_visible(False)

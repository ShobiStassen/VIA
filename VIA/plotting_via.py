#based on : https://github.com/theislab/scvelo/blob/1805ab4a72d3f34496f0ef246500a159f619d3a2/scvelo/plotting/velocity_embedding_grid.py#L27
import numpy as np
from sklearn.neighbors import NearestNeighbors
from datetime import datetime
from scipy.stats import norm as normal

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
import matplotlib.cm as cm
import pygam as pg
from typing import Optional, Union
from pyVIA.utils_via import *


def _make_knn_embeddedspace(embedding):
    # knn struct built in the embedded space to be used for drawing the lineage trajectories onto the 2D plot
    knn = hnswlib.Index(space='l2', dim=embedding.shape[1])
    knn.init_index(max_elements=embedding.shape[0], ef_construction=200, M=16)
    knn.add_items(embedding)
    knn.set_ef(50)
    return knn
def run_umap_hnsw(  X_input:ndarray , graph:csr_matrix, n_components=2, alpha: float = 1.0, negative_sample_rate: int = 5,
                  gamma: float = 1.0, spread:float=1.0, min_dist:float=0.1, init_pos:Union[str, ndarray]='spectral', random_state:int=0,
                    n_epochs:int=0, distance_metric: str = 'euclidean', layout:Optional[list]=None, cluster_membership:list=[])-> ndarray:
    '''

    :param X_input:
    :param graph:
    :param n_components:
    :param alpha:
    :param negative_sample_rate:
    :param gamma: Weight to apply to negative samples.
    :param spread: The effective scale of embedded points. In combination with min_dist this determines how clustered/clumped the embedded points are.
    :param min_dist: The effective minimum distance between embedded points. Smaller values will result in a more clustered/clumped embedding where nearby points on the manifold are drawn closer together, while larger values will result on a more even dispersal of points
    :param init_pos: either a string (default) 'spectral', 'via' (uses via graph to initialize). Or a n_cellx2 dimensional ndarray with initial coordinates
    :param random_state:
    :param n_epochs: The number of training epochs to be used in optimizing the low dimensional embedding. Larger values result in more accurate embeddings. If 0 is specified a value will be selected based on the size of the input dataset (200 for large datasets, 500 for small).
    :param distance_metric:
    :param layout: ndarray This is required if the init_pos is set to 'via'. layout should then = via0.graph_node_pos (which is a list of lists)
    :param cluster_membership:
    :return:
    '''
    ''''
    :param init_pos:
    :param
    :param layout via0.graph_node_pos
    :param cluster_membership = v0.labels
    
    
    '''
    #X_input = via0.data
    n_cells = X_input.shape[0]
    #graph = graph+graph.T
    #graph = via0.csr_full_graph
    print(f"{datetime.now()}\tComputing umap on sc-Viagraph s")

    from umap.umap_ import find_ab_params, simplicial_set_embedding
    #graph is a csr matrix
    #weight all edges as 1 in order to prevent umap from pruning weaker edges away
    layout_array = np.zeros(shape=(n_cells,2))

    if init_pos=='via':
        #list of lists [[x,y], [x1,y1], []]
        for i in range(n_cells):
            layout_array[i,0]=layout[cluster_membership[i]][0]
            layout_array[i, 1] = layout[cluster_membership[i]][1]
        init_pos = layout_array
        print('using via cluster graph to initialize embedding')

    a, b = find_ab_params(spread, min_dist)
    #print('a,b, spread, dist', a, b, spread, min_dist)
    t0 = time.time()
    m = graph.data.max()
    graph.data = np.clip(graph.data, np.percentile(graph.data, 10),
                                 np.percentile(graph.data, 90))
    #graph.data = 1 + graph.data/m
    #graph.data.fill(1)
    #print('average graph.data', round(np.mean(graph.data),4), round(np.max(graph.data),2))
    #graph.data = graph.data + np.mean(graph.data)

    #transpose =graph.transpose()

    # prod_matrix = result.multiply(transpose)
    #graph = graph + transpose  # - prod_matrix
    X_umap = simplicial_set_embedding(data=X_input, graph=graph, n_components=n_components, initial_alpha=alpha,
                                      a=a, b=b, n_epochs=n_epochs, metric_kwds={}, gamma=gamma, metric=distance_metric,
                                      negative_sample_rate=negative_sample_rate, init=init_pos,
                                      random_state=np.random.RandomState(random_state),
                                      verbose=1)
    return X_umap
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
    knn_hnsw = _make_knn_embeddedspace(embedding)
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


def animated_streamplot(via_coarse, embedding , density_grid=1,
 linewidth=0.5,min_mass = 1, cutoff_perc = None,scatter_size=500, scatter_alpha=0.2,marker_edgewidth=0.1, smooth_transition=1, smooth_grid=0.5, color_scheme = 'annotation', other_labels=[] , b_bias=20, n_neighbors_velocity_grid=None, fontsize=8, alpha_animate=0.7,
                        cmap_scatter = 'rainbow', cmap_stream='Blues_r', segment_length=1, saveto='/home/shobi/Trajectory/Datasets/animation.gif',use_sequentially_augmented=False):
    '''
    Draw Animated vector plots

    :param via_coarse:
    :param embedding:
    :param density_grid:
    :param linewidth:
    :param min_mass:
    :param cutoff_perc:
    :param scatter_size:
    :param scatter_alpha:
    :param marker_edgewidth:
    :param smooth_transition:
    :param smooth_grid:
    :param color_scheme: 'annotation', 'cluster', 'other'
    :param add_outline_clusters:
    :param cluster_outline_edgewidth:
    :param gp_color:
    :param bg_color:
    :param title:
    :param b_bias:
    :param n_neighbors_velocity_grid:
    :param fontsize:
    :param alpha_animate:
    :param cmap_scatter:
    :param cmap_stream: string of a cmap, default 'Blues_r' 'Greys_r'
    :param color_stream: string like 'white'. will override cmap_stream
    :param segment_length:

    :return:
    '''
    import tqdm

    import matplotlib.pyplot as plt
    from matplotlib.animation import FuncAnimation, writers
    from matplotlib.collections import LineCollection
    from pyVIA.windmap import Streamlines
    import matplotlib.patheffects as PathEffects

    #import cartopy.crs as ccrs

    V_emb = via_coarse._velocity_embedding(embedding, smooth_transition, b=b_bias, use_sequentially_augmented=use_sequentially_augmented)

    V_emb *= 100 #the velocity of the samples has shape (n_samples x 2).

    #interpolate the velocity along all grid points based on the velocities of the samples in V_emb
    X_grid, V_grid = compute_velocity_on_grid(
        X_emb=embedding,
        V_emb=V_emb,
        density=density_grid,
        smooth=smooth_grid,
        min_mass=min_mass,
        autoscale=False,
        adjust_for_stream=True,
        cutoff_perc=cutoff_perc, n_neighbors=n_neighbors_velocity_grid)
    print(f'{datetime.now()}\tInside animated. File will be saved to location {saveto}')

    '''
    #inspecting V_grid. most values are close to zero except those along data points of cells  
    print('U', V_grid.shape, V_grid[0].shape)
    print('sum is nan of each column', np.sum(np.isnan(V_grid[0]),axis=0))
    msk = np.isnan(V_grid[0])
    V_grid[0][msk]=0
    print('sum is nan of each column', np.sum(np.isnan(V_grid[0]), axis=0))
    print('max of each U column', np.max(V_grid[0], axis=0))
    print('max V', np.max(V_grid[1], axis=0))
    print('V')
    print( V_grid[1])
    '''
    #lengths = np.sqrt((V_grid ** 2).sum(0))

    fig = plt.figure(figsize=(10, 6))
    ax = plt.subplot(1, 1, 1)

    if color_scheme == 'time': ax.scatter(embedding[:, 0], embedding[:, 1], c=via_coarse.single_cell_pt_markov, alpha=scatter_alpha, zorder=0,
               s=scatter_size, linewidths=marker_edgewidth, cmap='viridis_r')
    else:
        if color_scheme == 'annotation': color_labels = via_coarse.true_label
        if color_scheme == 'cluster': color_labels = via_coarse.labels
        if color_scheme == 'other': color_labels = other_labels

        n_true = len(set(color_labels))
        lin_col = np.linspace(0, 1, n_true)
        col = 0
        cmap = matplotlib.cm.get_cmap(cmap_scatter) #'twilight' is nice too
        cmap = cmap(np.linspace(0.01, 0.95, n_true))
        for color, group in zip(lin_col, sorted(set(color_labels))):
            color_ = np.asarray(cmap[col]).reshape(-1, 4)

            color_[0,3]=scatter_alpha

            where = np.where(np.array(color_labels) == group)[0]

            ax.scatter(embedding[where, 0], embedding[where, 1], label=group,
                       c=color_,
                       alpha=scatter_alpha, zorder=0, s=scatter_size, linewidths=marker_edgewidth) #plt.cm.rainbow(color))

            x_mean = embedding[where, 0].mean()
            y_mean = embedding[where, 1].mean()
            ax.text(x_mean, y_mean, str(group), fontsize=fontsize, zorder=4,
                    path_effects=[PathEffects.withStroke(linewidth=linewidth, foreground='w')], weight='bold')
            col += 1

    lengths = []
    colors = []
    lines = []
    linewidths = []
    count  =0
    #X, Y, U, V = interpolate_static_stream(X_grid[0], X_grid[1], V_grid[0],V_grid[1])
    s = Streamlines(X_grid[0], X_grid[1], V_grid[0], V_grid[1])
    #s = Streamlines(X,Y,U,V)
    for streamline in s.streamlines:

        count +=1
        x, y = streamline
        #interpolate x, y data to handle nans
        x_ = np.array(x)
        nans, func_ = nan_helper(x_)
        x_[nans] = np.interp(func_(nans), func_(~nans), x_[~nans])


        y_ = np.array(y)
        nans, func_ = nan_helper(y_)
        y_[nans] = np.interp(func_(nans), func_(~nans), y_[~nans])


        # test=proj.transform_points(x=np.array(x),y=np.array(y),src_crs=proj)
        #points = np.array([x, y]).T.reshape(-1, 1, 2)
        points = np.array([x_, y_]).T.reshape(-1, 1, 2)
        #print('points')
        #print(points.shape)


        segments = np.concatenate([points[:-1], points[1:]], axis=1) #nx2x2
        #print('segments', segments.shape,len(segments))
        #print(segments[1,0,:])
        #print (segments[:,::2,:].shape)#segment 1, starting point x,y [11.13  17.181]
        #print('squeezed seg shape', np.squeeze(segments[:,::2,:]).shape)
        #C_locationbased = map_velocity_to_color(X_grid[0], X_grid[1], V_grid[0], V_grid[1], segments)
        n = len(segments)

        D = np.sqrt(((points[1:] - points[:-1]) ** 2).sum(axis=-1))/segment_length
        L = D.cumsum().reshape(n, 1) + np.random.uniform(0, 1)

        C = np.zeros((n, 4)) #3
        C[::-1] = (L * 1.5) % 1
        C[:,3]= alpha_animate
        #print('C shape', C.shape)
        #print('shape L', L.shape)
        #print('L')
        #print(L)
        #print('C', C)
        lw = L.flatten().tolist()
        #print('L list', L.flatten().tolist())
        line = LineCollection(segments, color=C, linewidth=1)# 0.1 when changing linewidths in update
        #line = LineCollection(segments, color=C_locationbased, linewidth=1)
        lengths.append(L)
        colors.append(C)
        linewidths.append(lw)
        lines.append(line)

        ax.add_collection(line)
    print('total number of stream lines', count)

    ax.set_xlim(min(X_grid[0]), max(X_grid[0]), ax.set_xticks([]))
    ax.set_ylim(min(X_grid[1]), max(X_grid[1]), ax.set_yticks([]))
    plt.tight_layout()
    #print('colors', colors)
    def update(frame_no):
        cmap = matplotlib.cm.get_cmap(cmap_stream)
        #cmap = cmap(np.linspace(0.8, 0.9, 100))
        for i in range(len(lines)):
            lengths[i] += 0.05
            # esthetic factors here by adding 0.1 and doing some clipping, 0.1 ensures no "hard blacks"
            colors[i][::-1] =  np.clip(0.1+(lengths[i] * 1.5) % 1,0.2,0.9)
            colors[i][:, 3] = alpha_animate
            temp = (lengths[i] * 1.5) % 1
            linewidths[i] = temp.flatten().tolist()
            #if i==5: print('temp', linewidths[i])
            '''
            if i%5 ==0:
                print('lengths',i, lengths[i])
                colors[i][::-1] = (lengths[i] * 1.5) % 1
                colors[i][:, 0] = 1
            '''

            #CMAP COLORS
            cmap_colors = [cmap(j) for j in colors[i][:,0]]
            '''
            cmap_colors = [cmap[int(j*100)] for j in colors[i][:, 0]]
            linewidths[i] = [f[0]*2 for f in cmap_colors]
            if i ==5: print('colors', [f[0]for f in cmap_colors])
            '''
            for row in range(colors[i].shape[0]):
                colors[i][row,:] = cmap_colors[row][0:4]
                #colors[i][row, 3] = (1-colors[i][row][0])*0.6#alpha_animate
                linewidths[i][row] = (1-colors[i][row][0])
                #if color_stream is not None: colors[i][row, :] =  matplotlib.colors.to_rgba_array(color_stream)[0] #monochrome is nice 1 or 0
            #if i == 5: print('lw', linewidths[i])
            colors[i][:, 3] = alpha_animate
            lines[i].set_linewidth(linewidths[i])
            lines[i].set_color(colors[i])
        pbar.update()

    n = 27

    animation = FuncAnimation(fig, update, frames=n, interval=20)
    pbar = tqdm.tqdm(total=n)
    pbar.close()
    animation.save(saveto, writer='imagemagick', fps=30)
    #animation.save('/home/shobi/Trajectory/Datasets/Toy3/wind_density_ffmpeg.mp4', writer='ffmpeg', fps=60)

    #fig.patch.set_visible(False)
    #ax.axis('off')
    plt.show()
    return


def via_streamplot(via_coarse, embedding , density_grid=0.5, arrow_size=.7, arrow_color = 'k',
arrow_style="-|>",  max_length=4, linewidth=1,min_mass = 1, cutoff_perc = 5,scatter_size=500, scatter_alpha=0.5,marker_edgewidth=0.1, density_stream = 2, smooth_transition=1, smooth_grid=0.5, color_scheme = 'annotation', add_outline_clusters=False, cluster_outline_edgewidth = 0.001,gp_color = 'white', bg_color='black' , dpi=300 , title='Streamplot', b_bias=20, n_neighbors_velocity_grid=None, other_labels=[],use_sequentially_augmented=False):

    """
   Construct vector streamplot on the embedding to show a fine-grained view of inferred directions in the trajectory

   Parameters
   ----------
   X_emb: np.ndarray of shape (n_samples, 2)
       umap or other 2-d embedding on which to project the directionality of cells

   scatter_size: int, default = 500

   linewidth: width of  lines in streamplot, defaolt = 1

   marker_edgewidth: width of outline arround each scatter point, default = 0.1

   color_scheme: str, default = 'annotation' corresponds to self.true_labels. Other options are 'time' (uses single-cell pseudotime) and 'cluster' (via cluster graph) and 'other'

   other_labels: list (will be used for the color scheme)

   b_bias: default = 20. higher value makes the forward bias of pseudotime stronger

   Returns
   -------
   streamplot matplotlib.pyplot instance of fine-grained trajectories drawn on top of scatter plot
   """

    import matplotlib.patheffects as PathEffects


    V_emb = via_coarse._velocity_embedding(embedding, smooth_transition,b=b_bias, use_sequentially_augmented=use_sequentially_augmented)

    V_emb *=20 #5


    X_grid, V_grid = compute_velocity_on_grid(
        X_emb=embedding,
        V_emb=V_emb,
        density=density_grid,
        smooth=smooth_grid,
        min_mass=min_mass,
        autoscale=False,
        adjust_for_stream=True,
        cutoff_perc=cutoff_perc, n_neighbors=n_neighbors_velocity_grid )

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
        #adapted from scanpy (scVelo utils adapts this from scanpy)
        gp_size = (2 * (scatter_size * cluster_outline_edgewidth *.1) + 0.1*scatter_size) ** 2

        bg_size = (2 * (scatter_size * cluster_outline_edgewidth)+ math.sqrt(gp_size)) ** 2

        ax.scatter(embedding[:, 0],embedding[:, 1], s=bg_size, marker=".", c=bg_color, zorder=-2)
        ax.scatter(embedding[:, 0], embedding[:, 1], s=gp_size, marker=".", c=gp_color, zorder=-1)

    if color_scheme == 'time':
        ax.scatter(embedding[:,0],embedding[:,1], c=via_coarse.single_cell_pt_markov,alpha=scatter_alpha,  zorder = 0, s=scatter_size, linewidths=marker_edgewidth, cmap = 'viridis_r')
    else:
        if color_scheme == 'annotation':color_labels = via_coarse.true_label
        if color_scheme == 'cluster': color_labels= via_coarse.labels
        if color_scheme == 'other':        color_labels = other_labels

        line = np.linspace(0, 1, len(set(color_labels)))
        for color, group in zip(line, sorted(set(color_labels))):
            where = np.where(np.array(color_labels) == group)[0]
            ax.scatter(embedding[where, 0], embedding[where, 1], label=group,
                        c=np.asarray(plt.cm.rainbow(color)).reshape(-1, 4),
                        alpha=scatter_alpha,  zorder = 0, s=scatter_size, linewidths=marker_edgewidth)

            x_mean = embedding[where, 0].mean()
            y_mean = embedding[where, 1].mean()
            ax.text(x_mean, y_mean, str(group), fontsize=5, zorder=4, path_effects = [PathEffects.withStroke(linewidth=1, foreground='w')], weight = 'bold')

    fig.patch.set_visible(False)
    ax.axis('off')
    ax.set_title(title)

def get_gene_expression(via0, gene_exp:pd.DataFrame, cmap:str='jet', dpi:int=150, marker_genes:list = [], linewidth:float = 2.0,n_splines:int=10, spline_order:int=4, fontsize_:int=8):
    '''

    :param gene_exp: dataframe where columns are features (gene)
    :param cmap: default: 'jet'
    :param dpi: default:150
    :param marker_genes: Default is to use all genes in gene_exp. other provide a list of marker genes that will be used from gene_exp.
    :param linewidth: default:2
    :param n_slines: default:10
    :param spline_order: default:4
    :return:  None
    '''

    if len(marker_genes) >0: gene_exp=gene_exp[marker_genes]
    sc_pt = via0.single_cell_pt_markov
    sc_bp_original = via0.single_cell_bp
    n_terminal_states = sc_bp_original.shape[1]

    palette = cm.get_cmap(cmap, n_terminal_states)
    cmap_ = palette(range(n_terminal_states))
    n_genes = gene_exp.shape[1]

    fig_nrows, mod = divmod(n_genes, 4)
    if mod == 0: fig_nrows = fig_nrows
    if mod != 0: fig_nrows += 1

    fig_ncols = 4
    fig, axs = plt.subplots(fig_nrows, fig_ncols, dpi=dpi)

    i_gene = 0  # counter for number of genes
    i_terminal = 0 #counter for terminal cluster
    # for i in range(n_terminal_states): #[0]

    for r in range(fig_nrows):
        for c in range(fig_ncols):
            if (i_gene < n_genes):
                for i_terminal in range(n_terminal_states):
                    sc_bp = sc_bp_original.copy()
                    if (len(np.where(sc_bp[:, i_terminal] > 0.9)[ 0]) > 0): # check in case this terminal state i cannot be reached (sc_bp is all 0)
                        gene_i = gene_exp.columns[i_gene]
                        loc_i = np.where(sc_bp[:, i_terminal] > 0.9)[0]
                        val_pt = [sc_pt[pt_i] for pt_i in loc_i]  # TODO,  replace with array to speed up

                        max_val_pt = max(val_pt)

                        loc_i_bp = np.where(sc_bp[:, i_terminal] > 0.000)[0]  # 0.001
                        loc_i_sc = np.where(np.asarray(sc_pt) <= max_val_pt)[0]

                        loc_ = np.intersect1d(loc_i_bp, loc_i_sc)

                        gam_in = np.asarray(sc_pt)[loc_]
                        x = gam_in.reshape(-1, 1)

                        y = np.asarray(gene_exp[gene_i])[loc_].reshape(-1, 1)

                        weights = np.asarray(sc_bp[:, i_terminal])[loc_].reshape(-1, 1)

                        if len(loc_) > 1:
                            geneGAM = pg.LinearGAM(n_splines=n_splines, spline_order=spline_order, lam=10).fit(x, y, weights=weights)
                            xval = np.linspace(min(sc_pt), max_val_pt, 100 * 2)
                            yg = geneGAM.predict(X=xval)

                        else:
                            print('loc_ has length zero')

                        if fig_nrows >1:
                            axs[r,c].plot(xval, yg, color=cmap_[i_terminal], linewidth=linewidth, zorder=3, label=f"Lineage:{via0.terminal_clusters[i_terminal]}")
                            axs[r, c].set_title(gene_i,fontsize=fontsize_)
                            # Set tick font size
                            for label in (axs[r,c].get_xticklabels() + axs[r,c].get_yticklabels()):
                                label.set_fontsize(fontsize_-1)
                            if i_gene == n_genes -1:
                                axs[r,c].legend(frameon=False, fontsize=fontsize_)
                                axs[r, c].set_xlabel('Time', fontsize=fontsize_)
                                axs[r, c].set_ylabel('Intensity', fontsize=fontsize_)
                            axs[r,c].spines['top'].set_visible(False)
                            axs[r,c].spines['right'].set_visible(False)
                        else:
                            axs[c].plot(xval, yg, color=cmap_[i_terminal], linewidth=linewidth, zorder=3,   label=f"Lineage:{via0.terminal_clusters[i_terminal]}")
                            axs[c].set_title(gene_i, fontsize=fontsize_)
                            # Set tick font size
                            for label in (axs[c].get_xticklabels() + axs[c].get_yticklabels()):
                                label.set_fontsize(fontsize_-1)
                            if i_gene == n_genes -1:
                                axs[c].legend(frameon=False,fontsize=fontsize_)
                                axs[ c].set_xlabel('Time', fontsize=fontsize_)
                                axs[ c].set_ylabel('Intensity', fontsize=fontsize_)
                            axs[c].spines['top'].set_visible(False)
                            axs[c].spines['right'].set_visible(False)
                i_gene+=1
            else:
                if fig_nrows > 1: axs[r,c].axis('off')
                else: axs[c].axis('off')
    return

def draw_trajectory_gams(via_coarse, via_fine, embedding: ndarray, idx:Optional[list]=None,
                         title_str:str= "Pseudotime", draw_all_curves:bool=True, arrow_width_scale_factor:float=15.0,
                         scatter_size:float=50, scatter_alpha:float=0.5,
                         linewidth:float=1.5, marker_edgewidth:float=1, cmap_pseudotime:str='viridis_r',dpi:int=150,highlight_terminal_states:bool=True):
    '''

    :param via_coarse:
    :param via_fine:
    :param embedding:
    :param idx:
    :param title_str:
    :param draw_all_curves:
    :param arrow_width_scale_factor:
    :param scatter_size:
    :param scatter_alpha:
    :param linewidth:
    :param marker_edgewidth:
    :param cmap_pseudotime:
    :param dpi:
    :param highlight_terminal_states:
    :return:
    '''
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
            print(f'{datetime.now()}\tSuper cluster {i} is a super terminal with sub_terminal cluster',
                  sub_terminal_clusters[terminal_count_])
            c_edge.append('yellow')  # ('yellow')
            if highlight_terminal_states == True:
                width_edge.append(2)
                super_cluster_label.append('TS' + str(sub_terminal_clusters[terminal_count_]))
            else:
                width_edge.append(0)
                super_cluster_label.append('')
            pen_color.append('black')
            # super_cluster_label.append('TS' + str(i))  # +'('+str(i)+')')
             # +'('+str(i)+')')
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



def draw_sc_lineage_probability_solo(via_coarse, via_fine, embedding, idx=None, cmap_name='plasma', dpi=150):
    # embedding is the full or downsampled 2D representation of the full dataset.
    # idx is the list of indices of the full dataset for which the embedding is available. if the dataset is very large the the user only has the visual embedding for a subsample of the data, then these idx can be carried forward
    # idx is the selected indices of the downsampled samples used in the visualization
    # G is the igraph knn (low K) used for shortest path in high dim space. no idx needed as it's made on full sample
    # knn_hnsw is the knn made in the embedded space used for query to find the nearest point in the downsampled embedding
    #   that corresponds to the single cells in the full graph
    if idx is None: idx = np.arange(0, via_coarse.nsamples)
    G = via_coarse.full_graph_shortpath
    knn_hnsw = _make_knn_embeddedspace(embedding)
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

def draw_piechart_graph(via0, type_data='pt', gene_exp='', title='', cmap=None, ax_text=True, dpi=150,headwidth_arrow = 0.1, alpha_edge=0.4, linewidth_edge=2, edge_color='darkblue',reference=None):
    '''
    :param via0 is class VIA (the same function also exists as a method of the class as it is called during the TI analysis when VIA is being generated,
    but we offer it outside for consistency with other plotting tools and for usage after teh TI is complete.
    :param type_data:string:  default 'pt' for pseudotime colored nodes. or 'gene'
    :param gene_exp: list of values (column of dataframe) corresponding to feature or gene expression to be used to color nodes
    :param title: string
    :param cmap: default None. automatically chooses coolwarm for gene expression or viridis_r for pseudotime
    :param ax_text: Bool default: True. Annotates each node with cluster number and population of membership
    :param dpi: int default 150
    :param headwidth_bundle: default = 0.1. width of arrowhead used to directed edges
    :param reference=None. list of labels for cluster composition
    :return: matplotlib figure with two axes that plot the clustergraph using edge bundling
             left axis shows the clustergraph with each node colored by annotated ground truth membership.
             right axis shows the same clustergraph with each node colored by the pseudotime or gene expression
    '''
    # type_data can be 'pt' pseudotime or 'gene' for gene expression


    from mpl_toolkits.axes_grid1 import make_axes_locatable
    f, ((ax, ax1)) = plt.subplots(1, 2, sharey=True, dpi=dpi)


    node_pos = via0.graph_node_pos

    node_pos = np.asarray(node_pos)
    if cmap is None: cmap = 'coolwarm' if type_data == 'gene' else 'viridis_r'

    if type_data == 'pt':
        pt = via0.scaled_hitting_times  # these are the final MCMC refined pt then slightly scaled at cluster level
        title_ax1 = "Pseudotime"

    if type_data == 'gene':
        pt = gene_exp
        title_ax1 = title
    if reference is None: reference_labels=via0.true_label
    else: reference_labels = reference
    n_groups = len(set(via0.labels))
    n_truegroups = len(set(reference_labels))
    group_pop = np.zeros([n_groups, 1])
    group_frac = pd.DataFrame(np.zeros([n_groups, n_truegroups]), columns=list(set(reference_labels)))
    via0.cluster_population_dict = {}
    for group_i in set(via0.labels):
        loc_i = np.where(via0.labels == group_i)[0]

        group_pop[group_i] = len(loc_i)  # np.sum(loc_i) / 1000 + 1
        via0.cluster_population_dict[group_i] = len(loc_i)
        true_label_in_group_i = list(np.asarray(reference_labels)[loc_i])
        for ii in set(true_label_in_group_i):
            group_frac[ii][group_i] = true_label_in_group_i.count(ii)

    line_true = np.linspace(0, 1, n_truegroups)
    color_true_list = [plt.cm.rainbow(color) for color in line_true]

    sct = ax.scatter(node_pos[:, 0], node_pos[:, 1],
                     c='white', edgecolors='face', s=group_pop, cmap='jet')

    bboxes = getbb(sct, ax)

    ax = plot_edge_bundle(ax, via0.hammer_bundle, layout=via0.graph_node_pos, CSM=via0.CSM,
                            velocity_weight=via0.velo_weight, pt=pt, headwidth_bundle=headwidth_arrow,
                            alpha_bundle=alpha_edge,linewidth_bundle=linewidth_edge, edge_color=edge_color)

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
    pie_size_ar = ((group_pop - np.min(group_pop)) / (np.max(group_pop) - np.min(group_pop)) + 0.5) / 10  # 10
    # print('pie_size_ar', pie_size_ar)

    for node_i in range(n_groups):

        cluster_i_loc = np.where(np.asarray(via0.labels) == node_i)[0]
        majority_true = via0.func_mode(list(np.asarray(reference_labels)[cluster_i_loc]))
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
        # pie_axs[node_i].text(0.5, 0.5, graph_node_label[node_i])
        pie_axs[node_i].text(0.5, 0.5, majority_true)

    patches, texts = pie_axs[node_i].pie(frac, wedgeprops={'linewidth': 0.0}, colors=color_true_list)
    labels = list(set(reference_labels))
    plt.legend(patches, labels, loc=(-5, -5), fontsize=6, frameon=False)

    ti = 'Cluster Composition. K=' + str(via0.knn) + '. ncomp = ' + str(via0.ncomp)  +'knnseq_'+str(via0.knn_sequential)# "+ is_sub
    ax.set_title(ti)
    ax.grid(False)
    ax.set_xticks([])
    ax.set_yticks([])

    title_list = [title_ax1]
    for i, ax_i in enumerate([ax1]):
        pt = via0.markov_hitting_times if type_data == 'pt' else gene_exp

        c_edge, l_width = [], []
        for ei, pti in enumerate(pt):
            if ei in via0.terminal_clusters:
                c_edge.append('red')
                l_width.append(1.5)
            else:
                c_edge.append('gray')
                l_width.append(0.0)

        gp_scaling = 1000 / max(group_pop)  # 500 / max(group_pop)
        # print(gp_scaling, 'gp_scaling')
        group_pop_scale = group_pop * gp_scaling * 0.5
        ax_i=plot_edge_bundle(ax_i, via0.hammer_bundle, layout=via0.graph_node_pos,CSM=via0.CSM, velocity_weight=via0.velo_weight, pt=pt,headwidth_bundle=headwidth_arrow, alpha_bundle=alpha_edge, linewidth_bundle=linewidth_edge, edge_color=edge_color)
        ## ax_i.plot(hammer_bundle.x, hammer_bundle.y, 'y', zorder=2, linewidth=2, color='gray', alpha=0.8)
        im1 = ax_i.scatter(node_pos[:, 0], node_pos[:, 1], s=group_pop_scale, c=pt, cmap=cmap,
                           edgecolors=c_edge,
                           alpha=1, zorder=3, linewidth=l_width)
        if ax_text:
            x_max_range = np.amax(node_pos[:, 0]) / 100
            y_max_range = np.amax(node_pos[:, 1]) / 100

            for ii in range(node_pos.shape[0]):
                ax_i.text(node_pos[ii, 0] + max(x_max_range, y_max_range),
                          node_pos[ii, 1] + min(x_max_range, y_max_range),
                          'C' + str(ii) + 'pop' + str(int(group_pop[ii][0])),
                          color='black', zorder=4)

        title_pt = title_list[i]
        ax_i.set_title(title_pt)
        ax_i.grid(False)
        ax_i.set_xticks([])
        ax_i.set_yticks([])

    divider = make_axes_locatable(ax1)
    cax = divider.append_axes('right', size='5%', pad=0.05)
    if type_data == 'pt':
        f.colorbar(im1, cax=cax, orientation='vertical', label='pseudotime')
    else:
        f.colorbar(im1, cax=cax, orientation='vertical', label='Gene expression')
    f.patch.set_visible(False)

    ax1.axis('off')
    ax.axis('off')
    plt.show()

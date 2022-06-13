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

import time

from datashader.bundling import connect_edges, hammer_bundle
def sigmoid_func(X):
    return 1 / (1 + np.exp(-X))

def logistic_function(X,par_slope=1):
    '''
    https://en.wikipedia.org/wiki/Generalised_logistic_function
    :param X: input matrix X on which elementwise logistic function will be performed
    :param par_slope:
    :return:
    '''
    return  1/ (1 + np.exp(par_slope*-X))

def cosine_sim(A,B):
    '''
    :param A: matrix with n_samples and n_var
    :param B: matrix with m_samples and n_var
    :return: matrix of cosine similarity between A and B
    https://towardsdatascience.com/cosine-similarity-matrix-using-broadcasting-in-python-2b1998ab3ff3
    '''
    #numerator
    num = np.dot(A, B.T)
    #print('num',num)
    #denominator
    p1 = np.sqrt(np.sum(A ** 2, axis=1))#[:, np.newaxis] when A and B are both matrices
    p2 = np.sqrt(np.sum(B ** 2))
    #print(p1, p2, p1*p2)
    #if A and B is a matrix rather than vector, use below
    #p1 = np.sqrt(np.sum(A ** 2, axis=1))[:, np.newaxis] when A and B are both matrices
    #p2 = np.sqrt(np.sum(B ** 2), axis = 1))[np.newaxis, :]
    return num / (p1 * p2)

def make_edgebundle(layout, graph,initial_bandwidth = 0.05, decay=0.9):
    '''
    # Perform Edgebundling of edges in clustergraph to return a hammer bundle. hb.x and hb.y contain all the x and y coords of the points that make up the edge lines.
    # each new line segment is separated by a nan value
    # https://datashader.org/_modules/datashader/bundling.html#hammer_bundle
    :param layout: force-directed layout coordinates of graph
    :param graph: igraph clustergraph
    :param initial_bandwidth: increasing bw increases merging of minor edges
    :param decay: increasing decay increases merging of minor edges #https://datashader.org/user_guide/Networks.html
    :return: hb hammerbundle class with hb.x and hb.y containing the coords
    '''

    data_node = [[node] + layout.coords[node] for node in range(graph.vcount())]
    nodes = pd.DataFrame(data_node, columns=['id', 'x', 'y'])
    nodes.set_index('id', inplace=True)

    edges = pd.DataFrame([e.tuple for e in graph.es], columns=['source', 'target'])
    edges['weight'] = graph.es['weight']
    hb = hammer_bundle(nodes, edges, weight='weight',initial_bandwidth = initial_bandwidth, decay=decay) #default bw=0.05, dec=0.7
    #fig, ax = plt.subplots(figsize=(8, 8))
    #ax.plot(hb.x, hb.y, 'y', zorder=1, linewidth=3)
    # hb.plot(x="x", y="y", figsize=(9,9))
    #plt.show()
    return hb

def plot_edge_bundle(ax, hammer_bundle, layout,CSM, velocity_weight, pt, alpha_bundle=1, linewidth_bundle=2, edge_color='darkblue',headwidth_bundle=0.1, arrow_frequency=0.05):
    '''

    :param ax: axis to plot on
    :param hammer_bundle: hammerbundle object with coordinates of all the edges to draw
    :param layout: coords of cluster nodes
    :param CSM: cosine similarity matrix. cosine similarity between the RNA velocity between neighbors and the change in gene expression between these neighbors. Only used when available
    :param velocity_weight: percentage weightage given to the RNA velocity based transition matrix
    :param pt: cluster-level pseudotime
    :param alpha_bundle: alpha when drawing lines
    :param linewidth_bundle: linewidth of bundled lines
    :param edge_color:
    :param headwidth_bundle: headwidth of arrows used in bundled edges
    :param arrow_frequency: min dist between arrows (bundled edges otherwise have overcrowding of arrows)
    :return: axis with bundled edges plotted
    '''

    x_ = [l[0] for l in layout ]
    y_ =  [l[1] for l in layout ]
    #min_x, max_x = min(x_), max(x_)
    #min_y, max_y = min(y_), max(y_)
    delta_x =  max(x_)- min(x_)

    delta_y = max(y_)- min(y_)

    layout = np.asarray(layout)
    # make a knn so we can find which clustergraph nodes the segments start and end at

    neigh = NearestNeighbors(n_neighbors=1)
    neigh.fit(layout)
    # get each segment. these are separated by nans.
    hbnp = hammer_bundle.to_numpy()
    splits = (np.isnan(hbnp[:, 0])).nonzero()[0] #location of each nan values
    edgelist_segments = []
    start = 0
    segments = []
    arrow_coords=[]
    for stop in splits:
        seg = hbnp[start:stop, :]
        segments.append(seg)
        start = stop

    n = 1  # every nth segment is plotted
    step = 1
    for seg in segments[::n]:
        do_arrow=True
        seg_weight = max(0.3, math.log(1+seg[-1,2]))
        #print('seg weight', seg_weight)
        seg = seg[:,0:2].reshape(-1,2)
        seg_p = seg[~np.isnan(seg)].reshape((-1, 2))

        start=neigh.kneighbors(seg_p[0, :].reshape(1, -1), return_distance=False)[0][0]
        end = neigh.kneighbors(seg_p[-1, :].reshape(1, -1), return_distance=False)[0][0]
        #print('start,end',[start, end])

        if ([start, end] in edgelist_segments)|([end,start] in edgelist_segments):
            do_arrow = False
        edgelist_segments.append([start,end])

        direction_ = infer_direction_piegraph(start_node=start, end_node=end, CSM=CSM, velocity_weight=velocity_weight, pt=pt)

        direction = -1 if direction_ <0 else 1


        ax.plot(seg_p[:, 0], seg_p[:, 1],linewidth=linewidth_bundle*seg_weight, alpha=alpha_bundle, color=edge_color )
        mid_point = math.floor(seg_p.shape[0] / 2)
        if len(arrow_coords)>0: #dont draw arrows in overlapping segments
            for v1 in arrow_coords:
                dist_ = dist_points(v1,v2=[seg_p[mid_point, 0], seg_p[mid_point, 1]])
                #print('dist between points', dist_)
                if dist_< arrow_frequency*delta_x: do_arrow=False
                if dist_< arrow_frequency*delta_y: do_arrow=False

        if do_arrow==True:
            ax.arrow(seg_p[mid_point, 0], seg_p[mid_point, 1],
                 seg_p[mid_point + (direction * step), 0] - seg_p[mid_point, 0],
                 seg_p[mid_point + (direction * step), 1] - seg_p[mid_point, 1],
                 lw=0, length_includes_head=False, head_width=headwidth_bundle, color=edge_color,shape='full', alpha= 0.6, zorder=5)
            arrow_coords.append([seg_p[mid_point, 0], seg_p[mid_point, 1]])
    return ax

def dist_points(v1, v2):
    #euclidean distance between two points (x,y) (x1,y1)
    x = (v1[0]-v2[0])*(v1[0]-v2[0])
    y = (v1[1]-v2[1])*(v1[1]-v2[1])
    return math.sqrt(x+y)

def infer_direction_piegraph(start_node, end_node, CSM, velocity_weight,pt, tanh_scaling_factor=1):
    '''
    infers directionality between start and end node using both the velocity metric and pseudotime change.
    The Level of influence from pseudotime vs. velocity is based on the velocity weight
    :param start_node: the start node of the edge
    :param end_node: the end node of the end
    :param CSM: Cosine similarity matrix. i,j entry is the Cosine similarity between the velocity vector of the i'th
            cell and the change in spliced gene expression from cell i to j
    :param velocity_weight: parameter between 0-1, 0 signifies 0 weight of velocity (CSM matrix)
    :param pt: the cluster level pseudotimes
    :param tanh_scaling_factor: tanh(kx), default k=1. slope of the tanh curve
    :return: value between -1 and 1. a negative value signifies that the start and end should be swapped. start, end = end, start
    '''
    if CSM is None:
        velocity_weight=csm_es= csm_se=0
    else:
        csm_se = CSM[start_node,end_node] #cosine similarity from start-to-end
        csm_es = CSM[end_node, start_node] #cosine similarity from end-to-start
        '''
        print('csm_start', csm_se)
        print('csm_end', -1 * csm_es)
        print('csm', csm_se + -1 * csm_es)
        '''
        # Note csm_es does not equal -csm_se because the velocity vector used in the dot product refers to the originating cell
    pt = pt * (3 / max(pt))  # bring the values to 0-3 so that the tanh function has a range of values
    #print('pt of end node', end_node, round(pt[end_node], 2), 'pt of start node', start_node, round(pt[start_node], 2))
    tanh_ = math.tanh((pt[end_node] - pt[start_node]) * tanh_scaling_factor)
    #print('tanh', tanh_)
    direction = velocity_weight*0.5*(csm_se+ (-1*csm_es)) + (1-velocity_weight)*tanh_
    #print('direction for start',start_node,' to end node',end_node,'is', direction)
    return direction
def get_projected_distances(loadings, gene_matrix, velocity_matrix, edgelist, current_pc):
    '''
    # revise the distance between the start cell and the neighbor based on expected location of the neighbor after one step in velocity/pseudotime
    # the start cell location is given by the "projected" distance implied by the velocity, and the end cell location is based on the current gene based location in PCA space
    # based on idea described in Veloviz (Atta Lyla 2022)
    :param loadings: PCA loadings adata.varm['PCs']
    :param gene_matrix: single-cell gene matrix of the filtered genes
    :param velocity_matrix: single-cell velocity from sc-Velo/veloctyo
    :param edgelist: list of tuples of graph (start, end)
    :param current_pc: PCs of the current gene space
    :return:
    '''
    #loadings = adata.varm['PCs']
    #print('loadings shape', loadings.shape)
    proj = gene_matrix + velocity_matrix

    proj[proj<0]=0
    #print('inside projected distances')
    #print(gene_matrix[0:3,0:20])#pcs one step ahead


    proj -= np.mean(proj,axis=0)
    proj[proj < 0] = 0
    #print('size projected', proj.shape)
    proj = np.matmul(proj, loadings)
    #print('size proj-pca', proj.shape)
    #checking computation approach used for proj. by using the same approach on the gene matrix without adding velocity, we see if we get the current PCs
    curr_loading_pca = gene_matrix - np.mean(gene_matrix,axis=0)
    curr_loading_pca= np.matmul(curr_loading_pca, loadings)


    new_edgelist = []
    closer=0
    farther = 0
    for (s,e) in edgelist:
        #dist_prime = np.linalg.norm(current_pc[s,:]- proj[e,:]) #revise the distance between start-cell and it's neighbor based on expected location of the neighbor after one step in velocity/pseudotime
        dist_prime = np.linalg.norm(proj[s, :] - current_pc[e, :])
        new_edgelist.append(dist_prime)
        '''
        if s==0: #just to see what is happening when you shift the PCs
            print('regular pca dist: start-end',s,'-',e, round(np.linalg.norm(current_pc[s, :] - current_pc[e, :]),2))#np.linalg.norm(current_pc[s,:]- current_pc[e,:])
            print('regular pca dist using loadings: start-end', s, '-', e, round(np.linalg.norm(curr_loading_pca[e,:]- curr_loading_pca[s,:]),2))#np.linalg.norm(curr_loading_pca[s,:]- curr_loading_pca[e,:])
            print('projected pca dist: start-end', s, '-', e, round(dist_prime,2))
            if dist_prime<np.linalg.norm(current_pc[e,:]- current_pc[s,:]):
                print('proj is closer')
                closer +=1
            else: farther +=1
        '''
    #print('distance between projected start and current neighbor location closer', closer)
    #print('farther', farther)

    weights = np.array(new_edgelist)
    weights = np.clip(weights, a_min=np.percentile(weights, 10),
                      a_max=None)  # the first neighbor is usually itself and hence has distance 0

    #print('min clip weights in utils get projected distsances', np.percentile(weights, 10))
    #print('clipped weights for projected distances', weights)
    #scaled to between 0.5 and 2
    weights = 0.5 + (weights - np.min(weights)) * (2 - 0.5) / (np.max(weights) - np.min(weights))
    weights = 1 / weights
    return weights
def stationary_probability_naive(A_velo):
    '''
    Find the stationary probability of the cluster-graph transition matrix
    :param A_velo: transition matrix of cluster graph based on velocity (cluster level)
    :return: stationary probability of each cluster
    '''

    n_clus = A_velo.shape[0]
    A = np.append(np.transpose(A_velo) - np.identity(n_clus), [[1 for i in range(n_clus)]], axis=0)

    b = np.transpose(np.array([0 for i in range(n_clus+1)]))
    b[n_clus]=1
    lin_alg = np.linalg.solve(np.transpose(A).dot(A), np.transpose(A).dot(b))

    print(np.round(lin_alg,3))
    return lin_alg

def stationary_probability_(A_velo):
    '''
    Find the stationary probability of the cluster-graph transition matrix
    :param A_velo: transition matrix of cluster graph based on velocity (cluster level)
    :return: stationary probability of each cluster
    '''

    n = A_velo.shape[0]

    A_velo /= np.max(A_velo)
    for i, r in enumerate(A_velo):
        if np.all(r == 0):
            A_velo[i, i] = 1
    # ensure probabilities sum to 1 along the rows, normalize across the rows
    A_velo = A_velo / A_velo.sum(axis=1).reshape((n, 1))
    #print('A_velo normed', np.round(A_velo,2))
    #print('A_velo colsum',np.sum(A_velo, axis=1))


    #print('using eigenvalue decomp')
    D, V_l = np.linalg.eig(A_velo.T)
    #print('D, V_l', D, V_l)
    D_r, V_r = np.linalg.eig(A_velo)
    #print('D_r, V_R', D_r, V_r)
    # Sort the eigenvalues and eigenvectors and take the real part
    #print("Sorting eigenvalues by their real part")
    #print('D.real', np.round(D.real, 2))
    p = np.flip(np.argsort(D.real))
    #print('p sorted and flipped', p)
    D, V_l, V_r = D[p], V_l[:, p], V_r[:, p]

    pi = np.abs(V_l[:, 0].real)

    pi /= np.sum(pi)

    print(f"{datetime.now()}\tStationary distribution normed {np.round(pi,3)}")
    sorted_pi = np.argsort(pi)
    velo_root_top3 = sorted_pi[0:3]
    print(f"{datetime.now()}\tTop 3 candidates for root: {np.round(sorted_pi[0:3],2)} with stationary prob (%) {np.round(pi[sorted_pi[0:3]]*100,2)}")
    print(f"{datetime.now()}\tTop 5 candidates for terminal: {np.flip(sorted_pi)[0:5]}")

    return pi, velo_root_top3

def velocity_root(stationary_probability, A_velo):
    '''
    #use the stationary probability combined with cluster-graph vertex properties to identify a likely candidate for the root cluster
    :param stationary_probability:
    :param A_velo:
    :return:
    '''


def run_umap_hnsw( X_input, graph, n_components=2, alpha: float = 1.0, negative_sample_rate: int = 5,
                  gamma: float = 1.0, spread=1.0, min_dist=0.1, init_pos='spectral', random_state=1, n_epochs=0, distance_metric: str = 'euclidean' ):

    print('Computing umap on sc-Viagraph')
    from umap.umap_ import find_ab_params, simplicial_set_embedding
    #graph is a csr matrix
    #weight all edges as 1 in order to prevent umap from pruning weaker edges away

    a, b = find_ab_params(spread, min_dist)
    #print('a,b, spread, dist', a, b, spread, min_dist)
    t0 = time.time()
    graph.data.fill(1)
    X_umap = simplicial_set_embedding(data=X_input, graph=graph, n_components=n_components, initial_alpha=alpha,
                                      a=a, b=b, n_epochs=n_epochs, metric_kwds={}, gamma=gamma, metric=distance_metric,
                                      negative_sample_rate=negative_sample_rate, init=init_pos,
                                      random_state=np.random.RandomState(random_state),
                                      verbose=1)
    return X_umap


def velocity_transition(A,V,G, slope =4):
    '''
    Reweighting the cluster level transition matrix based on the cosine similarities of velocity vectors
    relative to the change in gene expression from the ith cell to its neighbors in knn_gene graph (at cluster level)
    :param A: Adjacency of clustergraph
    :param V: velocity matrix, cluster average
    :param G: Gene expression matrix, cluster average
    :return A_vel: reweighted transition matrix of cluster graph
    '''
    #from scipy.spatial import distance
    CSM = np.zeros_like(A)
    A_velo = np.zeros_like(A)
    for i in range(A.shape[0]):
        delta_gene = G-G[i,:] #change in gene expression when going from i'th cell to other cells
        #print('shape delta_gene', delta_gene.shape, delta_gene)
        delta_gene = np.delete(delta_gene, obj=i, axis=0) #delete ith row to avoid divide by zero in csm

        CSM_i = cosine_sim(delta_gene,V[i,:]) #the cosine similarity between delta_gene and the i'th cell's velocity
        #for j in range(3): print('scipy cosine similarity', 1-distance.cosine(delta_gene[j,:],V[i,:]))

        # A_velo_: convert the Cosine similarity to a weight between 0-1, such that edges going from i to j that have negative csm, have a low weight near 0
        # this means that rather than reverse the implied directionality of the edge, we make the transition very unlikely by lowering the weight in the new transition matrix. In this way we also ensure that the Markov process is irreducible
        A_velo_i = logistic_function(CSM_i, par_slope=slope)
        A_velo_i = np.insert(A_velo_i,obj=i,values=0) #set the ith value to 0
        CSM_i = np.insert(CSM_i,obj=i,values=0)

        CSM[i,:] = CSM_i
        A_velo[i,:] = A_velo_i

    mask = A==0 #identify the non-neighbors
    #CSM[mask] = 0 #remove non-neighbor edges
    A_velo[mask]=0
    A_velo = np.multiply(A_velo,A) #multiply element-wise the edge-weight of the transition matrix A by the velocity-factor
    #print('A_velo col sum', np.sum(A_velo,axis=1))
    print(f"{datetime.now()}\t Looking for initial states")
    pi, velo_root_top3 = stationary_probability_(A_velo)

    #print('Avelo')
    #print(A_velo)
    #print('CSM')
    #print(CSM)
    return A_velo, CSM, velo_root_top3

def sc_CSM(A, V, G):
    '''
    :param A: single-cell csr knn graph with neighbors. v0.self.full_csr_matrix
    :param V: cell x velocity matrix (dim: n_samples x n_genes)
    :param G: cell x genes matrix (dim: n_samples x n_genes)
    :return:
    '''
    CSM = np.zeros_like(A)
    find_A = find(A)
    size_A = A.size
    time_0=  time.ctime()
    #print('single-cell computation of sc_CSM')
    for i in range(A.shape[0]):
        delta_gene = G-G[i,:] #change in gene expression when going from i'th cell to other cells
        #print('shape delta_gene', delta_gene.shape, delta_gene)
        delta_gene = np.delete(delta_gene, obj=i, axis=0) #delete ith row to avoid divide by zero in cosine similarity calculation
        CSM_i = cosine_sim(delta_gene,V[i,:]) #the cosine similarity between delta_gene and the i'th cell's velocity
        #for j in range(3): print('scipy cosine similarity', 1-distance.cosine(delta_gene[j,:],V[i,:]))

        # A_velo_: convert the Cosine similarity to a weight between 0-1, such that edges going from i to j that have negative csm, have a low weight near 0
        #this means that rather than reverse the implied directionality of the edge, we make the transition very unlikely by lowering the weight in the new transition matrix.
        #A_velo_i = logistic_function(CSM_i, par_slope=4)
        #A_velo_i = np.insert(A_velo_i,obj=i,values=0)
        CSM_i = np.insert(CSM_i,obj=i,values=0)
        CSM[i,:] = CSM_i
    #print('shape 0', CSM.shape)
    #print('time taken non csr computation of sc_CSM', round(time.ctime()-time_0),2)
    CSM_list = []
    time_1 = time.ctime()
    #print('time start CSR computation of sc_CSM', time_1)
    for i in range(size_A):
        start = find_A[0][i]
        end = find_A[1][i]
        #weight = find_A[2][i]
        delta_gene = G[end,:]-G[start,:]  # change in gene expression when going from start cell to end cells
        # print('shape delta_gene', delta_gene.shape, delta_gene)
        CSM_list = CSM_list.append(cosine_sim(delta_gene, V[start, :]))  # the cosine similarity between delta_gene and the i'th cell's velocity
        # for j in range(3): print('scipy cosine similarity', 1-distance.cosine(delta_gene[j,:],V[i,:]))

        # A_velo_: convert the Cosine similarity to a weight between 0-1, such that edges going from i to j that have negative csm, have a low weight near 0
        # this means that rather than reverse the implied directionality of the edge, we make the transition very unlikely by lowering the weight in the new transition matrix.
        # A_velo_i = logistic_function(CSM_i, par_slope=4)
        # A_velo_i = np.insert(A_velo_i,obj=i,values=0)
    CSM = csr_matrix((CSM_list, (np.array(find_A[0]), np.array(find_A[1]))),
                       shape=size_A)
    #print('shape 1', CSM.shape)
    #print('CSM.data', CSM.data)

    #print('time taken for CSR computation of sc_CSM', round(time.ctime() - time_1), 2)
    return CSM
def interpolate_stream(array):
    from scipy import interpolate
    x = np.arange(0, array.shape[1])
    y = np.arange(0, array.shape[0])
    # mask invalid values
    array = np.ma.masked_invalid(array)
    xx, yy = np.meshgrid(x, y)
    # get only the valid values
    x1 = xx[~array.mask]
    y1 = yy[~array.mask]
    newarr = array[~array.mask]

    array_corrected = interpolate.griddata((x1, y1), newarr.ravel(),
                               (xx, yy),
                               method='cubic')
    return array_corrected

def interpolate_density(a, density_factor=2):
    from scipy.interpolate import UnivariateSpline
    old_indices = np.arange(0, a.size) #len(a)
    new_length = int(len(a)*density_factor)
    new_indices = np.linspace(0, len(a) - 1, new_length)
    spl = UnivariateSpline(old_indices, a, k=3, s=0)
    new_array = spl(new_indices)
    return new_array


def nan_helper(y):
    """Helper to handle indices and logical indices of NaNs.

    Input:
        - y, 1d numpy array with possible NaNs
    Output:
        - nans, logical indices of NaNs
        - index, a function, with signature indices= index(logical_indices),
          to convert logical indices of NaNs to 'equivalent' indices
    Example:
        >>> # linear interpolation of NaNs
        >>> y= array([1, 1, 1, NaN, NaN, 2, 2, NaN, 0])
        >>> nans, x= nan_helper(y)
        >>> y[nans]= np.interp(x(nans), x(~nans), y[~nans])

    """

    return np.isnan(y), lambda z: z.nonzero()[0]

def quiver_autoscale(X_emb, V_emb):
    import matplotlib.pyplot as pl

    scale_factor = np.abs(X_emb).max()  # just so that it handles very large values
    fig, ax = pl.subplots()
    Q = ax.quiver(
        X_emb[:, 0] / scale_factor,
        X_emb[:, 1] / scale_factor,
        V_emb[:, 0],
        V_emb[:, 1],
        angles="xy",
        scale_units="xy",
        scale=None,
    )
    Q._init()
    fig.clf()
    pl.close(fig)
    return Q.scale / scale_factor
def compute_velocity_on_grid(
    X_emb,
    V_emb,
    density=None,
    smooth=None,
    n_neighbors=None,
    min_mass=None,
    autoscale=True,
    adjust_for_stream=True,
    cutoff_perc=None,
):
    #adapted from scVelo Volker Bergen Nature Biotechnology 2020
    #print(X_emb.shape, V_emb.shape)

    # remove invalid cells
    idx_valid = np.isfinite(X_emb.sum(1) + V_emb.sum(1))
    X_emb = X_emb[idx_valid]
    V_emb = V_emb[idx_valid]


    # prepare grid
    n_obs, n_dim = X_emb.shape
    density = 1 if density is None else density
    smooth = 0.5 if smooth is None else smooth #0.5

    grs = []
    for dim_i in range(n_dim):
        m, M = np.min(X_emb[:, dim_i]), np.max(X_emb[:, dim_i])
        m = m - 0.01 * np.abs(M - m)
        M = M + 0.01 * np.abs(M - m)
        gr = np.linspace(m, M, int(50 * density))
        grs.append(gr)
    #print('grs', grs)
    meshes_tuple = np.meshgrid(*grs)
    X_grid = np.vstack([i.flat for i in meshes_tuple]).T

    # estimate grid velocities
    if n_neighbors is None:
        n_neighbors = min(int(n_obs / 50), 20)
    nn = NearestNeighbors(n_neighbors=n_neighbors, n_jobs=-1)
    nn.fit(X_emb)
    dists, neighs = nn.kneighbors(X_grid)

    scale = np.mean([(g[1] - g[0]) for g in grs]) * smooth #diagonal distance of a grid-square

    weight = normal.pdf(x=dists, scale=scale)


    p_mass = weight.sum(1)


    V_grid = (V_emb[neighs] * weight[:, :, None]).sum(1)
    #print("V_grid in utils before norm")
    #print(V_grid)
    #print('V_grid intermediate 1', V_grid)
    V_grid /= np.maximum(1, p_mass)[:, None]

    #print('V_grid /= p_mass', V_grid)
    if min_mass is None:
        min_mass = 1

    if adjust_for_stream==True:

        X_grid = np.stack([np.unique(X_grid[:, 0]), np.unique(X_grid[:, 1])])
        ns = int(np.sqrt(len(V_grid[:, 0])))
        V_grid = V_grid.T.reshape(2, ns, ns)

        mass = np.sqrt((V_grid ** 2).sum(0))
        min_mass = 10 ** (min_mass - 6)  # default min_mass = 1e-5
        min_mass = np.clip(min_mass, None, np.max(mass) * 0.9)


        #min_mass = np.clip(0, None, np.max(mass) * 0.9999)
        cutoff = mass.reshape(V_grid[0].shape) < min_mass


        if cutoff_perc is None:
            cutoff_perc = 5
        length = np.sum(np.mean(np.abs(V_emb[neighs]), axis=1), axis=1).T
        length = length.reshape(ns, ns)
        cutoff |= length < np.percentile(length, cutoff_perc)

        V_grid[0][cutoff] = np.nan
    else:

        min_mass *= np.percentile(p_mass, 99) / 100

        X_grid, V_grid = X_grid[p_mass > min_mass], V_grid[p_mass > min_mass]

        #if autoscale:            V_grid /= 3 * quiver_autoscale(X_grid, V_grid)
    return X_grid, V_grid

def map_velocity_to_color(X1D, Y1D, U, V,segments):
    '''

    :param X: nx1 array of x-coors of grid
    :param Y: nx1 array of y-coors of grid
    :param U: nxn array of u-velocity on X-Y grid
    :param V:
    :return:
    '''
    #reshape U and V so we can match coords with velocities
    print(U,V)
    U_ = U.reshape((np.prod(U.shape),))
    V_ = V.reshape((np.prod(V.shape),))
    velo_coords = np.vstack((U_, V_)).T
    #velo_coords = velo_coords / velo_coords.max(axis=0)
    print('velo coords', velo_coords.shape)
    print(velo_coords)
    #reshape meshgrid to extract coords
    X, Y = np.meshgrid(X1D,Y1D)

    X = X.reshape((np.prod(X.shape),))
    Y = Y.reshape((np.prod(Y.shape),))

    coords = np.vstack((X, Y)).T #(nx2) array of grid coords
    neigh_graph = NearestNeighbors(n_neighbors=1)
    neigh_graph.fit(coords)
    print('number coords for grid', coords.shape)
    print(np.isnan(coords))

    seg_coors = np.squeeze(segments[:, ::2, :])
    print('seg coors', seg_coors.shape)
    neigh_seg = neigh_graph.kneighbors(seg_coors, return_distance=False)#[0]
    print('neighbors of seg points', neigh_seg)
    n= len(segments)
    u_seg = velo_coords[neigh_seg,1]

    print('u-seg', u_seg.shape)
    print(u_seg)
    C = np.zeros((n, 4))
    C[::-1] = 1-np.clip(u_seg,0.01,1)

    return C

def interpolate_static_stream(x, y, u,v):
    print('original U')
    print(u)
    from scipy.interpolate import griddata
    x, y = np.meshgrid(x, y)

    print('inside interpolate static stream')
    print('mesh original', x.shape, y.shape)
    #print(x, y)
    points = np.array((x.flatten(), y.flatten())).T
    u = np.nan_to_num(u.flatten())
    v = np.nan_to_num(v.flatten())
    xi = np.linspace(x.min(), x.max(), 25)
    yi = np.linspace(y.min(), y.max(), 25)
    X, Y = np.meshgrid(xi, yi)
    #print('new mesh', X.shape, Y.shape)
    #print('X new mesh')
    #print(X)
    #print('Y new mesh')
    #print(Y)

    U = griddata(points, u, (X, Y), method='cubic')
    V = griddata(points, v, (X, Y), method='cubic')
    print('interp static U', U.shape, U)
    print('interp static V', V.shape, V)
    return X, Y, U, V

def l2_norm(x: Union[ndarray, spmatrix], axis: int = 1) -> Union[float, ndarray]:


    """Calculate l2 norm along a given axis.
    Arguments
    ---------
    x
        Array to calculate l2 norm of.
    axis
        Axis along which to calculate l2 norm.
    Returns
    -------
    Union[float, ndarray]
        L2 norm along a given axis.
    """

    if issparse(x):
        return np.sqrt(x.multiply(x).sum(axis=axis).A1)
    elif x.ndim == 1:
        return np.sqrt(np.einsum("i, i -> ", x, x))
    elif axis == 0:
        return np.sqrt(np.einsum("ij, ij -> j", x, x))
    elif axis == 1:
        return np.sqrt(np.einsum("ij, ij -> i", x, x))
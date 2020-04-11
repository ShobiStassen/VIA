import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix, csgraph
import scipy
import igraph as ig
import leidenalg
import time
import hnswlib
import matplotlib.pyplot as plt
import matplotlib
import math
import multiprocessing
from scipy.sparse.csgraph import minimum_spanning_tree
from scipy import sparse
from sklearn.metrics.pairwise import euclidean_distances
import umap
import scanpy as sc
from MulticoreTSNE import MulticoreTSNE as TSNE
import random
from scipy.sparse.csgraph import connected_components
import pygam as pg
# version before translating chinese on Feb13
# jan2020 Righclick->GIT->Repository-> PUSH
def plot_sc_pb(ax, embedding, prob, ti):
    threshold = np.mean(prob) + 2 * np.std(prob)
    prob = [x if x < threshold else threshold for x in prob]

    cmap = matplotlib.cm.get_cmap('viridis')
    norm = matplotlib.colors.Normalize(vmin=0, vmax=threshold)
    prob = np.asarray(prob)
    c = cmap(norm(prob))
    c=c.reshape(-1,4)
    loc_c = np.where(prob <=0.3)[0]
    c[loc_c,3]=0.1
    loc_c = np.where((prob > 0.3)&(prob<=0.5))[0]
    c[loc_c,3]=0.2
    loc_c = np.where((prob > 0.5) & (prob <= 0.7))[0]
    c[loc_c, 3] = 0.5
    ax.scatter(embedding[:, 0], embedding[:, 1], c=c, s=10, cmap='viridis',
               edgecolors='none')  # , alpha=0.5)
    '''
    for i in range(embedding.shape[0]):
        rgba = cmap(norm(prob[i]))
        rgba = list(rgba)
        if prob[i]<0.1: rgba[3]=0.05
        else: list(rgba)[3] = 0.5
        rgba = tuple(rgba)
        
        #ax.scatter(embedding[i, 0], embedding[i, 1], c=np.asarray(rgba).reshape(1,-1), s=10, cmap='viridis',edgecolors='none')#, alpha=0.5)
        #ax.scatter(embedding[:, 0], embedding[:, 1], c=prob, s=10, cmap='viridis', alpha=0.5)
    '''
    #alpha_list = [0.7 if x >(np.mean(prob)-np.std(prob)) else 0.1 for x in prob]
    #ax.scatter(embedding[:, 0], embedding[:, 1], c=prob, s=10, cmap='viridis', alpha=0.5)
    ax.set_title('Target: ' + str(ti))
def simulate_multinomial(vmultinomial):
    r = np.random.uniform(0.0, 1.0)
    CS = np.cumsum(vmultinomial)
    CS = np.insert(CS, 0, 0)
    m = (np.where(CS < r))[0]
    nextState = m[len(m) - 1]
    return nextState

def sc_loc_ofsuperCluster_embeddedspace(embedding, p0, p1):
    # ci_list: single cell location of average location of supercluster based on embedded space hnsw
    knn_hnsw = hnswlib.Index(space='l2', dim=embedding.shape[1])
    knn_hnsw.init_index(max_elements=embedding.shape[0], ef_construction=200, M=16)
    knn_hnsw.add_items(embedding)
    knn_hnsw.set_ef(50)

    ci_list = []
    for ci in list(set(p0.labels)):
        if ci in p0.terminal_clusters:
            loc_i = np.where(np.asarray(p0.labels) == ci)[0]
            val_pt = [p1.single_cell_pt_markov[i] for i in loc_i]
            th_pt = np.percentile(val_pt, 50)  # 50
            loc_i = [loc_i[i] for i in range(len(val_pt)) if val_pt[i] >= th_pt]
            x = [embedding[xi, 0] for xi in loc_i]
            y = [embedding[yi, 1] for yi in loc_i]
        else:
            loc_i = np.where(np.asarray(p0.labels) == ci)[0]
            # temp = np.mean(adata_counts.obsm['X_pca'][:, 0:ncomps][loc_i], axis=0)
            x = [embedding[xi, 0] for xi in loc_i]
            y = [embedding[yi, 1] for yi in loc_i]

        labelsq, distancesq = knn_hnsw.knn_query(np.array([np.mean(x), np.mean(y)]), k=1)
        # labels, distances = p.knn_query(temp, k=1)
        ci_list.append(labelsq[0][0])
    return knn_hnsw, ci_list
def draw_sc_evolution_trajectory(p1, embedding, knn_hnsw):
    y_root = []
    x_root = []
    for ii, r_i in enumerate(p1.root):
        loc_i = np.where(np.asarray(p1.labels) == p1.root[ii])[0]
        x = [embedding[xi, 0] for xi in loc_i]
        y = [embedding[yi, 1] for yi in loc_i]

        labels_root, distances_root = knn_hnsw.knn_query(np.array([np.mean(x), np.mean(y)]), k=1)
        x_root.append(embedding[labels_root, 0][0])
        y_root.append(embedding[labels_root, 1][0])
        print('xyroots', x_root, y_root)

    # single-cell branch probability evolution probability
    for i, ti in enumerate(p1.terminal_clusters):
        root_i = p1.root[p1.connected_comp_labels[ti]]
        fig, ax = plt.subplots()
        plot_sc_pb(ax, embedding, p1.single_cell_bp[:, i], ti)
        loc_i = np.where(np.asarray(p1.labels) == ti)[0]
        val_pt = [p1.single_cell_pt_markov[i] for i in loc_i]
        th_pt = np.percentile(val_pt, 50)  # 50
        loc_i = [loc_i[i] for i in range(len(val_pt)) if val_pt[i] >= th_pt]
        x = [embedding[xi, 0] for xi in loc_i] #location of sc nearest to average location of terminal clus in the EMBEDDED space
        y = [embedding[yi, 1] for yi in loc_i]
        labels, distances = knn_hnsw.knn_query(np.array([np.mean(x), np.mean(y)]), k=1)
        x_sc = embedding[labels[0], 0]
        y_sc = embedding[labels[0], 1]
        ax.scatter(x_sc, y_sc, color='pink', zorder=3, label=str(ti), s=18)
        ax.text(x_sc + 0.5, y_sc + 0.5, 'TS' + str(ti), color='black')
        weights = p1.single_cell_bp[:, i]  # /np.sum(p1.single_cell_bp[:,i])
        weights[weights < 0.01] = 0
        weights[np.where(np.asarray(p1.labels) == root_i)[0]] = 0.9
        weights[np.where(np.asarray(p1.labels) == ti)[0]] = 1
        weights[labels[0]] = 10
        loc_z = np.where(weights > 0)[0]
        min_weight = np.min(weights[weights != 0])
        weights[weights == 0] = min_weight * 0.000001

        minx = min(x_root[p1.connected_comp_labels[i]], x_sc)  # np.min(x))
        maxx = max(x_root[p1.connected_comp_labels[i]], x_sc)  # np.max(x))
        xp = np.linspace(minx, maxx, 500)
        loc_i = np.where((embedding[:, 0] <= maxx) & (embedding[:, 0] >= minx))[0]
        loc_i = np.intersect1d(loc_i, loc_z)
        x_val = embedding[loc_i, 0].reshape(len(loc_i), -1)
        print('x-val shape0', x_val.shape)
        scGam = pg.LinearGAM(n_splines=10, spline_order=3, lam=10).fit(x_val, embedding[loc_i, 1],
                                                                       weights=weights[loc_i].reshape(len(loc_i),
                                                                                                      -1))
        ax.scatter(x_root[p1.connected_comp_labels[ti]], y_root[p1.connected_comp_labels[ti]], s=13, c='red')
        ax.text(x_root[p1.connected_comp_labels[ti]]+0.5, y_root[p1.connected_comp_labels[ti]]+0.5, 'TS' + str(ti), color='black')
        # XX = scGam.generate_X_grid(term=0, n=500)

        preds = scGam.predict(xp)
        ax.plot(xp, preds, linewidth=2, c='dimgray')
    return

def draw_sc_evolution_trajectory_pt(p1, embedding, knn_hnsw):
    y_root = []
    x_root = []
    for ii, r_i in enumerate(p1.root):
        loc_i = np.where(np.asarray(p1.labels) == p1.root[ii])[0]
        x = [embedding[xi, 0] for xi in loc_i]
        y = [embedding[yi, 1] for yi in loc_i]

        labels_root, distances_root = knn_hnsw.knn_query(np.array([np.mean(x), np.mean(y)]), k=1)
        x_root.append(embedding[labels_root, 0][0])
        y_root.append(embedding[labels_root, 1][0])
        print('xyroots', x_root, y_root)

    # single-cell branch probability evolution probability
    for i, ti in enumerate(p1.terminal_clusters):
        root_i = p1.root[p1.connected_comp_labels[ti]]
        xx_root = x_root[p1.connected_comp_labels[ti]]
        yy_root = y_root[p1.connected_comp_labels[ti]]
        fig, ax = plt.subplots()
        plot_sc_pb(ax, embedding, p1.single_cell_bp[:, i], ti)
        loc_i = np.where(np.asarray(p1.labels) == ti)[0]
        val_pt = [p1.single_cell_pt_markov[i] for i in loc_i]
        th_pt = np.percentile(val_pt, 50)  # 50
        loc_i = [loc_i[i] for i in range(len(val_pt)) if val_pt[i] >= th_pt]
        x = [embedding[xi, 0] for xi in loc_i] #location of sc nearest to average location of terminal clus in the EMBEDDED space
        y = [embedding[yi, 1] for yi in loc_i]
        labels, distances = knn_hnsw.knn_query(np.array([np.mean(x), np.mean(y)]), k=1)
        x_sc = embedding[labels[0], 0] #terminal sc location
        y_sc = embedding[labels[0], 1]
        ax.scatter(x_sc, y_sc, color='pink', zorder=3, label=str(ti), s=18)
        ax.text(x_sc + 0.5, y_sc + 0.5, 'TS' + str(ti), color='black')
        weights = p1.single_cell_bp[:, i]  # /np.sum(p1.single_cell_bp[:,i])
        weights[weights < 0.05] = 0
        weights[np.where(np.asarray(p1.labels) == root_i)[0]] = 0.9
        weights[np.where(np.asarray(p1.labels) == ti)[0]] = 1
        weights[labels[0]] = 1
        loc_z = np.where(weights > 0)[0]
        min_weight = np.min(weights[weights != 0])
        weights[weights == 0] = min_weight * 0.000001
        print('number of zeros in weights', np.sum([weights==0 ]))

        minx = min(x_root[p1.connected_comp_labels[i]], x_sc)  # np.min(x))
        maxx = max(x_root[p1.connected_comp_labels[i]], x_sc)  # np.max(x))
        if minx == x_sc: #the root-cell is on the RHS
            loc_i = np.where(embedding[:, 0] <=maxx)[0]
        else:
            loc_i = np.where(embedding[:, 0] >= minx)[0]
        xp = np.linspace(minx, maxx, len(loc_z))
        print('max sc_pt',max(p1.single_cell_pt_markov))
        xpt = np.linspace(0, max(p1.single_cell_pt_markov), len(loc_z)).reshape((len(loc_z),-1))
        loc_i = np.where((embedding[:, 0] <= maxx) & (embedding[:, 0] >= minx))[0]
        loc_pt= np.where(p1.single_cell_pt_markov<=np.percentile(val_pt, 95)) #doesnt help much
        loc_inter = np.intersect1d(loc_i, loc_z)
        x_val = embedding[loc_i, 0].reshape(len(loc_i), -1)
        print('x-val shape0', x_val.shape)
        #scGam = pg.LinearGAM(n_splines=10, spline_order=3, lam=10).fit(x_val, embedding[loc_i, 1], weights=weights[loc_i].reshape(len(loc_i), -1))
        Xin = np.asarray([p1.single_cell_pt_markov,embedding[:,0]]).T
        Xin = Xin[loc_inter,:]
        print('Xin shape', Xin.shape)
        n_reps = 100
        rep =np.repeat(np.array([[ max(val_pt),x_sc]]),n_reps,axis=0)
        rep = rep+np.random.normal(0,.1,rep.shape)
        Xin = np.concatenate((Xin,rep),axis=0)
        print('Xin shape', Xin.shape, xx_root)
        rep = np.repeat(np.array([[ 0,xx_root]]),n_reps,axis=0) +  np.random.normal(0,.1,rep.shape)
        Xin = np.concatenate((Xin, rep), axis=0)
        print('Xin shape', Xin.shape)
        weights = weights.reshape((embedding.shape[0], 1))[loc_inter, 0]
        weights = weights.reshape((len(loc_inter),1))
        print('weights shape', weights.shape)
        rep = np.repeat(np.array([[1]]), 2*n_reps, axis=0)
        print('rep shape', rep.shape)
        weights = np.concatenate((weights,rep),axis=0)

        print('weights shape', weights.shape, y_sc)
        rep = np.repeat(np.array([y_sc]), n_reps, axis=0)
        print('repy shape', rep.shape)
        yin=embedding[loc_inter,1]
        yin = yin.reshape((-1,1))
        print('yin shape', yin.shape)
        yin = np.concatenate((yin,rep),axis=0)
        print('yin shape', yin.shape)
        rep = np.repeat(np.array([yy_root]), n_reps, axis=0)
        yin = np.concatenate((yin, rep), axis=0)
        print('yin shape', yin.shape)
        Xin_yin = np.concatenate((Xin,yin), axis=1)
        Xin_yin = np.concatenate((Xin_yin, weights), axis=1)
        print('shape Xin_yin', Xin_yin.shape)
        temp_max = max(Xin_yin[:,1])
        temp_min = min(Xin_yin[:, 1])
        print('temp_ranges', temp_min, temp_max)
        n_bins = 20
        bin_width = (temp_max-temp_min)/n_bins
        final_input = np.zeros((1,4))
        print('final input', final_input)
        for i in range(n_bins):
            bin_start = temp_min + bin_width*i
            bin_end= bin_start+bin_width
            temp= Xin_yin[(Xin_yin[:,1]< bin_end)&(Xin_yin[:,1]>= bin_start)]
            print('bin', i, 'has temp shape before trimming', temp.shape)
            if temp.shape[1]>10:
                temp = temp[(temp[:,2]<=(np.mean(temp[:,2])+0.5*np.std(temp[:,2])))&(temp[:,2]>=(np.mean(temp[:,2])-0.5*np.std(temp[:,2])))]
                print('cutoff bin values', np.mean(temp[:, 2]) + np.std(temp[:, 2]))
            print('temp shape after trimming', temp.shape)

            final_input = np.concatenate((final_input, temp), axis=0)

        final_input = final_input[1:,:]
        final_input = final_input[(final_input[:,1]<=maxx) & (final_input[:,1]>=minx) ]
        print('final input', final_input.shape)
        print('final input', final_input)
        scGamx = pg.LinearGAM(n_splines=5, spline_order=3, lam=10).fit(final_input[:,0:2], final_input[:,2], weights=final_input[:,3])#25#fit(Xin, yin, weights=weights)#25
        #scGamx.gridsearch(Xin,  embedding[loc_z,1], weights=weights)
       # scGamy = pg.LinearGAM(n_splines=10, spline_order=3, lam=10).fit(p1.single_cell_pt_markov, embedding[:, 1],
                                                             #weights=weights.reshape(embedding.shape[0], -1))

        ax.scatter(xx_root, yy_root, s=13, c='red')
        ax.text(x_root[p1.connected_comp_labels[ti]]+0.5, y_root[p1.connected_comp_labels[ti]]+0.5, 'TS' + str(ti), color='black')
        XX = scGamx.generate_X_grid(term=1, n=500)
        print('xp shape before', xp.shape)
        print('xpt shape before', xpt.shape)
        xp_ = np.concatenate((xpt,xp),axis=1)
        print('xp shape', xp.shape)
        #XX=scGamx.generate_X_grid(term=1, n=500)

        #predsx = scGamx.predict(xp)
        #print('xx shape', XX.shape, XX)
        #print('predsx shape', predsx.shape)
        #predsy = scGamy.predict(predsx)
        Xin = pd.DataFrame(final_input[:,0:2]).sort_values(1).values #Xin
        #print('xin', Xin)# +  np.random.normal(0,.01,Xin.shape)
        yg = scGamx.predict(X=Xin)
        print('yg', yg.shape)
        #ax.plot(Xin[:,1], yg, linewidth=2, c='dimgray')
        Xin = Xin[:,1].reshape((len(yg),-1))
        print('final xin', Xin.shape)
        scGam = pg.LinearGAM(n_splines=5, spline_order=3, lam=1000).gridsearch(Xin, yg)
        XX = scGam.generate_X_grid(term=0, n=500)
        preds = scGam.predict(XX)
        ax.plot(XX, preds, linewidth=2, c='dimgray')
    return
def draw_sc_evolution_trajectory_dijkstra(p1, embedding, knn_hnsw,G):
    y_root = []
    x_root = []
    for ii, r_i in enumerate(p1.root):
        loc_i = np.where(np.asarray(p1.labels) == p1.root[ii])[0]
        x = [embedding[xi, 0] for xi in loc_i]
        y = [embedding[yi, 1] for yi in loc_i]

        labels_root, distances_root = knn_hnsw.knn_query(np.array([np.mean(x), np.mean(y)]), k=1)
        x_root.append(embedding[labels_root, 0][0])
        y_root.append(embedding[labels_root, 1][0])
        print('xyroots', x_root, y_root)


    # single-cell branch probability evolution probability
    for i, ti in enumerate(p1.terminal_clusters):
        print('i, ti, p1.root, p1.connected', i, ti, p1.root, p1.connected_comp_labels)
        root_i = p1.root[p1.connected_comp_labels[ti]]
        xx_root = x_root[p1.connected_comp_labels[ti]]
        yy_root = y_root[p1.connected_comp_labels[ti]]
        fig, ax = plt.subplots()
        plot_sc_pb(ax, embedding, p1.single_cell_bp[:, i], ti)

        loc_i = np.where(np.asarray(p1.labels) == ti)[0]
        val_pt = [p1.single_cell_pt_markov[i] for i in loc_i]
        th_pt = np.percentile(val_pt, 50)  # 50
        loc_i = [loc_i[i] for i in range(len(val_pt)) if val_pt[i] >= th_pt]
        x = [embedding[xi, 0] for xi in loc_i] #location of sc nearest to average location of terminal clus in the EMBEDDED space
        y = [embedding[yi, 1] for yi in loc_i]
        labels, distances = knn_hnsw.knn_query(np.array([np.mean(x), np.mean(y)]), k=1)
        x_sc = embedding[labels[0], 0] #terminal sc location
        y_sc = embedding[labels[0], 1]
        start_time = time.time()
        print('labels root and labels[0]',labels_root[0],labels[0])
        path = G.get_shortest_paths(labels_root[0][0], to=labels[0][0], weights='weight') #G is the knn of all sc points
        #formatted_float = "{:.2f}".format(a_float)
        print(f"get_shortest_paths time: {time.time()-start_time}")
        print('path', path)
        n_orange = len(path[0])
        orange_m = np.zeros((n_orange,3))
        for enum_point, point in enumerate(path[0]):
            #ax.scatter(embedding[point,0], embedding[point,1], color='orange', zorder=3, label=str(ti), s=22)
            ax.text(embedding[point,0], embedding[point,1],'D '+str(enum_point), color = 'blue', fontsize=8)
            orange_m[enum_point,0] = embedding[point,0]
            orange_m[enum_point, 1]=embedding[point, 1]
            orange_m[enum_point, 2] = p1.single_cell_pt_markov[point]#*p1.single_cell_pt_markov[point]
        from sklearn.neighbors import NearestNeighbors
        k_orange = 3
        nbrs = NearestNeighbors(n_neighbors=k_orange,  algorithm='ball_tree').fit(orange_m[:,0:])
        distances, indices = nbrs.kneighbors(orange_m[:,0:])
        row_list = []
        col_list = []
        dist_list = []

        for i_or in range(n_orange):
            for j_or in range(1,k_orange):
                row_list.append(i_or)
                col_list.append(indices[i_or,j_or])
                dist_list.append(distances[i_or,j_or])
        print('target number '+ str(ti))
        print('lists', len(row_list), len(col_list), len(dist_list))
        print('lists', (row_list), (col_list), (dist_list))

        orange_adjacency_knn = csr_matrix((np.array(dist_list), (np.array(row_list), np.array(col_list))),
                                       shape=(n_orange, n_orange))
        print('orange adj knn shape', orange_adjacency_knn.shape)
        #orange_adjacency_knn = (orange_adjacency_knn+orange_adjacency_knn.T)*.5
        orange_mst = orange_adjacency_knn#minimum_spanning_tree(orange_adjacency_knn)

        n_mst, comp_labels_mst = connected_components(csgraph=orange_mst, directed=False, return_labels=True)

        for enum_point, point in enumerate(path[0]):
            orange_m[enum_point, 2] = p1.single_cell_pt_markov[point] * p1.single_cell_pt_markov[point]*2
            print('single cell pt markov',p1.single_cell_pt_markov[point])

        while n_mst >1:
            comp_root = comp_labels_mst[0]
            print('comp-root', comp_root)
            min_ed = 9999999
            loc_comp_i = np.where(comp_labels_mst==comp_root)[0]
            loc_comp_noti = np.where(comp_labels_mst != comp_root)[0]
            print('compi', loc_comp_i)
            print('comp_noti', loc_comp_noti)
            orange_pt_val = [orange_m[cc,2] for cc in loc_comp_i]
            loc_comp_i_revised = [loc_comp_i[cc] for cc in range(len(orange_pt_val)) if orange_pt_val[cc]>=np.percentile(orange_pt_val,70)]
            print('Target:', ti)
            print('compi revised', loc_comp_i_revised)
            for nn_i in loc_comp_i_revised:

                ed = euclidean_distances(orange_m[nn_i,:].reshape(1,-1),orange_m[loc_comp_noti])

                if np.min(ed)< min_ed:
                    ed_where_min = np.where(ed[0] == np.min(ed))[0][0]
                    print('ed where min', ed_where_min,np.where(ed[0] == np.min(ed)))
                    min_ed = np.min(ed)
                    ed_loc_end = loc_comp_noti[ed_where_min]
                    ed_loc_start = nn_i
            print('min ed', min_ed)
            print('the closest pair of points', ed_loc_start, ed_loc_end)
            orange_mst[ed_loc_start, ed_loc_end] = min_ed
            n_mst, comp_labels_mst=connected_components(csgraph=orange_mst, directed=False, return_labels=True)




        if n_mst==1:
            print('orange mst shape', orange_mst.shape,orange_mst.shape[0])
            #orange_mst = minimum_spanning_tree(orange_mst)

            (orange_sources, orange_targets) = orange_mst.nonzero()
            orange_edgelist = list(zip(orange_sources.tolist(), orange_targets.tolist()))
            print('sources and targets', len(orange_sources), len(orange_targets))
            print('sources and targets', orange_sources, orange_targets)
            temp_list = []
            for i_or in range(len(orange_sources)):
                orange_x = [orange_m[:, 0][orange_sources[i_or]], orange_m[:, 0][orange_targets[i_or]]]
                orange_y = [orange_m[:, 1][orange_sources[i_or]], orange_m[:, 1][orange_targets[i_or]]]
                #ax.plot(orange_x, orange_y, color='purple') #visualize entire MST

            G_orange = ig.Graph(n=orange_mst.shape[0], edges= orange_edgelist,edge_attrs={'weight': orange_mst.data.tolist()}, )
            path_orange = G_orange.get_shortest_paths(0, to=orange_mst.shape[0]-1, weights='weight')[0]
            print('path orange', path_orange)
            len_path_orange = len(path_orange)

            for path_i in range(len_path_orange-1):
                path_x_start = orange_m[path_orange[path_i], 0]
                path_x_end = orange_m[path_orange[path_i+1], 0]
                orange_x = [orange_m[path_orange[path_i], 0], orange_m[path_orange[path_i+1], 0]]
                orange_minx = min(orange_x)
                orange_maxx = max(orange_x)

                path_y_start =  orange_m[path_orange[path_i ], 1]
                path_y_end =path_x_end = orange_m[path_orange[path_i+1], 0]
                orange_y = [orange_m[path_orange[path_i], 1], orange_m[path_orange[path_i + 1], 1]]
                orange_miny = min(orange_y)
                orange_maxy = max(orange_y)
                orange_embedding_sub = embedding[((embedding[:,0]<=orange_maxx) & (embedding[:,0]>=orange_minx))&((embedding[:,1]<=orange_maxy)&((embedding[:,1]>=orange_miny)))]
                print('orange sub size', orange_embedding_sub.shape)
                if (orange_maxy-orange_miny>5) | (orange_maxx-orange_minx>5):
                    orange_n_reps = 150
                else: orange_n_reps = 100
                or_reps  =np.repeat(np.array([[ orange_x[0],orange_y[0]]]),orange_n_reps,axis=0)
                orange_embedding_sub = np.concatenate((orange_embedding_sub, or_reps), axis=0)
                or_reps = np.repeat(np.array([[ orange_x[1],orange_y[1]]]), orange_n_reps, axis=0)
                orange_embedding_sub = np.concatenate((orange_embedding_sub, or_reps), axis=0)

                orangeGam = pg.LinearGAM(n_splines=8, spline_order=3, lam=10).fit(orange_embedding_sub[:,0], orange_embedding_sub[:,1])

                orange_GAM_xval = np.linspace(orange_minx, orange_maxx, 200)
                yg_orange = orangeGam.predict(X=orange_GAM_xval)




                #ax.plot(orange_x, orange_y, color='purple')
                ax.plot(orange_GAM_xval, yg_orange, color='dimgrey', linewidth = 2, zorder = 3, linestyle=(0, (5, 2, 1, 2)), dash_capstyle='round')
                step = 1

                     # , head_starts_at_zero = direction_arrow )

                #ax.plot(orange_GAM_xval, yg_orange, color='black', linewidth=2, zorder=3, linestyle = ':')
                cur_x1 = orange_GAM_xval[-1]
                cur_y1 = yg_orange[-1]
                cur_x2 = orange_GAM_xval[0]
                cur_y2 = yg_orange[0]
                if path_i >= 1:
                    for mmddi in range(2):
                        xy11= euclidean_distances(np.array([cur_x1, cur_y1]).reshape(1,-1),np.array([prev_x1, prev_y1]).reshape(1,-1))
                        xy12 = euclidean_distances(np.array([cur_x1, cur_y1]).reshape(1, -1),np.array([prev_x2, prev_y2]).reshape(1, -1))
                        xy21 = euclidean_distances(np.array([cur_x2, cur_y2]).reshape(1, -1), np.array([prev_x1, prev_y1]).reshape(1, -1))
                        xy22 = euclidean_distances(np.array([cur_x2, cur_y2]).reshape(1, -1), np.array([prev_x2, prev_y2]).reshape(1, -1))
                        mmdd_temp_array = np.asarray([xy11,xy12,xy21,xy22])
                        mmdd_loc = np.where(mmdd_temp_array==np.min(mmdd_temp_array))[0][0]
                        if mmdd_loc ==0:
                            ax.plot([cur_x1, prev_x1], [cur_y1, prev_y1], color='black',linestyle=(0, (5, 2, 1, 2)), dash_capstyle='round')
                        if mmdd_loc ==1:
                            ax.plot([cur_x1, prev_x2], [cur_y1, prev_y2], color='black',linestyle=(0, (5, 2, 1, 2)), dash_capstyle='round')
                        if mmdd_loc ==2:
                            ax.plot([cur_x2, prev_x1], [cur_y2, prev_y1], color='black',linestyle=(0, (5, 2, 1, 2)), dash_capstyle='round')
                        if mmdd_loc == 3:
                            ax.plot([cur_x2, prev_x2], [cur_y2, prev_y2], color='black',linestyle=(0, (5, 2, 1, 2)), dash_capstyle='round')
                    if (path_x_start>path_x_end): direction_arrow_orange = -1 #going LEFT
                    if (path_x_start<=path_x_end): direction_arrow_orange = 1  # going RIGHT

                    if (abs(path_x_start- path_x_end) > 2.5):# |(abs(orange_m[path_i, 2] - orange_m[path_i + 1, 1]) > 1)):
                        if (direction_arrow_orange==-1):#& :
                            ax.arrow(orange_GAM_xval[100], yg_orange[100], orange_GAM_xval[99] - orange_GAM_xval[100],
                                 yg_orange[99] - yg_orange[100], shape='full', lw=0,     length_includes_head=True,head_width=0.5, color='dimgray',zorder=3)
                            print('direction arrow', direction_arrow_orange, 'path', path_i, 'to', path_i+1)
                            print('x,y,dx,dy', orange_GAM_xval[100], yg_orange[100], orange_GAM_xval[99] - orange_GAM_xval[100],yg_orange[99] - yg_orange[100])

                        if (direction_arrow_orange==1):#&(abs(orange_m[path_i,0]-orange_m[path_i+1,0])>0.5):
                            ax.arrow(orange_GAM_xval[100], yg_orange[100], orange_GAM_xval[101] - orange_GAM_xval[100],
                                         yg_orange[101] - yg_orange[100], shape='full', lw=0, length_includes_head=True, head_width=0.5,
                                         color='dimgray', zorder=3)
                            print('direction arrow', direction_arrow_orange, 'path', path_i, 'to', path_i + 1)
                prev_x1 = cur_x1
                prev_y1 = cur_y1
                prev_x2= cur_x2
                prev_y2 = cur_y2



            '''
            
                #temp_list.append(orange_sources[i_or])
                #temp_list.append(orange_targets[i_or])
            '''
            #orange_x = orange_m[:,0][np.array(temp_list)]
            #orange_y = orange_m[:, 1][np.array(temp_list)]
            #print('number of x and y', len(orange_x))


        orange_m=pd.DataFrame(orange_m).sort_values(2).values
        print('orange m', orange_m)
        #ax.plot(orange_m[:,0],orange_m[:,1])
        ax.scatter(x_sc, y_sc, color='pink', zorder=3, label=str(ti), s=18)
        ax.text(x_sc + 0.5, y_sc + 0.5, 'TS ' + str(ti), color='black')
        weights = p1.single_cell_bp[:, i]  # /np.sum(p1.single_cell_bp[:,i])
        weights[weights < 0.05] = 0
        weights[np.where(np.asarray(p1.labels) == root_i)[0]] = 0.9
        weights[np.where(np.asarray(p1.labels) == ti)[0]] = 1
        weights[labels[0]] = 1
        loc_z = np.where(weights > 0)[0]
        min_weight = np.min(weights[weights != 0])
        weights[weights == 0] = min_weight * 0.000001
        print('number of zeros in weights', np.sum([weights==0 ]))

        minx = min(x_root[p1.connected_comp_labels[i]], x_sc)  # np.min(x))
        maxx = max(x_root[p1.connected_comp_labels[i]], x_sc)  # np.max(x))
        if minx == x_sc: #the root-cell is on the RHS
            loc_i = np.where(embedding[:, 0] <=maxx)[0]
        else:
            loc_i = np.where(embedding[:, 0] >= minx)[0]
        xp = np.linspace(minx, maxx, len(loc_z))
        print('max sc_pt',max(p1.single_cell_pt_markov))
        xpt = np.linspace(0, max(p1.single_cell_pt_markov), len(loc_z)).reshape((len(loc_z),-1))
        loc_i = np.where((embedding[:, 0] <= maxx) & (embedding[:, 0] >= minx))[0]
        loc_pt= np.where(p1.single_cell_pt_markov<=np.percentile(val_pt, 95)) #doesnt help much
        loc_inter = np.intersect1d(loc_i, loc_z)
        x_val = embedding[loc_i, 0].reshape(len(loc_i), -1)
        print('x-val shape0', x_val.shape)
        #scGam = pg.LinearGAM(n_splines=10, spline_order=3, lam=10).fit(x_val, embedding[loc_i, 1], weights=weights[loc_i].reshape(len(loc_i), -1))
        Xin = np.asarray([p1.single_cell_pt_markov,embedding[:,0]]).T
        Xin = Xin[loc_inter,:]
        print('Xin shape', Xin.shape)
        n_reps = 100
        rep =np.repeat(np.array([[ max(val_pt),x_sc]]),n_reps,axis=0)
        rep = rep+np.random.normal(0,.1,rep.shape)
        Xin = np.concatenate((Xin,rep),axis=0)
        print('Xin shape', Xin.shape, xx_root)
        rep = np.repeat(np.array([[ 0,xx_root]]),n_reps,axis=0) +  np.random.normal(0,.1,rep.shape)
        Xin = np.concatenate((Xin, rep), axis=0)
        print('Xin shape', Xin.shape)
        weights = weights.reshape((embedding.shape[0], 1))[loc_inter, 0]
        weights = weights.reshape((len(loc_inter),1))
        print('weights shape', weights.shape)
        rep = np.repeat(np.array([[1]]), 2*n_reps, axis=0)
        print('rep shape', rep.shape)
        weights = np.concatenate((weights,rep),axis=0)

        print('weights shape', weights.shape, y_sc)
        rep = np.repeat(np.array([y_sc]), n_reps, axis=0)
        print('repy shape', rep.shape)
        yin=embedding[loc_inter,1]
        yin = yin.reshape((-1,1))
        print('yin shape', yin.shape)
        yin = np.concatenate((yin,rep),axis=0)
        print('yin shape', yin.shape)
        rep = np.repeat(np.array([yy_root]), n_reps, axis=0)
        yin = np.concatenate((yin, rep), axis=0)
        print('yin shape', yin.shape)
        Xin_yin = np.concatenate((Xin,yin), axis=1)
        Xin_yin = np.concatenate((Xin_yin, weights), axis=1)
        print('shape Xin_yin', Xin_yin.shape)
        temp_max = max(Xin_yin[:,1])
        temp_min = min(Xin_yin[:, 1])
        print('temp_ranges', temp_min, temp_max)
        n_bins = 20
        bin_width = (temp_max-temp_min)/n_bins
        final_input = np.zeros((1,4))
        print('final input', final_input)
        for i in range(n_bins):
            bin_start = temp_min + bin_width*i
            bin_end= bin_start+bin_width
            temp= Xin_yin[(Xin_yin[:,1]< bin_end)&(Xin_yin[:,1]>= bin_start)]
            print('bin', i, 'has temp shape before trimming', temp.shape)
            if temp.shape[1]>10:
                temp = temp[(temp[:,2]<=(np.mean(temp[:,2])+0.5*np.std(temp[:,2])))&(temp[:,2]>=(np.mean(temp[:,2])-0.5*np.std(temp[:,2])))]
                print('cutoff bin values', np.mean(temp[:, 2]) + np.std(temp[:, 2]))
            print('temp shape after trimming', temp.shape)

            final_input = np.concatenate((final_input, temp), axis=0)

        final_input = final_input[1:,:]
        final_input = final_input[(final_input[:,1]<=maxx) & (final_input[:,1]>=minx) ]
        print('final input', final_input.shape)
        print('final input', final_input)
        scGamx = pg.LinearGAM(n_splines=5, spline_order=3, lam=10).fit(final_input[:,0:2], final_input[:,2], weights=final_input[:,3])#25#fit(Xin, yin, weights=weights)#25
        #scGamx.gridsearch(Xin,  embedding[loc_z,1], weights=weights)
       # scGamy = pg.LinearGAM(n_splines=10, spline_order=3, lam=10).fit(p1.single_cell_pt_markov, embedding[:, 1],
                                                             #weights=weights.reshape(embedding.shape[0], -1))

        ax.scatter(xx_root, yy_root, s=13, c='red')
        ax.text(x_root[p1.connected_comp_labels[ti]]+0.5, y_root[p1.connected_comp_labels[ti]]+0.5, 'root' + str(ti), color='black')
        XX = scGamx.generate_X_grid(term=1, n=500)
        print('xp shape before', xp.shape)
        print('xpt shape before', xpt.shape)
        xp_ = np.concatenate((xpt,xp),axis=1)
        print('xp shape', xp.shape)
        #XX=scGamx.generate_X_grid(term=1, n=500)

        #predsx = scGamx.predict(xp)
        #print('xx shape', XX.shape, XX)
        #print('predsx shape', predsx.shape)
        #predsy = scGamy.predict(predsx)
        Xin = pd.DataFrame(final_input[:,0:2]).sort_values(1).values #Xin
        #print('xin', Xin)# +  np.random.normal(0,.01,Xin.shape)
        yg = scGamx.predict(X=Xin)
        print('yg', yg.shape)
        #ax.plot(Xin[:,1], yg, linewidth=2, c='dimgray')
        Xin = Xin[:,1].reshape((len(yg),-1))
        print('final xin', Xin.shape)
        scGam = pg.LinearGAM(n_splines=5, spline_order=3, lam=1000).gridsearch(Xin, yg)
        XX = scGam.generate_X_grid(term=0, n=500)
        preds = scGam.predict(XX)
        #ax.plot(XX, preds, linewidth=2, c='dimgray')
    return
def get_biased_weights(edgelist, weights, pt, round_no=1):
    #print('weights', type(weights), weights)
    # small nu means less biasing (0.5 is quite mild)
    # larger nu (in our case 1/nu) means more aggressive biasing https://en.wikipedia.org/wiki/Generalised_logistic_function
    print(len(edgelist), len(weights))
    bias_weight = []
    if round_no==1:    b = 1  # 0.5
    else: b=20
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
    #print('loc hi weight', loc_high_weights)
    print('edges of high weight', [edgelist[i] for i in loc_high_weights])
    edgelist_hi = [edgelist[i] for i in loc_high_weights]

    for i in loc_high_weights:
        #print('loc of high weight along edgeweight', i)
        start = edgelist[i][0]
        end = edgelist[i][1]
        #print('start and end node', start, end)
        if (start in loc_high_pt) | (end in loc_high_pt):
            #print("found a high pt high weight node", (start, end), pt[start], pt[end])
            weights[i] = 0.5 * np.mean(weights)

    upper_lim = np.percentile(weights, 90)  # 80
    lower_lim = np.percentile(weights, 10)  # 20
    weights = [i if i <= upper_lim else upper_lim for i in weights]
    weights = [i if i >= lower_lim else lower_lim for i in weights]
    for i, (start, end) in enumerate(edgelist):
        #print('i, start, end', i, start, end)
        Pt_a = pt[start]
        Pt_b = pt[end]
        P_ab = weights[i]
        t_ab = Pt_a - Pt_b

        Bias_ab = K / ((C + math.exp(b * (t_ab + c)))) ** nu
        new_weight = (Bias_ab * P_ab)
        bias_weight.append(new_weight)
        #print('tab', t_ab, 'pab', P_ab, 'biased_pab', new_weight)
    print('original weights', len(weights),list(enumerate(zip(edgelist, weights))))
    print('bias weights', list(enumerate(zip(edgelist, bias_weight))))
    print('length bias weights', len(bias_weight))
    # bias_weight=np.asarray(bias_weight)
    # bias_weight = (bias_weight-np.min(bias_weight)+0.1)/(np.max(bias_weight)-np.min(bias_weight)+0.1)
    return list(bias_weight)





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
def draw_trajectory_gams(X_dimred, sc_supercluster_nn, cluster_labels, super_cluster_labels, super_edgelist, x_lazy, alpha_teleport,
                           projected_sc_pt, true_label, knn,  ncomp,terminal_clusters, super_terminal_clusters,title_str="hitting times", ):
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
    df_mean = df.groupby('cluster', as_index=False).mean()
    sub_cluster_isin_supercluster = df_mean[['cluster','super_cluster']]

    print('sub_cluster_isin_supercluster', sub_cluster_isin_supercluster)
    sub_cluster_isin_supercluster=    sub_cluster_isin_supercluster.sort_values(by='cluster')
    sub_cluster_isin_supercluster['int_supercluster'] = sub_cluster_isin_supercluster['super_cluster'].round(0).astype(int)
    print('sub_cluster_isin_supercluster', sub_cluster_isin_supercluster)
    final_super_terminal = super_terminal_clusters
    #for ti in terminal_clusters:
    #    final_super_terminal.append(sub_cluster_isin_supercluster.loc[sub_cluster_isin_supercluster['cluster']==ti,'int_supercluster'].values[0])
    #final_super_terminal = list(set(final_super_terminal))
    print('final_super_terminal', final_super_terminal)
    df_super_mean = df.groupby('super_cluster', as_index=False).mean()

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
        ax1.scatter(X_dimred[where, 0], X_dimred[where, 1], label=group, c=np.asarray(plt.cm.jet(color)).reshape(-1,4))
    ax1.legend(fontsize=6)
    ax1.set_title('true labels, ncomps:'+str(ncomp)+'. knn:'+str(knn))
    for e_i, (start, end) in enumerate(super_edgelist):

        if pt[start] >= pt[end]:
            temp = end
            end = start
            start = temp
        #print('edges', e_i, start, end, pt[start], pt[end])
        #print('df head', df.head())
        x_i_start = df[df['super_cluster'] == start]['x'].values#groupby('cluster').mean()['x'].values
        y_i_start = df[df['super_cluster'] == start]['y'].values#.groupby('cluster').mean()['y'].values
        x_i_end = df[df['super_cluster'] == end]['x'].values#.groupby('cluster').mean()['x'].values
        y_i_end = df[df['super_cluster'] == end]['y'].values#groupby('cluster').mean()['y'].values
        direction_arrow = 1
        # if np.mean(np.asarray(x_i_end)) < np.mean(np.asarray(x_i_start)): direction_arrow = -1

        super_start_x = X_dimred[sc_supercluster_nn[start],0]#df[df['super_cluster'] == start].mean()['x']
        super_end_x = X_dimred[sc_supercluster_nn[end],0]#df[df['super_cluster'] == end].mean()['x']
        super_start_y = X_dimred[sc_supercluster_nn[start],1]#df[df['super_cluster'] == start].mean()['y']
        super_end_y = X_dimred[sc_supercluster_nn[end],1]#df[df['super_cluster'] == end].mean()['y']

        if super_start_x > super_end_x: direction_arrow = -1
        ext_maxx = False
        minx = min(super_start_x, super_end_x)
        maxx = max(super_start_x, super_end_x)

        miny = min(super_start_y, super_end_y)
        maxy = max(super_start_y, super_end_y)

        x_val = np.concatenate([x_i_start, x_i_end])
        y_val = np.concatenate([y_i_start, y_i_end])

        idx_keep = np.where((x_val <= maxx) & (x_val >= minx))[0]#np.where((X_dimred[:,0]<=maxx) & (X_dimred[:,0]>=minx))#
        idy_keep =np.where((y_val <= maxy) & (y_val >= miny))[0] #np.where((X_dimred[:,1]<=maxy) & (X_dimred[:,1]>=miny))#

        idx_keep = np.intersect1d(idy_keep, idx_keep)

        x_val = x_val[idx_keep]#X_dimred[idx_keep,0]#
        y_val = y_val[idx_keep]#X_dimred[idx_keep,1]# y_val[idx_keep]
        print('start and end', start, '' , end)


        super_mid_x = (super_start_x + super_end_x) / 2
        super_mid_y = (super_start_y + super_end_y) / 2
        from scipy.spatial import distance

        # x_val = np.concatenate([x_i_start, x_i_end])
        #print('abs', abs(minx - maxx))
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



        for i in range(straight_level):  # DO THE SAME FOR A MIDPOINT TOO
            y_super = np.concatenate([y_super, y_super])
            x_super = np.concatenate([x_super, x_super])

        y_super_max = max(y_super)
        y_super_min = min(y_super)

        #print('xval', x_val, 'start/end', start, end)
        #print('yval', y_val, 'start/end', start, end)
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
        print('yval stats', np.max(y_val), maxy)
        print('yval stats', np.min(y_val), miny)
        #z = np.polyfit(x_val, y_val, 2)
        x_val = x_val.reshape((len(x_val), -1))
        y_val = y_val.reshape((len(y_val), -1))
        xp = np.linspace(minx, maxx, 500)
        #p = np.poly1d(z)
        gam50 = pg.LinearGAM(n_splines=4, spline_order=3, lam=10).gridsearch(x_val,y_val)
        #pg.ExpectileGAM(expectile=0.5, lam=.6).gridsearch(x_val, y_val)#pg.LinearGAM().fit(x_val, y_val)
        XX = gam50.generate_X_grid(term=0, n=500)

        preds = gam50.predict(XX)
        print('preds', preds.shape)
        #XX = gam50.generate_X_grid(term=0, n=500)
        #preds = gam.predict(xp)
        #print('gam returns', preds)
        #smooth = p(xp)
        if ext_maxx == False:
            idx_keep = np.where((xp <= (maxx)) & (xp >= (minx)))[0]  # minx+3
        else:
            idx_keep = np.where((xp <= (maxx)) & (xp >= (minx)))[0]  # maxx-3
        #ax2.plot(xp[idx_keep], preds[idx_keep], linewidth=3, c='black')
        cc = ['black','red','blue','yellow','pink'][ random.randint(0,4)]
        ax2.plot(XX, preds, linewidth=2, c='dimgray')
        #print('just drew this edge', start, end, 'This is the', e_i, 'th edge')
        # ax3.plot(xp[idx_keep], smooth[idx_keep], linewidth=3, c='black')
        med_loc = np.where(xp == np.median(xp[idx_keep]))[0]
        mean_temp = np.mean(xp[idx_keep])
        closest_val = xp[idx_keep][0]
        closest_loc = idx_keep[0]

        for i, xp_val in enumerate(xp[idx_keep]):

            if abs(xp_val - mean_temp) < abs(closest_val - mean_temp):
                closest_val = xp_val
                closest_loc = idx_keep[i]
        step = 1
        if direction_arrow == 1: #smooth instead of preds
            ax2.arrow(xp[closest_loc], preds[closest_loc], xp[closest_loc + step] - xp[closest_loc],
                      preds[closest_loc + step] - preds[closest_loc], shape='full', lw=0, length_includes_head=True,
                      head_width=.5, color='dimgray')  # , head_starts_at_zero = direction_arrow )

        else:
            ax2.arrow(xp[closest_loc], preds[closest_loc], xp[closest_loc - step] - xp[closest_loc],
                      preds[closest_loc - step] - preds[closest_loc], shape='full', lw=0, length_includes_head=True,
                      head_width=.5, color='dimgray')


    x_cluster = df_mean['x']
    y_cluster = df_mean['y']


    num_parc_group = len(set(cluster_labels))

    c_edge = []
    width_edge = []
    for i in range(num_parc_group):
        if i in final_super_terminal:
            width_edge.append(2.5)
            c_edge.append('yellow')
        else:
            width_edge.append(0)
            c_edge.append('black')


    ax2.scatter(x_cluster, y_cluster, c='red')

    for i, type in enumerate(pt_str):
        ax2.text(df_super_mean['x'][i], df_super_mean['y'][i], 'C'+str(i), weight='bold')

    for i in range(len(x_cluster)):
        ax2.text(x_cluster[i], y_cluster[i], 'c' + str(i))
    ax2.set_title('lazy:' + str(x_lazy) + ' teleport' + str(alpha_teleport) + 'super_knn:' + str(knn))
    # ax2.set_title('super_knn:' + str(knn) )
    ax2.scatter(X_dimred[:, 0], X_dimred[:, 1], c=projected_sc_pt, cmap='viridis_r', alpha=0.5)
    #ax2.scatter(df_super_mean['x'], df_super_mean['y'], c='black', s=60, edgecolors = c_edge, linewidth = width_edge)
    for i,c,w in zip(sc_supercluster_nn, c_edge, width_edge):
        ax2.scatter(X_dimred[i,0], X_dimred[i,1], c='black', s=60, edgecolors=c, linewidth=w)
    plt.title(title_str)

    return


def draw_trajectory_dimred(X_dimred, sc_supercluster_nn, cluster_labels, super_cluster_labels, super_edgelist, x_lazy, alpha_teleport,
                           projected_sc_pt, true_label, knn,  ncomp,terminal_clusters, super_terminal_clusters,title_str="hitting times", ):
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
    df_mean = df.groupby('cluster', as_index=False).mean()
    sub_cluster_isin_supercluster = df_mean[['cluster','super_cluster']]
    print('sub_cluster_isin_supercluster', sub_cluster_isin_supercluster)
    sub_cluster_isin_supercluster=    sub_cluster_isin_supercluster.sort_values(by='cluster')
    sub_cluster_isin_supercluster['int_supercluster'] = sub_cluster_isin_supercluster['super_cluster'].round(1).astype(int)
    print('sub_cluster_isin_supercluster', sub_cluster_isin_supercluster)
    final_super_terminal = super_terminal_clusters
    #for ti in terminal_clusters:
    #    final_super_terminal.append(sub_cluster_isin_supercluster.loc[sub_cluster_isin_supercluster['cluster']==ti,'int_supercluster'].values[0])
    #final_super_terminal = list(set(final_super_terminal))
    print('final_super_terminal', final_super_terminal)
    df_super_mean = df.groupby('super_cluster', as_index=False).mean()

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
        ax1.scatter(X_dimred[where, 0], X_dimred[where, 1], label=group, c=np.asarray(plt.cm.jet(color)).reshape(-1,4))
    ax1.legend(fontsize=6)
    ax1.set_title('true labels, ncomps:'+str(ncomp)+'. knn:'+str(knn))
    for e_i, (start, end) in enumerate(super_edgelist):

        if pt[start] >= pt[end]:
            temp = end
            end = start
            start = temp
        #print('edges', e_i, start, end, pt[start], pt[end])
        #print('df head', df.head())
        x_i_start = df[df['super_cluster'] == start].groupby('cluster').mean()['x'].values
        y_i_start = df[df['super_cluster'] == start].groupby('cluster').mean()['y'].values
        x_i_end = df[df['super_cluster'] == end].groupby('cluster').mean()['x'].values
        y_i_end = df[df['super_cluster'] == end].groupby('cluster').mean()['y'].values
        direction_arrow = 1
        # if np.mean(np.asarray(x_i_end)) < np.mean(np.asarray(x_i_start)): direction_arrow = -1

        #super_start_x = df[df['super_cluster'] == start].mean()['x']
        #super_end_x = df[df['super_cluster'] == end].mean()['x']
        #super_start_y = df[df['super_cluster'] == start].mean()['y']
        #super_end_y = df[df['super_cluster'] == end].mean()['y']

        super_start_x = X_dimred[sc_supercluster_nn[start], 0]  # df[df['super_cluster'] == start].mean()['x']
        super_end_x = X_dimred[sc_supercluster_nn[end], 0]  # df[df['super_cluster'] == end].mean()['x']
        super_start_y = X_dimred[sc_supercluster_nn[start], 1]  # df[df['super_cluster'] == start].mean()['y']
        super_end_y = X_dimred[sc_supercluster_nn[end], 1]  # df[df['super_cluster'] == end].mean()['y']

        if super_start_x > super_end_x: direction_arrow = -1
        ext_maxx = False
        minx = min(super_start_x, super_end_x)
        maxx = max(super_start_x, super_end_x)

        miny = min(super_start_y, super_end_y)
        maxy = max(super_start_y, super_end_y)


        x_val = np.concatenate([x_i_start, x_i_end])
        y_val = np.concatenate([y_i_start, y_i_end])

        idx_keep = np.where((x_val <= maxx) & (x_val >= minx))[0]
        idy_keep = np.where((y_val <= maxy) & (y_val >= miny))[0]
        print('len x-val before intersect', len(x_val))
        idx_keep = np.intersect1d(idy_keep, idx_keep)
        x_val = x_val[idx_keep]
        y_val = y_val[idx_keep]

        super_mid_x = (super_start_x + super_end_x) / 2
        super_mid_y = (super_start_y + super_end_y) / 2
        from scipy.spatial import distance

        # x_val = np.concatenate([x_i_start, x_i_end])
        #print('abs', abs(minx - maxx))
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



        for i in range(straight_level):  # DO THE SAME FOR A MIDPOINT TOO
            y_super = np.concatenate([y_super, y_super])
            x_super = np.concatenate([x_super, x_super])

        y_super_max = max(y_super)
        y_super_min = min(y_super)

        #print('xval', x_val, 'start/end', start, end)
        #print('yval', y_val, 'start/end', start, end)
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
        #print('just drew this edge', start, end, 'This is the', e_i, 'th edge')
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
        if direction_arrow == 1: #smooth instead of preds
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
    c_edge = []
    width_edge = []
    for i in range(num_parc_group):
        if i in final_super_terminal:
            width_edge.append(2.5)
            c_edge.append('yellow')
        else:
            width_edge.append(0)
            c_edge.append('black')


    ax2.scatter(x_cluster, y_cluster, c='red')

    for i, type in enumerate(pt_str):
        ax2.text(df_super_mean['x'][i], df_super_mean['y'][i], 'C'+str(i), weight='bold')

    for i in range(len(x_cluster)):
        ax2.text(x_cluster[i], y_cluster[i], pt_sub[i] + 'c' + str(i))
    ax2.set_title('lazy:' + str(x_lazy) + ' teleport' + str(alpha_teleport) + 'super_knn:' + str(knn))
    # ax2.set_title('super_knn:' + str(knn) )
    ax2.scatter(X_dimred[:, 0], X_dimred[:, 1], c=projected_sc_pt, cmap='viridis_r', alpha=0.5)
    ax2.scatter(df_super_mean['x'], df_super_mean['y'], c='black', s=60, edgecolors = c_edge, linewidth = width_edge)
    plt.title(title_str)

    return
def csr_mst(adjacency_matrix):

    Tcsr = adjacency_matrix.copy()
    Tcsr.data = -1 * Tcsr.data
    Tcsr.data = Tcsr.data - np.min(Tcsr.data)
    Tcsr.data = Tcsr.data + 1
    Tcsr = minimum_spanning_tree(Tcsr)  # adjacency_matrix)
    Tcsr = (Tcsr + Tcsr.T) * 0.5
    return Tcsr
def connect_all_components(Tcsr, cluster_graph_csr, adjacency_matrix):
    n_components, comp_labels = connected_components(csgraph=cluster_graph_csr, directed=False, return_labels=True)

    Td = Tcsr  # .todense()
    # Td[Td==0]=999.999
    while n_components > 1:
        sub_td = Td[comp_labels == 0, :][:, comp_labels != 0]
        print(min(sub_td.data))
        # locxy=np.where(Td ==np.min(sub_td.data))
        locxy = scipy.sparse.find(Td == np.min(sub_td.data))
        for i in range(len(locxy[0])):
            if (comp_labels[locxy[0][i]] == 0) & (comp_labels[locxy[1][i]] != 0):
                x = locxy[0][i]
                y = locxy[1][i]
        minval = adjacency_matrix[
            x, y]  # np.min(Td[comp_labels==0,:][:,comp_labels!=0])#np.min(Td[np.where(np.asarray(comp_labels)==0),np.where(comp_labels!=0)])
        print('inside reconnecting components', x, y, minval)
        cluster_graph_csr[x, y] = minval

        n_components, comp_labels = connected_components(csgraph=cluster_graph_csr, directed=False, return_labels=True)
        print('number of connected componnents after reconnecting ', n_components)
    return cluster_graph_csr
    '''
    Td = Tcsr.todense()
    Td[Td == 0] = 999.999
    while n_components > 1:
        sub_td = Td[comp_labels == 0, :][:, comp_labels != 0]
        locxy = np.where(Td == np.min(sub_td))
        for i in range(len(locxy[0])):
            if (comp_labels[locxy[0][i]] == 0) & (comp_labels[locxy[1][i]] != 0):
                x = locxy[0][i]
                y = locxy[1][i]
        minval = adjacency_matrix[x, y]  # np.min(Td[comp_labels==0,:][:,comp_labels!=0])#np.min(Td[np.where(np.asarray(comp_labels)==0),np.where(comp_labels!=0)])
        print('inside reconnecting components', x, y, minval)
        cluster_graph_csr[x, y] = minval
        n_components, comp_labels = connected_components(csgraph=cluster_graph_csr, directed=False,
                                                         return_labels=True)
        print('number of connected componnents after reconnecting ', n_components)
    '''


def local_pruning_clustergraph_mst(adjacency_matrix,  global_pruning_std=1, max_outgoing=30, preserve_disconnected = False, visual = False):
    # larger pruning_std factor means less pruning
    #the mst is only used to reconnect components that become disconnect due to pruning
    from scipy.sparse.csgraph import minimum_spanning_tree
    #Tcsr = adjacency_matrix.copy()

    Tcsr = csr_mst(adjacency_matrix)



    sources, targets = adjacency_matrix.nonzero()
    #original_edgelist = list(zip(sources, targets))

    initial_links_n = len(adjacency_matrix.data)
    #print('initial links n', adjacency_matrix, initial_links_n)
    adjacency_matrix = scipy.sparse.csr_matrix.todense(adjacency_matrix)
    n_components_0, comp_labels_0 = connected_components(csgraph=adjacency_matrix, directed=False, return_labels=True)
    print('number of components before pruning', n_components_0, comp_labels_0)
    #print('adjacency')
    # print(type(adjacency_matrix))
    row_list = []
    col_list = []
    weight_list = []
    neighbor_array = adjacency_matrix  # not listed in in any order of proximity

    n_cells = neighbor_array.shape[0]
    rowi = 0

    for i in range(neighbor_array.shape[0]):
        row = np.asarray(neighbor_array[i, :]).flatten()
        #print('row, row')
        n_nonz = np.sum(row>0)
        #print('n nonzero 1', n_nonz)
        n_nonz=min(n_nonz,max_outgoing)

        to_keep_index = np.argsort(row)[::-1][0:n_nonz]
        #print('to keep', to_keep_index)
        updated_nn_weights = list(row[to_keep_index])
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
    #print('mask is', mask)
    cluster_graph_csr.data = cluster_graph_csr.data / (np.std(cluster_graph_csr.data)) #normalize
    threshold_global = np.mean(cluster_graph_csr.data) - global_pruning_std* np.std(cluster_graph_csr.data)
    mask |= (cluster_graph_csr.data < (threshold_global))  # smaller Jaccard weight means weaker edge

    cluster_graph_csr.data[mask] = 0
    cluster_graph_csr.eliminate_zeros()
    print('shape of cluster graph', cluster_graph_csr.shape)
    #sources, targets = cluster_graph_csr.nonzero()
    n_components, comp_labels = connected_components(csgraph=cluster_graph_csr, directed=False, return_labels=True)
    print('number of connected components after pruning', n_components)
    if (preserve_disconnected ==True) & ( n_components>n_components_0): #preserve initial disconnected components
        Td = Tcsr#.todense()
        Td[Td == 0] = 999.999
        n_components_ = n_components
        while n_components_>n_components_0:
            for i in range(n_components_0):
                loc_x = np.where(comp_labels_0==i)[0]

                len_i = len(set(comp_labels[loc_x]))
                print('locx', loc_x, len_i)

                while len_i >1:
                    s = list(set(comp_labels[loc_x]))
                    loc_notxx = np.intersect1d(loc_x, np.where((comp_labels != s[0]))[0])
                    print('loc_notx', loc_notxx)
                    loc_xx = np.intersect1d(loc_x, np.where((comp_labels == s[0]))[0])
                    sub_td = Td[loc_xx,:][ :, loc_notxx]
                    print('subtd', np.min(sub_td))
                    locxy = np.where(Td == np.min(sub_td))
                    print('locxy and min', locxy, sub_td)
                    for i in range(len(locxy[0])):
                        if (comp_labels[locxy[0][i]]!=comp_labels[locxy[1][i]]):
                            x = locxy[0][i]
                            y= locxy[1][i]
                    minval = adjacency_matrix[x, y]
                    print('inside reconnecting components while preserving original ', x, y, minval)
                    cluster_graph_csr[x, y] = minval
                    n_components_, comp_labels = connected_components(csgraph=cluster_graph_csr, directed=False,
                                                                     return_labels=True)
                    loc_x = np.where(comp_labels_0 == i)[0]
                    len_i = len(set(comp_labels[loc_x]))
                print('number of connected componnents after reconnecting ', n_components_)
    if (n_components>1) & (preserve_disconnected==False):
        #Tcsr.data = Tcsr.data / (np.std(Tcsr.data))
        #Tcsr.data = 1/Tcsr.data
        cluster_graph_csr=connect_all_components(Tcsr, cluster_graph_csr, adjacency_matrix)
        n_components, comp_labels = connected_components(csgraph=cluster_graph_csr, directed=False, return_labels=True)
        '''
        Td = Tcsr#.todense()
        #Td[Td==0]=999.999
        while n_components >1:
            sub_td = Td[comp_labels==0,:][:,comp_labels!=0]
            print(min(sub_td.data))
            #locxy=np.where(Td ==np.min(sub_td.data))
            locxy = scipy.sparse.find(Td == np.min(sub_td.data))
            for i in range(len(locxy[0])):
                if (comp_labels[locxy[0][i]]==0) &(comp_labels[locxy[1][i]]!=0):
                    x = locxy[0][i]
                    y=locxy[1][i]
            minval=adjacency_matrix[x,y]#np.min(Td[comp_labels==0,:][:,comp_labels!=0])#np.min(Td[np.where(np.asarray(comp_labels)==0),np.where(comp_labels!=0)])
            print('inside reconnecting components', x,y,minval)
            cluster_graph_csr[x,y] = minval

            n_components, comp_labels = connected_components(csgraph=cluster_graph_csr, directed=False, return_labels=True)
            print('number of connected componnents after reconnecting ', n_components)
        '''
    sources, targets = cluster_graph_csr.nonzero()
    edgelist = list(zip(sources, targets))
    if global_pruning_std<0.5:
        print('edgelist after local and global pruning', len(edgelist))
    # cluster_graph_csr.data = locallytrimmed_sparse_vc.data / (np.std(locallytrimmed_sparse_vc.data))
    edgeweights = cluster_graph_csr.data / (np.std(cluster_graph_csr.data))

    trimmed_n = (initial_links_n - final_links_n) * 100 / initial_links_n
    trimmed_n_glob = (initial_links_n - len(edgeweights)) / initial_links_n
    if global_pruning_std < 0.5:
        print("percentage links trimmed from local pruning relative to start", trimmed_n)
        print("percentage links trimmed from global pruning relative to start", trimmed_n_glob)
    return edgeweights, edgelist, comp_labels
def local_pruning_clustergraph(adjacency_matrix, local_pruning_std=0.0, global_pruning_std=1, max_outgoing=30):
    # larger pruning_std factor means less pruning


    sources, targets = adjacency_matrix.nonzero()
    original_edgelist = list(zip(sources, targets))

    initial_links_n = len(adjacency_matrix.data)
    #print('initial links n', adjacency_matrix, initial_links_n)
    adjacency_matrix = scipy.sparse.csr_matrix.todense(adjacency_matrix)
    #print('adjacency')
    # print(type(adjacency_matrix))
    row_list = []
    col_list = []
    weight_list = []
    neighbor_array = adjacency_matrix  # not listed in in any order of proximity

    n_cells = neighbor_array.shape[0]
    rowi = 0

    for i in range(neighbor_array.shape[0]):
        row = np.asarray(neighbor_array[i, :]).flatten()
        #print('row, row')
        n_nonz = np.sum(row>0)
        #print('n nonzero 1', n_nonz)
        n_nonz=min(n_nonz,max_outgoing)

        to_keep_index = np.argsort(row)[::-1][0:n_nonz]
        #print('to keep', to_keep_index)
        updated_nn_weights = list(row[to_keep_index])
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
    #print('mask is', mask)
    cluster_graph_csr.data = cluster_graph_csr.data / (np.std(cluster_graph_csr.data)) #normalize
    threshold_global = np.mean(cluster_graph_csr.data) - global_pruning_std* np.std(cluster_graph_csr.data)
    mask |= (cluster_graph_csr.data < (threshold_global))  # smaller Jaccard weight means weaker edge

    cluster_graph_csr.data[mask] = 0
    cluster_graph_csr.eliminate_zeros()
    sources, targets = cluster_graph_csr.nonzero()
    edgelist = list(zip(sources, targets))
    if global_pruning_std<0.5:
        print('edgelist after local and global pruning', edgelist)


    # cluster_graph_csr.data = locallytrimmed_sparse_vc.data / (np.std(locallytrimmed_sparse_vc.data))
    edgeweights = cluster_graph_csr.data / (np.std(cluster_graph_csr.data))

    trimmed_n = (initial_links_n - final_links_n) * 100 / initial_links_n
    trimmed_n_glob = (initial_links_n - len(edgeweights)) / initial_links_n
    if global_pruning_std < 0.5:
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
                 too_big_factor=0.4, small_pop=10, jac_weighted_edges=True, knn=30, n_iter_leiden=5, random_seed=42,
                 num_threads=-1, distance='l2', time_smallpop=15, pseudotime=False,
                 root=0, path='/home/shobi/Trajectory/', super_cluster_labels=False,
                 super_node_degree_list=False, super_terminal_cells = False, x_lazy=0.95, alpha_teleport=0.99, root_str="root_cluster", preserve_disconnected = False, humanCD34=False):
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
        self.random_seed = random_seed  # enable reproducible Leiden clustering
        self.num_threads = num_threads  # number of threads used in KNN search/construction
        self.distance = distance  # Euclidean distance 'l2' by default; other options 'ip' and 'cosine'
        self.time_smallpop = time_smallpop
        self.pseudotime = pseudotime
        self.root = root
        self.path = path

        self.super_cluster_labels = super_cluster_labels
        self.super_node_degree_list = super_node_degree_list
        self.super_terminal_cells = super_terminal_cells
        self.x_lazy = x_lazy  # 1-x = probability of staying in same node
        self.alpha_teleport = alpha_teleport  # 1-alpha is probability of jumping
        self.root_str = root_str
        self.preserve_disconnected = preserve_disconnected
        self.humanCD34 = humanCD34

    def get_Q_transient_transition(self, sources, targets, bias_weights, absorbing_clusters):
        return

    def get_R_absorbing_transition(self, sources, targets, bias_weights, absorbing_clusters):
        return


    def get_terminal_clusters(self, A , markov_pt):

        pop_list= []
        out_list = []
        print('get terminal', set(self.labels),np.where(self.labels ==0))
        for i in list(set(self.labels)):
            pop_list.append(len(np.where(self.labels ==i)[0]))
        #print(pop_list)
        A_new = A.copy()
        for i in range(A.shape[0]):
            for j in range(A.shape[0]):
                A_new[i,j]=A[i,j]*(pop_list[i]+pop_list[j])/(pop_list[i]*pop_list[j])
        out_deg = A_new.sum(axis=1)
        #for pi, item in enumerate(out_deg):
        #    out_list.append(item/pop_list[i])
        out_deg = np.asarray(out_deg)
        print('out deg',  out_deg)
        n_ = A.shape[0]
        if n_<=10:
            loc_deg = np.where(out_deg<=np.percentile(out_deg,50))[0]
            print('low deg super', loc_deg)
            loc_pt =  np.where(markov_pt>=np.percentile(markov_pt,10))[0] #60 Ttoy
            print('high pt super', loc_pt)
        if (n_<=30) & (n_>10):
            loc_deg = np.where(out_deg<=np.percentile(out_deg,50))[0]#30 for Toy
            print('low deg super', loc_deg)
            loc_pt =  np.where(markov_pt>=np.percentile(markov_pt,20))[0] #60 Toy
            print('high pt super', loc_pt)
        if n_>30:
            loc_deg = np.where(out_deg <= np.percentile(out_deg, 25))[0] #15 Toy
            print('low deg', loc_deg)
            loc_pt = np.where(markov_pt >= np.percentile(markov_pt,30))[0] #60Toy
            print('high pt', loc_pt)
        terminal_clusters = list(set(loc_deg)&set(loc_pt))
        terminal_org = terminal_clusters.copy()
        print('original terminal_clusters', terminal_org)
        for terminal_i in terminal_org:
            #print('terminal state', terminal_i)
            count_nn = 0
            neigh_terminal = np.where(A[:, terminal_i] > 0)[0]
            if neigh_terminal.size > 0:
                for item in neigh_terminal:
                    #print('terminal state', terminal_i)
                    if item in terminal_clusters:
                        count_nn = count_nn+1
                if count_nn >=4:
                    terminal_clusters.remove(terminal_i)
                    print('terminal state', terminal_i,'had 4 or more neighboring terminal states')

        print('terminal_clusters', terminal_clusters)
        return terminal_clusters

    def make_absorbing_markov(self,A, pt):
        # cluster_graph is the vertex_cluster_graph made of sub_clusters in the finer iteration of PARC
        # pt is the pseudotime of each cluster in the graph
        absorbing_clusters =self.terminal_clusters
        n_s = len(absorbing_clusters)

        sources, targets =A
        sources, targets, bias_weights = get_biased_weights(sources, targets, weights, pt)
        Q = get_Q_transient_transition(sources, targets, bias_weights, absorbing_clusters)
        R = get_R_absorbing_transition(sources, targets, bias_weights, absorbing_clusters)
        n_t = Q.shape[0]
        I_t = np.identity(n_t)
        N = np.inv(I_t - Q)
        P_transition_absorbing = np.concatenate()  # put together Q, R, Is and 0s
        return Q, R, N, P_transition_absorbing
    def path_length_onbias(self, bias_edgelist, bias_edgeweights):
        #print('pre-dijkstra, but biased using original walk',list(zip(bias_edgeweights, bias_edgelist)))
        bias_edgeweights = 1 / np.asarray(bias_edgeweights)
        #print('dijkstra edge weights', list(zip(bias_edgeweights, bias_edgelist)))
        #print('dijkstra edge weights', list(zip(bias_edgeweights, bias_edgelist)))

        bias_g = ig.Graph(list(bias_edgelist), edge_attrs={'weight': list(bias_edgeweights)})
        paths = bias_g.shortest_paths_dijkstra(source=self.root, weights='weight')
        #print('paths ', paths)
        paths = np.asarray(paths)
        paths[paths>np.percentile(paths,95)] = np.mean(paths[paths<np.percentile(paths,95)])+np.std(paths[paths<np.percentile(paths,95)])
        return paths[0]

    def compute_hitting_time(self, sparse_graph, root, x_lazy, alpha_teleport, number_eig=0):
        # 1- alpha is the probabilty of teleporting
        # 1- x_lazy is the probability of staying in current state (be lazy)



        beta_teleport = 2 * (1 - alpha_teleport) / (2 - alpha_teleport)
        N = sparse_graph.shape[0]
        #print('adjacency in compute hitting', sparse_graph)
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
        #        print('deg matrix in compute hitting', deg)
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
        #print('stationary diagonal', S_diag)
        #print('degree diagonal', D_diag)
        L_dir = D_diag - A_bias.T
        print("L_eul_sym is symmetric", (L_dir == L_dir.T).all())
        L_eul = L_dir.dot(D_diag_recip)
        print("L_eul_sym is symmetric", (L_eul == L_eul.T).all())
        L_eul = L_eul.dot(S_diag)
        L_eul_sym = 0.5 * (L_eul + L_eul.T)
        print("L_eul_sym is symmetric", (L_eul_sym == L_eul_sym.T).all())
        return L_eul

    def pagerank_compute(self, P_bias, max_iterations=200):
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
            #old = p0.copy()
            p0 = p0.dot(P_bias)
            #delta = p0 - old
            #delta = math.sqrt(delta.dot(delta.T))

        p0 = p0[0] / np.sum(p0[0])
        #print('p0 stationary is', [('c' + str(i), pp0) for i, pp0 in enumerate(p0)])
        #print([('c' + str(i), pp0) for i, pp0 in enumerate(p0) if pp0>np.mean(p0)])
        upperlim = np.percentile(p0, 90)
        lowerlim = np.percentile(p0, 10)

        # upper_val = p0[p0 >upperlim]
        # upperlim = np.mean(upper_val)
        #print('upper lim', upperlim)
        if self.too_big_factor < 0.3:
            p0 = np.array([d if d <= upperlim else upperlim for d in p0])
            p0 = p0 / np.sum(p0)
        print('final stationary', [(i, pp0) for i, pp0 in enumerate(p0)])
        return p0
    def prob_reaching_terminal_state(self,terminal_state, all_terminal_states, A, root, pt,num_sim=300):
        print('root', root)
        print('terminal state target', terminal_state)
        n_states = A.shape[0]
        n_components, labels = connected_components(csgraph=csr_matrix(A),directed=False)
        print('the probability of Reaching matrix has', n_components,'connected components')
        A = A/(np.max(A))
        #A[A<=0.05]=0
        jj=0
        for row in A:
            if np.all(row==0): A[jj,jj]=1
            jj=jj+1

        P = A / A.sum(axis=1).reshape((n_states, 1))
        n_steps = int( 2* n_states)
        currentState = root
        state = np.zeros((1, n_states))
        state[0, currentState] = 1
        currentState = root
        state = np.zeros((1, n_states))
        state[0, currentState] = 1
        state_root = state.copy()
        neigh_terminal = np.where(A[:,terminal_state]>0)[0]
        non_nn_terminal_state = []
        for ts_i in all_terminal_states:
            if pt[ts_i] > pt[terminal_state]: non_nn_terminal_state.append(ts_i)

        for ts_i in all_terminal_states:
            if np.all(neigh_terminal!=ts_i): non_nn_terminal_state.append(ts_i)
            #print(ts_i, 'is a non-neighbor terminal state to the target terminal', terminal_state)

        cumstateChangeHist = np.zeros((1, n_states))
        cumstateChangeHist_all = np.zeros((1, n_states))
        count_reach_terminal_state = 0
        for i in range(num_sim):
            #distr_hist = [[0 for i in range(n_states)]]
            stateChangeHist = np.zeros((n_states, n_states))
            stateChangeHist[root, root] = 1
            state = state_root
            currentState = root
            stateHist = state
            terminal_state_found = False
            non_neighbor_terminal_state_reached = False
            #print('root', root)
            #print('terminal state target', terminal_state)

            x = 0
            while (x < n_steps) & ((terminal_state_found == False)):# & (non_neighbor_terminal_state_reached == False)):
                currentRow = np.ma.masked_values((P[currentState]), 0.0)
                nextState = simulate_multinomial(currentRow)
                #print('next state', nextState)
                if nextState == terminal_state:
                    terminal_state_found = True
                    #print('terminal state found at step', x)
                #if nextState in non_nn_terminal_state:
                    #non_neighbor_terminal_state_reached = True
                # Keep track of state changes
                stateChangeHist[currentState, nextState] += 1
                # Keep track of the state vector itself
                state = np.zeros((1, n_states))
                state[0, nextState] = 1.0
                # Keep track of state history
                stateHist = np.append(stateHist, state, axis=0)
                currentState = nextState
                x = x + 1
                    # calculate the actual distribution over the 3 states so far

                #print('stateChangeHist of sim num', i)
                #print(stateChangeHist)
                #print('Boolean stateChangeHist of sim num', i, )
                #print(stateChangeHist > 0)
                #print('cumStateChangeHist of sim num', i)
            if (terminal_state_found == True) :
                cumstateChangeHist = cumstateChangeHist + np.any(
                stateChangeHist > 0, axis=0)
                count_reach_terminal_state = count_reach_terminal_state+1
            cumstateChangeHist_all = cumstateChangeHist_all + np.any(
                    stateChangeHist > 0, axis=0)
                 #avoid division by zero on states that were never reached (e.g. terminal states that come after the target terminal state)
            #print('cumstateChangeHist')
            #print(cumstateChangeHist)
            #print(cumstateChangeHist_all)
        cumstateChangeHist_all[cumstateChangeHist_all == 0] = 1
        prob_ = cumstateChangeHist / cumstateChangeHist_all

        np.set_printoptions(precision=3)
        print('number of times Terminal state', terminal_state,'is found:', count_reach_terminal_state)
        print('prob', prob_)
        if count_reach_terminal_state==0:
            prob_[:,terminal_state]=0
        else:
            prob_[0,terminal_state]=0 #starting at the root, index=0
            prob_ = prob_/np.max(prob_)
        prob_[0, terminal_state]=1
        #for i in range(n_states):
            #print('cluster ', i, 'has likelihood', prob_[0,i])

        return prob_.tolist()[0]
    def directed_laplacian2(self, A_bias):
        #print('old Abias', A_bias)
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
            # print('shape teleport edge', teleport_edge.shape, teleport_edge)
            # print('old edge', np.sum(A_bias[i, :]))
            A_bias[i, :] = A_bias[i, :] + teleport_edge
            # print('new edge', np.sum(A_bias[i, :]))
        # print('new A bias ', A_bias.shape, A_bias)

        deg_val = A_bias.sum(axis=1)

        # print('new Abias', A_bias)

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
        hitting_array = hitting_array.flatten()
        return hitting_array / 1000, []
    def simulate_markov_sub(self, A, num_sim, hitting_array,q,root):
        n_states = A.shape[0]
        P = A / A.sum(axis=1).reshape((n_states, 1))
        #hitting_array = np.ones((P.shape[0], 1)) * 1000
        hitting_array_temp = np.zeros((P.shape[0], 1)).astype('float64')
        n_steps = int(2 * n_states)
        hitting_array_final = np.zeros((1, n_states))
        currentState = root

        print('root is', root)
        state = np.zeros((1, n_states))
        state[0, currentState] = 1
        state_root = state.copy()
        for i in range(num_sim):
            dist_list=[]
            # print(i, 'th simulation in Markov')
            #if i % 10 == 0: print(i, 'th simulation in Markov', time.ctime())
            state = state_root
            currentState = root
            stateHist = state
            for x in range(n_steps):
                currentRow = np.ma.masked_values((P[currentState]), 0.0)
                nextState = simulate_multinomial(currentRow)
                dist = A[currentState,nextState]

                dist = ( 1 / ((1 + math.exp((dist- 1)))))

                dist_list.append(dist)
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
                # totals = np.sum(stateHist, axis=0)
                # gt = np.sum(totals)
                # distrib = totals / gt
                # distrib = np.reshape(distrib, (1, n_states))
                # distr_hist = np.append(distr_hist, distrib, axis=0)
            for state_i in range(P.shape[0]):
                # print('first reach state', state_i, 'at step', np.where(stateHist[:, state_i] == 1)[0][0])
                first_time_at_statei = np.where(stateHist[:, state_i] == 1)[0]
                if len(first_time_at_statei) == 0:
                    # print('did not reach state', state_i,'setting dummy path length')
                    hitting_array_temp[state_i, 0] = n_steps + 1
                else:
                    total_dist = 0
                    for ff in range(first_time_at_statei[0]):
                        total_dist = dist_list[ff] + total_dist

                    hitting_array_temp[state_i, 0] = total_dist#first_time_at_statei[0]

            # hitting_array_temp[hitting_array_temp==(n_steps+1)] = np.mean(hitting_array_temp[hitting_array_temp!=n_steps+1])

            hitting_array = np.append(hitting_array, hitting_array_temp, axis=1)
            # print('hitting temp', hitting_array_temp)
            #if i % 100 == 0: print(i, 'th','has hitting temp', hitting_array_temp.flatten())
        hitting_array = hitting_array[:, 1:]

        q.append(hitting_array)#put(hitting_array)
        #return hitting_array
    def simulate_markov(self, A,root):

        n_states = A.shape[0]
        P = A / A.sum(axis=1).reshape((n_states, 1))
        #print('row normed P',P.shape, P, P.sum(axis=1))
        x_lazy = self.x_lazy  # 1-x is prob lazy
        alpha_teleport = self.alpha_teleport
        # bias_P is the transition probability matrix

        #P = x_lazy * P + (1 - x_lazy) * np.identity(n_states)
        #print(P, P.sum(axis=1))
        #P = alpha_teleport * P + ((1 - alpha_teleport) * (1 / n_states) * (np.ones((n_states, n_states))))
        #print('check prob of each row sum to one', P.sum(axis=1))

        currentState = root
        state = np.zeros((1, n_states))
        state[0, currentState] = 1
        state_root = state.copy()
        stateHist = state
        dfStateHist = pd.DataFrame(state)
        distr_hist = np.zeros([1, n_states])
        num_sim = 1000#1300

        ncpu = multiprocessing.cpu_count()
        if (ncpu == 1) | (ncpu == 2):
            n_jobs = 1
        elif ncpu > 2:
            n_jobs = min(ncpu - 1, 5)
        print('njobs', n_jobs)
        num_sim_pp = int(num_sim / n_jobs)  # num of simulations per process
        print('num_sim_pp', num_sim_pp)

        n_steps = int(2*n_states)

        jobs=[]

        manager = multiprocessing.Manager()

        q = manager.list()
        for i in range(n_jobs):
            hitting_array = np.ones((P.shape[0], 1)) * 1000
            process = multiprocessing.Process(target=self.simulate_markov_sub, args=(P, num_sim_pp, hitting_array, q,root))
            jobs.append(process)

        for j in jobs:
            j.start()

        for j in jobs:
            j.join()

        print('ended all multiprocesses, will retrieve and reshape')
        hitting_array = q[0]
        for qi in q[1:]:
            hitting_array = np.append(hitting_array, qi, axis=1)  # .get(), axis=1)
        print('finished getting from queue', hitting_array.shape)
        hitting_array_final = np.zeros((1, n_states))
        no_times_state_reached_array = np.zeros((1, n_states))

        for i in range(n_states):
            rowtemp = hitting_array[i, :]
            no_times_state_reached_array[0,i] = np.sum(rowtemp != (n_steps + 1))
        #upper_quart =np.percentile(no_times_state_reached_array,25)
        #loc_rarely_reached = np.where(no_times_state_reached_array<= upper_quart)
        #print('rarely reached clus', upper_quart, no_times_state_reached_array)
        for i in range(n_states):
            rowtemp = hitting_array[i, :]
            no_times_state_reached = np.sum(rowtemp != (n_steps + 1))
            if no_times_state_reached!= 0:
                #print('the number of times state ',i, 'has been reached is', no_times_state_reached )
                #if no_times_state_reached < upper_quart:perc = np.percentile(rowtemp[rowtemp != n_steps + 1], 20)
                perc = np.percentile(rowtemp[rowtemp != n_steps + 1], 15)+0.001
                #print('state ', i,' has perc' ,perc)

                #print('smaller than perc', rowtemp[rowtemp <= perc])



                # hitting_array_final[0, i] = np.min(rowtemp[rowtemp != (n_steps + 1)])
                hitting_array_final[0, i] = np.mean(rowtemp[rowtemp <= perc])
            else:
                hitting_array_final[0, i] = (n_steps + 1)

        # hitting_array=np.mean(hitting_array, axis=1)
        print('hitting from sim markov', [(i, val) for i, val in enumerate(hitting_array_final.flatten())])
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
        knn_sc = 30
        neighbor_array, distance_array = self.knn_struct.knn_query(self.data, k=knn_sc)
        print('shape of neighbor in project onto sc', neighbor_array.shape)
        labels = np.asarray(self.labels)
        sc_pt = np.zeros((len(self.labels),))

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
                mean_weight = mean_weight + hitting_time_clus_i * num_clus_i / knn_sc
                # print('mean weight',mean_weight)
            sc_pt[i] = mean_weight
            i = i + 1
        return sc_pt

    def project_branch_probability_sc(self, bp_array_clus):
        knn_sc = 30
        neighbor_array, distance_array = self.knn_struct.knn_query(self.data, k=knn_sc)
        print('shape of neighbor in project onto sc', neighbor_array.shape)
        labels = np.asarray(self.labels)

        weight_array = np.zeros((len(self.labels), len(list(set(self.labels)))))

        for row in neighbor_array:
            mean_weight = 0
            # print('row in neighbor array of cells', row, labels.shape)
            neighboring_clus = labels[row]
            # print('neighbor clusters labels', neighboring_clus)
            for clus_i in set(list(neighboring_clus)):
                #hitting_time_clus_i = df_graph[clus_i]
                num_clus_i = np.sum(neighboring_clus == clus_i)
                # print('hitting and num_clus for Clusi', hitting_time_clus_i, num_clus_i)
                wi = num_clus_i / knn_sc
                weight_array[row,clus_i] = wi
                # print('mean weight',mean_weight)
        print('shape weight array', weight_array)
        print(weight_array)
        bp_array_sc = weight_array.dot(bp_array_clus)
        self.single_cell_bp = bp_array_sc

        return


    def make_knn_struct(self, too_big=False, big_cluster=None):
        if self.knn > 190: print('please provide a lower K_in for KNN graph construction')
        ef_query = max(100, self.knn + 1)  # ef always should be >K. higher ef, more accuate query
        if too_big == False:
            num_dims = self.data.shape[1]
            n_elements = self.data.shape[0]
            p = hnswlib.Index(space=self.distance, dim=num_dims)  # default to Euclidean distance
            p.set_num_threads(self.num_threads)  # allow user to set threads used in KNN construction
            if n_elements < 10000:
                ef_param_const = min(n_elements - 10, 500)
                ef_query = ef_param_const
                print('setting ef_construction to', )
            else:
                ef_param_const = 200
            if num_dims > 30:
                p.init_index(max_elements=n_elements, ef_construction=ef_param_const,
                             M=48)  ## good for scRNA seq where dimensionality is high
            else:
                p.init_index(max_elements=n_elements, ef_construction=200, M=30, )
            p.add_items(self.data)
        if too_big == True:
            num_dims = big_cluster.shape[1]
            n_elements = big_cluster.shape[0]
            p = hnswlib.Index(space='l2', dim=num_dims)
            p.init_index(max_elements=n_elements, ef_construction=200, M=30)
            p.add_items(big_cluster)
        p.set_ef(ef_query)  # ef should always be > k
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
                to_keep = np.where(distlist <= np.mean(distlist) + self.dist_std_local * np.std(distlist))[0]  # 0*std
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
        if self.knn >=0.8*n_elements: k = int(0.5*n_elements)
        else: k = self.knn
        neighbor_array, distance_array = hnsw.knn_query(X_data, k=k)

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
            G_sim =  ig.Graph(n=n_elements, edges=list(new_edgelist), edge_attrs={'weight': sim_list_new})
        else:
            G_sim = ig.Graph(n=n_elements, edges=list(new_edgelist))
        G_sim.simplify(combine_edges='sum')
        resolution_parameter = 1
        if jac_weighted_edges == True:
            partition = leidenalg.find_partition(G_sim, leidenalg.ModularityVertexPartition, weights='weight',
                                                 n_iterations=self.n_iter_leiden, seed=self.random_seed)
        else:
            partition = leidenalg.find_partition(G_sim, leidenalg.ModularityVertexPartition,
                                                 n_iterations=self.n_iter_leiden, seed=self.random_seed)
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
            nw = w *(pop_s+pop_t)/ (pop_s * pop_t)  # *
            new_weights.append(nw)
            # print('old and new', w, nw)
            i = i + 1
            scale_factor = max(new_weights) - min(new_weights)
            wmin = min(new_weights)

            #wmax = max(new_weights)
        #print('weights before scaling', new_weights)
        new_weights = [(wi+wmin) / scale_factor for wi in new_weights]
        #print('weights after scaling', new_weights)
        sparse_clustergraph = csr_matrix((np.array(new_weights), (sources, targets)),
                                         shape=(n, n))
        # print('new weights', new_weights)
        # print(sparse_clustergraph)
        # print('reweighted sparse clustergraph')
        # print(sparse_clustergraph)
        sources, targets = sparse_clustergraph.nonzero()
        edgelist = list(zip(sources, targets))
        return sparse_clustergraph, edgelist
    def find_root_HumanCD34(self, graph_dense, PARC_labels_leiden, root_idx, true_labels):
        majority_truth_labels = np.empty((len(PARC_labels_leiden), 1), dtype=object)
        graph_node_label = []
        true_labels = np.asarray(true_labels)

        deg_list = graph_dense.sum(axis=1).reshape((1, -1)).tolist()[0]

        for ci, cluster_i in enumerate(sorted(list(set(PARC_labels_leiden)))):
            #print('cluster i', cluster_i)
            cluster_i_loc = np.where(np.asarray(PARC_labels_leiden) == cluster_i)[0]

            majority_truth = self.func_mode(list(true_labels[cluster_i_loc]))

            majority_truth_labels[cluster_i_loc] = str(majority_truth) + 'c' + str(cluster_i)

            graph_node_label.append(str(majority_truth) + 'c' + str(cluster_i))
            root = PARC_labels_leiden[root_idx]
        return graph_node_label, majority_truth_labels, deg_list, root

    def find_root(self, graph_dense, PARC_labels_leiden, root_str, true_labels, super_cluster_labels_sub, super_node_degree_list):
        majority_truth_labels = np.empty((len(PARC_labels_leiden), 1), dtype=object)
        graph_node_label = []
        min_deg = 1000
        super_min_deg = 1000
        found_super_and_sub_root = False
        found_any_root = False
        true_labels = np.asarray(true_labels)

        deg_list = graph_dense.sum(axis=1). reshape((1,-1)).tolist()[0]

        print('deg list', deg_list)# locallytrimmed_g.degree()

        for ci, cluster_i in enumerate(sorted(list(set(PARC_labels_leiden)))):
            print('cluster i' ,cluster_i)
            cluster_i_loc = np.where(np.asarray(PARC_labels_leiden) == cluster_i)[0]

            majority_truth = self.func_mode(list(true_labels[cluster_i_loc]))
            #print('cluster', cluster_i, 'has majority', majority_truth, 'with degree list', deg_list)
            if self.super_cluster_labels != False:
                super_majority_cluster = self.func_mode(list(np.asarray(super_cluster_labels_sub)[cluster_i_loc]))
                super_majority_cluster_loc = np.where(np.asarray(super_cluster_labels_sub) == super_majority_cluster)[0]
                super_majority_truth = self.func_mode(list(true_labels[super_majority_cluster_loc]))
                #print('spr node degree list sub',super_node_degree_list, super_majority_cluster)

                super_node_degree = super_node_degree_list[super_majority_cluster]

                if (root_str in majority_truth) & (root_str in super_majority_truth):
                    if super_node_degree < super_min_deg:
                        # if deg_list[cluster_i] < min_deg:
                        found_super_and_sub_root = True
                        root = cluster_i
                        found_any_root=True
                        min_deg = deg_list[ci]
                        super_min_deg = super_node_degree
                        print('new root is', root, ' with degree', min_deg, 'and super node degree',
                              super_min_deg)
            majority_truth_labels[cluster_i_loc] = str(majority_truth) + 'c' + str(cluster_i)

            graph_node_label.append(str(majority_truth) + 'c' + str(cluster_i))
        if (self.super_cluster_labels == False) | (found_super_and_sub_root == False):
            print('self.super_cluster_labels', super_cluster_labels_sub, ' foundsuper_cluster_sub and super root',
                  found_super_and_sub_root)
            for ic, cluster_i in enumerate(sorted(list(set(PARC_labels_leiden)))):
                cluster_i_loc = np.where(np.asarray(PARC_labels_leiden) == cluster_i)[0]
                print('cluster', cluster_i, 'set true lables', set(true_labels))
                true_labels = np.asarray(true_labels)

                majority_truth = self.func_mode(list(true_labels[cluster_i_loc]))
                print('cluster', cluster_i, 'has majority', majority_truth, 'with degree list', deg_list)
                if (root_str in majority_truth):
                    print('did not find a super and sub cluster with majority ', root_str)
                    if deg_list[ic] < min_deg:
                        root = cluster_i
                        found_any_root = True
                        min_deg = deg_list[ic]
                        print('new root is', root, ' with degree', min_deg)
        #print('len graph node label', graph_node_label)
        if found_any_root == False:
            print('setting arbitrary root', cluster_i)
            self.root = cluster_i
        return graph_node_label, majority_truth_labels, deg_list, root
    def full_graph_paths(self,X_data):
        neighbor_array, distance_array = self.knn_struct.knn_query(X_data, k=3)
        csr_array = self.make_csrmatrix_noselfloop(neighbor_array, distance_array)
        n_comp, comp_labels = connected_components(csr_array, return_labels=True)
        k_0 = 3
        while n_comp >1:
            k_0 = k_0+1
            neighbor_array, distance_array = self.knn_struct.knn_query(X_data, k=k_0)
            csr_array = self.make_csrmatrix_noselfloop(neighbor_array, distance_array)
            n_comp, comp_labels = connected_components(csr_array, return_labels=True)
        row_list = []
        neighbor_array = neighbor_array  # not listed in in any order of proximity
        print('size neighbor array', neighbor_array.shape)
        num_neigh = neighbor_array.shape[1]
        n_neighbors = neighbor_array.shape[1]
        n_cells = neighbor_array.shape[0]

        row_list.extend(list(np.transpose(np.ones((n_neighbors, n_cells)) * range(0, n_cells)).flatten()))
        col_list = neighbor_array.flatten().tolist()
        weight_list = (distance_array.flatten()).tolist()
        csr_full_graph = csr_matrix((np.array(weight_list), (np.array(row_list), np.array(col_list))),
                                    shape=(n_cells, n_cells))


        sources, targets = csr_full_graph.nonzero()
        edgelist = list(zip(sources.tolist(), targets.tolist()))
        Gr = ig.Graph(edgelist, edge_attrs={'weight': csr_full_graph.data.tolist()})
        Gr.simplify(combine_edges='sum')
        return Gr

    def run_subPARC(self):
        root_str = self.root_str
        X_data = self.data
        too_big_factor = self.too_big_factor
        small_pop = self.small_pop
        jac_std_global = self.jac_std_global
        jac_weighted_edges = self.jac_weighted_edges
        n_elements = X_data.shape[0]
        #if n_elements < 2000: self.knn = 10

        n_elements = X_data.shape[0]

        # print('number of k-nn is', knn, too_big_factor, 'small pop is', small_pop)

        neighbor_array, distance_array = self.knn_struct.knn_query(X_data, k=self.knn)
        csr_array = self.make_csrmatrix_noselfloop(neighbor_array, distance_array)


        #### construct full graph
        row_list = []
        neighbor_array = neighbor_array  # not listed in in any order of proximity
        print('size neighbor array', neighbor_array.shape)
        num_neigh = neighbor_array.shape[1]
        n_neighbors = neighbor_array.shape[1]
        n_cells = neighbor_array.shape[0]


        row_list.extend(list(np.transpose(np.ones((n_neighbors, n_cells)) * range(0, n_cells)).flatten()))
        col_list = neighbor_array.flatten().tolist()
        weight_list = (1. / (distance_array.flatten() + 0.05)).tolist()
        # if local_pruning_bool == True: print('share of neighbors discarded in local distance pruning %.1f' % (discard_count / neighbor_array.size))

        csr_full_graph = csr_matrix((np.array(weight_list), (np.array(row_list), np.array(col_list))),
                                    shape=(n_cells, n_cells))

        sources, targets = csr_full_graph.nonzero()
        edgelist = list(zip(sources.tolist(), targets.tolist()))
        G = ig.Graph(edgelist, edge_attrs={'weight': csr_full_graph.data.tolist()})

        sim_list = G.similarity_jaccard(pairs=edgelist)  # list of jaccard weights
        ig_fullgraph = ig.Graph(list(edgelist), edge_attrs={'weight': sim_list})
        ig_fullgraph.simplify(combine_edges='sum')
        inv_simlist = [1-i for i in sim_list]
        #full_graph_shortpath = ig.Graph(list(edgelist), edge_attrs={'weight': inv_simlist}) #the weights reflect distances
        #full_graph_shortpath.simplify(combine_edges='sum')
        #self.full_graph_shortpath = full_graph_shortpath
        self.full_graph_shortpath = self.full_graph_paths(X_data)
        ####


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
        print('Share of edges kept after Global Pruning %.2f' % (len(strong_locs) / len(sim_list)), '%')
        new_edgelist = list(edge_list_copy_array[strong_locs])
        sim_list_new = list(sim_list_array[strong_locs])

        G_sim = ig.Graph(n=n_elements, edges=list(new_edgelist), edge_attrs={'weight': sim_list_new})
        # print('average degree of graph is %.1f' % (np.mean(G_sim.degree())))
        G_sim.simplify(combine_edges='sum')  # "first"
        # print('average degree of SIMPLE graph is %.1f' % (np.mean(G_sim.degree())))
        print('commencing community detection')
        if jac_weighted_edges == True:
            start_leiden = time.time()
            # print('call leiden on weighted graph for ', self.n_iter_leiden, 'iterations')
            partition = leidenalg.find_partition(G_sim, leidenalg.ModularityVertexPartition, weights='weight',
                                                 n_iterations=self.n_iter_leiden, seed=self.random_seed)
            print(time.time() - start_leiden)
        else:
            start_leiden = time.time()
            # print('call leiden on unweighted graph', self.n_iter_leiden, 'iterations')
            partition = leidenalg.find_partition(G_sim, leidenalg.ModularityVertexPartition,
                                                 n_iterations=self.n_iter_leiden, seed=self.random_seed)
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

            #print('pop of new big labels', pop_list)
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
        time_smallpop = time.time()
        while (small_pop_exist) == True & (time.time()-time_smallpop<15):
            small_pop_list = []
            small_pop_exist = False
            for cluster in set(list(PARC_labels_leiden.flatten())):
                population = len(np.where(PARC_labels_leiden == cluster)[0])
                if population < small_pop:
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



        if self.pseudotime == True:

            ## Make cluster-graph (1)

            vc_graph = ig.VertexClustering(ig_fullgraph, membership=PARC_labels_leiden) #jaccard weights, bigger is better
            vc_graph_old = ig.VertexClustering(G_sim, membership=PARC_labels_leiden)

            # print('vc graph G_sim', vc_graph)

            vc_graph = vc_graph.cluster_graph(combine_edges='sum')
            vc_graph_old = vc_graph_old.cluster_graph(combine_edges='sum')
            #print('vc graph G_sim', vc_graph)
            #print('vc graph G_sim old', vc_graph_old)


            reweighted_sparse_vc, edgelist = self.recompute_weights(vc_graph, pop_list_raw)

            print('len old edge list', edgelist) #0.15 for CD34
            if self.humanCD34 ==False: global_pruning_std = 2 #toy data is usually simpler so we dont need to prune the links as the clusters are usually well separated such that spurious links dont exist
            else:  global_pruning_std = 0.15
            edgeweights, edgelist, comp_labels = local_pruning_clustergraph_mst(reweighted_sparse_vc, global_pruning_std= global_pruning_std, preserve_disconnected=self.preserve_disconnected)  #0.8 on 20knn and 40ncomp #0.15
            self.connected_comp_labels = comp_labels
            print('final comp labels set', set(comp_labels))
            #edgeweights_maxout, edgelist_maxout = local_pruning_clustergraph(reweighted_sparse_vc, local_pruning_std=0.0, global_pruning_std=0.15,  max_outgoing=4)
            row_list = []
            col_list = []
            #for (rowi, coli) in edgelist_maxout:
               # row_list.append(rowi)
               # col_list.append(coli)
            #temp_csr = csr_matrix((np.array(edgeweights_maxout), (np.array(row_list), np.array(col_list))),
                                          # shape=(n_cells, n_cells))
            #temp_csr = temp_csr.transpose().todense() +temp_csr.todense()
            #temp_csr = np.tril(temp_csr,-1) #elements along the main diagonal and above are set to zero
            #temp_csr = csr_matrix(temp_csr)
            #edgeweights_maxout  =temp_csr.data
            #sources, targets = temp_csr.nonzero()
            #edgelist_maxout = list(zip(sources.tolist(), targets.tolist()))
            #self.edgelist_maxout = edgelist_maxout
            #self.edgeweights_maxout = edgeweights_maxout
            #print('upper triangle edgelist', edgelist_maxout)
            #self.edgelist_maxout = set(tuple(sorted(l)) for l in edgelist_maxout) #only used for visualization, not for the hitting time computations
            print('len new edge list', edgelist)

            locallytrimmed_g = ig.Graph(edgelist, edge_attrs={'weight': edgeweights.tolist()})
            #print('locally trimmed_g', locallytrimmed_g)
            locallytrimmed_g = locallytrimmed_g.simplify(combine_edges='sum')
            #print('locally trimmed and simplified', locallytrimmed_g)

            locallytrimmed_sparse_vc = get_sparse_from_igraph(locallytrimmed_g, weight_attr='weight')
            layout = locallytrimmed_g.layout_fruchterman_reingold()  ##final layout based on locally trimmed


            #locallytrimmed_sparse_vc_copy = locallytrimmed_sparse_vc.copy()
            #dense_locallytrimmed = scipy.sparse.csr_matrix.todense(locallytrimmed_sparse_vc_copy)
            #degree_array = np.sum(dense_locallytrimmed, axis=1)

            # globally trimmed link
            sources, targets = locallytrimmed_sparse_vc.nonzero()
            edgelist_simple = list(zip(sources.tolist(), targets.tolist()))
            edgelist_unique = set(tuple(sorted(l)) for l in edgelist_simple)  # keep only one of (0,1) and (1,0)
            self.edgelist_unique = edgelist_unique
            self.edgelist = edgelist


            x_lazy = self.x_lazy
            alpha_teleport = self.alpha_teleport
            #locallytrimmed_sparse_vc = locallytrimmed_sparse_vc_copy  ##hitting times are computed based on the locally trimmed graph without any global pruning

            # number of components
            graph_dict = {}
            n_components, labels = connected_components(csgraph=locallytrimmed_sparse_vc, directed=False, return_labels=True)
            print('there are ', n_components,'components in the graph')
            df_graph = pd.DataFrame(locallytrimmed_sparse_vc.todense())
            df_graph['cc'] = labels
            df_graph['pt'] = float('NaN')
            df_graph['markov_pt']=float('NaN')
            df_graph['majority_truth'] = 'maj truth'
            df_graph['graph_node_label'] = 'node label'
            set_parc_labels =list(set(PARC_labels_leiden))
            set_parc_labels.sort()
            print('parc labels', set_parc_labels)
            terminal_clus = []
            node_deg_list=[]
            pd_columnnames_terminal = []
            self.root = []
            for comp_i in range(n_components):
                loc_compi = np.where(labels == comp_i)[0]
                print('loc_compi',loc_compi)

                a_i = df_graph.iloc[loc_compi][loc_compi].values
                a_i = csr_matrix(a_i, (a_i.shape[0],a_i.shape[0]))
                cluster_labels_subi = [x for x in loc_compi]
                sc_labels_subi =[PARC_labels_leiden[i] for i in range(len(PARC_labels_leiden)) if (PARC_labels_leiden[i] in cluster_labels_subi) ]
                sc_truelabels_subi = [self.true_label[i] for i in range(len(PARC_labels_leiden)) if(PARC_labels_leiden[i] in cluster_labels_subi)]
                if self.humanCD34==False:
                    if self.super_cluster_labels !=False:
                        super_labels_subi = [self.super_cluster_labels[i] for i in range(len(PARC_labels_leiden)) if(PARC_labels_leiden[i] in cluster_labels_subi)]
                        print('super node degree', self.super_node_degree_list)

                        graph_node_label, majority_truth_labels, node_deg_list_i, root_i= self.find_root(a_i, sc_labels_subi, root_str,    sc_truelabels_subi,
                                                                                         super_labels_subi,
                                                                                         self.super_node_degree_list)
                    else:
                        graph_node_label, majority_truth_labels,node_deg_list_i, root_i = self.find_root(a_i, sc_labels_subi, root_str,   sc_truelabels_subi,[],[])

                elif self.humanCD34==True:
                    graph_node_label, majority_truth_labels, node_deg_list_i, root_i = self.find_root_HumanCD34(a_i, sc_labels_subi,
                                                                                                root_str,
                                                                                                sc_truelabels_subi)
                self.root.append(root_i)
                for item in node_deg_list_i:
                    node_deg_list.append(item)
                print('a_i shape, true labels shape', a_i.shape, len(sc_truelabels_subi), len(sc_labels_subi))

                new_root_index_found = False
                for ii,llabel in enumerate(cluster_labels_subi):
                    if root_i ==llabel:
                        new_root_index = ii
                        new_root_index_found = True
                        print('new root index', new_root_index)
                if new_root_index_found == False:
                    print('cannot find the new root index')
                    new_root_index = 0
                hitting_times, roundtrip_times = self.compute_hitting_time(a_i, root=new_root_index,
                                                                       x_lazy=x_lazy, alpha_teleport=alpha_teleport)
                scaling_fac = 10/max(hitting_times)
                hitting_times = hitting_times*scaling_fac
                s_ai, t_ai = a_i.nonzero()
                edgelist_ai = list(zip(s_ai, t_ai))
                edgeweights_ai = a_i.data
                #print('edgelist ai', edgelist_ai)
                #print('edgeweight ai', edgeweights_ai)
                biased_edgeweights_ai = get_biased_weights(edgelist_ai, edgeweights_ai, hitting_times)


                # biased_sparse = csr_matrix((biased_edgeweights, (row, col)))
                adjacency_matrix_ai = np.zeros((a_i.shape[0], a_i.shape[0]))

                for i, (start, end) in enumerate(edgelist_ai):
                    adjacency_matrix_ai[start, end] = biased_edgeweights_ai[i]

                markov_hitting_times_ai = self.simulate_markov(adjacency_matrix_ai, new_root_index)  # +adjacency_matrix.T))
                scaling_fac = 10 / max(markov_hitting_times_ai)
                markov_hitting_times_ai = markov_hitting_times_ai * scaling_fac
                adjacency_matrix_csr_ai = sparse.csr_matrix(adjacency_matrix_ai)
                (sources, targets) = adjacency_matrix_csr_ai.nonzero()
                edgelist_ai = list(zip(sources, targets))
                weights_ai = adjacency_matrix_csr_ai.data
                bias_weights_2_ai = get_biased_weights(edgelist_ai, weights_ai, markov_hitting_times_ai, round_no=2)
                adjacency_matrix2_ai = np.zeros((adjacency_matrix_ai.shape[0], adjacency_matrix_ai.shape[0]))

                for i, (start, end) in enumerate(edgelist_ai):
                    adjacency_matrix2_ai[start, end] = bias_weights_2_ai[i]
                if self.super_terminal_cells == False:
                    terminal_clus_ai = self.get_terminal_clusters(adjacency_matrix2_ai, markov_hitting_times_ai)
                    for i in terminal_clus_ai:
                        terminal_clus.append(cluster_labels_subi[i])
                else:
                    print('super terminal cells', self.super_terminal_cells)
                    print('subi', cluster_labels_subi)
                    print([self.labels[ti] for ti in self.super_terminal_cells ]) # find the sub-cluster which contains the single-cell-superterminal
                    temp = [self.labels[ti] for ti in self.super_terminal_cells if self.labels[ti]  in cluster_labels_subi ]
                    terminal_clus_ai = []
                    for i in temp:
                        terminal_clus_ai.append(np.where(np.asarray(cluster_labels_subi)==i)[0][0])
                    print('terminal clus in this a_i', terminal_clus_ai)
                    for i in temp:
                        terminal_clus.append(i)
                    '''
                    terminal_clus_ai = []
                    print('super terminal cells', self.super_terminal_cells)
                    for ti in self.super_terminal_cells:
                        terminal_clus_ai.append(self.labels[ti])
                    terminal_clus_ai = list(set(terminal_clus_ai))
                    '''

                for target_terminal in terminal_clus_ai:

                    prob_ai = self.prob_reaching_terminal_state(target_terminal, terminal_clus_ai, adjacency_matrix2_ai,  new_root_index, pt=markov_hitting_times_ai, num_sim=100) #500 !!!! CHANGE BACK AFTER TESTING
                    df_graph['terminal_clus'+str(cluster_labels_subi[target_terminal])] = 0.0000000
                    pd_columnnames_terminal.append('terminal_clus'+str(cluster_labels_subi[target_terminal]))

                    print('prob ai for target termninal',target_terminal, prob_ai)
                    for k, prob_ii in enumerate(prob_ai):
                        #print('prob ii', prob_ii)
                        #df_graph.at['terminal_clus' + str(cluster_labels_subi[target_terminal])][cluster_labels_subi[k]] = prob_ii
                        df_graph.at[cluster_labels_subi[k],'terminal_clus' + str(cluster_labels_subi[target_terminal])] = prob_ii
                bp_array = df_graph[pd_columnnames_terminal].values

                for ei, ii in enumerate(loc_compi):
                    #print('ii',ii)
                    #df_graph['pt'][ii]=hitting_times[ei]
                    df_graph.at[ii,'pt'] = hitting_times[ei]
                    df_graph.at[ii,'graph_node_label'] = graph_node_label[ei]
                    #df_graph['graph_node_label'][ii] = graph_node_label[ei]
                    #df_graph['majority_truth'][ii] = graph_node_label[ei]
                    df_graph.at[ii,'majority_truth'] = graph_node_label[ei]
                    #df_graph['markov_pt'][ii] = markov_hitting_times_ai[ei]
                    df_graph.at[ii,'markov_pt'] = markov_hitting_times_ai[ei]
                #print('df_graph', df_graph)
                #print('adj2',adjacency_matrix2_ai)

            #print('graph node label', graph_node_label)
            #majority_truth_labels = list(majority_truth_labels.flatten())
            #print('locallytrimmed_g', locallytrimmed_g)


            locallytrimmed_g.vs["label"] = df_graph['graph_node_label'].values
            #hitting_times, roundtrip_times = self.compute_hitting_time(locallytrimmed_sparse_vc, root=root, x_lazy=x_lazy, alpha_teleport=alpha_teleport)
            hitting_times = df_graph['pt'].values

            self.hitting_times = hitting_times #* 1000
            self.markov_hitting_times = df_graph['markov_pt'].values
            self.terminal_clusters = terminal_clus
            print('terminal clusters', terminal_clus)
            self.node_degree_list = node_deg_list
            self.project_branch_probability_sc(bp_array)
            hitting_times = self.markov_hitting_times

            bias_weights_2_all = get_biased_weights(edgelist, edgeweights, self.markov_hitting_times, round_no=2)
            row_list = []
            col_list= []
            for (rowi, coli) in edgelist:
                row_list.append(rowi)
                col_list.append(coli)
            print('shape',a_i.shape[0], a_i.shape[0], row_list)
            temp_csr = csr_matrix((np.array(bias_weights_2_all), (np.array(row_list), np.array(col_list))),
                                           shape=(n_clus, n_clus))
            if self.humanCD34 == False:
                visual_global_pruning_std = 0.15
                max_outgoing = 4
            else:
                visual_global_pruning_std = 0
                max_outgoing = 2
            edgeweights_maxout_2, edgelist_maxout_2, comp_labels_2 = local_pruning_clustergraph_mst(temp_csr,#glob_std_pruning =0 and max_out = 2 for HumanCD34
                                                                                   global_pruning_std=visual_global_pruning_std,
                                                                                  max_outgoing=max_outgoing, preserve_disconnected=self.preserve_disconnected, visual = True) ##trying to use _MST as visual#maxoutoging =4
            print('edgelistmaxout2', edgelist_maxout_2)

            row_list = []
            col_list = []
            for (rowi, coli) in edgelist_maxout_2:
                row_list.append(rowi)
                col_list.append(coli)
            temp_csr = csr_matrix((np.array(edgeweights_maxout_2), (np.array(row_list), np.array(col_list))), shape=(n_clus, n_clus))
            temp_csr = temp_csr.transpose().todense() + temp_csr.todense()
            temp_csr = np.tril(temp_csr, -1)  # elements along the main diagonal and above are set to zero
            temp_csr = csr_matrix(temp_csr)
            edgeweights_maxout_2 = temp_csr.data
            scale_factor = max(edgeweights_maxout_2) - min(edgeweights_maxout_2)
            edgeweights_maxout_2 = [((wi + .1)*2.5 / scale_factor)+0.1 for wi in edgeweights_maxout_2]
            print('maxout2 edge', edgeweights_maxout_2)
            sources, targets = temp_csr.nonzero()
            edgelist_maxout_2 = list(zip(sources.tolist(), targets.tolist()))
            self.edgelist_maxout = edgelist_maxout_2
            self.edgeweights_maxout = edgeweights_maxout_2
            print('self edgelist final setting', len(edgeweights_maxout_2), edgelist_maxout_2)
            # plotting with threshold (thresholding the upper limit to make color palette more readable)
            #print('percentile', np.percentile(hitting_times, 95))
            remove_outliers = hitting_times#[hitting_times < np.percentile(hitting_times,               95)]  # hitting_times[hitting_times<np.mean(hitting_times)+np.std(hitting_times)]
            # print('mean hitting times', np.mean(hitting_times))
            threshold = np.percentile(remove_outliers, 95)  # np.mean(remove_outliers) + 1* np.std(remove_outliers)
            # print('threshold', threshold)
            th_hitting_times = [x if x < threshold else threshold for x in hitting_times]

            remove_outliers_low = hitting_times[hitting_times < (np.mean(hitting_times) - 0.3 * np.std(hitting_times))]
            threshold_low = np.mean(remove_outliers_low) - 0.3 * np.std(remove_outliers_low)
            threshold_low = np.percentile(remove_outliers_low, 5)
            #print('thresh low', threshold_low)
            th_hitting_times = [x if x > threshold_low else threshold_low for x in th_hitting_times]

            scaled_hitting_times = (th_hitting_times - np.min(th_hitting_times))
            scaled_hitting_times = scaled_hitting_times * (1000 / np.max(scaled_hitting_times))

            self.scaled_hitting_times = scaled_hitting_times
            self.single_cell_pt = self.project_hittingtimes_sc(self.hitting_times)
            #self.single_cell_pt_stationary_bias = self.project_hittingtimes_sc(self.stationary_hitting_times.flatten())
            self.single_cell_pt_markov = self.project_hittingtimes_sc(self.markov_hitting_times)
            #self.dijkstra_hitting_times = self.path_length_onbias(edgelist, biased_edgeweights)
            #print('dijkstra hitting times', [(i,j) for i,j in enumerate(self.dijkstra_hitting_times)])
            #self.single_cell_pt_dijkstra_bias = self.project_hittingtimes_sc(self.dijkstra_hitting_times)

            # threshold = np.mean(scaled_hitting_times)+0.25*np.std(scaled_hitting_times)
            threshold = int(threshold)
            scaled_hitting_times = scaled_hitting_times.astype(int)
            # print('scaled hitting times')
            # print(scaled_hitting_times)
            pal = ig.drawing.colors.AdvancedGradientPalette(['yellow', 'green', 'blue'], n=1001)

            all_colors = []
            #print('100 scaled hitting', scaled_hitting_times)
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
            svgpath_local = self.path + "vc_graph_locallytrimmed_Root" + str(self.root[0]) + "lazy" + str(
                x_lazy) + 'JacG' + str(self.jac_std_global) + 'toobig' + str(
                int(self.too_big_factor * 100)) + "M" + str(len(set(self.true_label))) + ".svg"
            # print('svglocal', svgpath_local)

            #ig.plot(locallytrimmed_g, svgpath_local, layout=layout ,edge_width=[e['weight'] * 1 for e in locallytrimmed_g.es], vertex_label=graph_node_label)
            # hitting_times = compute_hitting_time(sparse_vf, root=1)
            # print('final hitting times:', list(zip(range(len(hitting_times)), hitting_times)))

            # globallytrimmed_g.vs['color'] = [pal.get(i)[0:3] for i in scaled_hitting_times]
            # ig.plot(globallytrimmed_g,                    self.path+"/vc_graph_globallytrimmed_Root" + str(   root) + "Lazy" + str(x_lazy) + 'JacG' + str(self.jac_std_global) + 'toobig'+str(int(100*self.too_big_factor))+".svg", layout=layout,edge_width=[e['weight'] * .1 for e in globallytrimmed_g.es], vertex_label=graph_node_label, main ='lazy:'+str(x_lazy)+' alpha:'+str(alpha_teleport))
        self.graph_node_label = df_graph['graph_node_label'].values
        self.edgeweight = [e['weight'] * 1 for e in locallytrimmed_g.es]
        print('self edge weight', len(self.edgeweight), self.edgeweight)
        print('self edge list', len(self.edgelist_unique),self.edgelist_unique)
        self.graph_node_pos = layout.coords
        f, ((ax, ax1, ax2)) = plt.subplots(1, 3, sharey=True)

        self.draw_piechart_graph(ax,ax1,ax2)
        '''
        if self.super_terminal_cells !=False:
            for i in terminal_clus:
                fig, ax = plt.subplots()
                self.draw_evolution_probability(df_graph['terminal_clus' +str(i)],ax, i )
        '''
        plt.show()
        return

    def draw_piechart_graph(self,  ax,ax1,ax2,type_pt = 'original',):
        #f, ((ax, ax1,ax2)) = plt.subplots(1, 3, sharey=True)
        arrow_head_w = 0.3
        edgeweight_scale = 1
        # fig, ax = plt.subplots()
        node_pos = self.graph_node_pos
        edgelist = list(self.edgelist_maxout)
        edgeweight = self.edgeweights_maxout

        node_pos = np.asarray(node_pos)

        # #edgeweight = np.asarray(edgeweight)
        # loc_keep = np.where(edgeweight>=np.percentile(edgeweight,5))[0]
        # print('inside draw pie', loc_keep)
        # edgeweight = edgeweight[loc_keep]
        # edgelist = [edgelist[i] for i in loc_keep]
        # #node_pos=node_pos[loc_keep,:]
        # print(node_pos.shape, len(edgelist), len(edgeweight))
        #
        # edgeweight.tolist()

        graph_node_label = self.graph_node_label
        if type_pt == 'original': pt = self.scaled_hitting_times
        if type_pt =='biased_stationary': pt = self.biased_hitting_times_stationary
        if type_pt =='markov': pt = self.markov_hitting_times
        import matplotlib.lines as lines

        n_groups = len(set(self.labels))#node_pos.shape[0]
        n_truegroups = len(set(self.true_label))
        group_pop = np.zeros([n_groups, 1])
        group_frac = pd.DataFrame(np.zeros([n_groups, n_truegroups]), columns=list(set(self.true_label)))

        for group_i in set(self.labels):
            loc_i = np.where(self.labels == group_i)[0]
            #print('pop of group', group_i, 'is',len(loc_i) )
            group_pop[group_i] = len(loc_i)#np.sum(loc_i) / 1000 + 1
            true_label_in_group_i = list(np.asarray(self.true_label)[[loc_i]])
            for ii in set(true_label_in_group_i):
                group_frac[ii][group_i] = true_label_in_group_i.count(ii)
        group_frac = group_frac.div(group_frac.sum(axis=1), axis=0)

        line_true = np.linspace(0, 1, n_truegroups)
        color_true_list = [plt.cm.jet(color) for color in line_true]

        sct = ax.scatter(
            node_pos[:, 0], node_pos[:, 1],
            c='white', edgecolors='face', s=group_pop, cmap='jet')
        print('draw triangle edgelist', len(edgelist), edgelist)
        for e_i, (start, end) in enumerate(edgelist):
            if pt[start] > pt[end]:
                temp = start
                start = end
                end = temp

            ax.add_line(lines.Line2D([node_pos[start, 0], node_pos[end, 0]], [node_pos[start, 1], node_pos[end, 1]],
                                     color='grey', lw=edgeweight[e_i] * edgeweight_scale, alpha=0.2))
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
        plt.legend(patches, labels, loc=(-5, -5),fontsize=6)
        if self.too_big_factor >0.1: is_sub = ' super clusters'
        else: is_sub = ' sub clusters'
        ti= 'Reference Group Membership. K=' + str(self.knn) +'. ncomp = ' + str(self.ncomp) +is_sub
        ax.set_title(ti)

        title_list = ["PT on undirected original graph",
         "PT using Markov Simulation"]#, "PT using Dijkstra on digraph"] #"PT on digraph using closed-form approx",
        for i, ax_i in enumerate([ax1,ax2]):#,ax3]):
            print("drawing axis",i)
            if i ==0:  pt= self.hitting_times
            #if i ==1: pt= self.stationary_hitting_times
            if i==1: pt= self.markov_hitting_times
            #if i==2: pt = self.dijkstra_hitting_times

            for e_i, (start, end) in enumerate(edgelist):
                if pt[start] > pt[end]:
                    temp = start
                    start = end
                    end = temp

                ax_i.add_line(lines.Line2D([node_pos[start, 0], node_pos[end, 0]], [node_pos[start, 1], node_pos[end, 1]],
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

                    ax_i.arrow(xp[250], smooth[250], xp[250 + step] - xp[250], smooth[250 + step] - smooth[250],
                              shape='full', lw=0,
                              length_includes_head=True, head_width=arrow_head_w,
                              color='black')
                    # ax.plot(xp, smooth, linewidth=edgeweight[e_i], c='pink')
                else:
                    ax_i.arrow(xp[250], smooth[250], xp[250 - step] - xp[250],
                              smooth[250 - step] - smooth[250], shape='full', lw=0,
                              length_includes_head=True, head_width=arrow_head_w, color='black')
            c_edge = []
            l_width = []
            for ei,pti in enumerate(pt):
                if ei in self.terminal_clusters:
                    c_edge.append('red')
                    l_width.append(1.5)
                else:
                    c_edge.append('gray')
                    l_width.append(0.0)

            gp_scaling = 500/max(group_pop)
            print(gp_scaling,'gp_scaline')
            group_pop_scale = group_pop*gp_scaling
            #print('group pop scale pie', group_pop_scale)
            ax_i.scatter(node_pos[:, 0], node_pos[:, 1], s=group_pop_scale, c=pt, cmap='viridis_r',edgecolors=c_edge,
                         alpha=1, zorder=3, linewidth = l_width)
            for ii in range(node_pos.shape[0]):
                ax_i.text(node_pos[ii, 0]+0.5, node_pos[ii, 1]+0.5, 'c'+str(ii), color='black', zorder=4)

            title_pt = title_list[i]
            ax_i.set_title(title_pt)
        #plt.show()



    def draw_evolution_probability(self, prob_,ax_i, target_terminal):
        n_terminal = len(self.terminal_clusters)
        n_groups = len(set(self.labels))


        arrow_head_w = 0.3
        edgeweight_scale = 1
        # fig, ax = plt.subplots()
        node_pos = self.graph_node_pos
        edgelist = list(self.edgelist_maxout)
        edgeweight = self.edgeweights_maxout

        node_pos = np.asarray(node_pos)
        group_pop = np.zeros([n_groups, 1])
        # #edgeweight = np.asarray(edgeweight)
        # loc_keep = np.where(edgeweight>=np.percentile(edgeweight,5))[0]
        # print('inside draw pie', loc_keep)
        # edgeweight = edgeweight[loc_keep]
        # edgelist = [edgelist[i] for i in loc_keep]
        # #node_pos=node_pos[loc_keep,:]
        # print(node_pos.shape, len(edgelist), len(edgeweight))
        #
        # edgeweight.tolist()

        graph_node_label = self.graph_node_label
        pt = self.markov_hitting_times
        for group_i in set(self.labels):
            loc_i = np.where(self.labels == group_i)[0]
            #print('pop of group', group_i, 'is', len(loc_i))
            group_pop[group_i] = len(loc_i)  # np.sum(loc_i) / 1000 + 1
        import matplotlib.lines as lines

        n_groups = len(set(self.labels))#node_pos.shape[0]
        n_truegroups = len(set(self.true_label))



        line_true = np.linspace(0, 1, n_truegroups)

        for e_i, (start, end) in enumerate(edgelist):
            if pt[start] > pt[end]:
                temp = start
                start = end
                end = temp

            ax_i.add_line(lines.Line2D([node_pos[start, 0], node_pos[end, 0]], [node_pos[start, 1], node_pos[end, 1]],
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

                ax_i.arrow(xp[250], smooth[250], xp[250 + step] - xp[250], smooth[250 + step] - smooth[250],
                          shape='full', lw=0,
                          length_includes_head=True, head_width=arrow_head_w,
                          color='black')
                # ax.plot(xp, smooth, linewidth=edgeweight[e_i], c='pink')
            else:
                ax_i.arrow(xp[250], smooth[250], xp[250 - step] - xp[250],
                          smooth[250 - step] - smooth[250], shape='full', lw=0,
                          length_includes_head=True, head_width=arrow_head_w, color='black')
        c_edge = []
        l_width = []
        for ei,pti in enumerate(pt):
            if ei in self.terminal_clusters:
                c_edge.append('red')
                l_width.append(1.5)
            else:
                c_edge.append('gray')
                l_width.append(0.0)

        gp_scaling = 1000/np.max(group_pop)
        print('evolution scale', gp_scaling)
        group_pop_scale = group_pop*gp_scaling
        #print('group pop', group_pop_scale)
        threshold = np.mean(prob_)+1.5*np.std(prob_)
        if threshold >1:threshold = np.mean(prob_)+1*np.std(prob_)
        if threshold >1: threshold =1
        print('prob thresh', threshold)
        prob_ =[x if x < threshold else threshold for x in prob_]

        ax_i.scatter(node_pos[:, 0], node_pos[:, 1], s=group_pop_scale, c=prob_, cmap='viridis',edgecolors=c_edge,
                     alpha=0.5, zorder=3, linewidth = l_width)
        #for ii in range(node_pos.shape[0]):
            #ax.text(node_pos[ii, 0], node_pos[ii, 1], str(self.labels[i]), color='black', zorder=3)
        title_pt = str('target'+ str(target_terminal))
        ax_i.set_title(title_pt)


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
        self.ncomp = self.data.shape[1]
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

def mainHuman():

    dict_abb = {'Basophils':'BASO1', 'CD4+ Effector Memory': 'TCEL7','Colony Forming Unit-Granulocytes':'GRAN1','Colony Forming Unit-Megakaryocytic':'MEGA1','Colony Forming Unit-Monocytes':'MONO1','Common myeloid progenitors':"CMP",'Early B cells':"PRE_B2",'Eosinophils':"EOS2",
    'Erythroid_CD34- CD71+ GlyA-':"ERY2",'Erythroid_CD34- CD71+ GlyA+':"ERY3",'Erythroid_CD34+ CD71+ GlyA-':"ERY1",'Erythroid_CD34- CD71lo GlyA+':'ERY4','Granulocyte/monocyte progenitors':"GMP",'Hematopoietic stem cells_CD133+ CD34dim':"HSC1",'Hematopoietic stem cells_CD38- CD34+':"HSC2",
    'Mature B cells class able to switch':"B_a2",'Mature B cells class switched':"B_a4",'Mature NK cells_CD56- CD16- CD3-':"Nka3",'Monocytes':"MONO2",'Megakaryocyte/erythroid progenitors':"MEP",'Myeloid Dendritic Cells':'mDC','Naïve B cells':"B_a1",'Plasmacytoid Dendritic Cells':"pDC",'Pro B cells':'PRE_B3'}
    import palantir
    ncomps = 50 #40 ncomps and 20KNN works well
    knn = 30#30
    print('ncomp =', ncomps,' knn=', knn)
    nover_labels = pd.read_csv('/home/shobi/Trajectory/Datasets/HumanCD34/Nover_Cor_PredFine_notLogNorm.csv')['x'].values.tolist()
    nover_labels = [dict_abb[i] for i in nover_labels]
    parc53_labels = pd.read_csv('/home/shobi/Trajectory/Datasets/HumanCD34/Nover_Cor_Parc53_set1.csv')[
        'x'].values.tolist()

    parclabels_all = pd.read_csv('/home/shobi/Trajectory/Datasets/HumanCD34/parclabels_all_set1.csv')['parc'].values.tolist()
    parc_dict_nover = {}
    for i,c in enumerate(parc53_labels):
        parc_dict_nover[i]=dict_abb[c]
    parclabels_all = [parc_dict_nover[ll] for ll in parclabels_all]
    print('all', len(parclabels_all))


    ad = sc.read(
        '/home/shobi/Trajectory/Datasets/HumanCD34/human_cd34_bm_rep1.h5ad')
    # 5780 cells x 14651 genes Human Replicate 1. Male african american, 38 years
    print('h5ad  ad size', ad)
    colors = pd.Series(ad.uns['cluster_colors'])
    colors['10'] = '#0b128f'
    ct_colors = pd.Series(ad.uns['ct_colors'])

    ad.uns['iroot'] = np.flatnonzero(ad.obs_names == ad.obs['palantir_pseudotime'].idxmin())[0]
    print('iroot', np.flatnonzero(ad.obs_names == ad.obs['palantir_pseudotime'].idxmin())[0])

    tsne = pd.DataFrame(ad.obsm['tsne'], index=ad.obs_names, columns=['x', 'y'])
    tsnem = ad.obsm['tsne']


    revised_clus = ad.obs['clusters'].values.tolist().copy()
    loc_DCs = [i for i in range(5780) if ad.obs['clusters'].values.tolist()[i] == '7']
    for loc_i in loc_DCs:
        if ad.obsm['palantir_branch_probs'][loc_i, 5] > ad.obsm['palantir_branch_probs'][
            loc_i, 2]:  # if prob that cDC > pDC, then relabel as cDC
            revised_clus[loc_i] = '10'
    revised_clus = [int(i) for i in revised_clus]

    #ad.X: Filtered, normalized and log transformed count matrix
    #ad.raw: Filtered raw count matrix
    adata_counts = sc.AnnData(ad.X) #ad.X is filtered, lognormalized,scaled// ad.raw.X is the filtered but not pre-processed
    adata_counts.obs_names = ad.obs_names
    adata_counts.var_names = ad.var_names
    #sc.pp.recipe_zheng17(adata_counts, n_top_genes=1000, log=True) #using this or the .X scaled version is pretty much the same.
    sc.tl.pca(adata_counts, svd_solver='arpack', n_comps=ncomps)

    #tsnem = TSNE().fit_transform(adata_counts.obsm['X_pca'])
    f, (ax1, ax2, ax3) = plt.subplots(1, 3, sharey=True)

    line = np.linspace(0, 1, len(set(revised_clus)))

    for color, group in zip(line, set(revised_clus)):
        where = np.where(np.array(revised_clus) == group)[0]
        ax1.scatter(tsnem[where, 0],tsnem[where, 1], label=group, c=np.asarray(plt.cm.jet(color)).reshape(-1,4))
    ax1.legend()
    ax1.set_title('Palantir Phenograph Labels')

    import random
    import colorcet as cc
    marker =['x','+',(5,0), '>','o',(5,2)]
    line_nover=np.linspace(0, 1, len(set(nover_labels)))
    col_i = 0
    for color, group in zip(line_nover, set(nover_labels)):
        where = np.where(np.array(nover_labels) == group)[0]
        marker_x = marker[random.randint(0,5)]
        #ax2.scatter(tsnem[where, 0],tsnem[where, 1], label=group, c=plt.cm.nipy_spectral(color), marker = marker_x, alpha=0.5)

        ax2.scatter(tsnem[where, 0], tsnem[where, 1], label=group, c=cc.glasbey_dark[col_i], marker=marker_x,
                    alpha=0.5)
        col_i = col_i+1
    ax2.legend(fontsize=6)
    ax2.set_title('Novershtern Corr. Labels')

    line = np.linspace(0, 1, len(set(parclabels_all)))
    col_i=0
    for color, group in zip(line, set(parclabels_all)):
        where = np.where(np.array(parclabels_all) == group)[0]
        ax3.scatter(tsnem[where, 0], tsnem[where, 1], label=group, c=cc.glasbey_dark[col_i], alpha=0.5)
        col_i = col_i + 1
    ax3.legend()
    ax3.set_title('Parc53 Nover Labels')
    #plt.show()

    plt.figure(figsize=[5, 5])
    plt.title('palantir, ncomps = ' + str(ncomps)+' knn'+ str(knn))
    
    for group in set(revised_clus):
        loc_group = np.where(np.asarray(revised_clus) == group)[0]
        plt.scatter(tsnem[loc_group, 0], tsnem[loc_group, 1], s=5, color=colors[group], label=group)
    ax = plt.gca()
    ax.set_axis_off()
    ax.legend(fontsize=6)
    '''
    norm_df_pal = pd.DataFrame(ad.X)
    print('norm df', norm_df_pal)
    new = ['c' + str(i) for i in norm_df_pal.index]
    norm_df_pal.index = new
    pca_projections, _ = palantir.utils.run_pca(norm_df_pal, n_components=ncomps)

    sc.tl.pca(ad, svd_solver='arpack')
    dm_res = palantir.utils.run_diffusion_maps(pca_projections, n_components=ncomps, knn=knn)

    ms_data = palantir.utils.determine_multiscale_space(dm_res)  # n_eigs is determined using eigengap
    print('ms data', ms_data.shape)
    #tsne =  pd.DataFrame(tsnem)#palantir.utils.run_tsne(ms_data)
    tsne.index = new
    # print(type(tsne))
    str_true_label = pd.Series(revised_clus, index=norm_df_pal.index)

    palantir.plot.plot_cell_clusters(tsne, str_true_label)

    start_cell = 'c4823'  # '#C108 for M12 connected' #M8n1000d1000 start - c107 #c1001 for bifurc n2000d1000 #disconnected n1000 c108, "C1 for M10 connected" # c10 for bifurcating_m4_n2000d1000

    pr_res = palantir.core.run_palantir(ms_data, early_cell=start_cell, num_waypoints=1200, knn=knn)
    palantir.plot.plot_palantir_results(pr_res, tsne)
    plt.show()
    '''

    #print('xpca',norm_df['X_pca'])
    true_label = nover_labels#revised_clus
    p0 = PARC(adata_counts.obsm['X_pca'][:, 0:ncomps], true_label, jac_std_global=0.15, dist_std_local=1, knn=knn,
              too_big_factor=0.4,
              pseudotime=True, path="/home/shobi/Trajectory/Datasets/HumanCD34/", root=1,
              root_str=4823, humanCD34=True, preserve_disconnected=False)  # *.4
    p0.run_PARC()
    super_labels = p0.labels

    super_edges = p0.edgelist_maxout#p0.edgelist
    super_pt = p0.scaled_hitting_times  # pseudotime pt


    p = hnswlib.Index(space='l2', dim=adata_counts.obsm['X_pca'][:, 0:ncomps].shape[1])
    p.init_index(max_elements=adata_counts.obsm['X_pca'][:, 0:ncomps].shape[0], ef_construction=200, M=16)
    p.add_items(adata_counts.obsm['X_pca'][:, 0:ncomps])
    p.set_ef(50)
    tsi_list = [] #find the single-cell which is nearest to the average-location of a terminal cluster
    for tsi in p0.terminal_clusters:
        loc_i = np.where(np.asarray(p0.labels) == tsi)[0]
        temp = np.mean(adata_counts.obsm['X_pca'][:, 0:ncomps][loc_i], axis=0)
        labelsq, distances = p.knn_query(temp, k=1)
        print(labelsq[0])
        tsi_list.append(labelsq[0][0])


    p1 = PARC(adata_counts.obsm['X_pca'][:, 0:ncomps], true_label, jac_std_global=0.15, dist_std_local=1, knn=knn,
              too_big_factor=0.05,
              path="/home/shobi/Trajectory/Datasets/HumanCD34/", pseudotime=True, root=1,
              super_cluster_labels=super_labels, super_node_degree_list=p0.node_degree_list, super_terminal_cells=tsi_list, root_str=4823,
              x_lazy=0.99, alpha_teleport=0.99, humanCD34=True, preserve_disconnected=False)  # *.4
    p1.run_PARC()
    labels = p1.labels
    label_df = pd.DataFrame(labels, columns=['parc'])
    #label_df.to_csv('/home/shobi/Trajectory/Datasets/HumanCD34/parclabels.csv', index=False)
    gene_ids = adata_counts.var_names

    obs = ad.raw.X.toarray()
    print('shape obs', obs.shape)
    obs = pd.DataFrame(obs, columns=gene_ids)
    #    obs['parc']=p1.labels
    obs['louvain'] = revised_clus

    #obs_average = obs.groupby('parc', as_index=True).mean()
    obs_average = obs.groupby('louvain', as_index=True).mean()
    print(obs_average.head())
    #obs_average.to_csv('/home/shobi/Trajectory/Datasets/HumanCD34/louvain_palantir_average.csv', index=False)
    ad_obs = sc.AnnData(obs_average)
    ad_obs.var_names=gene_ids
    ad_obs.obs['parc']= [i for i in range(len(set(revised_clus)))]#p1.labels instaed of revised_clus

    #sc.write('/home/shobi/Trajectory/Datasets/HumanCD34/louvain_palantir_average.h5ad',ad_obs)
    print('start tsne')
    n_downsample = 10000
    if len(labels) > n_downsample:
        idx = np.random.randint(len(labels), size=2000)
        embedding = TSNE().fit_transform(adata_counts.obsm['X_pca'][idx, :])
    else:
        #embedding = TSNE().fit_transform(adata_counts.obsm['X_pca'][:,0:15])
        #print('tsne input size', adata_counts.obsm['X_pca'].shape)
        embedding =tsnem#umap.UMAP().fit_transform(adata_counts.obsm['X_pca'][:,0:20])
        idx = np.random.randint(len(labels), size=len(labels))
    print('end tsne')


    knn_hnsw, ci_list = sc_loc_ofsuperCluster_embeddedspace(embedding, p0, p1)
    '''
    p = hnswlib.Index(space='l2', dim=embedding.shape[1])
    p.init_index(max_elements=embedding.shape[0], ef_construction=200, M=16)
    p.add_items(embedding)
    p.set_ef(50)
    ci_list = []  # single cell location of average location of supercluster
    for ci in list(set(p0.labels)):
        loc_i = np.where(np.asarray(p0.labels) == ci)[0]
        #temp = np.mean(adata_counts.obsm['X_pca'][:, 0:ncomps][loc_i], axis=0)
        x = [embedding[xi, 0] for xi in loc_i]
        y = [embedding[yi, 1] for yi in loc_i]
        # maj = p0.majority_truth_labels[ti]
        labelsq, distancesq = p.knn_query(np.array([np.mean(x), np.mean(y)]), k=1)
        #labels, distances = p.knn_query(temp, k=1)
        ci_list.append(labelsq[0][0])
    '''

    #draw_trajectory_gams(embedding, ci_list, labels, super_labels, super_edges, p1.x_lazy, p1.alpha_teleport,
                           #p1.single_cell_pt, true_label, knn=p0.knn, terminal_clusters=p1.terminal_clusters, super_terminal_clusters=p0.terminal_clusters, title_str='Hitting times: Original Random walk', ncomp=ncomps)

    draw_trajectory_gams(embedding, ci_list, labels, super_labels, super_edges,
                           p1.x_lazy, p1.alpha_teleport, p1.single_cell_pt_markov, true_label, knn=p0.knn,terminal_clusters=p1.terminal_clusters,super_terminal_clusters=p0.terminal_clusters,
                           title_str='Hitting times: Markov Simulation on biased edges',ncomp=ncomps)
    plt.show()

    draw_trajectory_dimred(embedding, ci_list, labels, super_labels, super_edges,
                         p1.x_lazy, p1.alpha_teleport, p1.single_cell_pt_markov, true_label, knn=p0.knn,
                         terminal_clusters=p1.terminal_clusters, super_terminal_clusters=p0.terminal_clusters,
                         title_str='Hitting times: Markov Simulation on biased edges', ncomp=ncomps)
    plt.show()


    num_group = len(set(true_label))

    line = np.linspace(0, 1, num_group)
    lineP0 = np.linspace(0, 1, len(set(p0.labels)))
    lineP1 = np.linspace(0, 1, len(set(p1.labels)))

    f, (ax1, ax2, ax3) = plt.subplots(1, 3, sharey=True)
    ff, (ax11, ax22) = plt.subplots(1, 2, sharey=True)

    for color, group in zip(line, set(true_label)):
        if len(labels) > n_downsample:
            where = np.where(np.array(true_label)[idx] == group)[0]
        else:
            where = np.where(np.array(true_label) == group)[0]
        #ax1.scatter(embedding[where, 0], embedding[where, 1], label=group, c=plt.cm.jet(color))
        marker_x = marker[random.randint(0, 5)]
        ax1.scatter(embedding[where, 0], embedding[where, 1], label=group, c=cc.glasbey_dark[col_i], marker=marker_x,
                    alpha=0.5)
        col_i = col_i + 1


    ax1.legend(fontsize=6)
    ax1.set_title('true labels')


    for color, group in zip(lineP0, set(p0.labels)):
        if len(labels) > n_downsample:
            where = np.where(np.array(p0.labels)[idx] == group)[0]
        else:
            where = np.where(np.array(p0.labels) == group)[0]
        ax11.scatter(embedding[where, 0], embedding[where, 1], label=group, c=np.asarray(plt.cm.jet(color)).reshape(-1,4))
    ax11.legend(fontsize=6)
    ax11.set_title('p0 labels')

    for color, group in zip(lineP1, set(p1.labels)):
        if len(labels) > n_downsample:
            where = np.where(np.array(p1.labels)[idx] == group)[0]
        else:
            where = np.where(np.array(p1.labels) == group)[0]
        ax22.scatter(embedding[where, 0], embedding[where, 1], label=group, c=np.asarray(plt.cm.jet(color)).reshape(-1,4))
    ax22.legend(fontsize=6)
    ax22.set_title('p1 labels')



    ax3.set_title("Markov Sim PT ncomps:"+str(ncomps)+'. knn:'+str(knn))
    ax3.scatter(embedding[:, 0], embedding[:, 1], c=p1.single_cell_pt_markov, cmap='viridis_r')
    terminal_states = p1.terminal_clusters
    for ti in list(set(terminal_states)):
        print('ti', ti)
        loc_i = np.where(np.asarray(p1.labels)==ti)[0]
        x = [embedding[xi,0] for xi in loc_i]
        y = [embedding[yi, 1] for yi in loc_i]
        #maj = p0.majority_truth_labels[ti]
        labels, distances = knn_hnsw.knn_query(np.array([np.mean(x),np.mean(y)]), k=1)
        x = embedding[labels[0], 0]
        y = embedding[labels[0], 1]
        print('labels distances', labels, distances)
        #ax3.scatter(np.mean(x), np.mean(y), label='TS'+str(ti), c='red',s=10)
        ax3.scatter(x,y, label='TS' + str(ti), c='red', s=10,zorder=3)
        ax3.text(np.mean(x)+0.05, np.mean(y)+0.05, 'TS'+str(ti), color='black', zorder=3)
        ax3.legend(fontsize=6)

    ax2.set_title("terminal clus from P0 super clus:" + str(ncomps) + '. knn:' + str(knn))
    ax2.scatter(embedding[:, 0], embedding[:, 1], c=p1.single_cell_pt_markov, cmap='viridis_r')
    terminal_states = p0.terminal_clusters
    jj=0
    for ti in list(set(terminal_states)):
        loc_i = np.where(np.asarray(p0.labels) == ti)[0]
        val_pt = [p1.single_cell_pt_markov[i] for i in loc_i]
        th_pt = np.percentile(val_pt,0)#50
        loc_i = [loc_i[i] for i in range(len(val_pt)) if val_pt[i]>= th_pt]
        x = [embedding[xi, 0] for xi in loc_i]
        y = [embedding[yi, 1] for yi in loc_i]
        labels, distances = knn_hnsw.knn_query(np.array([np.mean(x), np.mean(y)]), k=1)
        x=embedding[labels[0], 0]
        y =embedding[labels[0], 1]
        #ax2.scatter(np.mean(x), np.mean(y), label='ts' + str(ti)+'M'+str(maj), c='red', s=15)
        ax2.scatter(x,y, label='TS' + str(ti), c='red', s=10)
        #ax3.scatter(x, y, label='TS' + str(ti), c='red', s=10)
        ax2.scatter(embedding[tsi_list[jj],0], embedding[tsi_list[jj],1], label='TS' + str(tsi_list[jj]), c='pink', s=10) #PCs HNSW
        ax3.scatter(embedding[tsi_list[jj], 0], embedding[tsi_list[jj], 1], label='TS' + str(p1.labels[tsi_list[jj]]), c='pink',s=10)
        jj=jj+1
        ax2.text(np.mean(x) + 0.05, np.mean(y) + 0.05, 'TS' + str(ti), color='black', zorder=3)
        ax2.legend(fontsize=6)

    draw_sc_evolution_trajectory_dijkstra(p1, embedding, knn_hnsw, p1.full_graph_shortpath)
    '''
    root = 4823
    loc_i = np.where(np.asarray(p0.labels) == p0.root[0])[0]
    x = [embedding[xi, 0] for xi in loc_i]
    y = [embedding[yi, 1] for yi in loc_i]
    labels_root, distances_root = p.knn_query(np.array([np.mean(x), np.mean(y)]), k=1)
    x_root = embedding[labels_root[0], 0]
    y_root = embedding[labels_root[0], 1]

    #single-cell branch probability evolution probability
    for i,ti in enumerate(p1.terminal_clusters):

        fig, ax = plt.subplots()
        plot_sc_pb(ax, embedding, p1.single_cell_bp[:, i], ti)
        loc_i = np.where(np.asarray(p1.labels) == ti)[0]
        val_pt = [p1.single_cell_pt_markov[i] for i in loc_i]
        th_pt = np.percentile(val_pt, 0)  # 50
        loc_i = [loc_i[i] for i in range(len(val_pt)) if val_pt[i] >= th_pt]
        x = [embedding[xi, 0] for xi in loc_i]
        y = [embedding[yi, 1] for yi in loc_i]_pt(
        labels, distances = p.knn_query(np.array([np.mean(x), np.mean(y)]), k=1)
        x_sc = embedding[labels[0], 0]
        y_sc = embedding[labels[0], 1]
        ax.scatter(x_sc,y_sc, color='pink', zorder=3,label = str(ti))
        ax.text(x_sc+0.5,y_sc+0.5, 'TS'+str(ti), color = 'black')
        weights = p1.single_cell_bp[:,i]#/np.sum(p1.single_cell_bp[:,i])
        weights[weights<0.01] = 0
        weights[np.where(np.asarray(p0.labels)==p0.root)[0]] = 0.9
        weights[np.where(np.asarray(p1.labels)==ti)[0]] = 1
        weights[labels[0]]=10
        loc_z = np.where(weights>0)[0]
        min_weight = np.min(weights[weights!=0])
        weights[weights==0] = min_weight*0.000001
        #print('weights', weights)
        minx = min(x_root, x_sc)#np.min(x))
        maxx = max(x_root, x_sc)#np.max(x))
        xp = np.linspace(minx, maxx, 500)
        loc_i = np.where((embedding[:,0]<=maxx) &(embedding[:,0]>=minx))[0]
        loc_i = np.intersect1d(loc_i, loc_z)
        x_val = embedding[loc_i,0].reshape(len(loc_i),-1)
        #print('x-val shape0', x_val.shape)
        scGam = pg.LinearGAM(n_splines=10, spline_order=3, lam=10).fit(x_val,embedding[loc_i,1], weights =weights[loc_i].reshape(len(loc_i),-1) )
        ax.scatter(x_root,y_root, s=10, c='red')
        #XX = scGam.generate_X_grid(term=0, n=500)

        preds = scGam.predict(xp)
        ax.plot(xp, preds, linewidth=2, c='dimgray')
    '''
    plt.show()

def mainToy():


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
        root_str = 'M1'#"T1_M1", "T2_M1"] #"T1_M1"
        if dataset == "Toy1":
            df_counts = pd.read_csv("/home/shobi/Trajectory/Datasets/Toy1/toy_bifurcating_M4_n2000d1000.csv",
                                    'rt', delimiter=",")
            df_ids = pd.read_csv("/home/shobi/Trajectory/Datasets/Toy1/toy_bifurcating_M4_n2000d1000_ids.csv",
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
        if dataset == "Toy4":
            df_counts = pd.read_csv("/home/shobi/Trajectory/Datasets/Toy4/toy_disconnected_M9_n1000d1000.csv", 'rt',
                                    delimiter=",")
            df_ids = pd.read_csv("/home/shobi/Trajectory/Datasets/Toy4/toy_disconnected_M9_n1000d1000_ids.csv", 'rt',
                                 delimiter=",")

        df_ids['cell_id_num'] = [int(s[1::]) for s in df_ids['cell_id']]
        print("shape", df_counts.shape, df_ids.shape)
        df_counts = df_counts.drop('Unnamed: 0', 1)
        df_ids = df_ids.sort_values(by=['cell_id_num'])
        df_ids = df_ids.reset_index(drop=True)

        true_label = df_ids['group_id']
        adata_counts = sc.AnnData(df_counts, obs=df_ids)
        # sc.pp.recipe_zheng17(adata_counts, n_top_genes=20) not helpful for toy data

    ncomps =100
    knn =50



    sc.tl.pca(adata_counts, svd_solver='arpack', n_comps=ncomps)
    '''
    print(np.flatnonzero(adata_counts.obs['group_id'] == 'T1_M1')[0])
    adata_counts.uns['iroot'] = np.flatnonzero(adata_counts.obs['group_id'] == 'T1_M1')[0]

    sc.pp.neighbors(adata_counts, n_neighbors=knn, n_pcs=ncomps)#4
    sc.tl.draw_graph(adata_counts)
    sc.pl.draw_graph(adata_counts, color='group_id', legend_loc='on data') #force-directed layout
    start_dfmap = time.time()
    sc.tl.diffmap(adata_counts, n_comps=ncomps)
    print('time taken to get diffmap given knn', time.time() - start_dfmap)
    sc.pp.neighbors(adata_counts, n_neighbors=knn, use_rep='X_diffmap')#4
    sc.tl.draw_graph(adata_counts)
    sc.pl.draw_graph(adata_counts, color='group_id', legend_loc='on data')
    sc.tl.leiden(adata_counts, resolution=1.0)
    sc.tl.paga(adata_counts, groups='leiden')
    #sc.pl.paga(adata_counts, color=['louvain','group_id'])

    sc.tl.dpt(adata_counts, n_dcs=ncomps)
    sc.pl.paga(adata_counts, color=['leiden', 'group_id', 'dpt_pseudotime'], title=['leiden (knn:'+str(knn)+' ncomps:'+str(ncomps)+')', 'group_id (ncomps:'+str(ncomps)+')','pseudotime (ncomps:'+str(ncomps)+')'])
    #X = df_counts.values
    
    import palantir
    print(palantir.__file__) #location of palantir source code
    #counts = palantir.io.from_csv("/home/shobi/Trajectory/Datasets/Toy4/toy_disconnected_M9_n1000d1000.csv")
    counts = palantir.io.from_csv("/home/shobi/Trajectory/Datasets/Toy3/toy_multifurcating_M8_n1000d1000.csv")
    #counts = palantir.io.from_csv("/home/shobi/Trajectory/Datasets/ToyCyclic/ToyCyclic_M5_n3000d1000.csv")
    print('counts',counts)
    str_true_label = true_label.tolist()
    str_true_label = [(i[1:]) for i in str_true_label]

    str_true_label = pd.Series(str_true_label, index=counts.index)
    norm_df = counts#palantir.preprocess.normalize_counts(counts)
    pca_projections, _ = palantir.utils.run_pca(norm_df, n_components=ncomps)
    dm_res = palantir.utils.run_diffusion_maps(pca_projections, n_components=ncomps, knn=knn)

    ms_data = palantir.utils.determine_multiscale_space(dm_res) #n_eigs is determined using eigengap

    tsne = palantir.utils.run_tsne(ms_data)

    palantir.plot.plot_cell_clusters(tsne, str_true_label)
    start_cell = 'C108'#C108 for M12 connected' #M8n1000d1000 start - c107 #c1001 for bifurc n2000d1000 #disconnected n1000 c108, "C1 for M10 connected" # c10 for bifurcating_m4_n2000d1000
    print('ms data', ms_data)
    pr_res = palantir.core.run_palantir(ms_data, start_cell, num_waypoints=500,knn=knn)
    palantir.plot.plot_palantir_results(pr_res, tsne)
    plt.show()
    '''
    #clusters = palantir.utils.determine_cell_clusters(pca_projections)


    from sklearn.decomposition import PCA
    pca = PCA(n_components=ncomps)
    pc = pca.fit_transform(df_counts)

    p0 = PARC(adata_counts.obsm['X_pca'][:, 0:ncomps], true_label, jac_std_global=0.15,dist_std_local=1, knn=knn, too_big_factor=0.3,
              pseudotime=True, path="/home/shobi/Trajectory/Datasets/" + dataset + "/", root=2,
              root_str=root_str, preserve_disconnected=True, humanCD34=False)  # *.4
    p0.run_PARC()
    super_labels = p0.labels

    super_edges = p0.edgelist
    super_pt = p0.scaled_hitting_times  # pseudotime pt
    #0.05 for p1 toobig

    p = hnswlib.Index(space='l2', dim=adata_counts.obsm['X_pca'][:, 0:ncomps].shape[1])
    p.init_index(max_elements=adata_counts.obsm['X_pca'][:, 0:ncomps].shape[0], ef_construction=200, M=16)
    p.add_items(adata_counts.obsm['X_pca'][:, 0:ncomps])
    p.set_ef(50)
    tsi_list = [] #find the single-cell which is nearest to the average-location of a terminal cluster in PCA space (
    for tsi in p0.terminal_clusters:
        loc_i = np.where(np.asarray(p0.labels) == tsi)[0]
        val_pt = [p0.single_cell_pt_markov[i] for i in loc_i]
        th_pt = np.percentile(val_pt, 50)  # 50
        loc_i = [loc_i[i] for i in range(len(val_pt)) if val_pt[i] >= th_pt]
        temp = np.mean(adata_counts.obsm['X_pca'][:, 0:ncomps][loc_i], axis=0)
        labelsq, distances = p.knn_query(temp, k=1)
        print(labelsq[0])
        tsi_list.append(labelsq[0][0])

    p1 = PARC(adata_counts.obsm['X_pca'][:, 0:ncomps], true_label, jac_std_global=1, dist_std_local=0.15, knn=knn,
              too_big_factor=0.05,
              path="/home/shobi/Trajectory/Datasets/" + dataset + "/", pseudotime=True, root=1,
              super_cluster_labels=super_labels, super_node_degree_list=p0.node_degree_list,
              super_terminal_cells=tsi_list, root_str=root_str,
              x_lazy=0.99, alpha_teleport=0.99, preserve_disconnected=True,humanCD34=False)  # *.4


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
        embedding = TSNE().fit_transform(pc)#(adata_counts.obsm['X_pca'])
        print('tsne input size', adata_counts.obsm['X_pca'].shape)
        #embedding = umap.UMAP().fit_transform(adata_counts.obsm['X_pca'])
        idx = np.random.randint(len(labels), size=len(labels))
    print('end tsne')


    knn_hnsw, ci_list = sc_loc_ofsuperCluster_embeddedspace(embedding, p0, p1)

    draw_trajectory_gams(embedding, ci_list, labels, super_labels, super_edges,
                           p1.x_lazy, p1.alpha_teleport, p1.single_cell_pt_markov, true_label, knn=p0.knn,
                           terminal_clusters=p1.terminal_clusters, super_terminal_clusters=p0.terminal_clusters,
                           title_str='Hitting times: Markov Simulation on biased edges', ncomp=ncomps)
    plt.show()
    draw_trajectory_dimred(embedding, ci_list, labels, super_labels, super_edges,
                           p1.x_lazy, p1.alpha_teleport, p1.single_cell_pt_markov, true_label, knn=p0.knn,
                           terminal_clusters=p1.terminal_clusters, super_terminal_clusters=p0.terminal_clusters,
                           title_str='Hitting times: Markov Simulation on biased edges', ncomp=ncomps)
    plt.show()

    num_group = len(set(true_label))
    line = np.linspace(0, 1, num_group)

    f, (ax1, ax3) = plt.subplots(1, 2, sharey=True)

    for color, group in zip(line, set(true_label)):
        if len(labels) > n_downsample:
            where = np.where(np.array(true_label)[idx] == group)[0]
        else:
            where = np.where(np.array(true_label) == group)[0]
        ax1.scatter(embedding[where, 0], embedding[where, 1], label=group, c=np.asarray(plt.cm.jet(color)).reshape(-1,4))
    ax1.legend(fontsize=6)
    ax1.set_title('true labels')

    num_parc_group = len(set(labels))


    ax3.set_title("Markov Sim PT ncomps:"+str(pc.shape[1])+'. knn:'+str(knn))
    ax3.scatter(embedding[:, 0], embedding[:, 1], c=p1.single_cell_pt_markov, cmap='viridis_r')
    plt.show()
    draw_sc_evolution_trajectory_pt(p1, embedding, knn_hnsw)

    plt.show()
def main():
    dataset = 'Human' #Toy
    if dataset =='Human': mainHuman()
    else: mainToy()

if __name__ == '__main__':

    main()


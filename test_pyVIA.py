import matplotlib.pyplot as plt
import pyVIA.examples as via #pyVIA.examples as via #pyVIA.examples as via#examples as via #Viav031 as via
import pandas as pd
import umap
import scanpy as sc
import numpy as np
import warnings
#import os
#print(os.path.abspath(via.__file__))


# This is a test script for some of the examples shown on the github page and as Jupyter Notebooks (https://github.com/ShobiStassen/VIA)
# change the foldername accordingly to the folder containing relevant data files
# Please visit https://github.com/ShobiStassen/VIA for more detailed examples

def run_Toy_multi(foldername = "/home/shobi/Trajectory/Datasets/Toy3/"):
    #### Test Toy Multifurcation
    via.main_Toy(ncomps=10, knn=30,dataset='Toy3', random_seed=2,foldername= foldername)

def run_Toy_discon(foldername = "/home/shobi/Trajectory/Datasets/Toy4/"):
    #### Test Toy Disconnected
    via.main_Toy(ncomps=10, knn=30,dataset='Toy4', random_seed=2,foldername= foldername)

def run_EB(foldername = '/home/shobi/Trajectory/Datasets/EB_Phate/'):
    #### Test Embryoid Body
    via.main_EB_clean(ncomps=30, knn=20, v0_random_seed=24, foldername=foldername)

def run_generic_wrapper(foldername = "/home/shobi/Trajectory/Datasets/Bcell/", knn=20, ncomps = 20):
    #### Test pre-B cell differentiation using generic VIA wrapper

    # Read the two files:
    # 1) the first file contains 200PCs of the Bcell filtered and normalized data for the first 5000 HVG.
    # 2)The second file contains raw count data for marker genes

    data = pd.read_csv(foldername+'Bcell_200PCs.csv')
    data_genes = pd.read_csv(foldername+'Bcell_markergenes.csv')
    data_genes = data_genes.drop(['Unnamed: 0'], axis=1)#cell
    true_label = data['time_hour']
    data = data.drop(['cell', 'time_hour'], axis=1)
    adata = sc.AnnData(data_genes)
    adata.obsm['X_pca'] = data.values

    # use UMAP or PHate to obtain embedding that is used for single-cell level visualization
    embedding = umap.UMAP(random_state=42, n_neighbors=15, init='random').fit_transform(data.values[:, 0:5])
    print('finished embedding')
    # list marker genes or genes of interest if known in advance. otherwise marker_genes = []
    marker_genes = ['Igll1', 'Myc', 'Slc7a5', 'Ldha', 'Foxo1', 'Lig4', 'Sp7']  # irf4 down-up
    # call VIA. We identify an early (suitable) start cell root = [42]. Can also set an arbitrary value
    via.via_wrapper(adata, true_label, embedding, knn=knn, ncomps=ncomps, jac_std_global=0.15, root=[42], dataset='',
                random_seed=1,v0_toobig=0.3, v1_toobig=0.1, marker_genes=marker_genes, piegraph_edgeweight_scalingfactor=1, piegraph_arrow_head_width=.1)

def run_faced_cell_cycle(foldername = '/home/shobi/Trajectory/Datasets/FACED/'):
    #FACED imaging cytometry based biophysical features
    df = pd.read_csv(foldername +'mcf7_38features.csv')
    df = df.drop('Unnamed: 0', 1)

    true_label = pd.read_csv(foldername+'mcf7_phases.csv')
    true_label = list(true_label['phase'].values.flatten())
    print('There are ', len(true_label), 'MCF7 cells and ', df.shape[1], 'features')

    ad = sc.AnnData(df)
    ad.var_names = df.columns
    # normalize features
    sc.pp.scale(ad)

    sc.tl.pca(ad, svd_solver='arpack')
    # Weight the top features (ranked by Mutual Information and Random Forest Classifier)
    X_in = ad.X
    df_X = pd.DataFrame(X_in)
    df_X.columns = df.columns

    df_X['Area'] = df_X['Area'] * 3
    df_X['Dry Mass'] = df_X['Dry Mass'] * 3
    df_X['Volume'] = df_X['Volume'] * 20

    X_in = df_X.values
    ad = sc.AnnData(df_X)
    # apply PCA
    sc.tl.pca(ad, svd_solver='arpack')
    ad.var_names = df_X.columns

    f, ax = plt.subplots(figsize=[20, 10])

    embedding = umap.UMAP().fit_transform(ad.obsm['X_pca'][:, 0:20])
    # phate_op = phate.PHATE()
    # embedding = phate_op.fit_transform(X_in)

    cell_dict = {'T1_M1': 'yellow', 'T2_M1': 'yellowgreen', 'T1_M2': 'orange', 'T2_M2': 'darkgreen', 'T1_M3': 'red',
                 'T2_M3': 'blue'}
    cell_phase_dict = {'T1_M1': 'G1', 'T2_M1': 'G1', 'T1_M2': 'S', 'T2_M2': 'S', 'T1_M3': 'M/G2', 'T2_M3': 'M/G2'}

    for key in list(set(true_label)):  # ['T1_M1', 'T2_M1','T1_M2', 'T2_M2','T1_M3', 'T2_M3']:
        loc = np.where(np.asarray(true_label) == key)[0]
        ax.scatter(embedding[loc, 0], embedding[loc, 1], c=cell_dict[key], alpha=.7, label=cell_phase_dict[key])
    plt.legend(markerscale=1.5, fontsize=14)
    plt.show()

    knn = 20
    jac_std_global = 0.5
    random_seed = 1
    root_user = ['T1_M1']
    v0 = via.VIA(X_in, true_label, jac_std_global=jac_std_global, dist_std_local=1, knn=knn,
                 too_big_factor=0.3, root_user=root_user, dataset='faced', random_seed=random_seed, is_coarse=True, preserve_disconnected=True,
                 preserve_disconnected_after_pruning=True,
                 pseudotime_threshold_TS=40)
    v0.run_VIA()

    v1 = via.VIA(X_in, true_label, jac_std_global=jac_std_global, dist_std_local=1, knn=knn,
                 too_big_factor=0.05,  root_user=root_user, is_coarse=False,
                 preserve_disconnected=True, dataset='faced',  random_seed=random_seed,
                  pseudotime_threshold_TS=40, via_coarse=v0)
    v1.run_VIA()


    via.draw_trajectory_gams(via_coarse=v0, via_fine=v1, embedding=embedding)
    plt.show()

    all_cols = ['Area', 'Volume', 'Dry Mass', 'Circularity', 'Orientation', 'Phase Entropy Skewness',
                'Phase Fiber Radial Distribution', 'Eccentricity', 'AspectRatio', 'Dry Mass Density', 'Dry Mass var',
                'Dry Mass Skewness', 'Peak Phase', 'Phase Var', 'Phase Skewness', 'Phase Kurtosis', 'Phase Range',
                'Phase Min', 'Phase Centroid Displacement', 'Phase STD Mean', 'Phase STD Variance',
                'Phase STD Skewness', 'Phase STD Kurtosis', 'Phase STD Centroid Displacement',
                'Phase STD Radial Distribution', 'Phase Entropy Mean', 'Phase Entropy Var', 'Phase Entropy Kurtosis',
                'Phase Entropy Centroid Displacement', 'Phase Entropy Radial Distribution',
                'Phase Fiber Centroid Displacement', 'Phase Fiber Pixel >Upper Percentile', 'Phase Fiber Pixel >Median',
                'Mean Phase Arrangement', 'Phase Arrangement Var', 'Phase Arrangement Skewness',
                'Phase Orientation Var', 'Phase Orientation Kurtosis']
    plot_n = 7
    fig, axs = plt.subplots(2, plot_n, figsize=[20, 10])  # (2,10)
    for enum_i, pheno_i in enumerate(all_cols[0:14]):  # [0:14]
        subset_ = df[pheno_i].values

        if enum_i >= plot_n:
            row = 1
            col = enum_i - plot_n
        else:
            row = 0
            col = enum_i
        ax = axs[row, col]
        v0.get_gene_expression_multi(ax=ax, gene_exp=subset_, title_gene=pheno_i)

    fig2, axs2 = plt.subplots(2, plot_n, figsize=[20, 10])
    for enum_i, pheno_i in enumerate(all_cols[2 * plot_n:2 * plot_n + 14]):
        subset_ = df[pheno_i].values

        if enum_i >= plot_n:
            row = 1
            col = enum_i - plot_n
        else:
            row = 0
            col = enum_i

        ax2 = axs2[row, col]
        v0.get_gene_expression_multi(ax=ax2, gene_exp=subset_, title_gene=pheno_i)

    plt.show()


def run_scATAC_Buenrostro_Hemato(foldername = '/home/shobi/Trajectory/Datasets/scATAC_Hemato/', knn=20):

    df = pd.read_csv(foldername+'scATAC_hemato_Buenrostro.csv', sep=',')
    print('number cells', df.shape[0])
    cell_types = ['GMP', 'HSC', 'MEP', 'CLP', 'CMP', 'LMuPP', 'MPP', 'pDC', 'mono', 'UNK']
    cell_dict = {'UNK': 'gray', 'pDC': 'purple', 'mono': 'gold', 'GMP': 'orange', 'MEP': 'red', 'CLP': 'aqua',
                 'HSC': 'black', 'CMP': 'moccasin', 'MPP': 'darkgreen', 'LMuPP': 'limegreen'}
    cell_annot = df['cellname'].values

    true_label = []
    found_annot = False
    #re-formatting labels (abbreviating the original annotations for better visualization on plot labels)
    for annot in cell_annot:
        for cell_type_i in cell_types:
            if cell_type_i in annot:
                true_label.append(cell_type_i)
                found_annot = True

        if found_annot == False:
            true_label.append('unknown')
        found_annot = False

    PCcol = ['PC1', 'PC2', 'PC3', 'PC4', 'PC5']
    embedding = umap.UMAP(n_neighbors=20, random_state=2, repulsion_strength=0.5).fit_transform(df[PCcol])

    fig, ax = plt.subplots(figsize=[20, 10])
    for key in cell_dict:
        loc = np.where(np.asarray(true_label) == key)[0]
        ax.scatter(embedding[loc, 0], embedding[loc, 1], c=cell_dict[key], alpha=0.7, label=key, s=90)

    plt.legend(fontsize='large', markerscale=1.3)
    plt.title('Original Annotations on UMAP embedding')
    plt.show()

    knn = knn
    random_seed = 4
    X_in = df[PCcol].values

    start_ncomp = 0
    root = [1200]  # HSC cell

    v0 = via.VIA(X_in, true_label, jac_std_global=0.5, dist_std_local=1, knn=knn,
                 too_big_factor=0.3, root_user=root, dataset='', random_seed=random_seed, is_coarse=True, preserve_disconnected=False)
    v0.run_VIA()
    via.via_streamplot(v0, embedding)
    plt.show()
    via.draw_sc_lineage_probability(v0, v0, embedding, scatter_size=5)
    plt.show()
    v1 = via.VIA(X_in, true_label, jac_std_global=0.15, dist_std_local=1, knn=knn,
                 too_big_factor=0.1, super_cluster_labels=v0.labels, root_user=root, is_coarse=False,
                 preserve_disconnected=True, dataset='',
                 random_seed=random_seed, via_coarse=v0)
    v1.run_VIA()

    df['via1'] = v1.labels
    df_mean = df.groupby('via1', as_index=False).mean()
    gene_dict = {'ENSG00000092067_LINE336_CEBPE_D_N1': 'CEBPE Eosophil (GMP/Mono)',
                 'ENSG00000102145_LINE2081_GATA1_D_N7': 'GATA1 (MEP)'}
    for key in gene_dict:
        v1.draw_piechart_graph(type_data='gene', gene_exp=df_mean[key].values, title=gene_dict[key])
        plt.show()
    # get knn-graph and locations of terminal states in the embedded space
        # draw overall pseudotime and main trajectories

    via.draw_trajectory_gams(v0, v1, embedding)
    plt.show()
    # draw trajectory and evolution probability for each lineage
    via.draw_sc_lineage_probability(v0, v1, embedding)
    plt.show()

def run_generic_discon(foldername ="/home/shobi/Trajectory/Datasets/Toy4/"):
    df_counts = pd.read_csv(foldername + "toy_disconnected_M9_n1000d1000.csv",
                            delimiter=",")
    df_ids = pd.read_csv(foldername + "toy_disconnected_M9_n1000d1000_ids.csv", delimiter=",")



    df_ids['cell_id_num'] = [int(s[1::]) for s in df_ids['cell_id']]

    df_counts = df_counts.drop('Unnamed: 0', 1)
    df_ids = df_ids.sort_values(by=['cell_id_num'])
    df_ids = df_ids.reset_index(drop=True)
    #true_label = df_ids['group_id']
    #true_label =['a' for i in true_label] #testing dummy true_label and overwriting the real true_labels
    #true_time = df_ids['true_time']
    adata_counts = sc.AnnData(df_counts, obs=df_ids)
    sc.tl.pca(adata_counts, svd_solver='arpack', n_comps=100)

    via.via_wrapper_disconnected(adata_counts, true_label=None, embedding=adata_counts.obsm['X_pca'][:, 0:2], root=[902,23],
                             preserve_disconnected=True, knn=10, ncomps=30, cluster_graph_pruning_std=1, random_seed=41)

    #order of roots input by the user does not matter. Via re-orders the roots so that the roots correspond to the components such that in self.root the ith root corresponds to the ith graph component

if __name__ == '__main__':

    warnings.filterwarnings('ignore')

    #run_Toy_multi(foldername="/home/shobi/Trajectory/Datasets/Toy3/")
    #run_Toy_discon()
    run_generic_discon()
    #run_EB(foldername = "/home/shobi/Trajectory/Datasets/EB_Phate/") #folder containing relevant data files
    #run_scATAC_Buenrostro_Hemato() #shows the main function calls within a typical VIA wrapper function
    #run_generic_wrapper(foldername = "/home/shobi/Trajectory/Datasets/Bcell/", knn=15, ncomps = 20)
    #run_faced_cell_cycle(foldername = '/home/shobi/Trajectory/Datasets/FACED/')

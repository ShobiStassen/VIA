# Via
VIA is a Trajectory Inference (TI) method that offers topology construction, pseudotimes, automated terminal state prediction and automated plotting of temporal gene dynamics along lineages. VIA combines lazy-teleporting random walks and Monte-Carlo Markov Chain simulations to overcome commonly encountered challenges such as 1) accurate terminal state and lineage inference, 2) ability to capture combination of cyclic, disconnected and tree-like structures, 3) scalability in feature and sample space. 

## Getting Started
### install using pip
We recommend setting up a new conda environment
```
conda create --name ViaEnv pip 
pip install pyVIA // tested on linux
```
### install by cloning repository and running setup.py (ensure dependencies are installed)
```
git clone https://github.com/ShobiStassen/VIA.git 
python3 setup.py install // cd into the directory of the cloned PARC folder containing setup.py and issue this command
```

### install dependencies separately if needed (linux)
If the pip install doesn't work, it usually suffices to first install all the requirements (using pip) and subsequently install VIA (also using pip)
We note that the latest version of leidenalg (0.8.0. released April 2020) is slower than its predecessor. Please ensure that the leidenalg installed is version 0.7.0 for the time being. Some of the examples use umap and/or phate so we make note of them below
```
pip install python-igraph, leidenalg==0.7.0, hnswlib, umap-learn, numpy>=1.17, scipy, pandas>=0.25, sklearn, termcolor, pygam, phate
pip install pyVIA
```
## Examples
### Human Embryoid 
save the [Raw data](https://drive.google.com/file/d/1yz3zR1KAmghjYB_nLLUZoIlKN9Ew4RHf/view?usp=sharing) matrix as 'EBdata.mat'. The cells in this file have been filtered for too small/large libraries by [Moon et al. 2019](https://nbviewer.jupyter.org/github/KrishnaswamyLab/PHATE/blob/master/Python/tutorial/EmbryoidBody.ipynb) 

The function main_EB_clean() preprocesses the cells (normalized by library size, sqrt transformation). It then calls VIA to: plot the pseudotimes, terminal states, lineage pathways and gene-clustermap.
```
import pyVia.core as via
via.main_EB_clean(ncomps=30, knn=20, p0_random_seed=20, foldername = '') # Most reasonable parameters of ncomps (10-200) and knn (15-50) work well
```
If you wish to run the data using UMAP or tsne, or require more control of the parameters/outputs, then use the following code:
```
import pyVia.core as via
#pre-process the data as needed and provide to via as a numpy array
#root_user is the index of the cell corresponding to a suitable start/root cell

v0 = via.VIA(input_data, time_labels, jac_std_global=0.15, dist_std_local=1, knn=knn,too_big_factor=p0_too_big, 
root_user=1, dataset='EB', random_seed=p0_random_seed, do_magic_bool=True, is_coarse=True, preserve_disconnected=True) 
v0.run_VIA()

tsi_list = get_loc_terminal_states(v0, input_data) #translate the terminal clusters found in v0 to the fine-grained run in v1

v1 = VIA(input_data, time_labels, jac_std_global=0.15, dist_std_local=1, knn=knn,
             too_big_factor=v1_too_big,super_cluster_labels=p0.labels, super_node_degree_list=v0.node_degree_list,
             super_terminal_cells=tsi_list, root_user=1,x_lazy=0.99, alpha_teleport=0.99, preserve_disconnected=True, dataset='EB',
             super_terminal_clusters=v0.terminal_clusters, random_seed=p0_random_seed)
v1.run_VIA()

#Plot the true and inferred times and pseudotimes
#Replace Y_phate with UMAP, TSNE embedding
f, (ax1, ax2) = plt.subplots(1, 2, sharey=True)
ax1.scatter(Y_phate[:, 0], Y_phate[:, 1], c=time_labels, s=5, cmap='viridis', alpha=0.5)
ax2.scatter(Y_phate[:, 0], Y_phate[:, 1], c=p1.single_cell_pt_markov, s=5, cmap='viridis', alpha=0.5)
ax1.set_title('Embyroid Data: Days')
ax2.set_title('Embyroid Data: Randomseed' + str(p0_random_seed))
plt.show()

#obtain the single-cell locations of the terminal clusters to be used for visualization of trajectories/lineages 
super_clus_ds_PCA_loc = via.sc_loc_ofsuperCluster_PCAspace(v0, v1, np.arange(0, len(v1.labels)))
#draw the overall lineage paths on the embedding
via.draw_trajectory_gams(Y_phate, super_clus_ds_PCA_loc, p1.labels, v0.labels, v0.edgelist_maxout,
                     v1.x_lazy, v1.alpha_teleport, v1.single_cell_pt_markov, time_labels, knn=v0.knn,
                     final_super_terminal=v1.revised_super_terminal_clusters,
                     sub_terminal_clusters=v1.terminal_clusters,
                     title_str='Markov Hitting Times (Gams)', ncomp=ncomps)

2D_knn_hnsw = via.make_knn_embeddedspace(Y_phate)
#draw the individual lineage paths and cell-fate probabilities at single-cell level 
via.draw_sc_evolution_trajectory_dijkstra(v1, Y_phate, 2D_knn_hnsw, v0.full_graph_shortpath,
                                      idx=np.arange(0, input_data.shape[0]), X_data=input_data)

plt.show()
```
![Output of VIA on Human Embryoid](https://github.com/ShobiStassen/VIA/blob/master/Figures/EB_fig0.png)

### Toy data (Multifurcation and Disconnected)
Two example [toy datasets](https://drive.google.com/drive/folders/1WQSZeNixUAB1Sm0Xf68ZnSLQXyep936l?usp=sharing) with annotations are generated using DynToy are provided. 
```
import pyVia.core as via
via.main_Toy(ncomps=10, knn=30,dataset='Toy3', random_seed=2,foldername = ".../Trajectory/Datasets/") #multifurcation
via.main_Toy(ncomps=10, knn=30,dataset='Toy4',random_seed=2,foldername =".../Trajectory/Datasets/") #2 disconnected trajectories
```
## Output of Multifurcating toy dataset
![Output of VIA on Human Embryoid](https://github.com/ShobiStassen/VIA/blob/master/Figures/Toy3_fig0.png)
## Output of disconnected toy dataset
![Output of VIA on Human Embryoid](https://github.com/ShobiStassen/VIA/blob/master/Figures/Toy4_fig0.png)

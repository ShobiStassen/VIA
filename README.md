pyVIA - [Multi-Omic Single-Cell Cartography](https://pyvia.readthedocs.io/en/latest/) 
====================================================

**Via 2.0** is our new single-cell trajectory inference method that explores **single-cell atlas-scale** data and **temporal studies** enabled by. In addition to the full functionality of earlier versions, Via 2.0 now offers

1. **Integration of metadata (e.g time-series labels):** Using sequential metadata (temporal labels from longitudinal studies, hierarchical information from phylogenetic trees, spatial distances relevant to spatial omics data) to guide the cartography.  Integrating RNA-velocity where applicable. 
2. **Higher Order Random Walks:** Leveraging higher order random walks with **memory** to highlight key end-to-end differentiation pathways along the atlas 
3. **Atlas View:** Via 2.0 offers a unique visualization of the predicted trajectory by intuitively merging the cell-cell graph connectivity with the high-resolution of single-cell embeddings.
4. **Generalizable and data modality agnostic** Via 2.0 still offers all the functionality of Via 1.0 across single-cell data modalities (scRNA-seq, imaging and flow cyometry, scATAC-seq) for types of topologies  (disconnected, cyclic, tree) to infer pseudotimes, automated terminal state prediction and automated plotting of temporal gene dynamics along lineages. 

<p align="center">
     <img src="https://github.com/ShobiStassen/VIA/blob/master/Figures/rtd_fig1.png?raw=true" width="750px" align="center", class="only-light" >
</p>


Via 2.0 extends the lazy-teleporting walks to higher order random walks with **memory** to allow better lineage detection, pathway recovery and preservation of global features in terms of computation and visualization. The cartographic approach combining high edge and spatial resolution produces informative and esthetically pleasing visualizations caled the Atlas View. 

If you find our work useful, please consider citing our **[paper](https://www.nature.com/articles/s41467-021-25773-3)** [![DOI](https://zenodo.org/badge/212254929.svg)](https://zenodo.org/badge/latestdoi/212254929). 


## Tutorials for Cartographic TI and Visualization using Via 2.0
Tutorials and **[videos](https://pyvia.readthedocs.io/en/latest/Tutorial%20Video.html)**  available on **[readthedocs](https://pyvia.readthedocs.io/en/latest/)** with step-by-step code for real and simulated datasets. Tutorials explain how to generate cartographic visualizations for TI, tune parameters, obtain various outputs and also understand the importance of *memory*.

#### :eight_spoked_asterisk:Cartography of Zebrafish gastrulation
<p align="center">
     <img src="https://github.com/ShobiStassen/VIA/blob/master/Figures/AtlasGallery/atlas_view_zebrahub.png?raw=true" alt="Trulli" width="80%" >
</p>

#### :eight_spoked_asterisk: windmaps of mouse gastrulation
<p align="center">
     <img src="https://github.com/ShobiStassen/VIA/blob/master/Figures/plasma_pijuansala_annotated.gif?raw=true" alt="Trulli" width="80%" >
</p>

:eight_spoked_asterisk: You can start with the **[The tutorial/Notebook](https://pyvia.readthedocs.io/en/latest/Basic%20tutorial.html)** for multifurcating data which shows a step-by-step use case. :eight_spoked_asterisk: 


**scATAC-seq dataset of Human Hematopoiesis represented by VIA graphs** *(click image to open interactive graph)*
<p align="center">
     <a href="https://shobistassen.github.io/toggle_data.html"><img width="100%" src="https://github.com/ShobiStassen/VIA/blob/master/Figures/scATAC_BuenrostroPCs_MainFig.png?raw=true"></a>
</p> 

#### :eight_spoked_asterisk: Fine-grained vector field without using RNA-velocity
Refer to the Jupiter Notebooks to plot these fine-grained vector fields of the sc-trajectories even when there is no RNA-velocity available.
<p align="center">
     <img src="https://github.com/ShobiStassen/VIA/blob/master/Figures/streamplotspng.png?raw=true" alt="Trulli" width="80%" >
</p> 

## Tutorials on [readthedocs](https://pyvia.readthedocs.io/en/latest/)
Please visit our **[readthedocs](https://pyvia.readthedocs.io/en/latest/)** for the latest tutorials and **[videos](https://pyvia.readthedocs.io/en/latest/Tutorial%20Video.html)** on usage and installation
notebook       | details         | dataset  | reference 
---------------| ---------------| ---------| ----------
[*Multifurcation: Starter Tutorial*](https://pyvia.readthedocs.io/en/latest/ViaJupyter_Toy_Multifurcating.html) | 4-leaf simulation | [4-leaf](https://github.com/ShobiStassen/VIA/tree/master/Datasets) | [DynToy](https://github.com/dynverse/dyntoy)
[*Disconnected*](https://pyvia.readthedocs.io/en/latest/ViaJupyter_Toy_Disconnected.html) | disconnected simulation | [Disconnected](https://github.com/ShobiStassen/VIA/tree/master/Datasets) | [DynToy](https://github.com/dynverse/dyntoy)
[*Zebrafish Gastrulation*](https://pyvia.readthedocs.io/en/latest/Zebrahub%20TI%20tutorial.html) | Time series of 120,000 cells | [Zebrahub](https://drive.google.com/drive/folders/1cr_mq94qZDoJLNDkAMukHYDGa4kkrMat?usp=drive_link)| Lange et al. (2023)
[*Mouse Gastrulation*](https://pyvia.readthedocs.io/en/latest/Via2.0%20Cartographic%20Mouse%20Gastrualation.html) | Time series of 90,000 cells | [Mouse data](https://drive.google.com/drive/folders/1EaYoQadm-s6gDJb_YUeeyxZ6LOITLqVK?usp=sharing)| Sala et al. (2019)
[*scRNA-seq Hematopoiesis*](https://pyvia.readthedocs.io/en/latest/ViaJupyter_scRNA_Hematopoiesis.html) | Human hematopoiesis (5780 cells) | [CD34 scRNA-seq](https://drive.google.com/file/d/1ZSZbMeTQQPfPBGcnfUNDNL4om98UiNcO/view?usp=sharing) | Setty et al. (2019)
[*FACED image-based*](https://pyvia.readthedocs.io/en/latest/Imaging%20Cytometry%20%28cell%20cycle%29.html) | 2036 MCF7 cells in cell cycle | [MCF7 FACED](https://github.com/ShobiStassen/VIA/tree/master/Datasets) | in-house data
[*scATAC-seq Hematopoiesis*]() | Human hematopoiesis | [scATAC-seq](https://github.com/ShobiStassen/VIA/tree/master/Datasets) | Buenrostro et al. (2018)
[*Human Embryoid*](https://github.com/ShobiStassen/VIA/blob/master/Jupyter%20Notebooks/ViaJupyter_EmbryoidBody.ipynb) | 16,825 ESCs | [EB scRNA-seq](https://drive.google.com/file/d/1yz3zR1KAmghjYB_nLLUZoIlKN9Ew4RHf/view?usp=sharing) and [embedding](https://github.com/ShobiStassen/VIA/tree/master/Datasets)| Moon et al. (2019)
## Datasets
Dataset are available in the [Datasets folder](https://github.com/ShobiStassen/VIA/tree/master/Datasets) (smaller files) with larger datasets [here](https://drive.google.com/drive/folders/1WQSZeNixUAB1Sm0Xf68ZnSLQXyep936l?usp=sharing).  

      
------------------------------------------------------

### To run on Windows:
All examples and tests have been run on Linux and MAC OS. We find there are somtimes small modifications required to run on a Windows OS (see below). Windows requires minor modifications in calling the code due to the way multiprocessing works in Windows compared to Linux:
```
#when running from an IDE you need to call the function in the following way to ensure the parallel processing works:
import os
import pyVIA.core as via
f= os.path.join(r'C:\Users\...\Documents'+'\\')
def main():
    via.main_Toy(ncomps=10, knn=30,dataset='Toy3', random_seed=2,foldername= f)    
if __name__ =='__main__':
    main()
    
#when running directly from terminal:
import os
import pyVIA.core as via
f= os.path.join(r'C:\Users\...\Documents'+'\\')
via.main_Toy(ncomps=10, knn=30,dataset='Toy3', random_seed=2,foldername= f)    

```

## Installation
### Linux Ubuntu 16.04 and Windows 10 Installation
We recommend setting up a new conda environment. You can use the examples below, the [Jupyter notebooks](https://github.com/ShobiStassen/VIA/tree/master/Jupyter%20Notebooks) and/or the [test script](https://github.com/ShobiStassen/VIA/blob/master/test_pyVIA.py) to make sure your installation works as expected.
```
conda create --name ViaEnv python=3.7 
pip install pyVIA // tested on linux Ubuntu 16.04 and Windows 10
```
This usually tries to install hnswlib, produces an error and automatically corrects itself by first installing pybind11 followed by hnswlib. To get a smoother installation, consider installing in the following order after creating a new conda environment:
```
pip install pybind11
pip install hnswlib
pip install pyVIA
```
### Install by cloning repository and running setup.py (ensure dependencies are installed)
```
git clone https://github.com/ShobiStassen/VIA.git 
python3 setup.py install // cd into the directory of the cloned VIA folder containing setup.py and issue this command
```

### MAC installation
The pie-chart cluster-graph plot does not render correctly for MACs for the time-being. All other outputs are as expected. 
```
conda create --name ViaEnv python=3.7 
pip install pybind11
conda install -c conda-forge hnswlib
pip install pyVIA
```

### Install dependencies separately if needed (linux ubuntu 16.04 and Windows 10)
If the pip install doesn't work, it usually suffices to first install all the requirements (using pip) and subsequently install VIA (also using pip). 
Note that on Windows if you do not have Visual C++ (required for hnswlib) you can install using [this link](https://www.scivision.dev/python-windows-visual-c-14-required/) . 
```
pip install pybind11, hnswlib, igraph, leidenalg>=0.7.0, umap-learn, numpy>=1.17, scipy, pandas>=0.25, sklearn, termcolor, pygam, phate, matplotlib,scanpy
pip install pyVIA
```
## Parameters and Attributes

### Parameters

| Input Parameter for class VIA | Description |
| ---------- |----------|
| `data` | (numpy.ndarray) n_samples x n_features. When using via_wrapper(), data is ANNdata object that has a PCA object adata.obsm['X_pca'][:, 0:ncomps] and ncomps is the number of components that will be used. |
| `true_label` | (list) 'ground truth' annotations or placeholder|
| `memory` | (float) default =5 higher memory means lineage pathways that deviate less from predecessors|
| `times_series` | (bool) default=False. whether or not sequential augmentation of the TI graph will be done based on time-series labels|
| `time_series_labels` | (list) list (length n_cells) of numerical values corresponding to sequential/chronological/hierarchical sequence|
| `knn` |  (optional, default = 30) number of K-Nearest Neighbors for HNSWlib KNN graph |
|`root_user`| root_user should be provided as a list containing roots corresponding to index (row number in cell matrix) of root cell. For most trajectories this is of the form [53] where 53 is the index of a sensible root cell, for multiple disconnected trajectories an arbitrary list of cells can be provided [1,506,1100], otherwise VIA arbitratily chooses cells. If the root cells of disconnected trajectories are known in advance, then the cells should be annotated with similar syntax to that of Example Dataset in Disconnected Toy Example 1b.|
| `dist_std_local` |  (optional, default = 1) local pruning threshold for PARC clustering stage: the number of standard deviations above the mean minkowski distance between neighbors of a given node. the higher the parameter, the more edges are retained|
| `edgepruning_clustering_resolution` |  (optional, default = 0.15) global level  graph pruning for PARC clustering stage. 0.1-1 provide reasonable pruning. higher value means less pruning. e.g. a value of 0.15 means all edges that are above mean(edgeweight)-0.15*std(edge-weights) are retained. We find both 0.15 and 'median' to yield good results resulting in pruning away ~ 50-60% edges |
| `too_big_factor` |  (optional, default = 0.4) if a cluster exceeds this share of the entire cell population, then the PARC will be run on the large cluster|
|`cluster_graph_pruning`| (optional, default =0.15) To retain more edges/connectivity in the graph underlying the trajectory computations, increase the value|
| `edgebundle_pruning` |  (optional) default value is the same as cluster_grap_pruning. Only impacts the visualized edges, not the underlying edges for computation and TI|
|`x_lazy`| (optional, default = 0.95) 1-x = probability of staying in same node (lazy). Values between 0.9-0.99 are reasonable|
|`alpha_teleport`| (optional, default = 0.99) 1-alpha is probability of jumping. Values between 0.95-0.99 are reasonable unless prior knowledge of teleportation 
| `distance` |  (optional, default = 'l2' euclidean) 'ip','cosine'|
| `random_seed` |  (optional, default = 42) The random seed to pass to Leiden|
| `pseudotime_threshold_TS` |  (optional, default = 30) Percentile threshold for potential node to qualify as Terminal State|
| `resolution_parameter` |  (optional, default = 1) Uses ModuliartyVP and RBConfigurationVertexPartition|
| `preserve_disconnected` |  (optional, default = True) If you do not think there should be any disconnected trajectories, set this to False |



| Attributes | Description |
| ---------- |----------|
| `labels` | (list) length n_samples of corresponding cluster labels |
| `single_cell_pt_markov` | (list) computed pseudotime|
|`embedding`|2d array representing a computed embedding|
|`single_cell_bp`|(array) computed single cell branch probabilities (lineage likelihoods). n_cells x n_terminal states. The columns each correspond to a terminal state, in the same order presented in the'terminal_clusters' attribute|
| `terminal clusters` | (list) terminal clusters found by VIA|
|`full_neighbor_array`|full_neighbor_array=v0.full_neighbor_array. KNN graph from first pass of via - neighbor array|
|`full_distance_array`|full_distance_array=v0.full_distance_array. KNN graph from first pass of via - edge weights|
|`ig_full_graph`|ig_full_graph=v0.ig_full_graph igraph of the KNN graph from first pass of via|
|`csr_full_graph`| csr_full_graph. If time_series is true, this is sequentially augmented.|
|`csr_array_locally_pruned`|csr_array_locally_pruned=v0.csr_array_locally_pruned. CSR matrix of the locally pruned KNN graph|

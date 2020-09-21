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
git clone https://github.com/ShobiStassen/PARC.git 
python3 setup.py install // cd into the directory of the cloned PARC folder containing setup.py and issue this command
```

### install dependencies separately if needed (linux)
If the pip install doesn't work, it usually suffices to first install all the requirements (using pip) and subsequently install parc (also using pip)
We note that the latest version of leidenalg (0.8.0. released April 2020) is slower than its predecessor. Please ensure that the leidenalg installed is version 0.7.0 for the time being.
```
pip install python-igraph, leidenalg==0.7.0, hnswlib, umap-learn
pip install pyVIA
```
## Examples
### Human Embryoid 
save the [Raw data](https://drive.google.com/file/d/1yz3zR1KAmghjYB_nLLUZoIlKN9Ew4RHf/view?usp=sharing) matrix as'EBdata.mat'  # the cells have been filtered for too small/large libraries by Moon et al. 2019 

The function main_EB_clean() preprocesses the cells (normalized by library size, sqrt transformation). It then calls VIA to: plot the pseudotimes, terminal states, lineage pathways and gene-clustermap.
```
import pyVia.core as via
via.main_EB_clean(ncomps=30, knn=20, p0_random_seed=20, foldername = '') # Most reasonable parameters of ncomps (10-200) and knn (15-50) work well
```
![Output of VIA on Human Embryoid](https://github.com/ShobiStassen/VIA/blob/master/Figures/EB_fig.png)

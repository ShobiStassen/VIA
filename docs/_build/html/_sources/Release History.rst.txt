Release History
===============

Version 0.1.82
-------------
- minor bugs in parameter naming conventions for gene trend plotting function

Version 0.1.77
-------------
- updated for compatibilty with umap-learn >=0.5.0
- bugfix of argument ``labels`` (predetermined cluster labels, overrides via's clustering): edited the api to clarify that this needs to be of type and size ``ndarray (nsamples, )``
- updated function via_umap() to run with simply ``via_umap(via_object = v0)`` or ``via_umap(via_object = v0, init='via')``

Version 0.1.73
-------------
- added lineage pathway visualization to improve the existing edge plotting function ``plot_edge_bundle()``
- for ``plot_edge_bundle``, the parameter ``lineage_pathway:list = []`` can be filled with cluster labels from the list of terminal cluster lineages in order to see the fine-grained lineage pathways along edges 
- example:  ``plot_edge_bundle(via_object=v0, lineage_pathway=[7,10,9], linewidth_bundle=0.5, headwidth_bundle=2, cmap='plasma',text_labels=True, show_milestones=True, scale_scatter_size_pop=True)``
- if you wish to recompute the edge based visualization with different resolution you have two options:
- 1. set attribute ``via_object.hammerbundle_milestone_dict = None`` and then rereun ``plot_edge_bundle()``
- 2. Or ``via_object.hammerbundle_milestone_dict=make_edgebundle_milestone(via_object=v0, n_milestones=40)`` followed by ``plot_edge_bundle(...)``

- Examine the single-cell lineage probabilities of each cell towards a particular terminal state using: 
- 1. see ``via_object.single_cell_bp_rownormed``
- 2. see ``via_object.single_cell_bp`` is actually lineage normalized so that the probabilities along a very rare lineage are not hidden. 

.. raw:: html

  <img src="https://github.com/ShobiStassen/VIA/blob/master/Figures/Toy3_lineage_path_edgebundle.png?raw=true" width="600px" align="center" </a>


Version 0.1.71
-------------
- small bugfixes in plotting functions

Version 0.1.70
-------------
- bugfix ``draw_sc_lineage_probability()`` to allow plotting multiple and single lineages in same plot

Version 0.1.68
-------------
- improved ``draw_sc_lineage_probability()`` to correctly assign subplots when ``marker_lineages = []`` is given by user as a subset of the terminal_clusters attribute of via. (corrections to the update in 0.1.57)
- control fontsize
- New Feature! VIA can autocompute via-umap and via-mds by passing ``do_compute_embedding = True`` and ``embedding_type = 'via-mds' or 'via-umap'`` when initializing a via object. e.g. ``v0 = via.VIA(X_data....)`` or after doing ``v0.run_via()``

Version 0.1.64
-------------
- Bugfix in ``via_mds()`` parameter ``saveto=''``

Version 0.1.62
-------------
- New Feature! Heatmap based gene trends (genes x pseudotime) for each lineage 
``plot_gene_trend_heatmaps(via_object, df_gene_exp:pd.DataFrame,...)``

- New Feature! annotate nodes in plot_edge_bundle() by setting text_labels = True uses true_labels as annotations. Optionally provide list of single cell annotations length n_samples to use instead of true_labels. Example figure below
``plot_edge_bundle(via_object, text_labels=True)``

.. raw:: html

  <img src="https://github.com/ShobiStassen/VIA/blob/master/Figures/milestoneplot_withannots.png?raw=true" width="600px" align="center" </a>

.. raw:: html

  <img src="https://github.com/ShobiStassen/VIA/blob/master/Figures/gene_pt_heatmap_example.png?raw=true" width="600px" align="center" </a>


Version 0.1.61
-------------
- Bug fix for import module in examples.py

Version 0.1.60
-------------
- Bug fix for root detection (the initialization for None was over-writing RNA-velocity predicted roots)

Version 0.1.59
-------------
- corrected the auto-scaling in ``draw_sc_lineage_probability()`` so that each subplot has the same colorbar scale

Version 0.1.58
-------------
- fix random_seed so pseudotime and branching probabilities are reproducible

Version 0.1.57
-------------
- optionally allow user to fix terminal states based on cell index or group label (corresponding to true_label)
- optionally allow user to plot only selected lineages (by corresponding terminal cluster number) in get_gene_expression and draw_sc_lineage_probability (marker_lineages = [2,8,10])

Version 0.1.56
-------------
- support via-guided embeddings. In particular fast via-mds and via-umap which can be adjusted using known time-series data if available


Release History
===============

Version 0.1.71
-------------
- small bugfixes in plotting functions

Version 0.1.70
-------------
- fixed `draw_sc_lineage_probability()` to allow plotting single lineage

Version 0.1.68
-------------
- improved ``draw_sc_lineage_probability()`` to correctly assign subplots when ``marker_lineages = []`` is given by user as a subset of the terminal_clusters attribute of via. (corrections to the update in 0.1.57)
- control fontsize
- New Feature! VIA can autocompute via-umap and via-mds by passing do_compute_embedding = True and embedding_type = 'via-mds' or 'via-umap' when initializing a via object. e.g. ``v0 = via.VIA(X_data....)`` or after doing ``v0.run_via()``

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

Version 0.1.61
-------------
- Bug fix for import module in examples.py

Version 0.1.60
-------------
- Bug fix for root detection (the initialization for None was over-writing RNA-velocity predicted roots)

Version 0.1.59
-------------
- corrected the auto-scaling in draw_sc_lineage_probability() so that each subplot has the same colorbar scale

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


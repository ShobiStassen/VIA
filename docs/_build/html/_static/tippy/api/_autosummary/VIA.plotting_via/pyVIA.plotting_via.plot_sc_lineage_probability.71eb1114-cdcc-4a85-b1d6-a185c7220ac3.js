selector_to_html = {"a[href=\"#pyvia-plotting-via-plot-sc-lineage-probability\"]": "<h1 class=\"tippy-header\" style=\"margin-top: 0;\">pyVIA.plotting_via.plot_sc_lineage_probability<a class=\"headerlink\" href=\"#pyvia-plotting-via-plot-sc-lineage-probability\" title=\"Permalink to this heading\">\uf0c1</a></h1>", "a[href=\"#pyVIA.plotting_via.plot_sc_lineage_probability\"]": "<dt class=\"sig sig-object py\" id=\"pyVIA.plotting_via.plot_sc_lineage_probability\">\n<span class=\"sig-prename descclassname\"><span class=\"pre\">pyVIA.plotting_via.</span></span><span class=\"sig-name descname\"><span class=\"pre\">plot_sc_lineage_probability</span></span><span class=\"sig-paren\">(</span><em class=\"sig-param\"><span class=\"n\"><span class=\"pre\">via_object</span></span></em>, <em class=\"sig-param\"><span class=\"n\"><span class=\"pre\">embedding</span></span><span class=\"o\"><span class=\"pre\">=</span></span><span class=\"default_value\"><span class=\"pre\">None</span></span></em>, <em class=\"sig-param\"><span class=\"n\"><span class=\"pre\">idx</span></span><span class=\"o\"><span class=\"pre\">=</span></span><span class=\"default_value\"><span class=\"pre\">None</span></span></em>, <em class=\"sig-param\"><span class=\"n\"><span class=\"pre\">cmap_name</span></span><span class=\"o\"><span class=\"pre\">=</span></span><span class=\"default_value\"><span class=\"pre\">'plasma'</span></span></em>, <em class=\"sig-param\"><span class=\"n\"><span class=\"pre\">dpi</span></span><span class=\"o\"><span class=\"pre\">=</span></span><span class=\"default_value\"><span class=\"pre\">150</span></span></em>, <em class=\"sig-param\"><span class=\"n\"><span class=\"pre\">scatter_size</span></span><span class=\"o\"><span class=\"pre\">=</span></span><span class=\"default_value\"><span class=\"pre\">None</span></span></em>, <em class=\"sig-param\"><span class=\"n\"><span class=\"pre\">marker_lineages</span></span><span class=\"o\"><span class=\"pre\">=</span></span><span class=\"default_value\"><span class=\"pre\">[]</span></span></em>, <em class=\"sig-param\"><span class=\"n\"><span class=\"pre\">fontsize</span></span><span class=\"o\"><span class=\"pre\">=</span></span><span class=\"default_value\"><span class=\"pre\">8</span></span></em>, <em class=\"sig-param\"><span class=\"n\"><span class=\"pre\">alpha_factor</span></span><span class=\"o\"><span class=\"pre\">=</span></span><span class=\"default_value\"><span class=\"pre\">0.9</span></span></em>, <em class=\"sig-param\"><span class=\"n\"><span class=\"pre\">majority_cluster_population_dict</span></span><span class=\"o\"><span class=\"pre\">=</span></span><span class=\"default_value\"><span class=\"pre\">None</span></span></em>, <em class=\"sig-param\"><span class=\"n\"><span class=\"pre\">cmap_sankey</span></span><span class=\"o\"><span class=\"pre\">=</span></span><span class=\"default_value\"><span class=\"pre\">'rainbow'</span></span></em>, <em class=\"sig-param\"><span class=\"n\"><span class=\"pre\">do_sankey</span></span><span class=\"o\"><span class=\"pre\">=</span></span><span class=\"default_value\"><span class=\"pre\">False</span></span></em><span class=\"sig-paren\">)</span><a class=\"reference internal\" href=\"../../../_modules/pyVIA/plotting_via.html#plot_sc_lineage_probability\"><span class=\"viewcode-link\"><span class=\"pre\">[source]</span></span></a></dt><dd><p>G is the igraph knn (low K) used for shortest path in high dim space. no idx needed as it\u2019s made on full sample\nknn_hnsw is the knn made in the embedded space used for query to find the nearest point in the downsampled embedding\nthat corresponds to the single cells in the full graph</p></dd>"}
skip_classes = ["headerlink", "sd-stretched-link"]

window.onload = function () {
    for (const [select, tip_html] of Object.entries(selector_to_html)) {
        const links = document.querySelectorAll(` ${select}`);
        for (const link of links) {
            if (skip_classes.some(c => link.classList.contains(c))) {
                continue;
            }

            tippy(link, {
                content: tip_html,
                allowHTML: true,
                arrow: true,
                placement: 'auto-start', maxWidth: 500, interactive: false,

            });
        };
    };
    console.log("tippy tips loaded!");
};

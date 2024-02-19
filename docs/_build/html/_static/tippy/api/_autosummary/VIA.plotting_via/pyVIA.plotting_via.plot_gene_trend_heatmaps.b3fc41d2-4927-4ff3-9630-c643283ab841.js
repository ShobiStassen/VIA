selector_to_html = {"a[href=\"#pyVIA.plotting_via.plot_gene_trend_heatmaps\"]": "<dt class=\"sig sig-object py\" id=\"pyVIA.plotting_via.plot_gene_trend_heatmaps\">\n<span class=\"sig-prename descclassname\"><span class=\"pre\">pyVIA.plotting_via.</span></span><span class=\"sig-name descname\"><span class=\"pre\">plot_gene_trend_heatmaps</span></span><span class=\"sig-paren\">(</span><em class=\"sig-param\"><span class=\"n\"><span class=\"pre\">via_object</span></span></em>, <em class=\"sig-param\"><span class=\"n\"><span class=\"pre\">df_gene_exp</span></span></em>, <em class=\"sig-param\"><span class=\"n\"><span class=\"pre\">marker_lineages</span></span><span class=\"o\"><span class=\"pre\">=</span></span><span class=\"default_value\"><span class=\"pre\">[]</span></span></em>, <em class=\"sig-param\"><span class=\"n\"><span class=\"pre\">fontsize</span></span><span class=\"o\"><span class=\"pre\">=</span></span><span class=\"default_value\"><span class=\"pre\">8</span></span></em>, <em class=\"sig-param\"><span class=\"n\"><span class=\"pre\">cmap</span></span><span class=\"o\"><span class=\"pre\">=</span></span><span class=\"default_value\"><span class=\"pre\">'viridis'</span></span></em>, <em class=\"sig-param\"><span class=\"n\"><span class=\"pre\">normalize</span></span><span class=\"o\"><span class=\"pre\">=</span></span><span class=\"default_value\"><span class=\"pre\">True</span></span></em>, <em class=\"sig-param\"><span class=\"n\"><span class=\"pre\">ytick_labelrotation</span></span><span class=\"o\"><span class=\"pre\">=</span></span><span class=\"default_value\"><span class=\"pre\">0</span></span></em>, <em class=\"sig-param\"><span class=\"n\"><span class=\"pre\">fig_width</span></span><span class=\"o\"><span class=\"pre\">=</span></span><span class=\"default_value\"><span class=\"pre\">7</span></span></em><span class=\"sig-paren\">)</span><a class=\"reference internal\" href=\"../../../_modules/pyVIA/plotting_via.html#plot_gene_trend_heatmaps\"><span class=\"viewcode-link\"><span class=\"pre\">[source]</span></span></a></dt><dd><p>Plot the gene trends on heatmap: a heatmap is generated for each lineage (identified by terminal cluster number). Default selects all lineages</p></dd>", "a[href=\"#pyvia-plotting-via-plot-gene-trend-heatmaps\"]": "<h1 class=\"tippy-header\" style=\"margin-top: 0;\">pyVIA.plotting_via.plot_gene_trend_heatmaps<a class=\"headerlink\" href=\"#pyvia-plotting-via-plot-gene-trend-heatmaps\" title=\"Permalink to this heading\">\uf0c1</a></h1>"}
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

selector_to_html = {"a[href=\"#pyvia-plotting-via-plot-viagraph\"]": "<h1 class=\"tippy-header\" style=\"margin-top: 0;\">pyVIA.plotting_via.plot_viagraph<a class=\"headerlink\" href=\"#pyvia-plotting-via-plot-viagraph\" title=\"Permalink to this heading\">\uf0c1</a></h1>", "a[href=\"#pyVIA.plotting_via.plot_viagraph\"]": "<dt class=\"sig sig-object py\" id=\"pyVIA.plotting_via.plot_viagraph\">\n<span class=\"sig-prename descclassname\"><span class=\"pre\">pyVIA.plotting_via.</span></span><span class=\"sig-name descname\"><span class=\"pre\">plot_viagraph</span></span><span class=\"sig-paren\">(</span><em class=\"sig-param\"><span class=\"n\"><span class=\"pre\">via_object</span></span></em>, <em class=\"sig-param\"><span class=\"n\"><span class=\"pre\">type_data</span></span><span class=\"o\"><span class=\"pre\">=</span></span><span class=\"default_value\"><span class=\"pre\">'gene'</span></span></em>, <em class=\"sig-param\"><span class=\"n\"><span class=\"pre\">df_genes</span></span><span class=\"o\"><span class=\"pre\">=</span></span><span class=\"default_value\"><span class=\"pre\">None</span></span></em>, <em class=\"sig-param\"><span class=\"n\"><span class=\"pre\">gene_list</span></span><span class=\"o\"><span class=\"pre\">=</span></span><span class=\"default_value\"><span class=\"pre\">''</span></span></em>, <em class=\"sig-param\"><span class=\"n\"><span class=\"pre\">arrow_head</span></span><span class=\"o\"><span class=\"pre\">=</span></span><span class=\"default_value\"><span class=\"pre\">0.1</span></span></em>, <em class=\"sig-param\"><span class=\"n\"><span class=\"pre\">edgeweight_scale</span></span><span class=\"o\"><span class=\"pre\">=</span></span><span class=\"default_value\"><span class=\"pre\">1.5</span></span></em>, <em class=\"sig-param\"><span class=\"n\"><span class=\"pre\">cmap</span></span><span class=\"o\"><span class=\"pre\">=</span></span><span class=\"default_value\"><span class=\"pre\">None</span></span></em>, <em class=\"sig-param\"><span class=\"n\"><span class=\"pre\">label_text</span></span><span class=\"o\"><span class=\"pre\">=</span></span><span class=\"default_value\"><span class=\"pre\">True</span></span></em>, <em class=\"sig-param\"><span class=\"n\"><span class=\"pre\">size_factor_node</span></span><span class=\"o\"><span class=\"pre\">=</span></span><span class=\"default_value\"><span class=\"pre\">1</span></span></em><span class=\"sig-paren\">)</span><a class=\"reference internal\" href=\"../../../_modules/pyVIA/plotting_via.html#plot_viagraph\"><span class=\"viewcode-link\"><span class=\"pre\">[source]</span></span></a></dt><dd><p>cluster level expression of gene/feature intensity\n:param via_object:\n:param type_data:\n:param gene_exp: pd.Dataframe size n_cells x genes. Otherwise defaults to plotting pseudotime\n:param gene_list: list of gene names corresponding to the column name\n:param arrow_head:\n:param edgeweight_scale:\n:param cmap:\n:param label_text: bool to add numeric values of the gene exp level\n:param size_factor_node size of graph nodes\n:return: fig, axs</p></dd>"}
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

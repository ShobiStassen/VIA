selector_to_html = {"a[href=\"#pyvia-plotting-via-get-gene-expression\"]": "<h1 class=\"tippy-header\" style=\"margin-top: 0;\">pyVIA.plotting_via.get_gene_expression<a class=\"headerlink\" href=\"#pyvia-plotting-via-get-gene-expression\" title=\"Permalink to this heading\">\uf0c1</a></h1>", "a[href=\"#pyVIA.plotting_via.get_gene_expression\"]": "<dt class=\"sig sig-object py\" id=\"pyVIA.plotting_via.get_gene_expression\">\n<span class=\"sig-prename descclassname\"><span class=\"pre\">pyVIA.plotting_via.</span></span><span class=\"sig-name descname\"><span class=\"pre\">get_gene_expression</span></span><span class=\"sig-paren\">(</span><em class=\"sig-param\"><span class=\"n\"><span class=\"pre\">via_object</span></span></em>, <em class=\"sig-param\"><span class=\"n\"><span class=\"pre\">gene_exp</span></span></em>, <em class=\"sig-param\"><span class=\"n\"><span class=\"pre\">cmap</span></span><span class=\"o\"><span class=\"pre\">=</span></span><span class=\"default_value\"><span class=\"pre\">'jet'</span></span></em>, <em class=\"sig-param\"><span class=\"n\"><span class=\"pre\">dpi</span></span><span class=\"o\"><span class=\"pre\">=</span></span><span class=\"default_value\"><span class=\"pre\">150</span></span></em>, <em class=\"sig-param\"><span class=\"n\"><span class=\"pre\">marker_genes</span></span><span class=\"o\"><span class=\"pre\">=</span></span><span class=\"default_value\"><span class=\"pre\">[]</span></span></em>, <em class=\"sig-param\"><span class=\"n\"><span class=\"pre\">linewidth</span></span><span class=\"o\"><span class=\"pre\">=</span></span><span class=\"default_value\"><span class=\"pre\">2.0</span></span></em>, <em class=\"sig-param\"><span class=\"n\"><span class=\"pre\">n_splines</span></span><span class=\"o\"><span class=\"pre\">=</span></span><span class=\"default_value\"><span class=\"pre\">10</span></span></em>, <em class=\"sig-param\"><span class=\"n\"><span class=\"pre\">spline_order</span></span><span class=\"o\"><span class=\"pre\">=</span></span><span class=\"default_value\"><span class=\"pre\">4</span></span></em>, <em class=\"sig-param\"><span class=\"n\"><span class=\"pre\">fontsize_</span></span><span class=\"o\"><span class=\"pre\">=</span></span><span class=\"default_value\"><span class=\"pre\">8</span></span></em>, <em class=\"sig-param\"><span class=\"n\"><span class=\"pre\">marker_lineages</span></span><span class=\"o\"><span class=\"pre\">=</span></span><span class=\"default_value\"><span class=\"pre\">[]</span></span></em>, <em class=\"sig-param\"><span class=\"n\"><span class=\"pre\">optional_title_text</span></span><span class=\"o\"><span class=\"pre\">=</span></span><span class=\"default_value\"><span class=\"pre\">''</span></span></em>, <em class=\"sig-param\"><span class=\"n\"><span class=\"pre\">cmap_dict</span></span><span class=\"o\"><span class=\"pre\">=</span></span><span class=\"default_value\"><span class=\"pre\">None</span></span></em><span class=\"sig-paren\">)</span><a class=\"reference internal\" href=\"../../../_modules/pyVIA/plotting_via.html#get_gene_expression\"><span class=\"viewcode-link\"><span class=\"pre\">[source]</span></span></a></dt><dd></dd>"}
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

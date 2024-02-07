selector_to_html = {"a[href=\"#pyVIA.plotting_via.plot_scatter\"]": "<dt class=\"sig sig-object py\" id=\"pyVIA.plotting_via.plot_scatter\">\n<span class=\"sig-prename descclassname\"><span class=\"pre\">pyVIA.plotting_via.</span></span><span class=\"sig-name descname\"><span class=\"pre\">plot_scatter</span></span><span class=\"sig-paren\">(</span><em class=\"sig-param\"><span class=\"n\"><span class=\"pre\">embedding</span></span></em>, <em class=\"sig-param\"><span class=\"n\"><span class=\"pre\">labels</span></span></em>, <em class=\"sig-param\"><span class=\"n\"><span class=\"pre\">cmap</span></span><span class=\"o\"><span class=\"pre\">=</span></span><span class=\"default_value\"><span class=\"pre\">'rainbow'</span></span></em>, <em class=\"sig-param\"><span class=\"n\"><span class=\"pre\">s</span></span><span class=\"o\"><span class=\"pre\">=</span></span><span class=\"default_value\"><span class=\"pre\">5</span></span></em>, <em class=\"sig-param\"><span class=\"n\"><span class=\"pre\">alpha</span></span><span class=\"o\"><span class=\"pre\">=</span></span><span class=\"default_value\"><span class=\"pre\">0.3</span></span></em>, <em class=\"sig-param\"><span class=\"n\"><span class=\"pre\">edgecolors</span></span><span class=\"o\"><span class=\"pre\">=</span></span><span class=\"default_value\"><span class=\"pre\">'None'</span></span></em>, <em class=\"sig-param\"><span class=\"n\"><span class=\"pre\">title</span></span><span class=\"o\"><span class=\"pre\">=</span></span><span class=\"default_value\"><span class=\"pre\">''</span></span></em>, <em class=\"sig-param\"><span class=\"n\"><span class=\"pre\">text_labels</span></span><span class=\"o\"><span class=\"pre\">=</span></span><span class=\"default_value\"><span class=\"pre\">True</span></span></em>, <em class=\"sig-param\"><span class=\"n\"><span class=\"pre\">color_dict</span></span><span class=\"o\"><span class=\"pre\">=</span></span><span class=\"default_value\"><span class=\"pre\">None</span></span></em>, <em class=\"sig-param\"><span class=\"n\"><span class=\"pre\">via_object</span></span><span class=\"o\"><span class=\"pre\">=</span></span><span class=\"default_value\"><span class=\"pre\">None</span></span></em>, <em class=\"sig-param\"><span class=\"n\"><span class=\"pre\">sc_index_terminal_states</span></span><span class=\"o\"><span class=\"pre\">=</span></span><span class=\"default_value\"><span class=\"pre\">None</span></span></em>, <em class=\"sig-param\"><span class=\"n\"><span class=\"pre\">true_labels</span></span><span class=\"o\"><span class=\"pre\">=</span></span><span class=\"default_value\"><span class=\"pre\">[]</span></span></em>, <em class=\"sig-param\"><span class=\"n\"><span class=\"pre\">show_legend</span></span><span class=\"o\"><span class=\"pre\">=</span></span><span class=\"default_value\"><span class=\"pre\">True</span></span></em><span class=\"sig-paren\">)</span><a class=\"reference internal\" href=\"../../../_modules/pyVIA/plotting_via.html#plot_scatter\"><span class=\"viewcode-link\"><span class=\"pre\">[source]</span></span></a></dt><dd><p>General scatter plotting tool for numeric and categorical labels on the single-cell level</p></dd>", "a[href=\"#pyvia-plotting-via-plot-scatter\"]": "<h1 class=\"tippy-header\" style=\"margin-top: 0;\">pyVIA.plotting_via.plot_scatter<a class=\"headerlink\" href=\"#pyvia-plotting-via-plot-scatter\" title=\"Permalink to this heading\">\uf0c1</a></h1>"}
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

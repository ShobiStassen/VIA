selector_to_html = {"a[href=\"#pyvia-plotting-via-plot-population-composition\"]": "<h1 class=\"tippy-header\" style=\"margin-top: 0;\">pyVIA.plotting_via.plot_population_composition<a class=\"headerlink\" href=\"#pyvia-plotting-via-plot-population-composition\" title=\"Permalink to this heading\">\uf0c1</a></h1>", "a[href=\"#pyVIA.plotting_via.plot_population_composition\"]": "<dt class=\"sig sig-object py\" id=\"pyVIA.plotting_via.plot_population_composition\">\n<span class=\"sig-prename descclassname\"><span class=\"pre\">pyVIA.plotting_via.</span></span><span class=\"sig-name descname\"><span class=\"pre\">plot_population_composition</span></span><span class=\"sig-paren\">(</span><em class=\"sig-param\"><span class=\"n\"><span class=\"pre\">via_object</span></span></em>, <em class=\"sig-param\"><span class=\"n\"><span class=\"pre\">time_labels</span></span><span class=\"o\"><span class=\"pre\">=</span></span><span class=\"default_value\"><span class=\"pre\">None</span></span></em>, <em class=\"sig-param\"><span class=\"n\"><span class=\"pre\">celltype_list</span></span><span class=\"o\"><span class=\"pre\">=</span></span><span class=\"default_value\"><span class=\"pre\">None</span></span></em>, <em class=\"sig-param\"><span class=\"n\"><span class=\"pre\">cmap</span></span><span class=\"o\"><span class=\"pre\">=</span></span><span class=\"default_value\"><span class=\"pre\">'rainbow'</span></span></em>, <em class=\"sig-param\"><span class=\"n\"><span class=\"pre\">legend</span></span><span class=\"o\"><span class=\"pre\">=</span></span><span class=\"default_value\"><span class=\"pre\">True</span></span></em>, <em class=\"sig-param\"><span class=\"n\"><span class=\"pre\">alpha</span></span><span class=\"o\"><span class=\"pre\">=</span></span><span class=\"default_value\"><span class=\"pre\">0.5</span></span></em>, <em class=\"sig-param\"><span class=\"n\"><span class=\"pre\">linewidth</span></span><span class=\"o\"><span class=\"pre\">=</span></span><span class=\"default_value\"><span class=\"pre\">0.2</span></span></em>, <em class=\"sig-param\"><span class=\"n\"><span class=\"pre\">n_intervals</span></span><span class=\"o\"><span class=\"pre\">=</span></span><span class=\"default_value\"><span class=\"pre\">20</span></span></em>, <em class=\"sig-param\"><span class=\"n\"><span class=\"pre\">xlabel</span></span><span class=\"o\"><span class=\"pre\">=</span></span><span class=\"default_value\"><span class=\"pre\">'time'</span></span></em>, <em class=\"sig-param\"><span class=\"n\"><span class=\"pre\">ylabel</span></span><span class=\"o\"><span class=\"pre\">=</span></span><span class=\"default_value\"><span class=\"pre\">''</span></span></em>, <em class=\"sig-param\"><span class=\"n\"><span class=\"pre\">title</span></span><span class=\"o\"><span class=\"pre\">=</span></span><span class=\"default_value\"><span class=\"pre\">'Cell</span> <span class=\"pre\">populations'</span></span></em>, <em class=\"sig-param\"><span class=\"n\"><span class=\"pre\">color_dict</span></span><span class=\"o\"><span class=\"pre\">=</span></span><span class=\"default_value\"><span class=\"pre\">None</span></span></em>, <em class=\"sig-param\"><span class=\"n\"><span class=\"pre\">fraction</span></span><span class=\"o\"><span class=\"pre\">=</span></span><span class=\"default_value\"><span class=\"pre\">True</span></span></em><span class=\"sig-paren\">)</span><a class=\"reference internal\" href=\"../../../_modules/pyVIA/plotting_via.html#plot_population_composition\"><span class=\"viewcode-link\"><span class=\"pre\">[source]</span></span></a></dt><dd></dd>"}
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

selector_to_html = {"a[href=\"#pyvia-plotting-via-animate-atlas\"]": "<h1 class=\"tippy-header\" style=\"margin-top: 0;\">pyVIA.plotting_via.animate_atlas<a class=\"headerlink\" href=\"#pyvia-plotting-via-animate-atlas\" title=\"Permalink to this heading\">\uf0c1</a></h1>", "a[href=\"#pyVIA.plotting_via.animate_atlas\"]": "<dt class=\"sig sig-object py\" id=\"pyVIA.plotting_via.animate_atlas\">\n<span class=\"sig-prename descclassname\"><span class=\"pre\">pyVIA.plotting_via.</span></span><span class=\"sig-name descname\"><span class=\"pre\">animate_atlas</span></span><span class=\"sig-paren\">(</span><em class=\"sig-param\"><span class=\"n\"><span class=\"pre\">hammerbundle_dict</span></span><span class=\"o\"><span class=\"pre\">=</span></span><span class=\"default_value\"><span class=\"pre\">None</span></span></em>, <em class=\"sig-param\"><span class=\"n\"><span class=\"pre\">via_object</span></span><span class=\"o\"><span class=\"pre\">=</span></span><span class=\"default_value\"><span class=\"pre\">None</span></span></em>, <em class=\"sig-param\"><span class=\"n\"><span class=\"pre\">linewidth_bundle</span></span><span class=\"o\"><span class=\"pre\">=</span></span><span class=\"default_value\"><span class=\"pre\">2</span></span></em>, <em class=\"sig-param\"><span class=\"n\"><span class=\"pre\">frame_interval</span></span><span class=\"o\"><span class=\"pre\">=</span></span><span class=\"default_value\"><span class=\"pre\">10</span></span></em>, <em class=\"sig-param\"><span class=\"n\"><span class=\"pre\">n_milestones</span></span><span class=\"o\"><span class=\"pre\">=</span></span><span class=\"default_value\"><span class=\"pre\">None</span></span></em>, <em class=\"sig-param\"><span class=\"n\"><span class=\"pre\">facecolor</span></span><span class=\"o\"><span class=\"pre\">=</span></span><span class=\"default_value\"><span class=\"pre\">'white'</span></span></em>, <em class=\"sig-param\"><span class=\"n\"><span class=\"pre\">cmap</span></span><span class=\"o\"><span class=\"pre\">=</span></span><span class=\"default_value\"><span class=\"pre\">'plasma_r'</span></span></em>, <em class=\"sig-param\"><span class=\"n\"><span class=\"pre\">extra_title_text</span></span><span class=\"o\"><span class=\"pre\">=</span></span><span class=\"default_value\"><span class=\"pre\">''</span></span></em>, <em class=\"sig-param\"><span class=\"n\"><span class=\"pre\">size_scatter</span></span><span class=\"o\"><span class=\"pre\">=</span></span><span class=\"default_value\"><span class=\"pre\">1</span></span></em>, <em class=\"sig-param\"><span class=\"n\"><span class=\"pre\">alpha_scatter</span></span><span class=\"o\"><span class=\"pre\">=</span></span><span class=\"default_value\"><span class=\"pre\">0.2</span></span></em>, <em class=\"sig-param\"><span class=\"n\"><span class=\"pre\">saveto</span></span><span class=\"o\"><span class=\"pre\">=</span></span><span class=\"default_value\"><span class=\"pre\">'/home/user/Trajectory/Datasets/animation_default.gif'</span></span></em>, <em class=\"sig-param\"><span class=\"n\"><span class=\"pre\">time_series_labels</span></span><span class=\"o\"><span class=\"pre\">=</span></span><span class=\"default_value\"><span class=\"pre\">None</span></span></em>, <em class=\"sig-param\"><span class=\"n\"><span class=\"pre\">sc_labels_numeric</span></span><span class=\"o\"><span class=\"pre\">=</span></span><span class=\"default_value\"><span class=\"pre\">None</span></span></em><span class=\"sig-paren\">)</span><a class=\"reference internal\" href=\"../../../_modules/pyVIA/plotting_via.html#animate_atlas\"><span class=\"viewcode-link\"><span class=\"pre\">[source]</span></span></a></dt><dd></dd>"}
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

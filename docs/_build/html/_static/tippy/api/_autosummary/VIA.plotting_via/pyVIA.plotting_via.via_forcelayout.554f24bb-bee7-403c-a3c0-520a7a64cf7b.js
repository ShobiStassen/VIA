selector_to_html = {"a[href=\"#pyVIA.plotting_via.via_forcelayout\"]": "<dt class=\"sig sig-object py\" id=\"pyVIA.plotting_via.via_forcelayout\">\n<span class=\"sig-prename descclassname\"><span class=\"pre\">pyVIA.plotting_via.</span></span><span class=\"sig-name descname\"><span class=\"pre\">via_forcelayout</span></span><span class=\"sig-paren\">(</span><em class=\"sig-param\"><span class=\"n\"><span class=\"pre\">X_pca</span></span></em>, <em class=\"sig-param\"><span class=\"n\"><span class=\"pre\">viagraph_full</span></span><span class=\"o\"><span class=\"pre\">=</span></span><span class=\"default_value\"><span class=\"pre\">None</span></span></em>, <em class=\"sig-param\"><span class=\"n\"><span class=\"pre\">k</span></span><span class=\"o\"><span class=\"pre\">=</span></span><span class=\"default_value\"><span class=\"pre\">10</span></span></em>, <em class=\"sig-param\"><span class=\"n\"><span class=\"pre\">n_milestones</span></span><span class=\"o\"><span class=\"pre\">=</span></span><span class=\"default_value\"><span class=\"pre\">2000</span></span></em>, <em class=\"sig-param\"><span class=\"n\"><span class=\"pre\">time_series_labels</span></span><span class=\"o\"><span class=\"pre\">=</span></span><span class=\"default_value\"><span class=\"pre\">[]</span></span></em>, <em class=\"sig-param\"><span class=\"n\"><span class=\"pre\">knn_seq</span></span><span class=\"o\"><span class=\"pre\">=</span></span><span class=\"default_value\"><span class=\"pre\">5</span></span></em>, <em class=\"sig-param\"><span class=\"n\"><span class=\"pre\">saveto</span></span><span class=\"o\"><span class=\"pre\">=</span></span><span class=\"default_value\"><span class=\"pre\">''</span></span></em>, <em class=\"sig-param\"><span class=\"n\"><span class=\"pre\">random_seed</span></span><span class=\"o\"><span class=\"pre\">=</span></span><span class=\"default_value\"><span class=\"pre\">0</span></span></em><span class=\"sig-paren\">)</span><a class=\"reference internal\" href=\"../../../_modules/pyVIA/plotting_via.html#via_forcelayout\"><span class=\"viewcode-link\"><span class=\"pre\">[source]</span></span></a></dt><dd><p>Compute force directed layout. #TODO not complete</p></dd>", "a[href=\"#pyvia-plotting-via-via-forcelayout\"]": "<h1 class=\"tippy-header\" style=\"margin-top: 0;\">pyVIA.plotting_via.via_forcelayout<a class=\"headerlink\" href=\"#pyvia-plotting-via-via-forcelayout\" title=\"Permalink to this heading\">\uf0c1</a></h1>"}
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

selector_to_html = {"a[href=\"#datasets-via-cell-cycle\"]": "<h1 class=\"tippy-header\" style=\"margin-top: 0;\">datasets_via.cell_cycle<a class=\"headerlink\" href=\"#datasets-via-cell-cycle\" title=\"Permalink to this heading\">\uf0c1</a></h1>", "a[href=\"#datasets_via.cell_cycle\"]": "<dt class=\"sig sig-object py\" id=\"datasets_via.cell_cycle\">\n<span class=\"sig-prename descclassname\"><span class=\"pre\">datasets_via.</span></span><span class=\"sig-name descname\"><span class=\"pre\">cell_cycle</span></span><span class=\"sig-paren\">(</span><em class=\"sig-param\"><span class=\"n\"><span class=\"pre\">foldername</span></span><span class=\"o\"><span class=\"pre\">=</span></span><span class=\"default_value\"><span class=\"pre\">'./'</span></span></em><span class=\"sig-paren\">)</span><a class=\"reference internal\" href=\"../../_modules/datasets_via.html#cell_cycle\"><span class=\"viewcode-link\"><span class=\"pre\">[source]</span></span></a></dt><dd><p>Load cell cycle data as AnnData object</p></dd>"}
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

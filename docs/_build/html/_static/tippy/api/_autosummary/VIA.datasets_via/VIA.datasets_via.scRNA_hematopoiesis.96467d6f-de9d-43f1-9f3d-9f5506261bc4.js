selector_to_html = {"a[href=\"#via-datasets-via-scrna-hematopoiesis\"]": "<h1 class=\"tippy-header\" style=\"margin-top: 0;\">VIA.datasets_via.scRNA_hematopoiesis<a class=\"headerlink\" href=\"#via-datasets-via-scrna-hematopoiesis\" title=\"Permalink to this heading\">\uf0c1</a></h1>", "a[href=\"#VIA.datasets_via.scRNA_hematopoiesis\"]": "<dt class=\"sig sig-object py\" id=\"VIA.datasets_via.scRNA_hematopoiesis\">\n<span class=\"sig-prename descclassname\"><span class=\"pre\">VIA.datasets_via.</span></span><span class=\"sig-name descname\"><span class=\"pre\">scRNA_hematopoiesis</span></span><span class=\"sig-paren\">(</span><em class=\"sig-param\"><span class=\"n\"><span class=\"pre\">foldername</span></span><span class=\"o\"><span class=\"pre\">=</span></span><span class=\"default_value\"><span class=\"pre\">'./'</span></span></em><span class=\"sig-paren\">)</span><a class=\"reference internal\" href=\"../../../_modules/VIA/datasets_via.html#scRNA_hematopoiesis\"><span class=\"viewcode-link\"><span class=\"pre\">[source]</span></span></a></dt><dd><p>Load scRNA seq Hematopoiesis data as AnnData object</p></dd>"}
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

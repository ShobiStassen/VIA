selector_to_html = {"a[href=\"#stavia-multi-omic-single-cell-cartography-for-spatial-and-temporal-atlases\"]": "<h1 class=\"tippy-header\" style=\"margin-top: 0;\">StaVia - Multi-Omic Single-Cell Cartography for Spatial and Temporal Atlases<a class=\"headerlink\" href=\"#stavia-multi-omic-single-cell-cartography-for-spatial-and-temporal-atlases\" title=\"Permalink to this heading\">\uf0c1</a></h1><p><strong>StaVia (Via 2.0)</strong> is our new single-cell trajectory inference method that explores <strong>single-cell atlas-scale</strong> data and <strong>temporal and spatial studies</strong> enabled by:</p>"}
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

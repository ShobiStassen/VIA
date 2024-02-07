selector_to_html = {"a[href=\"#via2-0-atlas-views-for-spatial-omics-data\"]": "<h1 class=\"tippy-header\" style=\"margin-top: 0;\">Via2.0 Atlas views for Spatial Omics data<a class=\"headerlink\" href=\"#via2-0-atlas-views-for-spatial-omics-data\" title=\"Permalink to this heading\">\uf0c1</a></h1><p><strong>Via 2.0</strong> offers sophisticated visualization tools aimed to capture connectivity and cellular resolution simultaneously. In light of rapidly emerging spatial omics datasets and atlases, we show that the Atlas view uniquely captures spatial and transcriptomic information.</p><p><strong>Atlas view of Preoptic cell classes of Merfish data (Moffitt 2018- 75K cells) compared to UMAP of the same</strong></p>"}
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

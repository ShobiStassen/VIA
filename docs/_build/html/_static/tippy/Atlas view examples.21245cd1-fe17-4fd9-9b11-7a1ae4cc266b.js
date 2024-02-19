selector_to_html = {"a[href=\"#via2-0-atlas-views\"]": "<h1 class=\"tippy-header\" style=\"margin-top: 0;\">Via2.0 Atlas views<a class=\"headerlink\" href=\"#via2-0-atlas-views\" title=\"Permalink to this heading\">\uf0c1</a></h1><p><strong>Via 2.0</strong> offers sophisticated visualization tools aimed to capture connectivity and cellular resolution simultaneously. We present several examples of single-cell atlases to show that the Atlas view uniquely illustrates developmental data. The Atlas Views of the Mouse Pup data are millions of cells large.</p><p><strong>Atlas view of Zebrahub (Lange 2023 - 120K cells) colored by major tissue type and stage</strong></p>"}
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

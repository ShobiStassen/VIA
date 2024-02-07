selector_to_html = {"a[href=\"#large-mouse-embryo-to-pup-developmental-atlas\"]": "<h1 class=\"tippy-header\" style=\"margin-top: 0;\">Large Mouse Embryo to Pup Developmental Atlas<a class=\"headerlink\" href=\"#large-mouse-embryo-to-pup-developmental-atlas\" title=\"Permalink to this heading\">\uf0c1</a></h1><p><strong>Via2.0</strong> Cartographic views of the timelapse mouse development dataset (Qiu et al., 2023) from gastrula to pup taken at 6 hours intervals spanning over 8 Million cells. Via2.0 integrates temporal information in order to preserve the global chronology of all major germ layers as they differentiate. The first figure shows 8 Millions cells whereas the subsequent figure show the Atlas View on specific subsets of the dataset.</p><p><strong>Atlas view of Mouse Embryo to Pup (Qiu 2023 - 8 Million cells E8.0 - E17.5) colored by major tissue type and stage</strong></p>"}
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

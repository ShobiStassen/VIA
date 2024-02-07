selector_to_html = {"a[href=\"#scatac-seq-human-hematopoiesis-click-to-open-interactive-via-graph\"]": "<h2 class=\"tippy-header\" style=\"margin-top: 0;\">scATAC-seq Human Hematopoiesis <a class=\"reference external\" href=\"https://shobistassen.github.io/toggle_data.html\">(click to open interactive VIA graph)</a><a class=\"headerlink\" href=\"#scatac-seq-human-hematopoiesis-click-to-open-interactive-via-graph\" title=\"Permalink to this heading\">\uf0c1</a></h2>", "a[href=\"#via-2-0-atlas-animations\"]": "<h1 class=\"tippy-header\" style=\"margin-top: 0;\">Via 2.0 Atlas Animations<a class=\"headerlink\" href=\"#via-2-0-atlas-animations\" title=\"Permalink to this heading\">\uf0c1</a></h1><p><strong>Via 2.0</strong> Animations to view the developmental progression of cell connectivity and differentiation flow</p><p><strong>Atlas view of Ascidian Protovert (Cao 2019) colored by stage</strong></p>"}
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

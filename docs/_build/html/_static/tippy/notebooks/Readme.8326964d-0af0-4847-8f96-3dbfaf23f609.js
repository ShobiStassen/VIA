selector_to_html = {"a[href=\"#tutorials-for-cartographic-ti-and-visualization-using-via-2-0\"]": "<h1 class=\"tippy-header\" style=\"margin-top: 0;\">Tutorials for Cartographic TI and Visualization using Via 2.0<a class=\"headerlink\" href=\"#tutorials-for-cartographic-ti-and-visualization-using-via-2-0\" title=\"Permalink to this heading\">\uf0c1</a></h1><p>Tutorials and <strong><a class=\"reference external\" href=\"https://pyvia.readthedocs.io/en/latest/Tutorial%20Video.html\">videos</a></strong>  available on <strong><a class=\"reference external\" href=\"https://pyvia.readthedocs.io/en/latest/\">readthedocs</a></strong> with step-by-step code for real and simulated datasets. Tutorials explain how to generate cartographic visualizations for TI, tune parameters, obtain various outputs and also understand the importance of <em>memory</em>.</p>"}
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

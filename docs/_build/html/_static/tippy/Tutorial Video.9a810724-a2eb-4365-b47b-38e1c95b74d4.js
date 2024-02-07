selector_to_html = {"a[href=\"#main-sections-covered-in-usage-tutorial\"]": "<h2 class=\"tippy-header\" style=\"margin-top: 0;\">Main sections covered in usage tutorial<a class=\"headerlink\" href=\"#main-sections-covered-in-usage-tutorial\" title=\"Permalink to this heading\">\uf0c1</a></h2>", "a[href=\"#via-implementaion-and-basic-usage\"]": "<h2 class=\"tippy-header\" style=\"margin-top: 0;\">VIA Implementaion and Basic Usage<a class=\"headerlink\" href=\"#via-implementaion-and-basic-usage\" title=\"Permalink to this heading\">\uf0c1</a></h2>", "a[href=\"#via-installation-with-mac-os-and-windows\"]": "<h2 class=\"tippy-header\" style=\"margin-top: 0;\">VIA Installation with Mac OS and Windows<a class=\"headerlink\" href=\"#via-installation-with-mac-os-and-windows\" title=\"Permalink to this heading\">\uf0c1</a></h2>", "a[href=\"#via-tutorial-videos\"]": "<h1 class=\"tippy-header\" style=\"margin-top: 0;\">VIA Tutorial Videos<a class=\"headerlink\" href=\"#via-tutorial-videos\" title=\"Permalink to this heading\">\uf0c1</a></h1><h2>VIA Installation with Mac OS and Windows<a class=\"headerlink\" href=\"#via-installation-with-mac-os-and-windows\" title=\"Permalink to this heading\">\uf0c1</a></h2>"}
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

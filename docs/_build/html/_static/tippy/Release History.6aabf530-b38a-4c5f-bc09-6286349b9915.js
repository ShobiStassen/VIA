selector_to_html = {"a[href=\"#version-0-1-60\"]": "<h2 class=\"tippy-header\" style=\"margin-top: 0;\">Version 0.1.60<a class=\"headerlink\" href=\"#version-0-1-60\" title=\"Permalink to this heading\">\uf0c1</a></h2>", "a[href=\"#version-0-1-58\"]": "<h2 class=\"tippy-header\" style=\"margin-top: 0;\">Version 0.1.58<a class=\"headerlink\" href=\"#version-0-1-58\" title=\"Permalink to this heading\">\uf0c1</a></h2>", "a[href=\"#version-0-1-62\"]": "<h2 class=\"tippy-header\" style=\"margin-top: 0;\">Version 0.1.62<a class=\"headerlink\" href=\"#version-0-1-62\" title=\"Permalink to this heading\">\uf0c1</a></h2><p><code class=\"docutils literal notranslate\"><span class=\"pre\">plot_gene_trend_heatmaps(via_object,</span> <span class=\"pre\">df_gene_exp:pd.DataFrame,...)</span></code></p>", "a[href=\"#version-0-1-73\"]": "<h2 class=\"tippy-header\" style=\"margin-top: 0;\">Version 0.1.73<a class=\"headerlink\" href=\"#version-0-1-73\" title=\"Permalink to this heading\">\uf0c1</a></h2>", "a[href=\"#version-0-1-57\"]": "<h2 class=\"tippy-header\" style=\"margin-top: 0;\">Version 0.1.57<a class=\"headerlink\" href=\"#version-0-1-57\" title=\"Permalink to this heading\">\uf0c1</a></h2>", "a[href=\"#version-0-1-59\"]": "<h2 class=\"tippy-header\" style=\"margin-top: 0;\">Version 0.1.59<a class=\"headerlink\" href=\"#version-0-1-59\" title=\"Permalink to this heading\">\uf0c1</a></h2>", "a[href=\"#version-0-1-82\"]": "<h2 class=\"tippy-header\" style=\"margin-top: 0;\">Version 0.1.82<a class=\"headerlink\" href=\"#version-0-1-82\" title=\"Permalink to this heading\">\uf0c1</a></h2>", "a[href=\"#version-0-1-71\"]": "<h2 class=\"tippy-header\" style=\"margin-top: 0;\">Version 0.1.71<a class=\"headerlink\" href=\"#version-0-1-71\" title=\"Permalink to this heading\">\uf0c1</a></h2>", "a[href=\"#version-0-1-56\"]": "<h2 class=\"tippy-header\" style=\"margin-top: 0;\">Version 0.1.56<a class=\"headerlink\" href=\"#version-0-1-56\" title=\"Permalink to this heading\">\uf0c1</a></h2>", "a[href=\"#version-0-1-70\"]": "<h2 class=\"tippy-header\" style=\"margin-top: 0;\">Version 0.1.70<a class=\"headerlink\" href=\"#version-0-1-70\" title=\"Permalink to this heading\">\uf0c1</a></h2>", "a[href=\"#version-0-1-64\"]": "<h2 class=\"tippy-header\" style=\"margin-top: 0;\">Version 0.1.64<a class=\"headerlink\" href=\"#version-0-1-64\" title=\"Permalink to this heading\">\uf0c1</a></h2>", "a[href=\"#version-0-1-68\"]": "<h2 class=\"tippy-header\" style=\"margin-top: 0;\">Version 0.1.68<a class=\"headerlink\" href=\"#version-0-1-68\" title=\"Permalink to this heading\">\uf0c1</a></h2>", "a[href=\"#version-0-1-77\"]": "<h2 class=\"tippy-header\" style=\"margin-top: 0;\">Version 0.1.77<a class=\"headerlink\" href=\"#version-0-1-77\" title=\"Permalink to this heading\">\uf0c1</a></h2>", "a[href=\"#version-0-1-61\"]": "<h2 class=\"tippy-header\" style=\"margin-top: 0;\">Version 0.1.61<a class=\"headerlink\" href=\"#version-0-1-61\" title=\"Permalink to this heading\">\uf0c1</a></h2>", "a[href=\"#release-history\"]": "<h1 class=\"tippy-header\" style=\"margin-top: 0;\">Release History<a class=\"headerlink\" href=\"#release-history\" title=\"Permalink to this heading\">\uf0c1</a></h1><h2>Version 0.1.82<a class=\"headerlink\" href=\"#version-0-1-82\" title=\"Permalink to this heading\">\uf0c1</a></h2>"}
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

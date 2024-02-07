selector_to_html = {"a[href=\"#visualize-output\"]": "<h2 class=\"tippy-header\" style=\"margin-top: 0;\">Visualize output<a class=\"headerlink\" href=\"#visualize-output\" title=\"Permalink to this heading\">\uf0c1</a></h2>", "a[href=\"#run-via\"]": "<h2 class=\"tippy-header\" style=\"margin-top: 0;\">Run VIA<a class=\"headerlink\" href=\"#run-via\" title=\"Permalink to this heading\">\uf0c1</a></h2><p>If you would like a finer grained version of VIA to get smaller clusters by 1) increasing the resolution parameter (e.g. to 2), lowering knn or lowering jac_std_global (with typical values between 0-1, with smaller values closer to zero resulting in more small clusters)</p>", "a[href=\"#load-data\"]": "<h2 class=\"tippy-header\" style=\"margin-top: 0;\">Load data<a class=\"headerlink\" href=\"#load-data\" title=\"Permalink to this heading\">\uf0c1</a></h2><p>Load the data using the function: datasets_via.cell_cycle_cyto_data(foldername=\u2019./\u2019) OR download the data from <a class=\"reference external\" href=\"https://github.com/ShobiStassen/VIA/tree/master/Datasets\">github</a>\nThe foldername can be changed to the path you want tp save the files for features and labels\nthe feature matrix and known phase labels. M1 denotes G1 phase, M2 denotes S phase and M3 denotes M/G2 phase</p>", "a[href=\"#physical-feature-dynamics\"]": "<h2 class=\"tippy-header\" style=\"margin-top: 0;\">Physical Feature Dynamics<a class=\"headerlink\" href=\"#physical-feature-dynamics\" title=\"Permalink to this heading\">\uf0c1</a></h2>", "a[href=\"#imaging-cytometry\"]": "<h1 class=\"tippy-header\" style=\"margin-top: 0;\">4. Imaging cytometry<a class=\"headerlink\" href=\"#imaging-cytometry\" title=\"Permalink to this heading\">\uf0c1</a></h1><p>FACED imaging cytometry based biophysical features: MCF7 Cell Cycle</p>"}
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

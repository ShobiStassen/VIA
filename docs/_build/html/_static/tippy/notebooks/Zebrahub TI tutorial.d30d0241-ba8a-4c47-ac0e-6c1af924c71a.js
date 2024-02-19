selector_to_html = {"a[href=\"#load-the-data\"]": "<h2 class=\"tippy-header\" style=\"margin-top: 0;\">Load the data<a class=\"headerlink\" href=\"#load-the-data\" title=\"Permalink to this heading\">\uf0c1</a></h2><p>Data has been filtered and library normalized and log1p. Data is annotated with time/stage, cluster labels, and cell type labels (coarse and fine).</p>", "a[href=\"#initialize-parameters\"]": "<h2 class=\"tippy-header\" style=\"margin-top: 0;\">Initialize parameters<a class=\"headerlink\" href=\"#initialize-parameters\" title=\"Permalink to this heading\">\uf0c1</a></h2>", "a[href=\"#set-terminal-states-optional\"]": "<h2 class=\"tippy-header\" style=\"margin-top: 0;\">Set terminal states (optional)<a class=\"headerlink\" href=\"#set-terminal-states-optional\" title=\"Permalink to this heading\">\uf0c1</a></h2>", "a[href=\"#initialize-and-run-the-via2-0-class\"]": "<h2 class=\"tippy-header\" style=\"margin-top: 0;\">Initialize and Run the Via2.0 Class<a class=\"headerlink\" href=\"#initialize-and-run-the-via2-0-class\" title=\"Permalink to this heading\">\uf0c1</a></h2>", "a[href=\"#differentiation-flow\"]": "<h2 class=\"tippy-header\" style=\"margin-top: 0;\">Differentiation Flow<a class=\"headerlink\" href=\"#differentiation-flow\" title=\"Permalink to this heading\">\uf0c1</a></h2><p>Generates a differentiation flow diagram based on the Via2.0 clustergraph. The second plot in each function call has the details by parc cluster and major cell population. In this case, we assign a different root node for each set of lineages.</p>", "a[href=\"#extract-some-annotation\"]": "<h2 class=\"tippy-header\" style=\"margin-top: 0;\">Extract some annotation<a class=\"headerlink\" href=\"#extract-some-annotation\" title=\"Permalink to this heading\">\uf0c1</a></h2>", "a[href=\"#find-a-suitable-root\"]": "<h2 class=\"tippy-header\" style=\"margin-top: 0;\">Find a suitable root<a class=\"headerlink\" href=\"#find-a-suitable-root\" title=\"Permalink to this heading\">\uf0c1</a></h2>", "a[href=\"#via-2-0-cartography-on-zebrahub-trajectory-inference\"]": "<h1 class=\"tippy-header\" style=\"margin-top: 0;\">2. Via 2.0 Cartography on Zebrahub (Trajectory Inference)<a class=\"headerlink\" href=\"#via-2-0-cartography-on-zebrahub-trajectory-inference\" title=\"Permalink to this heading\">\uf0c1</a></h1><p>This tutorial focuses on how to perform TI analysis of Zebrahub. See Tutorial #3 for Atlas View creation</p>", "a[href=\"#plot-single-cell-lineage-probabilities\"]": "<h2 class=\"tippy-header\" style=\"margin-top: 0;\">Plot single-cell lineage probabilities<a class=\"headerlink\" href=\"#plot-single-cell-lineage-probabilities\" title=\"Permalink to this heading\">\uf0c1</a></h2><p><em><strong>Memory</strong></em></p>", "a[href=\"#visualise-via-2-0-cluster-graph\"]": "<h2 class=\"tippy-header\" style=\"margin-top: 0;\">Visualise Via 2.0 Cluster Graph<a class=\"headerlink\" href=\"#visualise-via-2-0-cluster-graph\" title=\"Permalink to this heading\">\uf0c1</a></h2>"}
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

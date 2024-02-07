selector_to_html = {"a[href=\"#initialize-and-run-via\"]": "<h2 class=\"tippy-header\" style=\"margin-top: 0;\">Initialize and run VIA<a class=\"headerlink\" href=\"#initialize-and-run-via\" title=\"Permalink to this heading\">\uf0c1</a></h2>", "a[href=\"#visualize-trajectory-and-cell-progression\"]": "<h2 class=\"tippy-header\" style=\"margin-top: 0;\">Visualize trajectory and cell progression<a class=\"headerlink\" href=\"#visualize-trajectory-and-cell-progression\" title=\"Permalink to this heading\">\uf0c1</a></h2><p>Fine grained vector fields</p>", "a[href=\"#via-graph\"]": "<h2 class=\"tippy-header\" style=\"margin-top: 0;\">Via graph<a class=\"headerlink\" href=\"#via-graph\" title=\"Permalink to this heading\">\uf0c1</a></h2><p>Redraw the viagraph, finetune arrow head width</p>", "a[href=\"#using-rna-velocity\"]": "<h1 class=\"tippy-header\" style=\"margin-top: 0;\">6. Using RNA-velocity<a class=\"headerlink\" href=\"#using-rna-velocity\" title=\"Permalink to this heading\">\uf0c1</a></h1><p>When scRNA-velocity is available, it can be used to guide the trajectory inference and automate initial state prediction. However, because RNA velocitycan be misguided by(Bergen 2021) boosts in expression, variable transcription rates and data capture scope limited to steady-state populations only, users might find it useful to adjust the level of influence the RNA-velocity data should exercise on the inferred TI.</p><p>We use a familiar endocrine-genesis dataset (Bastidas-Ponce et al. (2019).) to demonstrate initial state prediction at the EP Ngn3 low cells and automatic captures of the 4 differentiated islets (alpha, beta, delta and epsilon). As mentioned, it us useful to control the level of influence of RNA-velocity relative to gene-gene distance and this is done using the velo_weight parameter.</p>", "a[href=\"#gene-trends\"]": "<h2 class=\"tippy-header\" style=\"margin-top: 0;\">Gene trends<a class=\"headerlink\" href=\"#gene-trends\" title=\"Permalink to this heading\">\uf0c1</a></h2><p>We can recover trends of islet-associated marker genes as they vary with pseudotime</p>", "a[href=\"#load-the-data\"]": "<h2 class=\"tippy-header\" style=\"margin-top: 0;\">Load the data<a class=\"headerlink\" href=\"#load-the-data\" title=\"Permalink to this heading\">\uf0c1</a></h2><p>The dataset consists of 2531 endocrine cells differentiating. We apply basic filtering and normalization using scanpy.</p>", "a[href=\"#draw-lineage-likelihoods\"]": "<h2 class=\"tippy-header\" style=\"margin-top: 0;\">Draw lineage likelihoods<a class=\"headerlink\" href=\"#draw-lineage-likelihoods\" title=\"Permalink to this heading\">\uf0c1</a></h2><p>These indicate potential pathways corresponding to the 4 islets (two types of Beta islets Lineage 5 and 12)</p>"}
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

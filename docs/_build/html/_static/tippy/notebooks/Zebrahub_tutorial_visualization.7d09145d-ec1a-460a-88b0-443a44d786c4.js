selector_to_html = {"a[href=\"#via-2-0-cartography-on-zebrahub-visualization\"]": "<h1 class=\"tippy-header\" style=\"margin-top: 0;\">3. Via 2.0 Cartography on Zebrahub (Visualization)<a class=\"headerlink\" href=\"#via-2-0-cartography-on-zebrahub-visualization\" title=\"Permalink to this heading\">\uf0c1</a></h1><p>This tutorial focuses on how to generate the Via2.0 Zebrahub Atlas View. The TI analysis is shown in Tutorial #2</p>", "a[href=\"#re-orient-optional\"]": "<h2 class=\"tippy-header\" style=\"margin-top: 0;\">Re-orient (optional)<a class=\"headerlink\" href=\"#re-orient-optional\" title=\"Permalink to this heading\">\uf0c1</a></h2><p>If you want to apply a rotation transformation on the embedding</p><p>#re-orient/rotate. This can be skipped.  Only use it if neceseary First plot out the atlas_embedding (code in next snipped) and see if you want to rotate it so the early cells are on the LHS.\nimport math\ntheta_deg = 180\ntheta_radians = math.sin(math.radians(theta_deg))\nR = np.array([[math.cos(theta_radians), -math.sin(theta_radians)],[math.sin(theta_radians),math.cos(theta_radians)]])\natlas_embedding_ = np.matmul(R,atlas_embedding.T)\natlas_embedding_=atlas_embedding.T\nv0.embedding = atlas_embedding_</p>", "a[href=\"#atlas-view-edges\"]": "<h2 class=\"tippy-header\" style=\"margin-top: 0;\">Atlas View Edges<a class=\"headerlink\" href=\"#atlas-view-edges\" title=\"Permalink to this heading\">\uf0c1</a></h2>", "a[href=\"#initialize-and-run-the-via2-0-class\"]": "<h2 class=\"tippy-header\" style=\"margin-top: 0;\">Initialize and Run the Via2.0 Class<a class=\"headerlink\" href=\"#initialize-and-run-the-via2-0-class\" title=\"Permalink to this heading\">\uf0c1</a></h2>", "a[href=\"#initialize-parameters\"]": "<h2 class=\"tippy-header\" style=\"margin-top: 0;\">Initialize parameters<a class=\"headerlink\" href=\"#initialize-parameters\" title=\"Permalink to this heading\">\uf0c1</a></h2>", "a[href=\"#extract-some-annotations\"]": "<h2 class=\"tippy-header\" style=\"margin-top: 0;\">Extract some annotations<a class=\"headerlink\" href=\"#extract-some-annotations\" title=\"Permalink to this heading\">\uf0c1</a></h2>", "a[href=\"#computing-the-atlas-view\"]": "<h2 class=\"tippy-header\" style=\"margin-top: 0;\">Computing the Atlas View<a class=\"headerlink\" href=\"#computing-the-atlas-view\" title=\"Permalink to this heading\">\uf0c1</a></h2><p>Once you have the single cell atlas embedding, you can compute the edge layout and generate the full Atlas View</p>", "a[href=\"#find-a-suitable-root\"]": "<h2 class=\"tippy-header\" style=\"margin-top: 0;\">Find a suitable root<a class=\"headerlink\" href=\"#find-a-suitable-root\" title=\"Permalink to this heading\">\uf0c1</a></h2>", "a[href=\"#plot-atlas-embedding\"]": "<h2 class=\"tippy-header\" style=\"margin-top: 0;\">Plot Atlas Embedding\u2019<a class=\"headerlink\" href=\"#plot-atlas-embedding\" title=\"Permalink to this heading\">\uf0c1</a></h2>", "a[href=\"#load-the-data\"]": "<h2 class=\"tippy-header\" style=\"margin-top: 0;\">Load the data<a class=\"headerlink\" href=\"#load-the-data\" title=\"Permalink to this heading\">\uf0c1</a></h2><p>Data has been filtered and library normalized.\nData is annotated with time/stage, cluster labels, and cell type labels (coarse and fine)</p>", "a[href=\"#set-terminal-states-optional\"]": "<h2 class=\"tippy-header\" style=\"margin-top: 0;\">Set terminal states (optional)<a class=\"headerlink\" href=\"#set-terminal-states-optional\" title=\"Permalink to this heading\">\uf0c1</a></h2>"}
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

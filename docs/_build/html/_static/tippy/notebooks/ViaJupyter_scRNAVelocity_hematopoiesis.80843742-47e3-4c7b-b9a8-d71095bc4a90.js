selector_to_html = {"a[href=\"#run-via\"]": "<h2 class=\"tippy-header\" style=\"margin-top: 0;\">Run VIA<a class=\"headerlink\" href=\"#run-via\" title=\"Permalink to this heading\">\uf0c1</a></h2>", "a[href=\"#different-embeddings\"]": "<h2 class=\"tippy-header\" style=\"margin-top: 0;\">Different embeddings<a class=\"headerlink\" href=\"#different-embeddings\" title=\"Permalink to this heading\">\uf0c1</a></h2><p>The directionality offered by the Via graph is consistent with the biology. However, when we project the trajectory onto different types of embeddings, we can sometimes see slightly misleading directions. We therefore show the fine-grained vector field of differentiation on the original tsne computed for this dataset and used in publications, a recomputed tsne on scanpy defaults and on umap.</p><p><strong>Original tsne</strong>\nYou will note that the CLP direction is into itself, however, the HSC2 point in the direction up towards CLP as we expect. Note that in the viagraph, the HSC2s clearly point up to the CLPS. Later we plot the finegrained vector field onto scanpy\u2019s default tsne and umap and see that the overall directinoality is largely consistent with our expectations and the Via graph. This highlights the importance of trying out a few parameters when choosing a suitable embedding.</p>", "a[href=\"#common-visualization-pitfalls\"]": "<h2 class=\"tippy-header\" style=\"margin-top: 0;\">Common visualization pitfalls<a class=\"headerlink\" href=\"#common-visualization-pitfalls\" title=\"Permalink to this heading\">\uf0c1</a></h2><p>We also take the opportunity to show that the underlying via graph is less susceptible to directionality artefacts that visualizations that project the trajectory onto umap/tsne/phate for interpretation. There is no doubt that 2D visualizations are useful and intuitive, but we find that can be instances where the visualized directionality is distorted by the 2D single cell embedding and it can therefore be useful to a) refer back to the via graph based abstraction and b) plot the trajectory on a few types of embeddings to overcome any distortions specific to a visualization method.</p>", "a[href=\"#bone-marrow-with-rna-velocity\"]": "<h1 class=\"tippy-header\" style=\"margin-top: 0;\">7. Bone marrow with RNA-velocity<a class=\"headerlink\" href=\"#bone-marrow-with-rna-velocity\" title=\"Permalink to this heading\">\uf0c1</a></h1><p>We use the familiar bone marrow dataset (Setty et al 2019) but show how to analyze this dataset using a combination of RNA-velocity and gene-gene similarities. Relying purely on RNA-velocity has been noted to be difficult on this dataset due to a boost in expression (Bergen 2021) which yields negative directionality. <strong>By allowing the RNA velocity and gene-gene based graph structure to work together, we can arrive at a more sensible analysis. In Via this is controlled by velo_weight.</strong></p>", "a[href=\"#lineage-paths\"]": "<h2 class=\"tippy-header\" style=\"margin-top: 0;\">Lineage Paths<a class=\"headerlink\" href=\"#lineage-paths\" title=\"Permalink to this heading\">\uf0c1</a></h2><p>See the differentiation likelihood from the HSC-1 towards each of the detected final states (outlined in Red as detected by VIA)</p>", "a[href=\"#load-data-and-pre-process\"]": "<h2 class=\"tippy-header\" style=\"margin-top: 0;\">Load data and pre-process<a class=\"headerlink\" href=\"#load-data-and-pre-process\" title=\"Permalink to this heading\">\uf0c1</a></h2><p><strong>The scvelo plot shows negative direction in some regions</strong></p>"}
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

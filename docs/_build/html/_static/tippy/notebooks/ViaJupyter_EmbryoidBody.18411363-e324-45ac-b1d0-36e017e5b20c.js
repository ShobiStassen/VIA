selector_to_html = {"a[href=\"#human-embryoid-body\"]": "<h1 class=\"tippy-header\" style=\"margin-top: 0;\">Human Embryoid Body<a class=\"headerlink\" href=\"#human-embryoid-body\" title=\"Permalink to this heading\">\uf0c1</a></h1><h2>Load gene expression data (16825 cells), annotations and perform normalization<a class=\"headerlink\" href=\"#load-gene-expression-data-16825-cells-annotations-and-perform-normalization\" title=\"Permalink to this heading\">\uf0c1</a></h2>", "a[href=\"#cluster-level-in-this-example-using-the-fine-grained-clusters-gene-heatmap-for-marker-genes-of-main-cell-types\"]": "<h3 class=\"tippy-header\" style=\"margin-top: 0;\">Cluster level (in this example using the fine-grained clusters) gene heatmap for marker genes of main cell types<a class=\"headerlink\" href=\"#cluster-level-in-this-example-using-the-fine-grained-clusters-gene-heatmap-for-marker-genes-of-main-cell-types\" title=\"Permalink to this heading\">\uf0c1</a></h3>", "a[href=\"#visualize-pseudotime-and-overall-pathways-towards-lineage-commitment-on-phate-t-sne-umap-embedding\"]": "<h2 class=\"tippy-header\" style=\"margin-top: 0;\">Visualize pseudotime and overall pathways towards lineage commitment on Phate/t-SNE/UMAP embedding<a class=\"headerlink\" href=\"#visualize-pseudotime-and-overall-pathways-towards-lineage-commitment-on-phate-t-sne-umap-embedding\" title=\"Permalink to this heading\">\uf0c1</a></h2><p>###Visalize individual lineage paths and lineage likelihoods on the embedding. The Cluster-pathways for each terminal state are also an output and can be used to construct cluster-level lineage specific heatmaps (the specific heatmaps by lineages are not shown below)</p>", "a[href=\"#run-via-first-iteration-is-coarse-grained-and-identifies-the-terminal-states\"]": "<h2 class=\"tippy-header\" style=\"margin-top: 0;\">Run Via (first iteration is coarse grained and identifies the terminal states)<a class=\"headerlink\" href=\"#run-via-first-iteration-is-coarse-grained-and-identifies-the-terminal-states\" title=\"Permalink to this heading\">\uf0c1</a></h2><p>###Run VIA in a second iteration for more granular view. The terminal states detected in first pass are transferred to this iteration as v0.terminal_clusters. One can choose to skip this step, but for mid-size datasets (tens of thousands of cells) it is inexpensive. Each pass takes 30-45 seconds for this dataset of 16825 cells.</p>", "a[href=\"#load-gene-expression-data-16825-cells-annotations-and-perform-normalization\"]": "<h2 class=\"tippy-header\" style=\"margin-top: 0;\">Load gene expression data (16825 cells), annotations and perform normalization<a class=\"headerlink\" href=\"#load-gene-expression-data-16825-cells-annotations-and-perform-normalization\" title=\"Permalink to this heading\">\uf0c1</a></h2>"}
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

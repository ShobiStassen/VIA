selector_to_html = {"a[href=\"#a-b-toy-data-multifurcation-and-disconnected\"]": "<h2 class=\"tippy-header\" style=\"margin-top: 0;\"><strong>1.a/b Toy data (Multifurcation and Disconnected)</strong><a class=\"headerlink\" href=\"#a-b-toy-data-multifurcation-and-disconnected\" title=\"Permalink to this heading\">\uf0c1</a></h2><p>Two examples <a class=\"reference external\" href=\"https://github.com/ShobiStassen/VIA/tree/master/Datasets\">toy datasets</a>  with annotations generated using DynToy are provided. For the step-by-step code within these wrappers, please see the corresponding Jupyter NBs.</p><p><strong>1.a/b Run on Linux</strong></p>", "a[href=\"#b-via-wrapper-for-generic-disconnected-trajectory\"]": "<h2 class=\"tippy-header\" style=\"margin-top: 0;\"><strong>2.b VIA wrapper for generic disconnected trajectory</strong><a class=\"headerlink\" href=\"#b-via-wrapper-for-generic-disconnected-trajectory\" title=\"Permalink to this heading\">\uf0c1</a></h2><p>A slightly different wrapper is called for the disconnected scenario. Refer to the Jupytern NB for a step-by-step tutorial.:</p>", "a[href=\"#examples-for-installation-checking\"]": "<h1 class=\"tippy-header\" style=\"margin-top: 0;\">Examples for installation checking<a class=\"headerlink\" href=\"#examples-for-installation-checking\" title=\"Permalink to this heading\">\uf0c1</a></h1><p>The examples below show how to run VIA on generic connected and disconnected data using wrapper functions and serve as a check for your installation. For more detailed guidelines on running VIA and plotting the results, please use the Notebooks. We also highlight a few difference in calling VIA when using Windows versus Linux. The data for the Jupyter Notebooks and Examples are available in the <a class=\"reference external\" href=\"https://github.com/ShobiStassen/VIA/tree/master/Datasets\">Datasets folder</a> (smaller files) with larger datasets <a class=\"reference external\" href=\"https://drive.google.com/drive/folders/1WQSZeNixUAB1Sm0Xf68ZnSLQXyep936l?usp=sharing\">here</a></p><p>A <a class=\"reference external\" href=\"https://github.com/ShobiStassen/VIA/blob/master/test_pyVIA.py\">test script</a> is available for some of the different datasets, please change the foldername accordingly to the folder containing relevant data files</p>", "a[href=\"#a-general-input-format-and-wrapper-function-uses-example-of-pre-b-cell-differentiation\"]": "<h2 class=\"tippy-header\" style=\"margin-top: 0;\"><strong>2.a General input format and wrapper function (uses example of pre-B cell differentiation)</strong><a class=\"headerlink\" href=\"#a-general-input-format-and-wrapper-function-uses-example-of-pre-b-cell-differentiation\" title=\"Permalink to this heading\">\uf0c1</a></h2><p>These wrapper functions are a good start but we highly recommend you look at the tutorials as you will be afforded a much higher degree of control without much added complexity. The below wrappers operate in the 2-iteration format (a coarse followed by a fine-grained), but this is not always needed and you will have more intuitive for the behaviour of your data by following the steps in the Tutorials. Nonetheless, the following wrappers are a great way to start to familiarize yourself with the various outputs from VIA.</p><p>Datasets and labels used in this example are provided in <a class=\"reference external\" href=\"https://github.com/ShobiStassen/VIA/tree/master/Datasets\">Datasets</a></p>"}
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

selector_to_html = {"a[href=\"#datasets-via-toy-multifurcating\"]": "<h1 class=\"tippy-header\" style=\"margin-top: 0;\">datasets_via.toy_multifurcating<a class=\"headerlink\" href=\"#datasets-via-toy-multifurcating\" title=\"Permalink to this heading\">\uf0c1</a></h1>", "a[href=\"#datasets_via.toy_multifurcating\"]": "<dt class=\"sig sig-object py\" id=\"datasets_via.toy_multifurcating\">\n<span class=\"sig-prename descclassname\"><span class=\"pre\">datasets_via.</span></span><span class=\"sig-name descname\"><span class=\"pre\">toy_multifurcating</span></span><span class=\"sig-paren\">(</span><em class=\"sig-param\"><span class=\"n\"><span class=\"pre\">foldername</span></span><span class=\"o\"><span class=\"pre\">=</span></span><span class=\"default_value\"><span class=\"pre\">'./'</span></span></em><span class=\"sig-paren\">)</span><a class=\"reference internal\" href=\"../../_modules/datasets_via.html#toy_multifurcating\"><span class=\"viewcode-link\"><span class=\"pre\">[source]</span></span></a></dt><dd><p>Load Toy_Multifurcating data as AnnData object</p><p>To access obs (label) as list, use AnnData.obs[\u2018group_id\u2019].values.tolist()</p></dd>"}
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

function onEachFeature(feature, layer) {
    var props = feature.properties;
    var countFormatted = Number(props.count).toLocaleString();

    var html = '<div style="font-family: Arial, sans-serif; min-width: 320px; max-width: 400px; font-size: 12px; padding: 6px;">' +
               '<b style="font-size: 14px; color: #2c3e50; display: block; margin-bottom: 5px;">📍 H3 Cell: ' + props.cell + ' (Res {{RES}})</b>' +
               '<b>📊 Total Images:</b> ' + countFormatted + '<br/>' +
               '<hr style="margin: 6px 0; border: 0; border-top: 1px solid #ddd;"/>' +
               '<b style="color: #16a085; text-transform: uppercase; font-size: 10px; display: block; margin-bottom: 4px;">Dominant Land Use / Cover:</b>' +
               '<ul style="margin: 0; padding-left: 14px; list-style-type: square; color: #34495e;">';

    try {
        var clusters = JSON.parse(props.clusters);
        for (var i = 0; i < clusters.length; i++) {
            var pct = Number(clusters[i][1]).toFixed(1);
            var cCount = Number(clusters[i][2]).toLocaleString();
            var label = clusters[i][0];
            var desc = clusters[i][3];
            var parent_label = clusters[i][4] || "";

            if (desc.length > 150) {
                desc = desc.substring(0, 147) + '...';
            }

            var parent_badge = parent_label ? ' <span style="font-size: 9px; background: #e0f2fe; color: #0369a1; padding: 1px 4px; border-radius: 3px; font-weight: bold; margin-left: 4px; vertical-align: middle;">' + parent_label + '</span>' : '';

            html += '<li style="margin-bottom: 6px;">' +
                    '<b>' + label + '</b>' + parent_badge + ': ' + pct + '% (' + cCount + ' images)' +
                    '<br/><span style="color: #7f8c8d; font-size: 10.5px; font-style: italic;">' + desc + '</span>' +
                    '</li>';
        }
    } catch(e) {
        html += '<li>Error parsing cluster details</li>';
    }

    html += '</ul></div>';
    layer.bindTooltip(html, {
        sticky: true,
        direction: "auto",
        opacity: 0.95
    });
}

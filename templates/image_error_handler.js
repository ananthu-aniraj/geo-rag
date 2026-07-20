const MAPILLARY_TOKEN = '{{MAPILLARY_TOKEN}}';

function handleImageError(img, photoId, platform) {
    if (img.dataset.retryAttempt) return;
    img.dataset.retryAttempt = '1';

    const platformLower = String(platform).toLowerCase().trim();
    const isMapillary = platformLower === 'mapillary' || img.src.includes('mapillary') || img.src.includes('fbcdn.net');
    const isKartaview = platformLower === 'kartaview' || img.src.includes('kartaview') || img.src.includes('openstreetcam');

    let cleanPhotoId = String(photoId).trim();
    if (cleanPhotoId.endsWith('.0')) {
        cleanPhotoId = cleanPhotoId.slice(0, -2);
    }

    if (isMapillary && cleanPhotoId && cleanPhotoId !== 'null' && cleanPhotoId !== 'undefined' && cleanPhotoId !== 'NaN') {
        const apiUrl = 'https://graph.mapillary.com/' + cleanPhotoId + '?fields=thumb_1024_url';
        fetch(apiUrl, {
            headers: { 'Authorization': 'OAuth ' + MAPILLARY_TOKEN }
        })
        .then(res => res.json())
        .then(resData => {
            if (resData.thumb_1024_url) {
                img.src = resData.thumb_1024_url;
                const link = img.closest('div').querySelector('a');
                if (link) link.href = resData.thumb_1024_url;
            }
        })
        .catch(err => console.error('Error fetching Mapillary fresh URL:', err));
    } else if (isKartaview && cleanPhotoId && cleanPhotoId !== 'null' && cleanPhotoId !== 'undefined' && cleanPhotoId !== 'NaN') {
        const apiUrl = 'https://api.openstreetcam.org/2.0/photo/' + cleanPhotoId;
        fetch(apiUrl)
        .then(res => res.json())
        .then(resData => {
            const data = resData.result && resData.result.data;
            if (data) {
                const freshUrl = data.fileurlLTh || data.fileurlTh || data.fileurl;
                if (freshUrl) {
                    img.src = freshUrl;
                    const link = img.closest('div').querySelector('a');
                    if (link) link.href = freshUrl;
                }
            }
        })
        .catch(err => console.error('Error fetching Kartaview fresh URL:', err));
    }
}

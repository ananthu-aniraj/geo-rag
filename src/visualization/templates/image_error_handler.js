const MAPILLARY_TOKEN = '__MAPILLARY_TOKEN__';
const MAP_NAME = '__MAP_NAME__';
let AUTO_DETECTED_BASE = '__AUTO_DETECTED_BASE__';
let DEFAULT_REPLACE_PREFIX = '__DEFAULT_REPLACE_PREFIX__';

function getDisplayUrl(url) {
    if (!url) return '';
    const findInput = document.getElementById('findPrefixInput');
    const replaceInput = document.getElementById('replacePrefixInput');
    const findVal = findInput ? findInput.value.trim() : '';
    const replaceVal = replaceInput ? replaceInput.value.trim() : '';

    if (findVal && replaceVal) {
        if (url.startsWith(findVal)) {
            return url.replace(findVal, replaceVal);
        }
        if (url.startsWith('file://' + findVal)) {
            return url.replace('file://' + findVal, replaceVal);
        }
        if (findVal.startsWith('file://') && url.startsWith(findVal.replace('file://', ''))) {
            return url.replace(findVal.replace('file://', ''), replaceVal);
        }
    }
    return url;
}

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
                const link = img.closest('div') ? img.closest('div').querySelector('a') : null;
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
                    const link = img.closest('div') ? img.closest('div').querySelector('a') : null;
                    if (link) link.href = freshUrl;
                }
            }
        })
        .catch(err => console.error('Error fetching Kartaview fresh URL:', err));
    }
}

function applyPrefixReplacement(popupElement) {
    if (!popupElement) return;

    // Handle iframes if present (e.g. legacy folium iframe popups)
    const iframes = popupElement.querySelectorAll('iframe');
    iframes.forEach(iframe => {
        try {
            const src = iframe.getAttribute('src') || '';
            if (src.includes('base64,')) {
                const b64 = src.split('base64,')[1];
                const decoded = decodeURIComponent(escape(window.atob(b64)));
                const div = document.createElement('div');
                div.innerHTML = decoded;
                div.style.width = '240px';
                div.style.maxHeight = '380px';
                div.style.overflowY = 'auto';
                iframe.parentNode.replaceChild(div, iframe);
                applyPrefixReplacement(div);
                return;
            }
        } catch (e) {
            console.warn('Could not decode iframe popup content:', e);
        }
    });

    const imgs = popupElement.querySelectorAll('img');
    imgs.forEach(img => {
        if (!img.dataset.origSrc) {
            img.dataset.origSrc = img.getAttribute('src') || '';
        }
        const newUrl = getDisplayUrl(img.dataset.origSrc);
        if (img.src !== newUrl) {
            delete img.dataset.retryAttempt;
            img.src = newUrl;
        }
    });

    const links = popupElement.querySelectorAll('a');
    links.forEach(a => {
        if (!a.dataset.origHref) {
            a.dataset.origHref = a.getAttribute('href') || '';
        }
        const newUrl = getDisplayUrl(a.dataset.origHref);
        if (a.href !== newUrl) {
            a.href = newUrl;
        }
    });
}

function updateAllPopupUrls() {
    const replaceInput = document.getElementById('replacePrefixInput');
    if (replaceInput && typeof localStorage !== 'undefined') {
        localStorage.setItem('geo_rag_replace_prefix', replaceInput.value.trim());
    }
    const popups = document.querySelectorAll('.leaflet-popup');
    popups.forEach(popupEl => {
        applyPrefixReplacement(popupEl);
    });
}

function initOfflineMapping() {
    let leafletMap = null;
    if (typeof MAP_NAME === 'string' && MAP_NAME && !MAP_NAME.includes('MAP_NAME') && window[MAP_NAME]) {
        leafletMap = window[MAP_NAME];
    }
    if (!leafletMap) {
        for (const key in window) {
            if (key.startsWith('map_') && window[key] && typeof window[key].on === 'function') {
                leafletMap = window[key];
                break;
            }
        }
    }

    if (!leafletMap) {
        if (!window._retryMapInitCount) window._retryMapInitCount = 0;
        if (window._retryMapInitCount < 10) {
            window._retryMapInitCount++;
            setTimeout(initOfflineMapping, 100);
        }
        return;
    }

    // Inject CSS styles if not already present
    if (!document.getElementById('offline-mapping-styles')) {
        const styleEl = document.createElement('style');
        styleEl.id = 'offline-mapping-styles';
        styleEl.textContent = `
            .offline-mapping-box {
                background: #ffffff;
                padding: 10px 14px;
                border-radius: 8px;
                box-shadow: 0 4px 12px rgba(0, 0, 0, 0.15);
                border: 1px solid #e2e8f0;
                font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
                color: #1e293b;
                z-index: 1000;
            }
            .offline-mapping-title {
                font-size: 11px;
                font-weight: 700;
                color: #64748b;
                text-transform: uppercase;
                letter-spacing: 0.5px;
                margin-bottom: 6px;
                display: flex;
                align-items: center;
                justify-content: space-between;
            }
            .offline-mapping-inputs {
                display: flex;
                gap: 6px;
                align-items: center;
            }
            .offline-mapping-input {
                height: 32px;
                padding: 4px 10px;
                border: 1px solid #cbd5e1;
                border-radius: 6px;
                font-family: inherit;
                font-size: 12px;
                width: 175px;
                outline: none;
                box-sizing: border-box;
                transition: border-color 0.15s ease;
            }
            .offline-mapping-input:focus {
                border-color: #3b82f6;
                box-shadow: 0 0 0 1px #3b82f6;
            }
            .offline-mapping-arrow {
                font-size: 13px;
                color: #64748b;
                font-weight: bold;
            }
        `;
        document.head.appendChild(styleEl);
    }

    // Add Leaflet Control if not already added
    if (!document.getElementById('findPrefixInput')) {
        const MappingControl = L.Control.extend({
            options: { position: 'topright' },
            onAdd: function() {
                const container = L.DomUtil.create('div', 'leaflet-bar offline-mapping-box');
                container.innerHTML = `
                    <div class="offline-mapping-title">
                        <span>Offline Image Server Mapping</span>
                    </div>
                    <div class="offline-mapping-inputs">
                        <input type="text" id="findPrefixInput" class="offline-mapping-input" placeholder="Detecting base path..." oninput="updateAllPopupUrls()">
                        <span class="offline-mapping-arrow">➔</span>
                        <input type="text" id="replacePrefixInput" class="offline-mapping-input" placeholder="e.g. http://localhost:8000/" oninput="updateAllPopupUrls()">
                    </div>
                `;
                L.DomEvent.disableClickPropagation(container);
                L.DomEvent.disableScrollPropagation(container);
                return container;
            }
        });
        leafletMap.addControl(new MappingControl());
    }

    // Auto-detect base file path prefix if not already provided
    let detectedBase = (typeof AUTO_DETECTED_BASE === 'string' && AUTO_DETECTED_BASE && !AUTO_DETECTED_BASE.includes('AUTO_DETECTED_BASE')) ? AUTO_DETECTED_BASE : '';
    if (!detectedBase) {
        const checkContentForPath = (content) => {
            if (!content || detectedBase) return;
            let text = '';
            if (typeof content === 'string') text = content;
            else if (content.innerHTML) text = content.innerHTML;
            else if (content.outerHTML) text = content.outerHTML;

            if (text.includes('base64,')) {
                try {
                    const b64 = text.split('base64,')[1].split('"')[0].split("'")[0];
                    text = decodeURIComponent(escape(window.atob(b64)));
                } catch (e) {}
            }
            const match = text.match(/(?:src|href)=["'](file:\/\/[^"']+|\/[^"']+)/);
            if (match) {
                const full = match[1];
                const lastSlash = full.lastIndexOf('/');
                if (lastSlash !== -1) {
                    detectedBase = full.substring(0, lastSlash + 1);
                }
            }
        };

        leafletMap.eachLayer(layer => {
            if (detectedBase) return;
            if (layer.getPopup && layer.getPopup()) {
                checkContentForPath(layer.getPopup().getContent());
            } else if (layer.eachLayer) {
                layer.eachLayer(subLayer => {
                    if (detectedBase) return;
                    if (subLayer.getPopup && subLayer.getPopup()) {
                        checkContentForPath(subLayer.getPopup().getContent());
                    }
                });
            }
        });
    }

    const findInput = document.getElementById('findPrefixInput');
    const replaceInput = document.getElementById('replacePrefixInput');

    if (findInput && detectedBase) {
        findInput.value = detectedBase;
    }

    let defaultReplace = (typeof DEFAULT_REPLACE_PREFIX === 'string' && DEFAULT_REPLACE_PREFIX && !DEFAULT_REPLACE_PREFIX.includes('DEFAULT_REPLACE_PREFIX')) ? DEFAULT_REPLACE_PREFIX : '';
    if (!defaultReplace && typeof localStorage !== 'undefined') {
        defaultReplace = localStorage.getItem('geo_rag_replace_prefix') || '';
    }
    if (replaceInput && defaultReplace) {
        replaceInput.value = defaultReplace;
    }

    // Listen for Leaflet popupopen event
    leafletMap.on('popupopen', function(e) {
        if (e && e.popup) {
            applyPrefixReplacement(e.popup.getElement());
        }
    });

    // Initial update in case any popup is open
    updateAllPopupUrls();
}

if (document.readyState === 'complete') {
    initOfflineMapping();
} else {
    window.addEventListener('load', initOfflineMapping);
}

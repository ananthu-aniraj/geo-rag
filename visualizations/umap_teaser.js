// Geo-RAG Lightweight Interactive UMAP Manifold Teaser
// Inspired by Cave-Umwelten & It's A Match
(function() {
  function initTeaser() {
    const container = document.getElementById('umap-teaser-container');
    if (!container || typeof UMAP_TEASER_DATA === 'undefined') return;

    const data = UMAP_TEASER_DATA;
    const bounds = data.bounds;
    let activeCategory = 'all';

    // Build controls & SVG markup
    container.innerHTML = `
      <div style="font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif; position: relative; width: 100%; border-radius: 12px; border: 1px solid rgba(128,128,128,0.2); background: rgba(0,0,0,0.02); overflow: hidden; box-shadow: 0 4px 20px rgba(0,0,0,0.06);">
        <div style="display: flex; flex-wrap: wrap; align-items: center; justify-content: space-between; padding: 10px 16px; background: rgba(128,128,128,0.06); border-bottom: 1px solid rgba(128,128,128,0.15); gap: 8px;">
          <div style="font-size: 13px; font-weight: 600; color: #555; display: flex; align-items: center; gap: 8px;">
            <span>🌌 Semantic Manifold Teaser (545 Landmark Centroids)</span>
          </div>
          <div style="display: flex; align-items: center; gap: 8px;">
            <select id="umap-cat-select" style="font-size: 12px; padding: 4px 8px; border-radius: 6px; border: 1px solid rgba(128,128,128,0.3); background: var(--md-default-bg-color, #fff); color: var(--md-default-fg-color, #333); cursor: pointer;">
              <option value="all">All 39 Parent Families</option>
              ${data.categories.map(c => `<option value="${c.name}">${c.name} (${c.points.length})</option>`).join('')}
            </select>
            <button id="umap-reset-btn" style="font-size: 11px; padding: 4px 10px; border-radius: 6px; border: 1px solid rgba(128,128,128,0.3); background: var(--md-default-bg-color, #fff); color: var(--md-default-fg-color, #333); cursor: pointer;">Reset View</button>
          </div>
        </div>
        <div style="position: relative; width: 100%; height: 500px; overflow: hidden; background: #0f141c; cursor: grab;" id="umap-viewport">
          <svg id="umap-svg" style="width: 100%; height: 100%; display: block;" viewBox="0 0 960 500">
            <g id="umap-canvas"></g>
          </svg>
          <div id="umap-tooltip" style="display: none; position: absolute; pointer-events: none; z-index: 100; max-width: 320px; background: rgba(15, 23, 42, 0.95); backdrop-filter: blur(12px); border: 1px solid rgba(255,255,255,0.15); border-radius: 10px; padding: 12px; box-shadow: 0 10px 30px rgba(0,0,0,0.5); color: #fff; font-size: 12px; transform: translate(-50%, -115%); transition: opacity 0.15s ease;">
            <div id="tt-img-wrap" style="width: 100%; height: 140px; border-radius: 6px; overflow: hidden; margin-bottom: 8px; background: #1e293b; display: none;">
              <img id="tt-img" style="width: 100%; height: 100%; object-fit: cover;" src="" alt="" />
            </div>
            <div style="display: flex; align-items: center; justify-content: space-between; gap: 6px; margin-bottom: 4px;">
              <span id="tt-parent" style="font-size: 10px; font-weight: 700; text-transform: uppercase; padding: 2px 6px; border-radius: 4px; background: #3b82f6; color: #fff;"></span>
              <span id="tt-id" style="font-size: 11px; opacity: 0.7;"></span>
            </div>
            <div id="tt-label" style="font-size: 13px; font-weight: 600; margin-bottom: 4px; color: #f8fafc;"></div>
            <div id="tt-desc" style="font-size: 11px; line-height: 1.4; opacity: 0.8; color: #cbd5e1;"></div>
          </div>
        </div>
        <div style="padding: 6px 16px; font-size: 11px; color: #888; background: rgba(128,128,128,0.06); border-top: 1px solid rgba(128,128,128,0.15); display: flex; justify-content: space-between;">
          <span>💡 Hover over centroids to inspect visual tags & photo previews. Drag to pan, scroll to zoom.</span>
          <span>⚡ Ultra-lightweight (136 KB payload, pure SVG)</span>
        </div>
      </div>
    `;

    const svg = document.getElementById('umap-svg');
    const canvas = document.getElementById('umap-canvas');
    const viewport = document.getElementById('umap-viewport');
    const tooltip = document.getElementById('umap-tooltip');
    const ttImgWrap = document.getElementById('tt-img-wrap');
    const ttImg = document.getElementById('tt-img');
    const ttParent = document.getElementById('tt-parent');
    const ttId = document.getElementById('tt-id');
    const ttLabel = document.getElementById('tt-label');
    const ttDesc = document.getElementById('tt-desc');
    const select = document.getElementById('umap-cat-select');
    const resetBtn = document.getElementById('umap-reset-btn');

    // Coordinate mapping into SVG viewBox (960 x 500)
    const pad = 40;
    const w = 960 - 2 * pad;
    const h = 500 - 2 * pad;
    const scaleX = w / (bounds.x_max - bounds.x_min);
    const scaleY = h / (bounds.y_max - bounds.y_min);

    function mapX(val) { return pad + (val - bounds.x_min) * scaleX; }
    function mapY(val) { return 500 - (pad + (val - bounds.y_min) * scaleY); }

    // Render points
    function renderPoints() {
      canvas.innerHTML = '';
      data.categories.forEach(cat => {
        const isDim = activeCategory !== 'all' && activeCategory !== cat.name;
        cat.points.forEach(pt => {
          const circle = document.createElementNS('http://www.w3.org/2000/svg', 'circle');
          const cx = mapX(pt.x);
          const cy = mapY(pt.y);
          const r = Math.min(7, Math.max(3.5, Math.sqrt(pt.count || 20) * 0.35));

          circle.setAttribute('cx', cx);
          circle.setAttribute('cy', cy);
          circle.setAttribute('r', r);
          circle.setAttribute('fill', cat.color);
          circle.setAttribute('opacity', isDim ? '0.12' : '0.85');
          circle.setAttribute('stroke', isDim ? 'none' : '#ffffff');
          circle.setAttribute('stroke-width', '0.8');
          circle.style.cursor = 'pointer';
          circle.style.transition = 'transform 0.15s ease, opacity 0.2s ease';

          circle.addEventListener('mouseenter', () => {
            circle.setAttribute('r', r * 1.7);
            circle.setAttribute('stroke-width', '2');
            circle.setAttribute('stroke', '#ffeb3b');
            circle.setAttribute('opacity', '1');

            ttParent.innerText = pt.parent;
            ttParent.style.background = cat.color;
            ttId.innerText = `ID: #${pt.id} (${pt.count.toLocaleString()} imgs)`;
            ttLabel.innerText = pt.label;
            ttDesc.innerText = pt.desc || 'No description available.';

            if (pt.img) {
              ttImg.src = pt.img;
              ttImgWrap.style.display = 'block';
            } else {
              ttImgWrap.style.display = 'none';
            }
            tooltip.style.display = 'block';
          });

          circle.addEventListener('mousemove', (e) => {
            const rect = viewport.getBoundingClientRect();
            const posX = Math.max(160, Math.min(rect.width - 160, e.clientX - rect.left));
            const posY = Math.max(20, e.clientY - rect.top);
            tooltip.style.left = posX + 'px';
            tooltip.style.top = posY + 'px';
          });

          circle.addEventListener('mouseleave', () => {
            circle.setAttribute('r', r);
            circle.setAttribute('stroke-width', '0.8');
            circle.setAttribute('stroke', isDim ? 'none' : '#ffffff');
            circle.setAttribute('opacity', isDim ? '0.12' : '0.85');
            tooltip.style.display = 'none';
          });

          canvas.appendChild(circle);
        });
      });
    }

    renderPoints();

    // Select filter
    select.addEventListener('change', (e) => {
      activeCategory = e.target.value;
      renderPoints();
    });

    // Pan & Zoom
    let isPanning = false;
    let startX = 0, startY = 0;
    let viewBox = { x: 0, y: 0, w: 960, h: 500 };

    function updateViewBox() {
      svg.setAttribute('viewBox', `${viewBox.x} ${viewBox.y} ${viewBox.w} ${viewBox.h}`);
    }

    viewport.addEventListener('mousedown', (e) => {
      if (e.target.tagName.toLowerCase() === 'circle') return;
      isPanning = true;
      startX = e.clientX;
      startY = e.clientY;
      viewport.style.cursor = 'grabbing';
    });

    window.addEventListener('mousemove', (e) => {
      if (!isPanning) return;
      const dx = (e.clientX - startX) * (viewBox.w / viewport.clientWidth);
      const dy = (e.clientY - startY) * (viewBox.h / viewport.clientHeight);
      viewBox.x -= dx;
      viewBox.y -= dy;
      startX = e.clientX;
      startY = e.clientY;
      updateViewBox();
    });

    window.addEventListener('mouseup', () => {
      isPanning = false;
      viewport.style.cursor = 'grab';
    });

    viewport.addEventListener('wheel', (e) => {
      e.preventDefault();
      const zoomFactor = e.deltaY > 0 ? 1.15 : 0.85;
      const rect = viewport.getBoundingClientRect();
      const mouseX = e.clientX - rect.left;
      const mouseY = e.clientY - rect.top;

      const svgX = viewBox.x + (mouseX / rect.width) * viewBox.w;
      const svgY = viewBox.y + (mouseY / rect.height) * viewBox.h;

      const newW = Math.max(120, Math.min(3000, viewBox.w * zoomFactor));
      const newH = Math.max(62.5, Math.min(1562.5, viewBox.h * zoomFactor));

      viewBox.x = svgX - (mouseX / rect.width) * newW;
      viewBox.y = svgY - (mouseY / rect.height) * newH;
      viewBox.w = newW;
      viewBox.h = newH;
      updateViewBox();
    }, { passive: false });

    resetBtn.addEventListener('click', () => {
      viewBox = { x: 0, y: 0, w: 960, h: 500 };
      activeCategory = 'all';
      select.value = 'all';
      updateViewBox();
      renderPoints();
    });
  }

  if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', initTeaser);
  } else {
    initTeaser();
  }
})();

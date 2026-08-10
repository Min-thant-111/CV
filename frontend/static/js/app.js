/* ──────────────────────────────────────────────────
   Edge-CV Traffic Intelligence — Frontend App
   Handles: file upload, SSE streaming, UI updates
   ────────────────────────────────────────────────── */

(function () {
  'use strict';

  /* ── DOM References ─────────────────────────────── */
  const dropZone        = document.getElementById('drop-zone');
  const fileInput       = document.getElementById('file-input');
  const fileInfo        = document.getElementById('file-info');
  const fileName        = document.getElementById('file-name');
  const fileSize        = document.getElementById('file-size');
  const analyseBtn      = document.getElementById('analyse-btn');
  const btnLabel        = document.getElementById('btn-label');
  const btnSpinner      = document.getElementById('btn-spinner');
  const statusMessage   = document.getElementById('status-message');
  const progressBar     = document.getElementById('progress-bar');
  const progressLabel   = document.getElementById('progress-label');
  const frameLabel      = document.getElementById('frame-label');
  const headerBadge     = document.getElementById('header-status-badge');

  // Metadata
  const metaGrid    = document.getElementById('meta-grid');
  const metaRes     = document.getElementById('meta-res');
  const metaFps     = document.getElementById('meta-fps');
  const metaDur     = document.getElementById('meta-dur');
  const metaFrames  = document.getElementById('meta-frames');

  // Signal
  const lampRed     = document.getElementById('lamp-red');
  const lampYellow  = document.getElementById('lamp-yellow');
  const lampGreen   = document.getElementById('lamp-green');
  const signalState = document.getElementById('signal-state');
  const signalDur   = document.getElementById('signal-duration');
  const signalReason= document.getElementById('signal-reason');

  // Density
  const gaugeArc      = document.getElementById('gauge-arc');
  const densityPct    = document.getElementById('density-pct');
  const densityBadge  = document.getElementById('density-level-badge');
  const vehicleCount  = document.getElementById('vehicle-count');

  // Vehicle breakdown
  const CLASSES = ['car', 'motorcycle', 'bus', 'truck'];

  /* ── State ──────────────────────────────────────── */
  let selectedFile = null;
  let activeSSE    = null;

  /* ── Gauge Constants ────────────────────────────── */
  // Arc path: M 20 100 A 80 80 0 0 1 180 100  →  circumference ≈ 251
  const GAUGE_TOTAL = 251;

  /* ── File Selection ─────────────────────────────── */
  dropZone.addEventListener('click', () => fileInput.click());
  dropZone.addEventListener('keydown', (e) => {
    if (e.key === 'Enter' || e.key === ' ') fileInput.click();
  });

  dropZone.addEventListener('dragover', (e) => {
    e.preventDefault();
    dropZone.classList.add('drag-over');
  });
  dropZone.addEventListener('dragleave', () => dropZone.classList.remove('drag-over'));
  dropZone.addEventListener('drop', (e) => {
    e.preventDefault();
    dropZone.classList.remove('drag-over');
    if (e.dataTransfer.files.length) handleFileSelected(e.dataTransfer.files[0]);
  });

  fileInput.addEventListener('change', () => {
    if (fileInput.files.length) handleFileSelected(fileInput.files[0]);
  });

  function handleFileSelected(file) {
    selectedFile = file;
    fileName.textContent = file.name;
    fileSize.textContent = formatBytes(file.size);
    fileInfo.classList.remove('hidden');
    analyseBtn.disabled = false;
    resetResults();
  }

  /* ── Upload & Analyse ───────────────────────────── */
  analyseBtn.addEventListener('click', () => {
    if (!selectedFile) return;
    startAnalysis(selectedFile);
  });

  function startAnalysis(file) {
    analyseBtn.disabled = true;
    btnLabel.textContent = 'Uploading…';
    btnSpinner.classList.remove('hidden');

    setStatus('Uploading video…', 'processing');
    setProgress(0);

    const formData = new FormData();
    formData.append('video', file);

    const xhr = new XMLHttpRequest();
    xhr.open('POST', '/api/upload');

    xhr.upload.addEventListener('progress', (e) => {
      if (e.lengthComputable) {
        const pct = Math.round((e.loaded / e.total) * 100);
        setProgress(pct, `Uploading… ${pct}%`);
      }
    });

    xhr.addEventListener('load', () => {
      if (xhr.status === 202) {
        const data = JSON.parse(xhr.responseText);
        btnLabel.textContent = 'Processing…';
        setStatus('Upload complete. Starting analysis pipeline…', 'processing');
        setProgress(0);
        openSSEStream(data.job_id);
      } else {
        let msg = 'Upload failed.';
        try { msg = JSON.parse(xhr.responseText).error || msg; } catch (_) {}
        setStatus(`Error: ${msg}`, 'error');
        resetButton();
      }
    });

    xhr.addEventListener('error', () => {
      setStatus('Network error during upload. Is the server running?', 'error');
      resetButton();
    });

    xhr.send(formData);
  }

  /* ── SSE Stream ─────────────────────────────────── */
  function openSSEStream(jobId) {
    if (activeSSE) activeSSE.close();

    activeSSE = new EventSource(`/api/stream/${jobId}`);

    activeSSE.addEventListener('update', (e) => {
      const data = JSON.parse(e.data);
      handleEvent(data);
    });

    activeSSE.addEventListener('done', (e) => {
      const data = JSON.parse(e.data);
      handleEvent(data);
      setStatus('✅ Analysis complete!', 'done');
      setProgress(100, '100%');
      resetButton(true);
      activeSSE.close();
    });

    activeSSE.addEventListener('error', (e) => {
      let msg = 'Pipeline error occurred.';
      try { msg = JSON.parse(e.data).message || msg; } catch (_) {}
      setStatus(`❌ Error: ${msg}`, 'error');
      resetButton();
      activeSSE.close();
    });

    activeSSE.onerror = () => {
      // SSE connection closed after done — expected
    };
  }

  /* ── Event Handler ──────────────────────────────── */
  function handleEvent(data) {
    switch (data.type) {
      case 'status':
        setStatus(data.message, 'processing');
        break;

      case 'metadata':
        metaRes.textContent    = `${data.width} × ${data.height}`;
        metaFps.textContent    = `${data.fps} FPS`;
        metaDur.textContent    = `${data.duration}s`;
        metaFrames.textContent = data.total_frames;
        metaGrid.classList.remove('hidden');
        break;

      case 'frame':
      case 'done':
        updateDensityPanel(data);
        updateSignalPanel(data);
        updateBreakdown(data.class_counts || {}, data.vehicle_count || 0);
        if (data.progress_pct !== undefined) {
          setProgress(data.progress_pct,
            `${data.progress_pct}%`,
            `Frame ${data.frame_index} / ${data.total_frames}`
          );
        }
        setStatus(
          `Processing frame ${data.frame_index} | ` +
          `${data.vehicle_count} vehicle${data.vehicle_count !== 1 ? 's' : ''} detected | ` +
          `MQTT ${data.mqtt_published ? '✓' : '–'}`,
          'processing'
        );
        break;
    }
  }

  /* ── UI Update: Density ─────────────────────────── */
  function updateDensityPanel(data) {
    const pct   = data.density_percentage || 0;
    const level = (data.density_level || 'LOW').toUpperCase();

    densityPct.textContent = `${pct}%`;
    vehicleCount.textContent = data.vehicle_count ?? '—';

    // Gauge arc: dasharray = (pct/100) * GAUGE_TOTAL
    const filled = Math.round((pct / 100) * GAUGE_TOTAL);
    gaugeArc.setAttribute('stroke-dasharray', `${filled} ${GAUGE_TOTAL - filled}`);

    // Colour by level
    const levelKey = level.toLowerCase();
    gaugeArc.className.baseVal = `gauge-fill gauge-${levelKey}`;
    densityPct.className = `gauge-pct level-${levelKey}`;

    densityBadge.textContent = level;
    densityBadge.className = `density-level-badge level-badge-${levelKey}`;
  }

  /* ── UI Update: Signal ──────────────────────────── */
  function updateSignalPanel(data) {
    const sig    = (data.signal || 'RED').toUpperCase();
    const dur    = data.green_duration ?? '—';
    const reason = data.reason || '';

    signalState.textContent  = sig;
    signalDur.textContent    = `${dur} seconds GREEN`;
    signalReason.textContent = reason;

    // Colour
    const colours = { GREEN: '#22c55e', YELLOW: '#facc15', RED: '#f43f5e' };
    signalState.style.color = colours[sig] || '#e2eaf8';

    // Activate correct lamp
    lampRed.className    = 'tl-lamp tl-red'    + (sig === 'RED'    ? ' active-red'    : '');
    lampYellow.className = 'tl-lamp tl-yellow' + (sig === 'YELLOW' ? ' active-yellow' : '');
    lampGreen.className  = 'tl-lamp tl-green'  + (sig === 'GREEN'  ? ' active-green'  : '');
  }

  /* ── UI Update: Breakdown ────────────────────────── */
  function updateBreakdown(counts, total) {
    const maxCount = Math.max(total, 1);
    CLASSES.forEach((cls) => {
      const c   = counts[cls] || 0;
      const pct = Math.round((c / maxCount) * 100);
      const countEl = document.getElementById(`count-${cls}`);
      const barEl   = document.getElementById(`bar-${cls}`);
      if (countEl) countEl.textContent = c;
      if (barEl)   barEl.style.width   = `${pct}%`;
    });
  }

  /* ── Helpers ─────────────────────────────────────── */
  function setStatus(msg, state) {
    statusMessage.textContent = msg;

    const badgeMap = {
      idle:       'badge-idle',
      processing: 'badge-processing',
      done:       'badge-done',
      error:      'badge-error',
    };
    headerBadge.className = `badge ${badgeMap[state] || 'badge-idle'}`;
    headerBadge.textContent = state.toUpperCase();
  }

  function setProgress(pct, label, frameText) {
    const clampedPct = Math.min(100, Math.max(0, pct));
    progressBar.style.width = `${clampedPct}%`;
    progressLabel.textContent = label || `${clampedPct}%`;
    if (frameText !== undefined) frameLabel.textContent = frameText;
  }

  function resetButton(done = false) {
    btnSpinner.classList.add('hidden');
    btnLabel.textContent = done ? 'Analyse Another' : 'Analyse Video';
    analyseBtn.disabled = !selectedFile;
  }

  function resetResults() {
    setStatus('File selected. Press "Analyse Video" to start.', 'idle');
    setProgress(0, '0%', '—');
    metaGrid.classList.add('hidden');
    lampRed.className    = 'tl-lamp tl-red';
    lampYellow.className = 'tl-lamp tl-yellow';
    lampGreen.className  = 'tl-lamp tl-green';
    signalState.textContent  = '—';
    signalDur.textContent    = '— s';
    signalReason.textContent = 'No signal yet.';
    densityPct.textContent   = '0%';
    densityBadge.textContent = '—';
    densityBadge.className   = 'density-level-badge';
    vehicleCount.textContent = '—';
    gaugeArc.setAttribute('stroke-dasharray', `0 ${GAUGE_TOTAL}`);
    CLASSES.forEach((cls) => {
      const countEl = document.getElementById(`count-${cls}`);
      const barEl   = document.getElementById(`bar-${cls}`);
      if (countEl) countEl.textContent = '—';
      if (barEl)   barEl.style.width   = '0%';
    });
  }

  function formatBytes(bytes) {
    if (bytes < 1024)       return `${bytes} B`;
    if (bytes < 1048576)    return `${(bytes / 1024).toFixed(1)} KB`;
    return `${(bytes / 1048576).toFixed(1)} MB`;
  }

})();

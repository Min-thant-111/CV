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
  const sourceVideo     = document.getElementById('source-video');
  const sourceEmpty     = document.getElementById('source-empty');
  const sourceCaption   = document.getElementById('source-caption');
  const outputVideo     = document.getElementById('output-video');
  const outputPreview   = document.getElementById('output-preview');
  const outputEmpty     = document.getElementById('output-empty');
  const outputCaption   = document.getElementById('output-caption');

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
  const signalDensitySummary = document.getElementById('signal-density-summary');
  const timingBreakdown = document.getElementById('timing-breakdown');
  const timingBase = document.getElementById('timing-base');
  const timingDemand = document.getElementById('timing-demand');
  const timingPath = document.getElementById('timing-path');
  const timingCap = document.getElementById('timing-cap');
  const processingTime = document.getElementById('processing-time');

  // Density
  const gaugeArc      = document.getElementById('gauge-arc');
  const densityPct    = document.getElementById('density-pct');
  const densityBadge  = document.getElementById('density-level-badge');
  const vehicleCount  = document.getElementById('vehicle-count');
  const detectedPeakCount = document.getElementById('detected-peak-count');
  const averageVehicleCount = document.getElementById('average-vehicle-count');
  const activeVehicleCount = document.getElementById('active-vehicle-count');
  const densityPathCount = document.getElementById('density-path-count');

  // Vehicle breakdown
  const CLASSES = ['car', 'motorcycle', 'bus', 'truck'];

  /* ── State ──────────────────────────────────────── */
  let selectedFile = null;
  let activeSSE    = null;
  let selectedObjectUrl = null;
  let activeJobId = null;
  let statusPollTimer = null;
  let jobFinished = false;
  let previewPollTimer = null;
  let previewPollBusy = false;
  let previewVersion = 0;
  let previewJobId = null;
  let previewObjectUrl = null;

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
    if (selectedObjectUrl) URL.revokeObjectURL(selectedObjectUrl);
    selectedObjectUrl = URL.createObjectURL(file);
    if (/\.(avi|mkv|mov)$/i.test(file.name)) {
      clearVideo('source', 'This format needs browser conversion. The preview will appear after upload.');
    } else {
      showVideo('source', selectedObjectUrl, `${file.name} (local preview)`);
    }
    clearVideo('output', 'The annotated result will appear here after analysis.');
  }

  /* ── Upload & Analyse ───────────────────────────── */
  analyseBtn.addEventListener('click', () => {
    if (!selectedFile) return;
    startAnalysis(selectedFile);
  });

  function startAnalysis(file) {
    stopJobMonitoring();
    jobFinished = false;
    analyseBtn.disabled = true;
    btnLabel.textContent = 'Uploading…';
    btnSpinner.classList.remove('hidden');

    setStatus('Uploading video…', 'processing');
    setProgress(0);
    clearVideo('output', 'Analysis is running. The processed video will appear automatically.');

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
        showPlayableVideo('source', data.source_url, data.filename);
        if (selectedObjectUrl) {
          URL.revokeObjectURL(selectedObjectUrl);
          selectedObjectUrl = null;
        }
        btnLabel.textContent = 'Processing…';
        setStatus('Upload complete. Starting analysis pipeline…', 'processing');
        setProgress(0);
        startOutputPreview(data.job_id);
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
    activeJobId = jobId;

    activeSSE = new EventSource(`/api/stream/${jobId}`);

    activeSSE.addEventListener('update', (e) => {
      const data = JSON.parse(e.data);
      handleEvent(data);
    });

    activeSSE.addEventListener('done', (e) => {
      const data = JSON.parse(e.data);
      completeAnalysis(data);
    });

    activeSSE.addEventListener('error', (e) => {
      // Custom pipeline errors contain data. Native connection errors do not;
      // polling continues so a transient SSE disconnect cannot lose completion.
      if (e.data) {
        let msg = 'Pipeline error occurred.';
        try { msg = JSON.parse(e.data).message || msg; } catch (_) {}
        failAnalysis(msg);
      }
    });

    statusPollTimer = window.setInterval(() => pollJobStatus(jobId), 1000);
  }

  async function pollJobStatus(jobId) {
    if (jobFinished || jobId !== activeJobId) return;
    try {
      const response = await fetch(`/api/status/${jobId}`, { cache: 'no-store' });
      if (!response.ok) return;
      const job = await response.json();
      if (job.status === 'done') completeAnalysis(job.result || {});
      if (job.status === 'error') {
        failAnalysis((job.result && job.result.message) || 'Pipeline error occurred.');
      }
    } catch (_) {
      // The next poll retries while SSE remains the primary live channel.
    }
  }

  async function completeAnalysis(data) {
    if (jobFinished) return;
    jobFinished = true;
    handleEvent(data);
    setStatus('✅ Analysis complete!', 'done');
    setProgress(100, '100%');
    resetButton(true);
    stopJobMonitoring();
    if (data.output_url) {
      await showPlayableVideo(
        'output', `${data.output_url}?v=${Date.now()}`, 'Annotated analysis result'
      );
    } else {
      clearVideo('output', 'Analysis completed, but no processed video URL was returned.');
    }
  }

  function failAnalysis(message) {
    if (jobFinished) return;
    jobFinished = true;
    setStatus(`❌ Error: ${message}`, 'error');
    resetButton();
    stopJobMonitoring();
  }

  function stopJobMonitoring() {
    if (activeSSE) activeSSE.close();
    activeSSE = null;
    if (statusPollTimer !== null) window.clearInterval(statusPollTimer);
    statusPollTimer = null;
    activeJobId = null;
    stopOutputPreview(false);
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
          `Estimated ${data.vehicle_count} vehicle${data.vehicle_count !== 1 ? 's' : ''}` +
          (data.detected_peak_vehicle_count !== undefined
            ? ` | ${data.detected_peak_vehicle_count} detected peak`
            : '') +
          (data.average_vehicle_count !== undefined
            ? ` | ${data.average_vehicle_count} average`
            : '') +
          (data.active_vehicle_count !== undefined
            ? ` | ${data.active_vehicle_count} active now`
            : '') +
          ` | ` +
          `MQTT ${data.mqtt_published ? '✓' : '–'}` +
          (data.eta_seconds !== undefined
            ? ` | ETA ${formatDuration(data.eta_seconds)}`
            : ''),
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
    detectedPeakCount.textContent = data.detected_peak_vehicle_count ?? '—';
    averageVehicleCount.textContent = data.average_vehicle_count ?? '—';
    activeVehicleCount.textContent = data.active_vehicle_count ?? '—';

    // Gauge arc: dasharray = (pct/100) * GAUGE_TOTAL
    const filled = Math.round((pct / 100) * GAUGE_TOTAL);
    const boundedFill = Math.max(0, Math.min(GAUGE_TOTAL, filled));
    gaugeArc.setAttribute('stroke-dasharray', `${boundedFill} ${GAUGE_TOTAL - boundedFill}`);
    if (data.road_path_count !== undefined) {
      const confidence = data.road_path_confidence;
      densityPathCount.textContent = confidence !== undefined
        ? `${data.road_path_count} (${Math.round(confidence * 100)}% confidence)`
        : data.road_path_count;
    } else {
      densityPathCount.textContent = '—';
    }

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

    signalState.textContent  = sig;
    signalDur.textContent    = `${dur} seconds GREEN`;
    const level = String(data.density_level || 'low').toLowerCase();
    const levelLabel = level.charAt(0).toUpperCase() + level.slice(1);
    const vehicles = data.vehicle_count ?? 0;
    const paths = data.road_path_count ?? 1;
    signalDensitySummary.innerHTML =
      `<strong class="level-${level}">${levelLabel} traffic density</strong>` +
      `<span>${data.density_percentage ?? 0}%</span> with ` +
      `<strong>${vehicles}</strong> vehicle${vehicles === 1 ? '' : 's'} across ` +
      `<strong>${paths}</strong> road path${paths === 1 ? '' : 's'}.`;

    const hasBreakdown = data.base_duration !== undefined;
    timingBreakdown.classList.toggle('hidden', !hasBreakdown);
    if (hasBreakdown) {
      timingBase.textContent = `${data.base_duration}s`;
      timingDemand.textContent = `+ ${data.vehicle_demand_duration}s`;
      timingPath.textContent = `+ ${data.per_path_queue_duration}s`;
    }
    const wasCapped = data.uncapped_duration > data.green_duration;
    timingCap.classList.toggle('hidden', !wasCapped);
    if (wasCapped) {
      timingCap.textContent =
        `${data.uncapped_duration}s calculated · capped safely at ${data.green_duration}s`;
    }
    const isDone = data.type === 'done' && data.processing_seconds !== undefined;
    processingTime.classList.toggle('hidden', !isDone);
    if (isDone) {
      processingTime.textContent = `Processed in ${formatDuration(data.processing_seconds)}`;
    }

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

  /* Video playback */
  function stopOutputPreview(hide = false) {
    previewJobId = null;
    if (previewPollTimer !== null) window.clearInterval(previewPollTimer);
    previewPollTimer = null;
    if (hide) {
      outputPreview.classList.add('hidden');
      outputPreview.removeAttribute('src');
      if (previewObjectUrl) URL.revokeObjectURL(previewObjectUrl);
      previewObjectUrl = null;
    }
  }

  function startOutputPreview(jobId) {
    stopOutputPreview(true);
    previewJobId = jobId;
    previewVersion = 0;

    const poll = async () => {
      if (previewPollBusy || previewJobId !== jobId) return;
      previewPollBusy = true;
      try {
        const response = await fetch(
          `/api/preview/${jobId}?since=${previewVersion}`,
          { cache: 'no-store' }
        );
        if (response.status === 204) return;
        if (!response.ok) throw new Error(`HTTP ${response.status}`);
        const version = Number(response.headers.get('X-Preview-Version') || 0);
        const frameNumber = Number(response.headers.get('X-Preview-Frame') || 0);
        const blob = await response.blob();
        if (previewJobId !== jobId) return;
        const objectUrl = URL.createObjectURL(blob);
        if (previewObjectUrl) URL.revokeObjectURL(previewObjectUrl);
        previewObjectUrl = objectUrl;
        previewVersion = version;
        outputPreview.src = objectUrl;
        outputPreview.classList.remove('hidden');
        outputVideo.classList.add('hidden');
        outputEmpty.classList.add('hidden');
        outputCaption.textContent = frameNumber >= 0
          ? `Live analysed frame ${frameNumber}`
          : 'Preparing vehicle detection…';
        outputCaption.classList.remove('hidden');
      } catch (_) {
        // A later poll retries; metric updates continue over SSE.
      } finally {
        previewPollBusy = false;
      }
    };

    poll();
    previewPollTimer = window.setInterval(poll, 750);
  }

  function showVideo(kind, url, caption) {
    const video = kind === 'source' ? sourceVideo : outputVideo;
    const empty = kind === 'source' ? sourceEmpty : outputEmpty;
    const captionEl = kind === 'source' ? sourceCaption : outputCaption;
    if (kind === 'output') stopOutputPreview(true);
    video.src = url;
    video.classList.remove('hidden');
    empty.classList.add('hidden');
    captionEl.textContent = caption;
    captionEl.classList.remove('hidden');
    video.onerror = () => {
      video.classList.add('hidden');
      empty.textContent = 'The browser could not decode this video. Check the server conversion error.';
      empty.classList.remove('hidden');
    };
    video.load();
  }

  function playbackUrl(url) {
    const value = String(url || '');
    if (/^\/media\/(uploads|outputs)\//.test(value)) {
      return value.replace('/media/', '/media/play/');
    }
    return value;
  }

  async function showPlayableVideo(kind, url, caption) {
    const playable = playbackUrl(url);
    const separator = playable.includes('?') ? '&' : '?';
    const checkedUrl = `${playable}${separator}media_check=${Date.now()}`;
    clearVideo(kind, 'Preparing browser-compatible video…');
    try {
      const response = await fetch(checkedUrl, {
        headers: { Range: 'bytes=0-0' },
        cache: 'no-store',
      });
      if (!response.ok) {
        let detail = `HTTP ${response.status}`;
        try { detail = (await response.json()).error || detail; } catch (_) {}
        throw new Error(detail);
      }
      await response.arrayBuffer();
      showVideo(kind, checkedUrl, caption);
    } catch (error) {
      clearVideo(kind, `Video preparation failed: ${error.message}`);
    }
  }

  function clearVideo(kind, message) {
    const video = kind === 'source' ? sourceVideo : outputVideo;
    const empty = kind === 'source' ? sourceEmpty : outputEmpty;
    const captionEl = kind === 'source' ? sourceCaption : outputCaption;
    if (kind === 'output') stopOutputPreview(true);
    video.onerror = null;
    video.pause();
    video.removeAttribute('src');
    video.load();
    video.classList.add('hidden');
    captionEl.classList.add('hidden');
    empty.textContent = message;
    empty.classList.remove('hidden');
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
    signalDensitySummary.textContent = 'No signal yet.';
    timingBreakdown.classList.add('hidden');
    timingCap.classList.add('hidden');
    processingTime.classList.add('hidden');
    densityPct.textContent   = '0%';
    densityBadge.textContent = '—';
    densityBadge.className   = 'density-level-badge';
    vehicleCount.textContent = '—';
    detectedPeakCount.textContent = '—';
    averageVehicleCount.textContent = '—';
    activeVehicleCount.textContent = '—';
    densityPathCount.textContent = '—';
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

  function formatDuration(seconds) {
    const total = Math.max(0, Math.round(Number(seconds) || 0));
    const minutes = Math.floor(total / 60);
    const remainder = total % 60;
    return minutes ? `${minutes}m ${remainder}s` : `${remainder}s`;
  }


})();

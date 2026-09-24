/**
 * main.js - Agent 3: UI/UX & Browser Integration
 * Full Hybrid Architecture: Supports local Python MJPEG backend AND 100% In-Browser Netlify Client-Side MediaPipe AI.
 */

document.addEventListener('DOMContentLoaded', () => {
  // DOM Elements
  const mjpegStream = document.getElementById('mjpegStream');
  const clientVideo = document.getElementById('clientVideo');
  const overlayCanvas = document.getElementById('overlayCanvas');
  const ctx = overlayCanvas ? overlayCanvas.getContext('2d') : null;

  const reticle = document.getElementById('antiGravityReticle');
  const reticleLabel = document.getElementById('reticleLabel');
  const translationCard = document.getElementById('translationCard');
  const translationIcon = document.getElementById('translationIcon');
  const translationText = document.getElementById('translationText');
  const translationSubtext = document.getElementById('translationSubtext');
  const confidenceNum = document.getElementById('confidenceNum');
  const meterFill = document.getElementById('meterFill');
  const gateBadge = document.getElementById('gateBadge');
  const spatialLockBadge = document.getElementById('spatialLockBadge');
  const diagStabilityVal = document.getElementById('diagStabilityVal');
  const stabilityBar = document.getElementById('stabilityBar');
  const diagGateState = document.getElementById('diagGateState');
  const headerLockText = document.getElementById('headerLockText');
  const fpsCounter = document.getElementById('fpsCounter');
  const toggleStreamModeBtn = document.getElementById('toggleStreamModeBtn');
  const chips = document.querySelectorAll('.sign-chip');

  // DOM Elements - Mode Switcher & Panels
  const tabLiveDetection = document.getElementById('tabLiveDetection');
  const tabPracticeTraining = document.getElementById('tabPracticeTraining');
  const liveDetectionPanel = document.getElementById('liveDetectionPanel');
  const practiceTrainingPanel = document.getElementById('practiceTrainingPanel');

  // DOM Elements - Practice Panel
  const practiceTargetSelect = document.getElementById('practiceTargetSelect');
  const practiceTargetIcon = document.getElementById('practiceTargetIcon');
  const practiceTargetTitle = document.getElementById('practiceTargetTitle');
  const practiceTargetDesc = document.getElementById('practiceTargetDesc');
  const practiceTypeBadge = document.getElementById('practiceTypeBadge');
  const practiceMatchPercent = document.getElementById('practiceMatchPercent');
  const practiceMeterFill = document.getElementById('practiceMeterFill');
  const fingerChecklistGrid = document.getElementById('fingerChecklistGrid');
  const coachFeedbackText = document.getElementById('coachFeedbackText');
  const holdChallengeBar = document.getElementById('holdChallengeBar');
  const holdProgress = document.getElementById('holdProgress');
  const holdText = document.getElementById('holdText');
  const practiceSuccessBanner = document.getElementById('practiceSuccessBanner');
  const btnNextPracticeSign = document.getElementById('btnNextPracticeSign');

  let audioCtx = null;
  let lastVerifiedId = null;
  let practiceCatalog = [];
  let currentPracticeIndex = 2; // Default to I Love You
  let holdStartTime = null;
  const HOLD_DURATION_MS = 1400; // Hold 1.4s to pass
  let hasPassedCurrentSign = false;

  // Client-Side AI Engine Instances
  let stabilizer = null;
  let classifier = null;
  let isClientCameraActive = false;
  let currentFacingMode = 'user';
  let mediaPipeHands = null;
  let mediaPipeCamera = null;
  let clientFps = 0;
  let clientFrameCount = 0;
  let clientFpsTimer = performance.now();

  if (window.AntiGravityStabilizerJS && window.ZeroErrorClassifierJS) {
    stabilizer = new window.AntiGravityStabilizerJS();
    classifier = new window.ZeroErrorClassifierJS();
  }

  // Web Audio Synth for Zero-Error Chime
  function playVerificationChime() {
    try {
      if (!audioCtx) {
        audioCtx = new (window.AudioContext || window.webkitAudioContext)();
      }
      if (audioCtx.state === 'suspended') audioCtx.resume();

      const osc = audioCtx.createOscillator();
      const gain = audioCtx.createGain();
      osc.type = 'sine';
      osc.frequency.setValueAtTime(587.33, audioCtx.currentTime); // D5
      osc.frequency.exponentialRampToValueAtTime(880.00, audioCtx.currentTime + 0.12); // A5

      gain.gain.setValueAtTime(0.08, audioCtx.currentTime);
      gain.gain.exponentialRampToValueAtTime(0.001, audioCtx.currentTime + 0.28);

      osc.connect(gain);
      gain.connect(audioCtx.destination);
      osc.start();
      osc.stop(audioCtx.currentTime + 0.3);
    } catch (e) {}
  }

  // Sound Synth for Practice Success Chime
  function playTriumphChime() {
    try {
      if (!audioCtx) {
        audioCtx = new (window.AudioContext || window.webkitAudioContext)();
      }
      if (audioCtx.state === 'suspended') audioCtx.resume();

      const now = audioCtx.currentTime;
      [523.25, 659.25, 783.99, 1046.50].forEach((freq, i) => {
        const osc = audioCtx.createOscillator();
        const gain = audioCtx.createGain();
        osc.type = 'triangle';
        osc.frequency.setValueAtTime(freq, now + i * 0.08);

        gain.gain.setValueAtTime(0.12, now + i * 0.08);
        gain.gain.exponentialRampToValueAtTime(0.001, now + i * 0.08 + 0.4);

        osc.connect(gain);
        gain.connect(audioCtx.destination);
        osc.start(now + i * 0.08);
        osc.stop(now + i * 0.08 + 0.42);
      });
    } catch (e) {}
  }

  // Speech Synthesizer for Verified Sign
  function speakSign(text) {
    if (!('speechSynthesis' in window)) return;
    try {
      window.speechSynthesis.cancel();
      const cleanText = text.replace(/[^\w\s]/gi, '').trim();
      const utter = new SpeechSynthesisUtterance(cleanText);
      utter.lang = 'id-ID';
      utter.rate = 1.05;
      utter.pitch = 1.0;
      window.speechSynthesis.speak(utter);
    } catch (e) {}
  }

  // =============================================================
  // Client-Side Browser Camera & MediaPipe Engine (Netlify Ready)
  // =============================================================

  // MediaPipe Hand Connection lines
  const HAND_CONNECTIONS = [
    [0, 1], [1, 2], [2, 3], [3, 4],          // Thumb
    [0, 5], [5, 6], [6, 7], [7, 8],          // Index
    [9, 10], [10, 11], [11, 12], [0, 9],     // Middle
    [13, 14], [14, 15], [15, 16], [0, 13],   // Ring
    [0, 17], [17, 18], [18, 19], [19, 20],   // Pinky
    [5, 9], [9, 13], [13, 17]                // Palm Knuckles
  ];

  async function startClientCamera() {
    if (!clientVideo || !overlayCanvas || !ctx) return;
    console.log('[Netlify AI] Activating client-side browser webcam...');

    if (window.location.protocol === 'file:') {
      console.warn('[Camera] Protokol file:/// memblokir akses webcam browser.');
      overlayCanvas.width = 640;
      overlayCanvas.height = 480;
      overlayCanvas.style.display = 'block';
      if (mjpegStream) mjpegStream.style.display = 'none';
      renderStandbyGuide(640, 480, "BUKA DENGAN: HTTP://LOCALHOST:5000");
      return;
    }

    try {
      if (clientVideo.srcObject) {
        clientVideo.srcObject.getTracks().forEach(t => t.stop());
      }

      if (!navigator.mediaDevices || !navigator.mediaDevices.getUserMedia) {
        throw new Error('getUserMedia not supported in this context');
      }

      const stream = await navigator.mediaDevices.getUserMedia({
        video: {
          width: { ideal: 640 },
          height: { ideal: 480 },
          facingMode: currentFacingMode
        },
        audio: false
      });

      clientVideo.srcObject = stream;
      await clientVideo.play();

      overlayCanvas.width = clientVideo.videoWidth || 640;
      overlayCanvas.height = clientVideo.videoHeight || 480;

      isClientCameraActive = true;
      if (mjpegStream) mjpegStream.style.display = 'none';
      overlayCanvas.style.display = 'block';

      initMediaPipeHands();
    } catch (err) {
      console.warn('[Camera] Browser webcam access failed:', err);
      overlayCanvas.width = 640;
      overlayCanvas.height = 480;
      overlayCanvas.style.display = 'block';
      if (mjpegStream) mjpegStream.style.display = 'none';

      let msg = "IZINKAN AKSES KAMERA DI BROWSER";
      if (err.name === 'NotAllowedError' || err.name === 'PermissionDeniedError') {
        msg = "AKSES KAMERA DITOLAK (IZINKAN DI IKON GEMBOK BROWSER)";
      } else if (err.name === 'NotFoundError' || err.name === 'DevicesNotFoundError') {
        msg = "KAMERA TIDAK DITEMUKAN DI PERANGKAT";
      } else if (err.name === 'NotReadableError' || err.name === 'TrackStartError') {
        msg = "KAMERA SEDANG DIPAKAI APLIKASI LAIN";
      }
      renderStandbyGuide(640, 480, msg);
    }
  }

  function initMediaPipeHands() {
    if (typeof Hands === 'undefined') {
      console.warn('[MediaPipe] Hands script not yet available from CDN, retrying in 500ms...');
      setTimeout(initMediaPipeHands, 500);
      return;
    }

    if (!mediaPipeHands) {
      mediaPipeHands = new Hands({
        locateFile: (file) => `https://cdn.jsdelivr.net/npm/@mediapipe/hands/${file}`
      });

      mediaPipeHands.setOptions({
        maxNumHands: 1,
        modelComplexity: 1,
        minDetectionConfidence: 0.60,
        minTrackingConfidence: 0.60
      });

      mediaPipeHands.onResults(onHandResults);
    }

    if (typeof Camera !== 'undefined' && !mediaPipeCamera) {
      mediaPipeCamera = new Camera(clientVideo, {
        onFrame: async () => {
          if (isClientCameraActive && clientVideo && clientVideo.readyState >= 2) {
            await mediaPipeHands.send({ image: clientVideo });
          }
        },
        width: 640,
        height: 480
      });
      mediaPipeCamera.start();
    } else {
      // Fallback RAF loop if Camera helper isn't loaded
      requestAnimationFrame(runClientFrameLoop);
    }
  }

  async function runClientFrameLoop() {
    if (isClientCameraActive && clientVideo && clientVideo.readyState >= 2) {
      try {
        await mediaPipeHands.send({ image: clientVideo });
      } catch (e) {}
    }
    if (isClientCameraActive) {
      requestAnimationFrame(runClientFrameLoop);
    }
  }

  function onHandResults(results) {
    if (!ctx || !overlayCanvas) return;
    const w = overlayCanvas.width;
    const h = overlayCanvas.height;

    // Calculate FPS
    clientFrameCount++;
    const now = performance.now();
    if (now - clientFpsTimer >= 1000) {
      clientFps = Math.round((clientFrameCount * 1000) / (now - clientFpsTimer));
      clientFrameCount = 0;
      clientFpsTimer = now;
    }

    // 1. Draw mirrored camera frame
    ctx.save();
    ctx.clearRect(0, 0, w, h);
    ctx.translate(w, 0);
    ctx.scale(-1, 1);
    ctx.drawImage(clientVideo, 0, 0, w, h);

    if (results.multiHandLandmarks && results.multiHandLandmarks.length > 0 && stabilizer && classifier) {
      const rawLandmarks = results.multiHandLandmarks[0];

      // 2. Anti-Gravity Spatial Stabilization
      const stabRes = stabilizer.process(rawLandmarks);
      const targetId = practiceCatalog[currentPracticeIndex] ? practiceCatalog[currentPracticeIndex].id : null;

      // 3. Zero-Error Classifier Inference (>88%)
      const pred = classifier.predict(stabRes.canonicalPoints, stabRes.isLocked, targetId, stabRes.filteredRaw);

      // 4. Render skeleton directly on user's real hand 1:1
      const ptsPx = stabRes.filteredRaw.map(p => ({
        x: Math.round(p[0] * w),
        y: Math.round(p[1] * h)
      }));

      const isVerified = pred.verified;
      const boneCol = isVerified ? '#00f0b4' : '#00b4f0';
      const nodeCol = isVerified ? '#00ffa3' : '#00e1ff';

      // Draw bones
      ctx.lineWidth = 3;
      ctx.strokeStyle = boneCol;
      ctx.lineCap = 'round';
      HAND_CONNECTIONS.forEach(([s, e]) => {
        if (ptsPx[s] && ptsPx[e]) {
          ctx.beginPath();
          ctx.moveTo(ptsPx[s].x, ptsPx[s].y);
          ctx.lineTo(ptsPx[e].x, ptsPx[e].y);
          ctx.stroke();
        }
      });

      // Draw nodes
      ptsPx.forEach((p, idx) => {
        const isTip = [4, 8, 12, 16, 20].includes(idx);
        ctx.beginPath();
        ctx.arc(p.x, p.y, isTip ? 6 : 4, 0, 2 * Math.PI);
        ctx.fillStyle = isTip ? '#ffffff' : nodeCol;
        ctx.fill();
        ctx.strokeStyle = nodeCol;
        ctx.lineWidth = 1.5;
        ctx.stroke();
      });

      // 5. Draw Dynamic Corner Reticle around hand
      const xs = ptsPx.map(p => p.x);
      const ys = ptsPx.map(p => p.y);
      const bx1 = Math.max(0, Math.min(...xs) - 20);
      const by1 = Math.max(0, Math.min(...ys) - 20);
      const bx2 = Math.min(w - 1, Math.max(...xs) + 20);
      const by2 = Math.min(h - 1, Math.max(...ys) + 20);
      const cLen = Math.min(24, Math.floor((bx2 - bx1) / 4));

      ctx.lineWidth = 2.5;
      ctx.strokeStyle = nodeCol;

      // Top-left
      ctx.beginPath(); ctx.moveTo(bx1, by1 + cLen); ctx.lineTo(bx1, by1); ctx.lineTo(bx1 + cLen, by1); ctx.stroke();
      // Top-right
      ctx.beginPath(); ctx.moveTo(bx2 - cLen, by1); ctx.lineTo(bx2, by1); ctx.lineTo(bx2, by1 + cLen); ctx.stroke();
      // Bottom-left
      ctx.beginPath(); ctx.moveTo(bx1, by2 - cLen); ctx.lineTo(bx1, by2); ctx.lineTo(bx1 + cLen, by2); ctx.stroke();
      // Bottom-right
      ctx.beginPath(); ctx.moveTo(bx2 - cLen, by2); ctx.lineTo(bx2, by2); ctx.lineTo(bx2, by2 - cLen); ctx.stroke();

      ctx.restore(); // Restore unmirrored transform for text

      // 6. Holographic Badge above hand (Unmirrored text)
      const mirroredBx1 = w - bx2; // Convert back to unmirrored screen space
      const badgeY = Math.max(30, by1 - 12);
      ctx.font = 'bold 15px "JetBrains Mono", monospace';
      ctx.fillStyle = isVerified ? '#00ffa3' : '#00e1ff';
      ctx.shadowColor = 'rgba(0,0,0,0.8)';
      ctx.shadowBlur = 4;
      ctx.fillText(`${pred.output_text} (${(pred.confidence * 100).toFixed(1)}%)`, Math.max(10, mirroredBx1), badgeY);
      ctx.shadowBlur = 0;

      // Update telemetry UI
      pred.fps = clientFps;
      pred.stability_score = stabRes.stabilityScore;
      pred.is_locked = stabRes.isLocked;
      updateUI(pred);

    } else {
      ctx.restore();
      // Standby guide overlay
      renderStandbyGuide(w, h, "ARAHKAN TANGAN KE DEPAN KAMERA");
      updateUI({
        verified: false,
        output_text: "Menunggu Tangan...",
        confidence: 0.0,
        threshold: 0.88,
        icon: "👋",
        candidate_label: "Menunggu Tangan",
        status: "STANDBY_NO_HAND",
        stability_score: 0.0,
        is_locked: false,
        fps: clientFps,
        practice_eval: null
      });
    }
  }

  function renderStandbyGuide(w, h, msg) {
    if (!ctx) return;
    const cx = Math.floor(w / 2);
    const cy = Math.floor(h / 2);
    const boxW = Math.floor(w * 0.40);
    const boxH = Math.floor(h * 0.55);
    const x1 = cx - Math.floor(boxW / 2);
    const y1 = cy - Math.floor(boxH / 2);
    const x2 = cx + Math.floor(boxW / 2);
    const y2 = cy + Math.floor(boxH / 2);

    ctx.strokeStyle = '#00e1ff';
    ctx.lineWidth = 1.5;
    ctx.setLineDash([8, 6]);
    ctx.strokeRect(x1, y1, boxW, boxH);
    ctx.setLineDash([]);

    ctx.font = '600 13px "JetBrains Mono", monospace';
    ctx.fillStyle = '#00e1ff';
    ctx.textAlign = 'center';
    ctx.fillText(msg, cx, cy);
    ctx.textAlign = 'start';
  }

  // Toggle Camera Source / Switch Front-Back Camera
  if (toggleStreamModeBtn) {
    toggleStreamModeBtn.addEventListener('click', () => {
      if (!isClientCameraActive) {
        startClientCamera();
      } else {
        // Toggle front/back camera on mobile
        currentFacingMode = currentFacingMode === 'user' ? 'environment' : 'user';
        startClientCamera();
      }
    });
  }

  // Stream Fallback: If MJPEG video fails or not available, seamlessly launch in-browser camera
  window.handleStreamFallback = function(imgEl) {
    if (imgEl) imgEl.style.display = 'none';
    console.log('[Stream Fallback] Python MJPEG unavailable. Launching Client-Side Browser Camera...');
    startClientCamera();
  };

  // If deployed to Netlify (hostname does not match localhost), start client camera immediately
  const isNetlifyOrStatic = window.location.hostname !== 'localhost' && window.location.hostname !== '127.0.0.1';
  if (isNetlifyOrStatic) {
    console.log('[Netlify Detected] Launching 100% In-Browser MediaPipe Camera...');
    startClientCamera();
  }

  // =============================================================
  // UI Tabs, Practice Catalog & Telemetry
  // =============================================================

  // Tab Switching
  tabLiveDetection.addEventListener('click', () => {
    tabLiveDetection.classList.add('active');
    tabPracticeTraining.classList.remove('active');
    liveDetectionPanel.style.display = 'block';
    practiceTrainingPanel.style.display = 'none';
  });

  tabPracticeTraining.addEventListener('click', () => {
    tabPracticeTraining.classList.add('active');
    tabLiveDetection.classList.remove('active');
    practiceTrainingPanel.style.display = 'block';
    liveDetectionPanel.style.display = 'none';
  });

  function populatePracticeSelect() {
    if (!practiceTargetSelect) return;
    practiceTargetSelect.innerHTML = '';
    practiceCatalog.forEach((item, idx) => {
      const opt = document.createElement('option');
      opt.value = item.id;
      opt.textContent = `${item.icon} ${item.label}`;
      if (item.id === 'i_love_you') {
        opt.selected = true;
        currentPracticeIndex = idx;
      }
      practiceTargetSelect.appendChild(opt);
    });

    if (practiceCatalog.length > 0) {
      updatePracticeCard(practiceCatalog[currentPracticeIndex]);
    }
  }

  // Load Practice Catalog (Instantly from Classifier + Fallback Fetch)
  function loadPracticeCatalog() {
    if (classifier && classifier.CLASSES) {
      practiceCatalog = classifier.CLASSES;
      populatePracticeSelect();
    }

    fetch('/api/practice/catalog')
      .then(res => res.ok ? res.json() : null)
      .then(data => {
        if (data && data.catalog && data.catalog.length > 0) {
          practiceCatalog = data.catalog;
          populatePracticeSelect();
        }
      })
      .catch(() => {});
  }

  function updatePracticeCard(item) {
    if (!item) return;
    practiceTargetIcon.textContent = item.icon;
    practiceTargetTitle.textContent = item.label;
    practiceTargetDesc.textContent = item.instruction || 'Bentuk posisi jari sesuai instruksi.';
    practiceTypeBadge.textContent = (item.type || 'GENERAL').toUpperCase();
    hasPassedCurrentSign = false;
    practiceSuccessBanner.style.display = 'none';
    holdProgress.style.width = '0%';
    holdText.textContent = 'Tahan posisi selama 1.4 detik';

    fetch('/api/practice/select', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ target_id: item.id })
    }).catch(() => {});
  }

  practiceTargetSelect.addEventListener('change', (e) => {
    const foundIdx = practiceCatalog.findIndex(c => c.id === e.target.value);
    if (foundIdx !== -1) {
      currentPracticeIndex = foundIdx;
      updatePracticeCard(practiceCatalog[currentPracticeIndex]);
    }
  });

  btnNextPracticeSign.addEventListener('click', () => {
    if (practiceCatalog.length === 0) return;
    currentPracticeIndex = (currentPracticeIndex + 1) % practiceCatalog.length;
    practiceTargetSelect.value = practiceCatalog[currentPracticeIndex].id;
    updatePracticeCard(practiceCatalog[currentPracticeIndex]);
  });

  chips.forEach(chip => {
    chip.style.cursor = 'pointer';
    chip.addEventListener('click', () => {
      const chipId = chip.getAttribute('data-id');
      const foundIdx = practiceCatalog.findIndex(c => c.id === chipId);
      if (foundIdx !== -1) {
        currentPracticeIndex = foundIdx;
        practiceTargetSelect.value = practiceCatalog[currentPracticeIndex].id;
        updatePracticeCard(practiceCatalog[currentPracticeIndex]);
        tabPracticeTraining.click();
      }
    });
  });

  // Telemetry Polling Loop (Active only when using Python MJPEG backend)
  let isPolling = false;
  async function pollTelemetry() {
    if (isClientCameraActive) {
      return; // Handled directly in client loop
    }
    if (isPolling) {
      setTimeout(pollTelemetry, 60);
      return;
    }
    isPolling = true;
    try {
      const resp = await fetch('/api/telemetry');
      if (resp.ok) {
        const data = await resp.json();
        requestAnimationFrame(() => updateUI(data));
      } else {
        // If server gives non-200, fallback to client camera
        if (!isClientCameraActive) startClientCamera();
      }
    } catch (err) {
      // If server unreachable, fallback to client camera
      if (!isClientCameraActive) startClientCamera();
    } finally {
      isPolling = false;
      if (!isClientCameraActive) {
        setTimeout(pollTelemetry, 75);
      }
    }
  }

  // Update UI with incoming telemetry & practice feedback
  function updateUI(data) {
    if (!data) return;

    const conf = Math.max(0, Math.min(1, data.confidence || 0));
    const confPercent = (conf * 100).toFixed(1);
    const isVerified = data.verified === true;
    const isLocked = data.is_locked === true;
    const stability = Math.max(0, Math.min(1, data.stability_score || 0));

    // 1. Confidence Meter & Number
    confidenceNum.textContent = `${confPercent}%`;
    meterFill.style.width = `${Math.min(100, conf * 100)}%`;

    // 2. Anti-Gravity Floating Reticle
    if (isLocked) {
      reticle.classList.add('locked');
      reticleLabel.textContent = 'ANTI-GRAVITY FLOATING ZONE [LOCKED]';
      spatialLockBadge.textContent = 'LOCKED';
      spatialLockBadge.style.color = 'var(--neon-green)';
      spatialLockBadge.style.borderColor = 'var(--neon-green)';
      headerLockText.textContent = 'LOCKED (100%)';
      headerLockText.style.color = 'var(--neon-green)';
    } else {
      reticle.classList.remove('locked');
      reticleLabel.textContent = 'CALIBRATING SPATIAL LOCK...';
      spatialLockBadge.textContent = 'CALIBRATING';
      spatialLockBadge.style.color = 'var(--neon-amber)';
      spatialLockBadge.style.borderColor = 'var(--neon-amber)';
      headerLockText.textContent = 'CALIBRATING';
      headerLockText.style.color = 'var(--neon-amber)';
    }

    // 3. Strict Zero-Error Translation Card (Live Mode)
    if (isVerified) {
      translationCard.classList.add('verified');
      translationIcon.textContent = data.icon || '✨';
      translationText.textContent = data.output_text;
      translationSubtext.textContent = `Strict Threshold Passed (${confPercent}% > 88.0%) & Spatial Locked.`;
      
      gateBadge.textContent = 'VERIFIED ACCURATE';
      gateBadge.style.color = 'var(--neon-green)';
      gateBadge.style.borderColor = 'var(--neon-green)';

      diagGateState.textContent = 'VERIFIED (>88%)';
      diagGateState.style.color = 'var(--neon-green)';

      if (lastVerifiedId !== data.output_text) {
        playVerificationChime();
        speakSign(data.output_text);
        lastVerifiedId = data.output_text;
      }
    } else {
      translationCard.classList.remove('verified');
      translationIcon.textContent = '⏳';
      translationText.textContent = 'Signal Unclear / Analyzing';
      translationSubtext.textContent = `Confidence: ${confPercent}%. Zero-Error Protocol active: strictly zero guessing when <= 88.0%.`;

      gateBadge.textContent = 'ZERO-ERROR GATE (<=88%)';
      gateBadge.style.color = 'var(--neon-amber)';
      gateBadge.style.borderColor = 'var(--neon-amber)';

      diagGateState.textContent = 'GATED (<=88%)';
      diagGateState.style.color = 'var(--neon-amber)';
      lastVerifiedId = null;
    }

    // 4. Stability Index
    const stabPercent = (stability * 100).toFixed(1);
    diagStabilityVal.textContent = `${stabPercent}%`;
    stabilityBar.style.width = `${stabPercent}%`;

    // 5. FPS Counter
    if (data.fps) {
      fpsCounter.textContent = `${data.fps} FPS`;
    }

    // 6. Highlight active sign chip
    chips.forEach(chip => {
      if (isVerified && data.output_text.toLowerCase().includes(chip.textContent.slice(2).trim().toLowerCase())) {
        chip.classList.add('active');
      } else {
        chip.classList.remove('active');
      }
    });

    // 7. Practice Mode Real-Time Evaluation
    if (data.practice_eval) {
      renderPracticeFeedback(data.practice_eval, isLocked);
    }
  }

  // Render Practice Mode Training Analytics
  function renderPracticeFeedback(evalData, isLocked) {
    if (!evalData) return;

    const matchAcc = Math.max(0, Math.min(100, evalData.accuracy_percent || 0));
    practiceMatchPercent.textContent = `${matchAcc}%`;
    practiceMeterFill.style.width = `${matchAcc}%`;

    // Render 5-Finger Checklist
    if (evalData.finger_eval && fingerChecklistGrid) {
      fingerChecklistGrid.innerHTML = '';
      evalData.finger_eval.forEach(f => {
        const card = document.createElement('div');
        card.className = `finger-card ${f.is_correct ? 'correct' : 'incorrect'}`;
        card.innerHTML = `
          <div class="finger-name">${f.finger}</div>
          <span class="finger-status-badge">${f.is_correct ? 'SESUAI' : 'BELUM'}</span>
          <div class="finger-hint">${f.hint}</div>
        `;
        fingerChecklistGrid.appendChild(card);
      });
    }

    // AI Coach Guidance Tip
    if (evalData.feedback_tips && evalData.feedback_tips.length > 0) {
      coachFeedbackText.textContent = evalData.feedback_tips[0];
    } else {
      coachFeedbackText.textContent = 'Posisikan jari-jari Anda menghadap kamera...';
    }

    // Hold-to-Succeed Challenge Logic
    if (evalData.is_perfect && isLocked && !hasPassedCurrentSign) {
      if (!holdStartTime) {
        holdStartTime = performance.now();
      }
      const elapsed = performance.now() - holdStartTime;
      const progressRatio = Math.min(1.0, elapsed / HOLD_DURATION_MS);
      const remainingSec = ((HOLD_DURATION_MS - elapsed) / 1000).toFixed(1);

      holdProgress.style.width = `${progressRatio * 100}%`;
      holdText.textContent = `Tahan posisi: ${remainingSec}s lagi...`;

      if (progressRatio >= 1.0) {
        hasPassedCurrentSign = true;
        holdStartTime = null;
        holdProgress.style.width = '100%';
        holdText.textContent = 'SEMPURNA! TANTANGAN SELESAI 🎉';
        practiceSuccessBanner.style.display = 'flex';
        playTriumphChime();
      }
    } else if (!hasPassedCurrentSign) {
      holdStartTime = null;
      holdProgress.style.width = '0%';
      holdText.textContent = 'Tahan posisi selama 1.4 detik saat &ge;92% sesuai';
    }
  }

  // Initialize
  loadPracticeCatalog();
  if (!isNetlifyOrStatic) {
    pollTelemetry();
  }
});

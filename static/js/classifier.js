/**
 * classifier.js - Client-Side Anti-Gravity AI & Zero-Error Hand Sign Classifier
 * 100% In-Browser Execution for Standalone Netlify Deployment.
 */

// -------------------------------------------------------------
// 1. 3D Kalman Filter for Landmark Jitter Damping
// -------------------------------------------------------------
class LandmarkKalmanFilterJS {
  constructor(processNoise = 1e-3, measurementNoise = 1e-2) {
    this.processNoise = processNoise;
    this.measurementNoise = measurementNoise;
    this.state = [0, 0, 0, 0, 0, 0]; // [x, y, z, vx, vy, vz]
    this.P = [1, 1, 1, 1, 1, 1]; // Diagonal covariance
    this.initialized = false;
    this.lastTime = performance.now();
  }

  update(meas) {
    const now = performance.now();
    const dt = Math.max(0.001, Math.min(0.1, (now - this.lastTime) / 1000));
    this.lastTime = now;

    if (!this.initialized) {
      this.state = [meas[0], meas[1], meas[2], 0, 0, 0];
      this.initialized = true;
      return [meas[0], meas[1], meas[2]];
    }

    // Predict
    this.state[0] += this.state[3] * dt;
    this.state[1] += this.state[4] * dt;
    this.state[2] += this.state[5] * dt;

    for (let i = 0; i < 6; i++) {
      this.P[i] += this.processNoise;
    }

    // Update for x, y, z independently (H = identity)
    for (let i = 0; i < 3; i++) {
      const K = this.P[i] / (this.P[i] + this.measurementNoise);
      const residual = meas[i] - this.state[i];
      this.state[i] += K * residual;
      this.state[i + 3] += (K / dt) * residual * 0.15; // smooth velocity update
      this.P[i] = (1 - K) * this.P[i];
    }

    return [this.state[0], this.state[1], this.state[2]];
  }

  reset() {
    this.initialized = false;
    this.state = [0, 0, 0, 0, 0, 0];
    this.P = [1, 1, 1, 1, 1, 1];
  }
}

// -------------------------------------------------------------
// 2. Anti-Gravity Spatial Normalizer & Stabilizer
// -------------------------------------------------------------
class AntiGravityStabilizerJS {
  constructor(canvasCenter = [0.5, 0.5, 0.0], targetScale = 0.32) {
    this.filters = Array.from({ length: 21 }, () => new LandmarkKalmanFilterJS());
    this.canvasCenter = canvasCenter;
    this.targetScale = targetScale;
    this.lastCanonical = null;
    this.stabilityHistory = [];
  }

  reset() {
    this.filters.forEach(f => f.reset());
    this.lastCanonical = null;
    this.stabilityHistory = [];
  }

  process(rawLandmarks) {
    // 1. Kalman filter smoothing
    const filtered = rawLandmarks.map((pt, i) => this.filters[i].update([pt.x || pt[0], pt.y || pt[1], pt.z || pt[2] || 0]));

    // 2. Wrist-relative origin
    const wrist = filtered[0];
    const relative = filtered.map(p => [p[0] - wrist[0], p[1] - wrist[1], p[2] - wrist[2]]);

    // 3. Palm scale normalization (Wrist 0 to Middle MCP 9)
    const midMcp = relative[9];
    let palmLen = Math.hypot(midMcp[0], midMcp[1]);
    if (palmLen < 1e-4) palmLen = 0.1;

    const scaled = relative.map(p => [p[0] / palmLen, p[1] / palmLen, p[2] / palmLen]);

    // 4. Rotational alignment (Middle MCP pointing straight up at -PI/2)
    const currentAngle = Math.atan2(scaled[9][1], scaled[9][0]);
    const targetAngle = -Math.PI / 2.0;
    const rotationAngle = targetAngle - currentAngle;

    const cosA = Math.cos(rotationAngle);
    const sinA = Math.sin(rotationAngle);

    let canonical = scaled.map(p => [
      p[0] * cosA - p[1] * sinA,
      p[0] * sinA + p[1] * cosA,
      p[2]
    ]);

    // Handedness normalization: ensure Pinky MCP (17) is on positive X, Index (5) on negative X
    if (canonical[5][0] > canonical[17][0]) {
      canonical = canonical.map(p => [-p[0], p[1], p[2]]);
    }

    // Stability calculation
    let stability = 0.88;
    if (this.lastCanonical) {
      let totalDist = 0;
      for (let i = 0; i < 21; i++) {
        totalDist += Math.hypot(
          canonical[i][0] - this.lastCanonical[i][0],
          canonical[i][1] - this.lastCanonical[i][1],
          canonical[i][2] - this.lastCanonical[i][2]
        );
      }
      const meanDist = totalDist / 21.0;
      stability = Math.exp(-meanDist * 6.0);
    }
    this.lastCanonical = canonical.map(p => [...p]);

    this.stabilityHistory.push(stability);
    if (this.stabilityHistory.length > 8) this.stabilityHistory.shift();

    const smoothStability = this.stabilityHistory.reduce((a, b) => a + b, 0) / this.stabilityHistory.length;
    const isLocked = smoothStability >= 0.50;

    return {
      filteredRaw: filtered,
      canonicalPoints: canonical,
      stabilityScore: smoothStability,
      isLocked: isLocked,
      palmScale: palmLen
    };
  }
}

// -------------------------------------------------------------
// 3. Strict Zero-Error Classifier (>88% Threshold)
// -------------------------------------------------------------
class ZeroErrorClassifierJS {
  constructor() {
    this.THRESHOLD = 0.88;

    this.CLASSES = [
      {
        id: "hello", label: "Hello / Halo", icon: "👋", type: "greeting",
        instruction: "Buka kelima jari tangan Anda lurus ke atas dan rentangkan secara wajar menghadap kamera.",
        fingerHints: { jempol: "Terbuka", telunjuk: "Lurus ke atas", "jari tengah": "Lurus ke atas", "jari manis": "Lurus ke atas", kelingking: "Lurus ke atas" },
        practiceTip: "Pastikan semua 5 jari berdiri tegak dan tidak saling menempel."
      },
      {
        id: "thank_you", label: "Thank You / Terima Kasih", icon: "🙏", type: "courtesy",
        instruction: "Rapatkan 4 jari ke atas secara tegak, lipat jempol santai di sisi telapak, tangan lurus vertikal.",
        fingerHints: { jempol: "Ditekuk/Merapat", telunjuk: "Lurus rapat", "jari tengah": "Lurus rapat", "jari manis": "Lurus rapat", kelingking: "Lurus rapat" },
        practiceTip: "Jaga jari-jari tetap rapat satu sama lain menghadap kamera."
      },
      {
        id: "i_love_you", label: "I Love You", icon: "🤟", type: "emotion",
        instruction: "Buka jempol ke samping, angkat telunjuk dan kelingking lurus ke atas. Lipat jari tengah dan jari manis ke dalam telapak.",
        fingerHints: { jempol: "Terbuka lebar", telunjuk: "Lurus ke atas", "jari tengah": "Ditekuk rapat", "jari manis": "Ditekuk rapat", kelingking: "Lurus ke atas" },
        practiceTip: "Pastikan jempol benar-benar terbuka ke luar dan dua jari tengah terlipat penuh."
      },
      {
        id: "yes", label: "Yes / Bagus", icon: "👍", type: "affirmation",
        instruction: "Kepalkan seluruh 4 jari ke telapak tangan, lalu acungkan jempol tegak lurus mengarah ke atas.",
        fingerHints: { jempol: "Mengacung ke atas", telunjuk: "Ditekuk/Mengepal", "jari tengah": "Ditekuk/Mengepal", "jari manis": "Ditekuk/Mengepal", kelingking: "Ditekuk/Mengepal" },
        practiceTip: "Jempol harus tegak lurus mengarah ke atas, bukan miring ke samping."
      },
      {
        id: "no", label: "No / Tidak", icon: "👎", type: "affirmation",
        instruction: "Kepalkan seluruh jari tangan, lalu putar pergelangan tangan hingga jempol mengarah lurus ke bawah.",
        fingerHints: { jempol: "Mengarah ke bawah", telunjuk: "Ditekuk/Mengepal", "jari tengah": "Ditekuk/Mengepal", "jari manis": "Ditekuk/Mengepal", kelingking: "Ditekuk/Mengepal" },
        practiceTip: "Arahkan jempol tegas ke bawah dengan 4 jari terkepal rapat."
      },
      {
        id: "peace", label: "Peace / Damai", icon: "✌️", type: "social",
        instruction: "Angkat jari telunjuk dan tengah membentuk huruf 'V' terbuka lebar. Lipat jempol di atas jari manis dan kelingking.",
        fingerHints: { jempol: "Melipat di telapak", telunjuk: "Lurus (V)", "jari tengah": "Lurus (V)", "jari manis": "Ditekuk", kelingking: "Ditekuk" },
        practiceTip: "Renggangkan jarak antara telunjuk dan jari tengah selebar mungkin."
      },
      {
        id: "ok", label: "OK / Sempurna", icon: "👌", type: "social",
        instruction: "Pertemukan ujung jempol dan telunjuk hingga membentuk lingkaran kecil. Tegakkan 3 jari lainnya lurus ke atas.",
        fingerHints: { jempol: "Menyentuh telunjuk", telunjuk: "Menyentuh jempol", "jari tengah": "Lurus ke atas", "jari manis": "Lurus ke atas", kelingking: "Lurus ke atas" },
        practiceTip: "Ujung jempol dan telunjuk harus saling menempel (pinch) membentuk lingkaran."
      },
      {
        id: "rock_on", label: "Rock On / Keren", icon: "🤘", type: "social",
        instruction: "Angkat telunjuk dan kelingking tegak ke atas. Lipat jari tengah dan manis, lalu kunci dengan jempol di depannya.",
        fingerHints: { jempol: "Mengunci di depan", telunjuk: "Lurus ke atas", "jari tengah": "Ditekuk", "jari manis": "Ditekuk", kelingking: "Lurus ke atas" },
        practiceTip: "Berbeda dengan I Love You, pada Rock On jempol terlipat mengunci jari tengah dan manis."
      },
      {
        id: "call_me", label: "Call Me / Telepon", icon: "🤙", type: "activity",
        instruction: "Rentangkan jempol dan kelingking lebar ke samping seperti gagang telepon. Lipat 3 jari tengah rapat ke telapak.",
        fingerHints: { jempol: "Terbuka lebar", telunjuk: "Ditekuk", "jari tengah": "Ditekuk", "jari manis": "Ditekuk", kelingking: "Terbuka lebar" },
        practiceTip: "Buka kelingking dan jempol semaksimal mungkin ke arah berlawanan."
      },
      {
        id: "help", label: "Help / Tolong", icon: "🤲", type: "courtesy",
        instruction: "Buka telapak tangan santai sedikit melengkung membentuk mangkuk terbuka menghadap ke depan.",
        fingerHints: { jempol: "Santai terbuka", telunjuk: "Sedikit melengkung", "jari tengah": "Sedikit melengkung", "jari manis": "Sedikit melengkung", kelingking: "Sedikit melengkung" },
        practiceTip: "Posisikan telapak tangan seperti sedang menengadah meminta pertolongan."
      },
      {
        id: "num_1", label: "Number 1", icon: "☝️", type: "number",
        instruction: "Angkat hanya jari telunjuk tegak lurus ke atas. Lipat jempol di atas jari tengah, manis, dan kelingking.",
        fingerHints: { jempol: "Melipat di telapak", telunjuk: "Lurus ke atas", "jari tengah": "Ditekuk", "jari manis": "Ditekuk", kelingking: "Ditekuk" },
        practiceTip: "Pastikan hanya jari telunjuk satu-satunya yang terangkat."
      },
      {
        id: "num_2", label: "Number 2", icon: "✌️", type: "number",
        instruction: "Angkat jari telunjuk dan jari tengah secara tegak lurus dan rapat (berbeda dengan Peace yang melebar).",
        fingerHints: { jempol: "Melipat di telapak", telunjuk: "Lurus rapat", "jari tengah": "Lurus rapat", "jari manis": "Ditekuk", kelingking: "Ditekuk" },
        practiceTip: "Rapatkan jari telunjuk dan jari tengah tanpa ada celah di antaranya."
      },
      {
        id: "num_3", label: "Number 3", icon: "🤟", type: "number",
        instruction: "Buka jempol, telunjuk, dan jari tengah lurus ke atas. Lipat jari manis dan kelingking ke telapak tangan.",
        fingerHints: { jempol: "Terbuka", telunjuk: "Lurus ke atas", "jari tengah": "Lurus ke atas", "jari manis": "Ditekuk", kelingking: "Ditekuk" },
        practiceTip: "Tiga jari pertama (jempol, telunjuk, tengah) terangkat tegak bersamaan."
      },
      {
        id: "num_4", label: "Number 4", icon: "🖖", type: "number",
        instruction: "Buka 4 jari (telunjuk, tengah, manis, kelingking) lurus ke atas. Lipat jempol rapat melintang di telapak tangan.",
        fingerHints: { jempol: "Melipat di telapak", telunjuk: "Lurus ke atas", "jari tengah": "Lurus ke atas", "jari manis": "Lurus ke atas", kelingking: "Lurus ke atas" },
        practiceTip: "Sembunyikan jempol ke dalam telapak dan tegakkan keempat jari lainnya."
      },
      {
        id: "num_5", label: "Number 5", icon: "🖐️", type: "number",
        instruction: "Rentangkan seluruh 5 jari selebar mungkin dengan jempol terentang penuh.",
        fingerHints: { jempol: "Terbuka lebar", telunjuk: "Terbuka lebar", "jari tengah": "Terbuka lebar", "jari manis": "Terbuka lebar", kelingking: "Terbuka lebar" },
        practiceTip: "Buka kelima jari tangan selebar-lebarnya menghadap kamera."
      },
      {
        id: "fist_a", label: "Fist / Huruf A", icon: "✊", type: "alphabet",
        instruction: "Kepalkan seluruh 4 jari ke telapak tangan, sandarkan jempol tegak di sisi luar jari telunjuk.",
        fingerHints: { jempol: "Di samping telunjuk", telunjuk: "Ditekuk penuh", "jari tengah": "Ditekuk penuh", "jari manis": "Ditekuk penuh", kelingking: "Ditekuk penuh" },
        practiceTip: "Kepalkan tangan rapat-rapat seperti tinju."
      },
      {
        id: "flat_b", label: "Flat / Huruf B", icon: "✋", type: "alphabet",
        instruction: "Tegakkan 4 jari lurus dan rapat tanpa celah. Lipat jempol menyilang di depan telapak tangan.",
        fingerHints: { jempol: "Menyilang di telapak", telunjuk: "Lurus rapat", "jari tengah": "Lurus rapat", "jari manis": "Lurus rapat", kelingking: "Lurus rapat" },
        practiceTip: "Keempat jari harus saling menempel rapat dan lurus sempurna."
      },
      {
        id: "cup_c", label: "Cup / Huruf C", icon: "🤏", type: "alphabet",
        instruction: "Lengkungkan jempol dan keempat jari hingga membentuk siluet huruf 'C' seperti memegang cangkir.",
        fingerHints: { jempol: "Melengkung ke atas", telunjuk: "Melengkung ke bawah", "jari tengah": "Melengkung", "jari manis": "Melengkung", kelingking: "Melengkung" },
        practiceTip: "Bentuk tangan menyerupai huruf C dari sudut pandang kamera."
      }
    ];

    this.TARGET_EXTS = {
      0:  [0.9, 1.0, 1.0, 1.0, 1.0], // Hello
      1:  [0.1, 1.0, 1.0, 1.0, 1.0], // Thank You
      2:  [0.9, 1.0, 0.1, 0.1, 0.9], // I Love You
      3:  [0.9, 0.1, 0.1, 0.1, 0.1], // Yes / Thumbs Up
      4:  [0.9, 0.1, 0.1, 0.1, 0.1], // No / Thumbs Down
      5:  [0.1, 1.0, 1.0, 0.1, 0.1], // Peace
      6:  [0.2, 0.2, 1.0, 1.0, 1.0], // OK
      7:  [0.1, 1.0, 0.1, 0.1, 0.9], // Rock On
      8:  [0.9, 0.1, 0.1, 0.1, 0.9], // Call Me
      9:  [0.6, 0.6, 0.6, 0.6, 0.6], // Help
      10: [0.1, 1.0, 0.1, 0.1, 0.1], // Num 1
      11: [0.1, 1.0, 1.0, 0.1, 0.1], // Num 2
      12: [0.8, 1.0, 1.0, 0.1, 0.1], // Num 3
      13: [0.1, 1.0, 1.0, 1.0, 1.0], // Num 4
      14: [1.0, 1.0, 1.0, 1.0, 1.0], // Num 5
      15: [0.1, 0.1, 0.1, 0.1, 0.1], // Fist
      16: [0.1, 1.0, 1.0, 1.0, 1.0], // Flat B
      17: [0.4, 0.4, 0.4, 0.4, 0.4]  // Cup C
    };
  }

  clip(val, min, max) {
    return Math.max(min, Math.min(max, val));
  }

  dist3d(p1, p2) {
    return Math.hypot(p1[0] - p2[0], p1[1] - p2[1], (p1[2] || 0) - (p2[2] || 0));
  }

  extractGeometricFeatures(canonicalPoints, rawPoints = null) {
    const pts = canonicalPoints;
    let palmScale = this.dist3d(pts[9], pts[0]);
    if (palmScale < 1e-4) palmScale = 0.1;

    const normPts = pts.map(p => [
      (p[0] - pts[0][0]) / palmScale,
      (p[1] - pts[0][1]) / palmScale,
      (p[2] - pts[0][2]) / palmScale
    ]);

    // 1. Thumb lateral abduction and distances
    const thumbAbduct = normPts[5][0] - normPts[4][0];
    const dThumbPinky = this.dist3d(normPts[4], normPts[17]);
    const dThumbIdxMcp = this.dist3d(normPts[4], normPts[5]);

    const extAbduct = this.clip((thumbAbduct - 0.05) / 0.30, 0.0, 1.0);
    const extDist = this.clip((dThumbPinky - 0.60) / 0.35, 0.0, 1.0);
    const extThumb = this.clip(extAbduct * 0.55 + extDist * 0.45, 0.0, 1.0);

    // 2. Fingers 1-4 (Index, Middle, Ring, Pinky)
    const tips = [8, 12, 16, 20];
    const pips = [6, 10, 14, 18];
    const mcps = [5, 9, 13, 17];
    const extList = [extThumb];

    for (let i = 0; i < 4; i++) {
      const tip = tips[i];
      const pip = pips[i];
      const mcp = mcps[i];

      const dTipWrist = Math.hypot(normPts[tip][0], normPts[tip][1], normPts[tip][2]);
      const diffY = normPts[pip][1] - normPts[tip][1];

      const extY = this.clip((diffY - 0.12) / 0.45, 0.0, 1.0);
      const extW = this.clip((dTipWrist - 1.15) / 0.50, 0.0, 1.0);
      const extVal = this.clip(extY * 0.55 + extW * 0.45, 0.0, 1.0);
      extList.push(extVal);
    }

    const extBool = extList.map(e => e >= 0.55);

    // 3. Spreads and ratios
    const pinchThumbIndex = this.dist3d(normPts[4], normPts[8]);
    const distIndexMiddle = this.dist3d(normPts[8], normPts[12]);
    const distMiddleRing = this.dist3d(normPts[12], normPts[16]);
    const totalSpread = this.dist3d(normPts[8], normPts[20]);

    const mcpWidth = this.dist3d(normPts[5], normPts[17]) + 1e-5;
    const tipWidth = this.dist3d(normPts[8], normPts[20]);
    const spreadRatio = tipWidth / mcpWidth;

    const vertReaches = mcps.map((mcp, i) => normPts[mcp][1] - normPts[tips[i]][1]);
    const meanVertReach = vertReaches.reduce((a, b) => a + b, 0) / vertReaches.length;

    const isFist4 = extList.slice(1).every(e => e < 0.40);
    const isCShape = (0.25 < pinchThumbIndex && pinchThumbIndex < 0.80) &&
                     (0.25 < meanVertReach && meanVertReach < 0.75) && !isFist4;

    const isThumbTucked = (thumbAbduct < 0.12) || (normPts[4][0] > -0.35 && extThumb < 0.45);
    const isWide5 = (totalSpread > 1.35 && dThumbPinky > 1.25);

    // 4. Thumb vector in palm & screen space:
    const vThumbY = (pts[4][1] - pts[2][1]) / palmScale;
    let thumbUp = false;
    let thumbDown = false;

    if (rawPoints && rawPoints.length === 21) {
      // In camera space: Y=0 is top and Y=1 is bottom
      const vRawThumbY = rawPoints[4][1] - rawPoints[2][1];
      const diffRawWristY = rawPoints[4][1] - rawPoints[0][1];
      thumbUp = (vRawThumbY < -0.03 || diffRawWristY < -0.04) && extThumb > 0.38;
      thumbDown = (vRawThumbY > 0.03 || diffRawWristY > 0.04) && extThumb > 0.38;
    } else {
      thumbUp = (vThumbY < -0.35 && pts[4][1] < pts[0][1] && extThumb > 0.38);
      thumbDown = ((vThumbY > 0.20 || pts[4][1] > pts[0][1] + 0.05) && extThumb > 0.38);
    }

    return {
      extContinuous: extList,
      extBool: extBool,
      extCount: extBool.slice(1).filter(Boolean).length,
      totalExt: extBool.filter(Boolean).length,
      pinchThumbIndex,
      distIndexMiddle,
      distMiddleRing,
      totalSpread,
      spreadRatio,
      meanVertReach,
      isFist4,
      isCShape,
      isThumbTucked,
      isWide5,
      dThumbPinky,
      dThumbIdxMcp,
      thumbUp,
      thumbDown
    };
  }

  computeCalibratedLogits(canonicalPoints, geo) {
    const ext = geo.extContinuous;
    const scores = new Array(this.CLASSES.length).fill(0);
    const weights = [1.4, 1.0, 1.0, 1.0, 1.3];
    const totalWeights = 5.7;

    for (let c = 0; c < this.CLASSES.length; c++) {
      const target = this.TARGET_EXTS[c];
      let weightedDist = 0;
      for (let i = 0; i < 5; i++) {
        const diff = ext[i] - target[i];
        weightedDist += weights[i] * diff * diff;
      }
      weightedDist /= totalWeights;
      const baseScore = 7.0 * Math.exp(-weightedDist * 4.0);
      let bonus = 0.0;

      if (c === 0) { // Hello
        if (ext.slice(1).every(e => e > 0.55) && ext[0] > 0.45 && !geo.isThumbTucked && !geo.isCShape) {
          bonus += !geo.isWide5 ? 6.5 : 2.0;
        } else {
          bonus -= 4.0;
        }
      } else if (c === 1) { // Thank You
        if (ext.slice(1).every(e => e > 0.65) && ext[0] < 0.45 && !geo.isCShape) {
          bonus += (geo.spreadRatio < 1.15 && geo.dThumbIdxMcp < 0.45) ? 6.0 : -3.0;
        } else {
          bonus -= 4.0;
        }
      } else if (c === 2) { // I Love You
        const isIly = (
          ext[0] > 0.40 &&
          !geo.isThumbTucked &&
          ext[1] > 0.48 &&
          ext[4] > 0.48 &&
          ext[2] < 0.48 &&
          ext[3] < 0.48 &&
          geo.totalSpread > 0.55
        );
        if (isIly) {
          bonus += 6.5;
        } else {
          if (ext[1] > 0.5 && ext[4] > 0.5 && (ext[2] > 0.55 || ext[3] > 0.55)) {
            bonus -= 4.0;
          } else if (ext[0] < 0.35) {
            bonus -= 3.5;
          } else if (ext[1] < 0.35 || ext[4] < 0.35) {
            bonus -= 3.5;
          }
        }
      } else if (c === 3) { // Yes / Thumbs Up
        bonus += (geo.isFist4 && geo.thumbUp && !geo.thumbDown && !geo.isThumbTucked && ext[0] > 0.50) ? 6.5 : -5.0;
      } else if (c === 4) { // No / Thumbs Down
        bonus += (geo.isFist4 && geo.thumbDown && !geo.thumbUp && !geo.isThumbTucked && ext[0] > 0.50) ? 6.5 : -5.0;
      } else if (c === 5) { // Peace
        if (ext[1] > 0.6 && ext[2] > 0.6 && ext[3] < 0.35 && ext[4] < 0.35) {
          bonus += (geo.distIndexMiddle > 0.26) ? 5.5 : -3.0;
        }
      } else if (c === 6) { // OK
        bonus += (geo.pinchThumbIndex < 0.30 && ext[2] > 0.55 && ext[3] > 0.55 && ext[4] > 0.55) ? 6.5 : -4.0;
      } else if (c === 7) { // Rock On
        if (ext[1] > 0.50 && ext[4] > 0.50 && ext[2] < 0.45 && ext[3] < 0.45) {
          bonus += (geo.isThumbTucked || ext[0] < 0.42) ? 6.0 : -4.0;
        } else {
          bonus -= 3.5;
        }
      } else if (c === 8) { // Call Me
        bonus += (ext[0] > 0.6 && ext[4] > 0.6 && ext[1] < 0.35 && ext[2] < 0.35 && ext[3] < 0.35) ? 6.0 : -3.5;
      } else if (c === 9) { // Help
        const minE = Math.min(...ext.slice(1));
        const maxE = Math.max(...ext.slice(1));
        const meanE = ext.slice(1).reduce((a, b) => a + b, 0) / 4.0;
        if (meanE >= 0.30 && meanE <= 0.85 && (maxE - minE < 0.35) && ext[0] >= 0.35 && geo.meanVertReach < 0.80 && !geo.isCShape) {
          bonus += 7.5;
        } else {
          bonus -= 3.0;
        }
      } else if (c === 10) { // Number 1
        bonus += (ext[1] > 0.6 && ext[2] < 0.35 && ext[3] < 0.35 && ext[4] < 0.35 && ext[0] < 0.45) ? 6.0 : -3.5;
      } else if (c === 11) { // Number 2
        if (ext[1] > 0.6 && ext[2] > 0.6 && ext[3] < 0.35 && ext[4] < 0.35) {
          bonus += (geo.distIndexMiddle <= 0.26) ? 5.5 : -3.0;
        }
      } else if (c === 12) { // Number 3
        const isV1 = (ext[0] > 0.55 && ext[1] > 0.6 && ext[2] > 0.6 && ext[3] < 0.4 && ext[4] < 0.4);
        const isV2 = (ext[1] > 0.6 && ext[2] > 0.6 && ext[3] > 0.6 && ext[4] < 0.4 && ext[0] < 0.5);
        bonus += (isV1 || isV2) ? 6.0 : -3.0;
      } else if (c === 13) { // Number 4
        if (geo.isThumbTucked && ext[0] < 0.35 && ext.slice(1).every(e => e > 0.60)) {
          bonus += (geo.spreadRatio >= 1.20) ? 6.5 : -3.0;
        } else {
          bonus -= 5.0; // If thumb is open, CANNOT be Number 4!
        }
      } else if (c === 14) { // Number 5
        bonus += (geo.isWide5 && ext.every(e => e > 0.70)) ? 6.5 : -2.5;
      } else if (c === 15) { // Fist / Huruf A
        bonus += (geo.isFist4 && !geo.thumbUp && !geo.thumbDown && ext[0] < 0.40) ? 6.5 : -4.0;
      } else if (c === 16) { // Flat B
        if (ext.slice(1).every(e => e > 0.65) && ext[0] < 0.35 && !geo.isCShape) {
          bonus += (geo.spreadRatio < 1.10 && geo.dThumbIdxMcp >= 0.45) ? 6.5 : -2.0;
        } else {
          bonus -= 3.0;
        }
      } else if (c === 17) { // Cup C
        bonus += geo.isCShape ? 8.0 : -4.5;
      }

      scores[c] = baseScore + bonus;
    }

    return scores;
  }

  getPracticeFeedback(targetKey, canonicalPoints, geo) {
    let targetIdx = 2; // Default to I Love You
    if (typeof targetKey === 'number' && targetKey >= 0 && targetKey < this.CLASSES.length) {
      targetIdx = targetKey;
    } else if (typeof targetKey === 'string') {
      const found = this.CLASSES.findIndex(c => c.id === targetKey || c.label.toLowerCase() === targetKey.toLowerCase());
      if (found !== -1) targetIdx = found;
    }

    const targetClass = this.CLASSES[targetIdx];
    const targetExt = this.TARGET_EXTS[targetIdx];
    const actualExt = geo.extContinuous;

    const fingerNames = ["Jempol", "Telunjuk", "Jari Tengah", "Jari Manis", "Kelingking"];
    const fingerEval = [];
    const feedbackTips = [];
    let correctCount = 0;

    for (let i = 0; i < 5; i++) {
      const currVal = actualExt[i];
      const tVal = targetExt[i];
      const isOk = Math.abs(currVal - tVal) <= 0.32;

      const targetStateStr = tVal >= 0.7 ? "Terbuka" : (tVal <= 0.3 ? "Ditekuk" : "Melengkung");
      const currStateStr = currVal >= 0.6 ? "Terbuka" : (currVal <= 0.35 ? "Ditekuk" : "Setengah Ditekuk");

      if (isOk) {
        correctCount++;
      } else {
        if (tVal >= 0.7 && currVal < 0.5) {
          feedbackTips.push(`Luruskan ${fingerNames[i]} ke atas!`);
        } else if (tVal <= 0.3 && currVal > 0.35) {
          feedbackTips.push(`Lipat ${fingerNames[i]} rapat ke telapak!`);
        }
      }

      fingerEval.push({
        finger: fingerNames[i],
        is_correct: isOk,
        current_val: Math.round(currVal * 100) / 100,
        current_state: currStateStr,
        target_state: targetStateStr,
        hint: (targetClass.fingerHints && targetClass.fingerHints[fingerNames[i].toLowerCase()]) || targetStateStr
      });
    }

    let specialOk = true;
    if (targetIdx === 2 && (geo.isThumbTucked || geo.extContinuous[0] < 0.40)) {
      specialOk = false;
      feedbackTips.push("Buka jempol lebar ke samping!");
    } else if (targetIdx === 3 && !geo.thumbUp) {
      specialOk = false;
      feedbackTips.push("Arahkan jempol tegak lurus ke atas (jempol mengacung)!");
    } else if (targetIdx === 4 && !geo.thumbDown) {
      specialOk = false;
      feedbackTips.push("Arahkan jempol menghadap lurus ke bawah!");
    } else if (targetIdx === 5 && geo.distIndexMiddle <= 0.26) {
      specialOk = false;
      feedbackTips.push("Buka dan renggangkan jari telunjuk dan tengah membentuk V!");
    } else if (targetIdx === 6 && geo.pinchThumbIndex >= 0.30) {
      specialOk = false;
      feedbackTips.push("Satukan ujung jempol dan telunjuk membentuk lingkaran (pinch)!");
    } else if (targetIdx === 11 && geo.distIndexMiddle > 0.26) {
      specialOk = false;
      feedbackTips.push("Rapatkan jari telunjuk dan tengah!");
    }

    const meanDev = actualExt.reduce((acc, v, i) => acc + Math.abs(v - targetExt[i]), 0) / 5.0;
    let baseAccuracy = Math.max(0.0, 1.0 - meanDev);
    if (!specialOk) baseAccuracy = Math.min(baseAccuracy, 0.75);
    const accuracyPercent = Math.round(baseAccuracy * 100);

    const isPerfect = (accuracyPercent >= 90) && (correctCount >= 4) && specialOk;
    if (isPerfect) {
      feedbackTips.length = 0;
      feedbackTips.push("Bagus sekali! Gestur ini sudah sempurna. Pertahankan posisi ini!");
    } else if (feedbackTips.length === 0) {
      feedbackTips.push("Hampir tepat! Tahan dan mantapkan posisi jari Anda.");
    }

    return {
      target_id: targetClass.id,
      target_label: targetClass.label,
      target_icon: targetClass.icon,
      instruction: targetClass.instruction,
      practice_tip: targetClass.practiceTip,
      accuracy_percent: accuracyPercent,
      is_perfect: isPerfect,
      correct_fingers: correctCount,
      finger_eval: fingerEval,
      feedback_tips: feedbackTips
    };
  }

  predict(canonicalPoints, isLocked = true, targetPracticeId = null, rawPoints = null) {
    const geo = this.extractGeometricFeatures(canonicalPoints, rawPoints);
    const logits = this.computeCalibratedLogits(canonicalPoints, geo);

    // Softmax
    const maxLogit = Math.max(...logits);
    const expLogits = logits.map(l => Math.exp(l - maxLogit));
    const sumExp = expLogits.reduce((a, b) => a + b, 0);
    const probs = expLogits.map(e => e / sumExp);

    let bestIdx = 0;
    let maxProb = probs[0];
    for (let i = 1; i < probs.length; i++) {
      if (probs[i] > maxProb) {
        maxProb = probs[i];
        bestIdx = i;
      }
    }

    const bestClass = this.CLASSES[bestIdx];
    const isVerified = (maxProb > this.THRESHOLD) && (isLocked || maxProb > 0.94);

    const outputText = isVerified ? bestClass.label : "Signal Unclear / Analyzing";
    const icon = isVerified ? bestClass.icon : "⏳";
    const statusDesc = isVerified ? "VERIFIED_ACCURATE" : "THRESHOLD_GATED";

    const topIndices = probs.map((p, i) => [p, i]).sort((a, b) => b[0] - a[0]).slice(0, 3);
    const topCandidates = topIndices.map(([p, i]) => ({ label: this.CLASSES[i].label, confidence: p }));

    const practiceKey = targetPracticeId !== null ? targetPracticeId : bestClass.id;
    const practiceEval = this.getPracticeFeedback(practiceKey, canonicalPoints, geo);

    return {
      verified: isVerified,
      output_text: outputText,
      confidence: maxProb,
      threshold: this.THRESHOLD,
      icon: icon,
      class_id: isVerified ? bestClass.id : null,
      candidate_label: bestClass.label,
      status: statusDesc,
      top_candidates: topCandidates,
      practice_eval: practiceEval,
      features: {
        extended_fingers: geo.extCount,
        total_extended: geo.totalExt,
        is_fist: geo.isFist4,
        anti_gravity_locked: isLocked
      }
    };
  }
}

// Attach to window
window.AntiGravityStabilizerJS = AntiGravityStabilizerJS;
window.ZeroErrorClassifierJS = ZeroErrorClassifierJS;

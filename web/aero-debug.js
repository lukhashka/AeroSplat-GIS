// ═══════════════════════════════════════════════════════════════════
//  AeroSplat-GIS — Diagnostics console
//
//  A black screen in a 3DGS viewer has a dozen indistinguishable
//  causes: the data never arrived, it arrived NaN, the scales are
//  log-space, every splat is behind the camera, every splat projects
//  to 0.01 px, the alpha cutoff eats everything, the shader failed to
//  compile, the GL context died. From the outside all of those look
//  identical — dark blue and silent.
//
//  This module makes them distinguishable. It does three things the
//  browser can't do for you:
//
//   1. STATS ON THE ACTUAL BYTES. Not what scene-meta claims — what
//      is really in the Float32Array, per attribute, after the wire.
//   2. A CPU MIRROR OF THE VERTEX SHADER. The same quaternion → Σ →
//      W·Σ·Wᵀ → Jacobian → Σ' chain, in JS, on a sample of splats,
//      reporting how many land in the frustum, how big they project,
//      and what peak alpha they reach. This is the verification
//      harness AeroSplat-viewer-review.md §5 asked for, and it is
//      what turns "nothing renders" into a specific sentence.
//   3. A SYNTHETIC TEST SCENE. Seven hand-built Gaussians with known
//      geometry. If they render and your data doesn't, the shader is
//      fine and the pipeline is wrong — which halves the search space
//      in one click.
//
//  Load AFTER appGLSL.js (classic script, no module). It reads the
//  viewer's top-level bindings by name and degrades gracefully if a
//  binding is missing.
//
//  Toggle: F9, or Ctrl+Shift+D.
// ═══════════════════════════════════════════════════════════════════

(function () {
    'use strict';

    // ── Configuration ──────────────────────────────────────────────
    const PROBE_SAMPLE      = 4096;      // splats pushed through the CPU mirror
    const STATS_SAMPLE      = 250000;    // splats included in attribute statistics
    const LOG_CAPACITY      = 400;
    const FRAME_WINDOW      = 60;        // frames averaged for the frame graph
    const PIXEL_PROBE_SIZE  = 96;        // readPixels square, at framebuffer scale

    // ── Bindings to the viewer ─────────────────────────────────────
    // appGLSL.js declares these with const/let at script top level, which
    // puts them in the global declarative record: reachable by name from a
    // later classic script, but NOT as window properties. So resolve by
    // name inside a try, and let bind() override anything explicitly.
    const overrides = {};

    function look(name) {
        if (name in overrides) return overrides[name]();
        try {
            return (0, eval)(name);
        } catch {
            return undefined;
        }
    }

    const ref = {
        get renderer()   { return look('renderer'); },
        get scene()      { return look('scene'); },
        get sceneRoot()  { return look('sceneRoot'); },
        get camera()     { return look('camera'); },
        get controls()   { return look('controls'); },
        get activeMesh() { return look('activeMesh'); },
        get sceneMeta()  { return look('sceneMeta'); },
        get mode()       { return look('currentMode'); },
        get dataset()    { return look('currentDataset'); },
        get viewportPx() { return look('viewportPx'); },
        get sortGen()    { return look('sortGeneration'); },
        get sortBusy()   { return look('sortInFlight'); }
    };

    // ── State ──────────────────────────────────────────────────────
    const state = {
        open: true,
        collapsed: {},
        logs: [],
        marks: [],
        markStart: 0,
        payload: null,        // { floatArr, meta, mode, stats }
        probe: null,
        verdict: { level: 'idle', text: 'No load recorded yet.' },
        frames: new Float32Array(FRAME_WINDOW),
        frameIdx: 0,
        lastFrameT: performance.now(),
        fps: 0,
        sorts: 0,
        lastSortMs: 0,
        lastWorkerMs: 0,
        lastSortAt: 0,
        glErrors: [],
        contextLost: false,
        pixelProbe: null,
        overridesActive: {}
    };

    // ═══════════════════════════════════════════════════════════════
    //  Logging
    // ═══════════════════════════════════════════════════════════════
    function log(level, msg, data) {
        state.logs.push({ t: performance.now(), level, msg, data });
        if (state.logs.length > LOG_CAPACITY) state.logs.shift();
        paintLog();
    }

    // Three.js reports shader compile failures through console.error and
    // nothing else. A viewer whose shader failed to link looks exactly
    // like a viewer whose data is empty, so mirror the console in here.
    ['error', 'warn'].forEach((kind) => {
        const original = console[kind].bind(console);
        console[kind] = function (...args) {
            const text = args.map(a => {
                if (a instanceof Error) return a.message;
                if (typeof a === 'object') { try { return JSON.stringify(a); } catch { return String(a); } }
                return String(a);
            }).join(' ');
            state.logs.push({ t: performance.now(), level: kind, msg: text.slice(0, 2000) });
            if (state.logs.length > LOG_CAPACITY) state.logs.shift();
            if (/shader|program|glsl|compile|link/i.test(text)) {
                setVerdict('bad', 'Shader failed to compile or link — see the log below.');
            }
            paintLog();
            original(...args);
        };
    });

    function mark(label) {
        const now = performance.now();
        if (!state.marks.length) state.markStart = now;
        const prev = state.marks.length ? state.marks[state.marks.length - 1].at : now;
        state.marks.push({ label, at: now, delta: now - prev });
        paintSection('load');
    }

    function resetMarks() {
        state.marks = [];
        state.markStart = performance.now();
    }

    function setVerdict(level, text, detail) {
        state.verdict = { level, text, detail };
        paintVerdict();
    }

    // ═══════════════════════════════════════════════════════════════
    //  Attribute statistics over the real payload
    // ═══════════════════════════════════════════════════════════════
    function seriesStats(get, n) {
        let min = Infinity, max = -Infinity, sum = 0, nonFinite = 0;
        for (let i = 0; i < n; i++) {
            const v = get(i);
            if (!Number.isFinite(v)) { nonFinite++; continue; }
            if (v < min) min = v;
            if (v > max) max = v;
            sum += v;
        }
        const good = n - nonFinite;
        return {
            min: good ? min : NaN,
            max: good ? max : NaN,
            mean: good ? sum / good : NaN,
            nonFinite
        };
    }

    function percentile(sortedArr, p) {
        if (!sortedArr.length) return NaN;
        const i = Math.min(sortedArr.length - 1, Math.max(0, Math.round((p / 100) * (sortedArr.length - 1))));
        return sortedArr[i];
    }

    function analyzeSplatPayload(floatArr, meta) {
        const stride = meta.stride_floats || 16;
        const N = meta.count;
        const step = Math.max(1, Math.ceil(N / STATS_SAMPLE));
        const sampled = [];
        for (let i = 0; i < N; i += step) sampled.push(i);
        const n = sampled.length;

        const at = (k, off) => floatArr[sampled[k] * stride + off];

        const stats = {
            sampled: n,
            of: N,
            x: seriesStats(k => at(k, 0), n),
            y: seriesStats(k => at(k, 1), n),
            z: seriesStats(k => at(k, 2), n),
            opacity: seriesStats(k => at(k, 3), n),
            scaleX: seriesStats(k => at(k, 4), n),
            scaleY: seriesStats(k => at(k, 5), n),
            scaleZ: seriesStats(k => at(k, 6), n),
            r: seriesStats(k => at(k, 12), n),
            g: seriesStats(k => at(k, 13), n),
            b: seriesStats(k => at(k, 14), n)
        };

        // Quaternion norms — the shader normalises, so a non-unit norm is
        // harmless, but an all-zero quaternion is not: normalize(0) is
        // undefined and takes the whole rotation matrix with it.
        let qMin = Infinity, qMax = -Infinity, qZero = 0, w1 = 0;
        for (let k = 0; k < n; k++) {
            const w = at(k, 8), x = at(k, 9), y = at(k, 10), z = at(k, 11);
            const nn = Math.hypot(w, x, y, z);
            if (nn < 1e-8) qZero++;
            if (nn < qMin) qMin = nn;
            if (nn > qMax) qMax = nn;
            if (Math.abs(w) > 0.99) w1++;
        }
        stats.quat = { min: qMin, max: qMax, zero: qZero, nearIdentityFrac: w1 / n };

        // Degenerate geometry counts.
        let zeroScale = 0, negScale = 0, opacityDead = 0, opacityOut = 0;
        const scaleSamples = [];
        for (let k = 0; k < n; k++) {
            const sx = at(k, 4), sy = at(k, 5), sz = at(k, 6);
            if (sx <= 0 || sy <= 0 || sz <= 0) zeroScale++;
            if (sx < 0 || sy < 0 || sz < 0) negScale++;
            const o = at(k, 3);
            if (o < 1 / 255) opacityDead++;
            if (o < 0 || o > 1) opacityOut++;
            scaleSamples.push(Math.max(sx, sy, sz));
        }
        scaleSamples.sort((a, b) => a - b);
        stats.degenerate = { zeroScale, negScale, opacityDead, opacityOut };
        stats.scalePct = {
            p01: percentile(scaleSamples, 1),
            p50: percentile(scaleSamples, 50),
            p99: percentile(scaleSamples, 99)
        };

        // The scale sanity ratio: largest Gaussian axis versus scene extent.
        // A healthy 3DGS scene sits around 1e-4 … 1e-2 of the extent. Orders
        // of magnitude below that is an un-exponentiated log-scale that came
        // out positive; orders above is an exp() applied to metric values.
        stats.scaleVsExtent = meta.extent ? stats.scalePct.p50 / meta.extent : NaN;

        return stats;
    }

    function analyzeDensePayload(floatArr, meta) {
        const stride = 6;
        const N = meta.count;
        const step = Math.max(1, Math.ceil(N / STATS_SAMPLE));
        const sampled = [];
        for (let i = 0; i < N; i += step) sampled.push(i);
        const n = sampled.length;
        const at = (k, off) => floatArr[sampled[k] * stride + off];
        return {
            sampled: n,
            of: N,
            x: seriesStats(k => at(k, 0), n),
            y: seriesStats(k => at(k, 1), n),
            z: seriesStats(k => at(k, 2), n),
            r: seriesStats(k => at(k, 3), n),
            g: seriesStats(k => at(k, 4), n),
            b: seriesStats(k => at(k, 5), n)
        };
    }

    // ═══════════════════════════════════════════════════════════════
    //  CPU mirror of the vertex shader (review §5)
    //
    //  Same chain, same order, same constants. If this disagrees with
    //  what you see on screen, one of the two is wrong and you now have
    //  two independent implementations to diff instead of one to stare
    //  at. If it agrees and the screen is empty, the numbers below tell
    //  you which stage empties it.
    // ═══════════════════════════════════════════════════════════════
    const CUTOFF   = 3.5;
    const DILATION = 0.3;

    function quatToMat3(w, x, y, z) {
        const n = Math.hypot(w, x, y, z) || 1;
        w /= n; x /= n; y /= n; z /= n;
        // Row-major here (plain JS arrays), unlike GLSL's column-major
        // constructor — the §A1 trap only exists on the shader side.
        return [
            1 - 2 * (y * y + z * z), 2 * (x * y - w * z),     2 * (x * z + w * y),
            2 * (x * y + w * z),     1 - 2 * (x * x + z * z), 2 * (y * z - w * x),
            2 * (x * z - w * y),     2 * (y * z + w * x),     1 - 2 * (x * x + y * y)
        ];
    }

    function mat3Mul(a, b) {
        const o = new Array(9);
        for (let r = 0; r < 3; r++) {
            for (let c = 0; c < 3; c++) {
                o[r * 3 + c] = a[r * 3] * b[c] + a[r * 3 + 1] * b[3 + c] + a[r * 3 + 2] * b[6 + c];
            }
        }
        return o;
    }

    function mat3T(m) {
        return [m[0], m[3], m[6], m[1], m[4], m[7], m[2], m[5], m[8]];
    }

    function runProbe() {
        const meta = state.payload && state.payload.meta;
        const mesh = ref.activeMesh;
        const cam  = ref.camera;
        const rndr = ref.renderer;
        if (!meta || !mesh || !cam || !rndr || state.payload.mode !== 'splat') {
            state.probe = null;
            return null;
        }

        const floatArr = state.payload.floatArr;
        const stride   = meta.stride_floats || 16;
        const N        = meta.count;

        cam.updateMatrixWorld();
        mesh.updateMatrixWorld(true);

        const view = new THREE.Matrix4().copy(cam.matrixWorld).invert();
        const modelView = new THREE.Matrix4().multiplyMatrices(view, mesh.matrixWorld);
        const mv = modelView.elements;   // column-major

        // W = mat3(modelViewMatrix), read back into row-major.
        const W = [
            mv[0], mv[4], mv[8],
            mv[1], mv[5], mv[9],
            mv[2], mv[6], mv[10]
        ];
        const Wt = mat3T(W);

        const proj = cam.projectionMatrix.elements;
        const vw = ref.viewportPx ? ref.viewportPx.x : rndr.domElement.width;
        const vh = ref.viewportPx ? ref.viewportPx.y : rndr.domElement.height;
        const fx = proj[0] * vw * 0.5;
        const fy = proj[5] * vh * 0.5;
        const tanFovX = 1 / proj[0];
        const tanFovY = 1 / proj[5];

        const mult = (mesh.material.uniforms.uScaleMultiplier || { value: 1 }).value;

        const step = Math.max(1, Math.ceil(N / PROBE_SAMPLE));
        let tested = 0, behind = 0, onScreen = 0, subPixel = 0, alphaDead = 0, degenerate = 0;
        const radii = [];
        const alphas = [];
        const depths = [];

        for (let i = 0; i < N; i += step) {
            const s = i * stride;
            const px = floatArr[s], py = floatArr[s + 1], pz = floatArr[s + 2];
            const op = floatArr[s + 3];
            const sx = floatArr[s + 4] * mult, sy = floatArr[s + 5] * mult, sz = floatArr[s + 6] * mult;
            const qw = floatArr[s + 8], qx = floatArr[s + 9], qy = floatArr[s + 10], qz = floatArr[s + 11];
            tested++;

            if (!Number.isFinite(px + py + pz + op + sx + sy + sz + qw + qx + qy + qz)) {
                degenerate++;
                continue;
            }

            // Camera-space centre.
            const cx = mv[0] * px + mv[4] * py + mv[8]  * pz + mv[12];
            const cy = mv[1] * px + mv[5] * py + mv[9]  * pz + mv[13];
            const cz = mv[2] * px + mv[6] * py + mv[10] * pz + mv[14];
            depths.push(-cz);

            if (cz >= -0.1) { behind++; continue; }

            // Σ = R S² Rᵀ, then Σ_cam = W Σ Wᵀ.
            const R = quatToMat3(qw, qx, qy, qz);
            const S2 = [sx * sx, 0, 0, 0, sy * sy, 0, 0, 0, sz * sz];
            const cov3 = mat3Mul(mat3Mul(R, S2), mat3T(R));
            const covCam = mat3Mul(mat3Mul(W, cov3), Wt);

            const limX = 1.3 * tanFovX * (-cz);
            const limY = 1.3 * tanFovY * (-cz);
            const xc = Math.max(-limX, Math.min(limX, cx));
            const yc = Math.max(-limY, Math.min(limY, cy));

            const iz = 1 / cz, iz2 = iz * iz;
            const fx2 = fx * fx, fy2 = fy * fy;

            let c00 = fx2 * iz2 * covCam[0]
                    - 2 * fx2 * xc * iz2 * iz * covCam[6]
                    + fx2 * xc * xc * iz2 * iz2 * covCam[8];
            let c11 = fy2 * iz2 * covCam[4]
                    - 2 * fy2 * yc * iz2 * iz * covCam[7]
                    + fy2 * yc * yc * iz2 * iz2 * covCam[8];
            let c01 = fx * fy * iz2 * covCam[1]
                    - fx * fy * yc * iz2 * iz * covCam[2]
                    - fx * fy * xc * iz2 * iz * covCam[7]
                    + fx * fy * xc * yc * iz2 * iz2 * covCam[8];

            const detOrig = c00 * c11 - c01 * c01;
            c00 += DILATION;
            c11 += DILATION;
            const detDil = c00 * c11 - c01 * c01;

            const peakAlpha = op * Math.sqrt(Math.max(detOrig, 0) / Math.max(detDil, 1e-9));
            alphas.push(peakAlpha);
            if (peakAlpha < 1 / 255) alphaDead++;

            const mid = 0.5 * (c00 + c11);
            const dif = 0.5 * (c00 - c11);
            const disc = Math.sqrt(Math.max(dif * dif + c01 * c01, 0));
            const l1 = Math.max(mid + disc, 1e-6);
            const radiusPx = CUTOFF * Math.sqrt(l1);
            radii.push(radiusPx);
            if (radiusPx < 0.5) subPixel++;

            // Frustum test on the centre, widened by the sprite radius.
            const clipW = -cz;
            const ndcX = (proj[0] * cx + proj[8] * cz) / clipW;
            const ndcY = (proj[5] * cy + proj[9] * cz) / clipW;
            const marginX = radiusPx / (vw * 0.5);
            const marginY = radiusPx / (vh * 0.5);
            if (ndcX > -1 - marginX && ndcX < 1 + marginX &&
                ndcY > -1 - marginY && ndcY < 1 + marginY) onScreen++;
        }

        radii.sort((a, b) => a - b);
        alphas.sort((a, b) => a - b);
        depths.sort((a, b) => a - b);

        // Overdraw estimate. Splat radius is extremely heavy-tailed in an
        // aerial scene (a handful of near-camera splats can be 50-100x the
        // median radius, i.e. 2500-10000x the AREA), so a raw mean over the
        // sample is dominated by whichever few huge splats happened to land
        // in it and swings wildly run to run. Median area is robust to
        // that; report it as the primary "typical" overdraw factor and keep
        // the mean available separately, labeled as what it is.
        const areas = radii.map(r => Math.PI * r * r);
        const medianArea = percentile(areas, 50);
        const meanArea = areas.length ? areas.reduce((a, b) => a + b, 0) / areas.length : 0;
        const onScreenFrac = tested ? onScreen / tested : 0;
        // Expected total covered pixels (double-counting overlap — this is
        // "ink", not unique pixels) if every on-screen splat in the FULL
        // population (not just the sample) painted its footprint, divided
        // by the framebuffer's real pixel count. >1 means overdraw; a
        // scene with heavy alpha-blended overlap is expected to sit well
        // above 1 even when it looks and performs fine.
        const screenPixels = Math.max(vw * vh, 1);
        const overdrawMedian = (medianArea * N * onScreenFrac) / screenPixels;
        const overdrawMean   = (meanArea   * N * onScreenFrac) / screenPixels;

        const probe = {
            tested, behind, onScreen, subPixel, alphaDead, degenerate,
            onScreenFrac,
            radius: { p05: percentile(radii, 5), p50: percentile(radii, 50), p95: percentile(radii, 95), max: radii.length ? radii[radii.length - 1] : NaN },
            alpha:  { p05: percentile(alphas, 5), p50: percentile(alphas, 50), p95: percentile(alphas, 95) },
            depth:  { near: percentile(depths, 1), p50: percentile(depths, 50), far: percentile(depths, 99) },
            overdrawMedian,   // robust "typical" overdraw factor — read this one
            overdrawMean      // outlier-sensitive; large gap from overdrawMedian itself is informative
        };
        state.probe = probe;
        deriveVerdict();
        return probe;
    }

    // The verdict is the whole point of the panel: one sentence naming the
    // stage that is losing the image, chosen from the probe's numbers
    // rather than from a guess.
    function deriveVerdict() {
        const p = state.payload;
        const pr = state.probe;

        if (window.__AEROSPLAT_FATAL__) return setVerdict('bad', 'WebGL context creation failed — see the console error above for full detail and remediation steps. If chrome://gpu says "GPU process crashed too many times with software GL", this is Chrome\'s own stability blocklist (common in VMs/remote desktops/containers with no real GPU); try chrome://flags/#ignore-gpu-blocklist. Not a data or shader problem — nothing else in this panel is meaningful until it\'s fixed.');
        if (state.contextLost) return setVerdict('bad', 'WebGL context lost. The GPU dropped the context — usually a driver timeout from too much overdraw. Lower SUPERSAMPLE or load with ?step=4.');
        if (!p) return setVerdict('idle', 'No payload recorded. Call AeroDebug.onPayload() from loadSpatialData.');
        if (!p.meta.count) return setVerdict('bad', 'Server returned 0 splats. The .pt is empty or everything was filtered — check the server console.');

        if (p.mode === 'splat' && p.stats) {
            const st = p.stats;
            if (st.degenerate.negScale) return setVerdict('bad', `${st.degenerate.negScale} sampled splats have negative scale — the server did not exponentiate a log-space scale tensor. Check activations in scene-meta.`);
            if (Number.isFinite(st.scaleVsExtent)) {
                if (st.scaleVsExtent < 1e-6) return setVerdict('bad', `Median splat scale is ${st.scaleVsExtent.toExponential(1)}× the scene extent — every Gaussian is far below one pixel. Almost certainly a log-space scale that stayed positive and so defeated the auto-detect.`);
                if (st.scaleVsExtent > 0.5) return setVerdict('bad', `Median splat scale is ${st.scaleVsExtent.toFixed(2)}× the scene extent — exp() was probably applied to already-metric scales. Set scale_is_log = False on the server.`);
            }
            if (st.quat.zero > st.sampled * 0.5) return setVerdict('bad', 'Over half the quaternions are zero-length. normalize() is undefined there and the rotation matrix goes with it.');
            if (st.degenerate.opacityOut > st.sampled * 0.5) return setVerdict('bad', 'Most opacities fall outside [0,1] — logits reached the wire unactivated.');
        }

        if (!pr) return setVerdict('warn', 'Payload looks sane. Run the projection probe to check what reaches the screen.');

        if (pr.behind === pr.tested) return setVerdict('bad', 'Every sampled splat is behind the camera. Either the up-axis rotation is wrong or the camera is inside/underneath the scene — press "Frame scene".');
        if (pr.onScreen === 0) return setVerdict('bad', 'No sampled splat projects inside the frustum. The scene exists but the camera is not pointed at it.');
        if (pr.subPixel > pr.tested * 0.95) return setVerdict('bad', `${Math.round(100 * pr.subPixel / pr.tested)}% of splats project below half a pixel. Geometry is placed correctly but rendered too small to see — a scale-encoding problem, not a camera problem.`);
        if (pr.alphaDead > pr.tested * 0.95) return setVerdict('bad', 'Peak alpha is under 1/255 for nearly every splat, so the fragment shader discards them all. Opacity is arriving near zero.');
        if (pr.overdrawMedian < 0.02) return setVerdict('warn', `Splats are on screen but cover an estimated ${(pr.overdrawMedian * 100).toFixed(2)}% of it. Expect a nearly empty frame; check the scene scale.`);
        if (pr.overdrawMedian > 15 && state.fps > 0 && state.fps < 30) return setVerdict('warn', `Heavy overdraw (median splat footprint covers the frame ~${pr.overdrawMedian.toFixed(0)}× over) combined with ${state.fps.toFixed(0)} fps — fill rate is the likely bottleneck, not a bug. Try a lower SUPERSAMPLE, a smaller MAX_PIXEL_RATIO, or ?step=2 to render fewer splats.`);

        setVerdict('ok', `${pr.onScreen} of ${pr.tested} sampled splats project on screen, median radius ${pr.radius.p50.toFixed(2)} px, median peak alpha ${pr.alpha.p50.toFixed(3)}. Rendering path looks healthy.`);
    }

    // ═══════════════════════════════════════════════════════════════
    //  Framebuffer probe — is the GPU actually writing colour?
    //
    //  Separates "drew nothing" from "drew something invisible". A
    //  readPixels of the centre of the frame, counted against the known
    //  clear colour.
    // ═══════════════════════════════════════════════════════════════
    function runPixelProbe() {
        const rndr = ref.renderer;
        if (!rndr) return null;
        const gl = rndr.getContext();
        rndr.render(ref.scene, ref.camera);   // guarantee a fresh frame in the buffer

        const w = Math.min(PIXEL_PROBE_SIZE, gl.drawingBufferWidth);
        const h = Math.min(PIXEL_PROBE_SIZE, gl.drawingBufferHeight);
        const x = Math.floor((gl.drawingBufferWidth - w) / 2);
        const y = Math.floor((gl.drawingBufferHeight - h) / 2);
        const buf = new Uint8Array(w * h * 4);
        gl.readPixels(x, y, w, h, gl.RGBA, gl.UNSIGNED_BYTE, buf);

        const bg = ref.scene && ref.scene.background ? ref.scene.background : null;
        const br = bg ? Math.round(bg.r * 255) : 0;
        const bgc = bg ? Math.round(bg.g * 255) : 0;
        const bb = bg ? Math.round(bg.b * 255) : 0;

        let differing = 0, sumR = 0, sumG = 0, sumB = 0;
        const total = w * h;
        for (let i = 0; i < total; i++) {
            const r = buf[i * 4], g = buf[i * 4 + 1], b = buf[i * 4 + 2];
            sumR += r; sumG += g; sumB += b;
            if (Math.abs(r - br) > 2 || Math.abs(g - bgc) > 2 || Math.abs(b - bb) > 2) differing++;
        }
        state.pixelProbe = {
            region: `${w}×${h} @ centre`,
            differing,
            total,
            frac: differing / total,
            mean: [sumR / total, sumG / total, sumB / total],
            clear: [br, bgc, bb]
        };
        log('info', `Pixel probe: ${differing}/${total} pixels differ from clear colour.`);
        paintSection('pixels');
        return state.pixelProbe;
    }

    // ═══════════════════════════════════════════════════════════════
    //  Synthetic test scene (review §5, golden-value form)
    //
    //  Seven Gaussians with hand-chosen geometry, injected through the
    //  viewer's own buildSplatMesh so they travel the exact path real
    //  data travels. What you should see, with the camera framed:
    //
    //    · three long needles along world X (red), Y (green), Z (blue)
    //    · one isotropic white blob at the origin
    //    · one flat disc in the XY plane (yellow)
    //    · two 45°-tilted needles (magenta, cyan) leaning opposite ways
    //
    //  Needles pointing down the wrong axis means the quaternion
    //  convention or quatToMat is wrong. Tilted needles leaning the same
    //  way as each other means the cross term lost its sign. Nothing at
    //  all means the shader or the texture path is broken, not the data.
    // ═══════════════════════════════════════════════════════════════
    function buildTestPayload(extent) {
        const L = extent * 0.35;      // needle half-length, in COLMAP units
        const T = extent * 0.01;      // needle thickness
        const R2 = Math.SQRT1_2;

        const defs = [
            // pos,               scale,          quat(w,x,y,z),        rgb
            [[0, 0, 0], [L, T, T], [1, 0, 0, 0], [1.0, 0.2, 0.2]],   // X needle
            [[0, 0, 0], [T, L, T], [1, 0, 0, 0], [0.2, 1.0, 0.2]],   // Y needle
            [[0, 0, 0], [T, T, L], [1, 0, 0, 0], [0.3, 0.5, 1.0]],   // Z needle
            [[0, 0, 0], [T * 4, T * 4, T * 4], [1, 0, 0, 0], [1, 1, 1]], // isotropic
            [[0, 0, 0], [L * 0.6, L * 0.6, T], [1, 0, 0, 0], [1.0, 0.9, 0.2]], // disc
            [[L * 0.5, 0, 0], [L * 0.5, T, T], [R2, 0, 0, R2], [1.0, 0.2, 1.0]], // +45° about Z
            [[-L * 0.5, 0, 0], [L * 0.5, T, T], [R2, 0, 0, -R2], [0.2, 1.0, 1.0]] // −45° about Z
        ];

        const perRow = 512, stride = 16;
        const padded = perRow;      // one texture row is plenty
        const arr = new Float32Array(padded * stride);
        defs.forEach((d, i) => {
            const s = i * stride;
            arr[s] = d[0][0]; arr[s + 1] = d[0][1]; arr[s + 2] = d[0][2];
            arr[s + 3] = 0.95;
            arr[s + 4] = d[1][0]; arr[s + 5] = d[1][1]; arr[s + 6] = d[1][2];
            arr[s + 8] = d[2][0]; arr[s + 9] = d[2][1]; arr[s + 10] = d[2][2]; arr[s + 11] = d[2][3];
            arr[s + 12] = d[3][0]; arr[s + 13] = d[3][1]; arr[s + 14] = d[3][2];
        });
        return { arr, count: defs.length, padded };
    }

    function injectTestScene() {
        const meta = ref.sceneMeta;
        if (!meta) { log('error', 'No scene-meta yet — load real data once first, so the test scene inherits its frame.'); return; }
        const build = look('buildSplatMesh');
        const dispose = look('disposeActiveMesh');
        if (typeof build !== 'function') { log('error', 'buildSplatMesh is not reachable; bind it via AeroDebug.bind({ buildSplatMesh: () => buildSplatMesh }).'); return; }

        const { arr, count } = buildTestPayload(meta.extent);
        const testMeta = Object.assign({}, meta, {
            count,
            padded_count: 512,
            stride_floats: 16,
            texture_width: 2048,
            splats_per_row: 512
        });

        if (typeof dispose === 'function') dispose();
        const mesh = build(arr, testMeta);
        ref.sceneRoot.add(mesh);
        if (overrides.setActiveMesh) overrides.setActiveMesh(mesh);

        state.payload = { floatArr: arr, meta: testMeta, mode: 'splat', stats: analyzeSplatPayload(arr, testMeta), synthetic: true };
        log('info', 'Test scene injected: 3 axis needles, 1 blob, 1 disc, 2 tilted needles.');
        runProbe();
        paintAll();
    }

    // ═══════════════════════════════════════════════════════════════
    //  Render-state overrides
    // ═══════════════════════════════════════════════════════════════
    function setUniform(name, value) {
        const m = ref.activeMesh;
        if (!m || !m.material.uniforms || !m.material.uniforms[name]) return false;
        m.material.uniforms[name].value = value;
        return true;
    }

    let axesHelper = null;
    function toggleAxes(on) {
        const root = ref.sceneRoot;
        const meta = ref.sceneMeta;
        if (!root) return;
        if (on && !axesHelper) {
            const r = meta ? meta.extent : 1;
            axesHelper = new THREE.AxesHelper(r);
            root.add(axesHelper);
        } else if (!on && axesHelper) {
            root.remove(axesHelper);
            axesHelper.geometry.dispose();
            axesHelper.material.dispose();
            axesHelper = null;
        }
    }

    let boundsHelper = null;
    function toggleBounds(on) {
        const root = ref.sceneRoot;
        const meta = ref.sceneMeta;
        if (!root || !meta) return;
        if (on && !boundsHelper) {
            const g = new THREE.BoxGeometry(meta.extent * 2, meta.extent * 2, meta.extent * 2);
            boundsHelper = new THREE.LineSegments(
                new THREE.EdgesGeometry(g),
                new THREE.LineBasicMaterial({ color: 0x00ff33 })
            );
            g.dispose();
            root.add(boundsHelper);
        } else if (!on && boundsHelper) {
            root.remove(boundsHelper);
            boundsHelper.geometry.dispose();
            boundsHelper.material.dispose();
            boundsHelper = null;
        }
    }

    function frameScene() {
        const meta = ref.sceneMeta, cam = ref.camera, ctl = ref.controls;
        if (!meta || !cam) return;
        const r = meta.extent * meta.scene_scale;
        if (ctl) ctl.target.set(0, 0, 0);
        cam.position.set(0, r * 1.2, r * 1.8);
        cam.near = Math.max(r * 0.001, 0.01);
        cam.far = r * 1000;
        cam.updateProjectionMatrix();
        cam.lookAt(0, 0, 0);
        if (ctl) ctl.update();
        log('info', `Camera reframed to scene radius ${r.toFixed(2)}.`);
        runProbe();
    }

    // ═══════════════════════════════════════════════════════════════
    //  GPU capability snapshot
    // ═══════════════════════════════════════════════════════════════
    function gpuInfo() {
        const rndr = ref.renderer;
        if (!rndr) return {};
        const gl = rndr.getContext();
        const dbg = gl.getExtension('WEBGL_debug_renderer_info');
        const psr = gl.getParameter(gl.ALIASED_POINT_SIZE_RANGE);
        return {
            webgl2: !!rndr.capabilities.isWebGL2,
            vendor: dbg ? gl.getParameter(dbg.UNMASKED_VENDOR_WEBGL) : 'hidden',
            renderer: dbg ? gl.getParameter(dbg.UNMASKED_RENDERER_WEBGL) : 'hidden',
            maxTextureSize: gl.getParameter(gl.MAX_TEXTURE_SIZE),
            maxVertexTextureUnits: gl.getParameter(gl.MAX_VERTEX_TEXTURE_IMAGE_UNITS),
            maxVaryingVectors: gl.getParameter(gl.MAX_VARYING_VECTORS),
            pointSizeRange: `${psr[0]}–${psr[1]}`,
            floatTextures: !!(rndr.capabilities.isWebGL2 || gl.getExtension('OES_texture_float')),
            floatLinear: !!gl.getExtension('OES_texture_float_linear'),
            colorBufferFloat: !!(gl.getExtension('EXT_color_buffer_float') || gl.getExtension('WEBGL_color_buffer_float')),
            pixelRatio: rndr.getPixelRatio(),
            drawingBuffer: `${gl.drawingBufferWidth}×${gl.drawingBufferHeight}`,
            // Splat-count ceilings this build actually has. Both matter and
            // they are not the same number — see the panel note.
            texelCeiling: gl.getParameter(gl.MAX_TEXTURE_SIZE) * 512,
            floatIndexCeiling: 16777216 / 4
        };
    }

    // ═══════════════════════════════════════════════════════════════
    //  Panel
    // ═══════════════════════════════════════════════════════════════
    const CSS = `
    #aero-dbg {
        position: fixed; top: 0; right: 0; bottom: 0; width: 440px;
        background: rgba(3, 7, 10, 0.94);
        border-left: 1px solid rgba(0, 255, 51, 0.35);
        color: #b8d8c0; font: 11px/1.45 ui-monospace, "SF Mono", Menlo, Consolas, monospace;
        z-index: 100000; overflow-y: auto; overflow-x: hidden;
        backdrop-filter: blur(3px);
    }
    #aero-dbg::-webkit-scrollbar { width: 8px; }
    #aero-dbg::-webkit-scrollbar-thumb { background: rgba(0,255,51,0.25); }
    #aero-dbg h1 {
        margin: 0; padding: 10px 12px; font-size: 11px; letter-spacing: 2px;
        color: #00ff33; border-bottom: 1px solid rgba(0,255,51,0.25);
        display: flex; justify-content: space-between; align-items: center;
        position: sticky; top: 0; background: rgba(3,7,10,0.98); z-index: 2;
    }
    #aero-dbg h1 span.hint { color: #3a5a44; letter-spacing: 0; font-weight: normal; }
    .adbg-verdict {
        padding: 10px 12px; border-bottom: 1px solid rgba(0,255,51,0.15);
        font-size: 11.5px; line-height: 1.5;
    }
    .adbg-verdict .tag {
        display: inline-block; padding: 1px 6px; margin-right: 7px;
        border-radius: 2px; font-weight: bold; letter-spacing: 1px; font-size: 10px;
    }
    .adbg-verdict.ok   { background: rgba(0,255,51,0.07); color: #7fe89a; }
    .adbg-verdict.ok .tag   { background: #0a5c22; color: #b6ffc8; }
    .adbg-verdict.warn { background: rgba(255,190,0,0.07); color: #f0d089; }
    .adbg-verdict.warn .tag { background: #6b4c00; color: #ffe6a8; }
    .adbg-verdict.bad  { background: rgba(255,50,50,0.09); color: #ff9b9b; }
    .adbg-verdict.bad .tag  { background: #7a1414; color: #ffc9c9; }
    .adbg-verdict.idle { color: #6e8a78; }
    .adbg-verdict.idle .tag { background: #22332a; color: #8fae9a; }
    .adbg-sec { border-bottom: 1px solid rgba(0,255,51,0.12); }
    .adbg-sec > header {
        padding: 6px 12px; color: #00ff33; cursor: pointer; user-select: none;
        letter-spacing: 1.5px; font-size: 10px; display: flex; gap: 8px; align-items: baseline;
    }
    .adbg-sec > header:hover { background: rgba(0,255,51,0.06); }
    .adbg-sec > header .caret { color: #3a5a44; width: 8px; }
    .adbg-sec > header .note  { color: #46705a; letter-spacing: 0; margin-left: auto; }
    .adbg-body { padding: 2px 12px 10px; }
    .adbg-sec.closed .adbg-body { display: none; }
    table.adbg { width: 100%; border-collapse: collapse; }
    table.adbg td { padding: 1.5px 0; vertical-align: top; }
    table.adbg td.k { color: #5d8570; width: 44%; padding-right: 8px; }
    table.adbg td.v { color: #cfe9d6; text-align: right; font-variant-numeric: tabular-nums; }
    table.adbg td.v.bad  { color: #ff8080; }
    table.adbg td.v.warn { color: #ffcc66; }
    table.adbg td.v.ok   { color: #7fe89a; }
    .adbg-btns { display: flex; flex-wrap: wrap; gap: 5px; margin: 7px 0 3px; }
    .adbg-btns button, .adbg-tog {
        background: rgba(0,255,51,0.06); color: #8fe3a5;
        border: 1px solid rgba(0,255,51,0.3); border-radius: 2px;
        font: inherit; font-size: 10px; padding: 4px 7px; cursor: pointer;
    }
    .adbg-btns button:hover { background: rgba(0,255,51,0.16); color: #d6ffe2; }
    .adbg-tog { display: flex; align-items: center; gap: 5px; }
    .adbg-tog input { accent-color: #00ff33; margin: 0; }
    .adbg-slider { display: flex; align-items: center; gap: 7px; margin: 5px 0; }
    .adbg-slider input[type=range] { flex: 1; accent-color: #00ff33; }
    .adbg-slider label { color: #5d8570; min-width: 92px; }
    .adbg-slider output { color: #cfe9d6; min-width: 42px; text-align: right; }
    .adbg-log { max-height: 190px; overflow-y: auto; font-size: 10px; }
    .adbg-log div { padding: 1px 0; border-bottom: 1px solid rgba(255,255,255,0.03); white-space: pre-wrap; word-break: break-word; }
    .adbg-log .lvl-error { color: #ff8080; }
    .adbg-log .lvl-warn  { color: #ffcc66; }
    .adbg-log .lvl-info  { color: #86ad95; }
    .adbg-log .t { color: #3f6350; }
    .adbg-bar { height: 3px; background: rgba(0,255,51,0.12); margin: 3px 0 6px; }
    .adbg-bar > i { display: block; height: 100%; background: #00ff33; }
    #aero-dbg-tab {
        position: fixed; top: 0; right: 0; z-index: 100001;
        background: #041008; color: #00ff33; border: 1px solid rgba(0,255,51,0.4);
        border-right: none; border-top: none; padding: 5px 9px; cursor: pointer;
        font: 10px ui-monospace, monospace; letter-spacing: 1px;
    }
    #aero-dbg canvas.spark { width: 100%; height: 30px; display: block; margin: 4px 0; }
    `;

    let root = null, tab = null;

    function el(html) {
        const d = document.createElement('div');
        d.innerHTML = html.trim();
        return d.firstElementChild;
    }

    function num(v, digits) {
        if (v === undefined || v === null) return '—';
        if (typeof v !== 'number') return String(v);
        if (!Number.isFinite(v)) return String(v);
        if (v !== 0 && (Math.abs(v) < 1e-3 || Math.abs(v) >= 1e6)) return v.toExponential(2);
        return v.toFixed(digits === undefined ? 3 : digits);
    }

    function rows(pairs) {
        return '<table class="adbg">' + pairs.map(([k, v, cls]) =>
            `<tr><td class="k">${k}</td><td class="v ${cls || ''}">${v}</td></tr>`).join('') + '</table>';
    }

    function section(id, title, note) {
        const s = el(`<section class="adbg-sec" data-sec="${id}">
            <header><span class="caret">▾</span><span>${title}</span><span class="note">${note || ''}</span></header>
            <div class="adbg-body"></div>
        </section>`);
        s.querySelector('header').addEventListener('click', () => {
            s.classList.toggle('closed');
            state.collapsed[id] = s.classList.contains('closed');
        });
        if (state.collapsed[id]) s.classList.add('closed');
        return s;
    }

    function build() {
        const style = document.createElement('style');
        style.textContent = CSS;
        document.head.appendChild(style);

        tab = el('<button id="aero-dbg-tab">DIAG</button>');
        tab.addEventListener('click', () => toggle());
        document.body.appendChild(tab);

        root = el('<aside id="aero-dbg"></aside>');
        root.appendChild(el(`<h1><span>AEROSPLAT · DIAGNOSTICS</span><span class="hint">F9</span></h1>`));
        root.appendChild(el('<div class="adbg-verdict idle" id="adbg-verdict"></div>'));

        ['actions', 'probe', 'pixels', 'data', 'load', 'frame', 'camera', 'sort', 'gpu', 'log']
            .forEach(id => root.appendChild(section(id, SECTION_TITLES[id], '')));

        document.body.appendChild(root);
        buildActions();
        paintAll();
    }

    const SECTION_TITLES = {
        actions: 'CONTROLS',
        probe:   'PROJECTION PROBE',
        pixels:  'FRAMEBUFFER',
        data:    'PAYLOAD',
        load:    'LOAD TIMELINE',
        frame:   'FRAME',
        camera:  'CAMERA',
        sort:    'DEPTH SORT',
        gpu:     'GPU',
        log:     'LOG'
    };

    function body(id) {
        return root.querySelector(`[data-sec="${id}"] .adbg-body`);
    }

    function buildActions() {
        const b = body('actions');
        b.innerHTML = `
            <div class="adbg-btns">
                <button data-act="probe">Run probe</button>
                <button data-act="pixels">Read pixels</button>
                <button data-act="frame">Frame scene</button>
                <button data-act="test">Test scene</button>
                <button data-act="reload">Reload data</button>
                <button data-act="dump">Copy dump</button>
            </div>
            <div class="adbg-slider">
                <label>scale ×</label>
                <input type="range" id="adbg-scale" min="-2" max="2" step="0.02" value="0">
                <output id="adbg-scale-out">1.00</output>
            </div>
            <div class="adbg-btns">
                <label class="adbg-tog"><input type="checkbox" data-tog="axes"> axes</label>
                <label class="adbg-tog"><input type="checkbox" data-tog="bounds"> bounds</label>
                <label class="adbg-tog"><input type="checkbox" data-tog="opaque"> force α=1</label>
                <label class="adbg-tog"><input type="checkbox" data-tog="autoprobe"> auto-probe</label>
            </div>
        `;

        b.querySelectorAll('button[data-act]').forEach(btn => {
            btn.addEventListener('click', () => {
                const a = btn.dataset.act;
                // The viewer sets this if WebGL context creation itself
                // failed — every button below assumes a working renderer/
                // scene/camera, none of which exist in that state. One
                // clear message here beats a different cryptic error per
                // button (see the appGLSL.js note next to the flag).
                if (window.__AEROSPLAT_FATAL__) {
                    log('error', 'Viewer failed to initialize (WebGL context creation failed — see the console error above this panel\'s own log). Fix that first; nothing here will work until it does.');
                    return;
                }
                try {
                    if (a === 'probe')  { runProbe(); paintAll(); }
                    if (a === 'pixels') runPixelProbe();
                    if (a === 'frame')  frameScene();
                    if (a === 'test')   injectTestScene();
                    if (a === 'reload') { const f = look('loadSpatialData'); if (typeof f === 'function') { resetMarks(); f(); } else { log('warn', 'loadSpatialData is not reachable — bind it via AeroDebug.bind({ loadSpatialData: () => loadSpatialData }).'); } }
                    if (a === 'dump')   copyDump();
                } catch (e) {
                    log('error', `Action '${a}' failed: ${e.message}`);
                }
            });
        });

        // Log-scale slider: a linear 0.05–3 range cannot express the factors
        // that actually matter when a scale encoding is wrong, which are
        // powers of ten. Exponent −2…2 covers ×0.01 to ×100.
        const slider = b.querySelector('#adbg-scale');
        const out = b.querySelector('#adbg-scale-out');
        slider.addEventListener('input', () => {
            const v = Math.pow(10, parseFloat(slider.value));
            out.textContent = v < 10 ? v.toFixed(2) : v.toFixed(0);
            setUniform('uScaleMultiplier', v);
            state.overridesActive.scale = v;
            if (state.autoProbe) { runProbe(); paintSection('probe'); paintVerdict(); }
        });

        b.querySelectorAll('input[data-tog]').forEach(cb => {
            cb.addEventListener('change', () => {
                const on = cb.checked;
                if (cb.dataset.tog === 'axes')   toggleAxes(on);
                if (cb.dataset.tog === 'bounds') toggleBounds(on);
                if (cb.dataset.tog === 'opaque') {
                    state.overridesActive.opaque = on;
                    log('info', on
                        ? 'Forcing α=1 is a shader change; wire uForceOpaque into fragmentShaderGS to use it.'
                        : 'α override off.');
                }
                if (cb.dataset.tog === 'autoprobe') state.autoProbe = on;
            });
        });
    }

    // ── Painters ───────────────────────────────────────────────────
    function paintVerdict() {
        if (!root) return;
        const v = state.verdict;
        const box = root.querySelector('#adbg-verdict');
        box.className = 'adbg-verdict ' + v.level;
        const tagText = { ok: 'OK', warn: 'CHECK', bad: 'FAULT', idle: 'IDLE' }[v.level];
        box.innerHTML = `<span class="tag">${tagText}</span>${v.text}`;
    }

    function paintProbe() {
        const b = body('probe');
        const p = state.probe;
        if (!p) {
            b.innerHTML = `<div style="color:#5d8570">The probe re-runs the vertex shader's projection on the CPU for a sample of splats, so you can see what the GPU was given. Press "Run probe".</div>`;
            return;
        }
        const pct = (n) => `${n} <span style="color:#46705a">(${(100 * n / p.tested).toFixed(1)}%)</span>`;

        // Colour the overdraw factor by what it actually predicts: too low
        // means the scene may be nearly invisible; too high (combined with
        // a low measured fps, if we have one) means fill rate is plausibly
        // why the frame rate is low. Neither is a fixed universal threshold
        // — a dense aerial 3DGS scene legitimately overdraws a lot even
        // when it renders fine — so this is advisory, not a hard fault.
        let overdrawCls = 'ok';
        if (p.overdrawMedian < 0.02) overdrawCls = 'warn';
        else if (p.overdrawMedian > 15 && state.fps > 0 && state.fps < 30) overdrawCls = 'warn';

        b.innerHTML = rows([
            ['splats sampled', `${p.tested} of ${state.payload.meta.count}`],
            ['behind camera', pct(p.behind), p.behind === p.tested ? 'bad' : ''],
            ['inside frustum', pct(p.onScreen), p.onScreen === 0 ? 'bad' : (p.onScreenFrac < 0.02 ? 'warn' : 'ok')],
            ['sub-pixel (&lt;0.5px)', pct(p.subPixel), p.subPixel > p.tested * 0.9 ? 'bad' : ''],
            ['α below 1/255', pct(p.alphaDead), p.alphaDead > p.tested * 0.9 ? 'bad' : ''],
            ['non-finite', pct(p.degenerate), p.degenerate ? 'bad' : ''],
            ['radius px p05/p50/p95', `${num(p.radius.p05, 2)} / ${num(p.radius.p50, 2)} / ${num(p.radius.p95, 2)}`],
            ['radius px max', num(p.radius.max, 1)],
            ['peak α p05/p50/p95', `${num(p.alpha.p05, 3)} / ${num(p.alpha.p50, 3)} / ${num(p.alpha.p95, 3)}`],
            ['view depth near/p50/far', `${num(p.depth.near, 1)} / ${num(p.depth.p50, 1)} / ${num(p.depth.far, 1)}`],
            ['overdraw factor (median)', `${num(p.overdrawMedian, 1)}×`, overdrawCls],
            ['overdraw factor (mean)', `${num(p.overdrawMean, 1)}×`,
                p.overdrawMean > p.overdrawMedian * 5 ? 'warn' : '']
        ]);
        if (p.overdrawMean > p.overdrawMedian * 5) {
            b.innerHTML += `<div style="color:#46705a;margin-top:6px">Mean is far above median: a handful of huge near-camera splats (see radius max, above) are dominating the mean — the median is the more representative "typical" number here.</div>`;
        }
    }

    function paintPixels() {
        const b = body('pixels');
        const p = state.pixelProbe;
        if (!p) {
            b.innerHTML = `<div style="color:#5d8570">Reads the centre of the framebuffer and counts pixels that differ from the clear colour. Separates "nothing was drawn" from "something was drawn invisibly".</div>`;
            return;
        }
        b.innerHTML = rows([
            ['region', p.region],
            ['pixels ≠ clear', `${p.differing} / ${p.total}`, p.differing === 0 ? 'bad' : 'ok'],
            ['fraction', `${(p.frac * 100).toFixed(2)}%`],
            ['mean RGB', p.mean.map(v => v.toFixed(1)).join(', ')],
            ['clear RGB', p.clear.join(', ')]
        ]);
    }

    function paintData() {
        const b = body('data');
        const pl = state.payload;
        if (!pl) { b.innerHTML = '<div style="color:#5d8570">No payload recorded.</div>'; return; }
        const m = pl.meta, s = pl.stats;

        let html = rows([
            ['dataset / mode', `${m.dataset} · ${m.mode}${pl.synthetic ? ' · SYNTHETIC' : ''}`],
            ['step', m.step],
            ['count', (m.count || 0).toLocaleString()],
            ['padded count', (m.padded_count || 0).toLocaleString()],
            ['stride floats', m.stride_floats],
            ['payload bytes', `${(pl.floatArr.byteLength / 1048576).toFixed(1)} MB`],
            ['dropped non-finite', m.dropped_nonfinite, m.dropped_nonfinite ? 'warn' : ''],
            ['dropped outliers', m.dropped_outliers, m.dropped_outliers ? 'warn' : ''],
            ['frame source', m.frame_source],
            ['centre (metric)', (m.center || []).map(v => num(v, 2)).join(', ')],
            ['extent (metric)', num(m.extent, 3)],
            ['scene_scale', num(m.scene_scale, 5)],
            ['viewer radius', num((m.extent || 0) * (m.scene_scale || 0), 2)],
            ['activations', m.activations ? `log=${m.activations.scale_is_log} logit=${m.activations.opacity_is_logit}` : '—']
        ]);

        if (pl.mode === 'splat' && s) {
            const scaleCls = !Number.isFinite(s.scaleVsExtent) ? ''
                : (s.scaleVsExtent < 1e-6 || s.scaleVsExtent > 0.5) ? 'bad'
                : (s.scaleVsExtent < 1e-5 || s.scaleVsExtent > 0.1) ? 'warn' : 'ok';
            html += `<div style="color:#00ff33;margin:9px 0 3px;letter-spacing:1px;font-size:10px">MEASURED ON THE WIRE (${s.sampled.toLocaleString()} sampled)</div>`;
            html += rows([
                ['x min/max', `${num(s.x.min, 2)} / ${num(s.x.max, 2)}`],
                ['y min/max', `${num(s.y.min, 2)} / ${num(s.y.max, 2)}`],
                ['z min/max', `${num(s.z.min, 2)} / ${num(s.z.max, 2)}`],
                ['opacity min/mean/max', `${num(s.opacity.min, 3)} / ${num(s.opacity.mean, 3)} / ${num(s.opacity.max, 3)}`,
                    (s.opacity.min < 0 || s.opacity.max > 1) ? 'bad' : ''],
                ['scale p01/p50/p99', `${num(s.scalePct.p01)} / ${num(s.scalePct.p50)} / ${num(s.scalePct.p99)}`],
                ['scale p50 ÷ extent', num(s.scaleVsExtent), scaleCls],
                ['negative scales', s.degenerate.negScale, s.degenerate.negScale ? 'bad' : 'ok'],
                ['zero scales', s.degenerate.zeroScale, s.degenerate.zeroScale ? 'warn' : ''],
                ['α &lt; 1/255 on wire', s.degenerate.opacityDead, s.degenerate.opacityDead > s.sampled * 0.5 ? 'bad' : ''],
                ['quat norm min/max', `${num(s.quat.min, 4)} / ${num(s.quat.max, 4)}`],
                ['zero-length quats', s.quat.zero, s.quat.zero ? 'bad' : 'ok'],
                ['near-identity quats', `${(s.quat.nearIdentityFrac * 100).toFixed(1)}%`],
                ['rgb min/max', `${num(Math.min(s.r.min, s.g.min, s.b.min), 3)} / ${num(Math.max(s.r.max, s.g.max, s.b.max), 3)}`]
            ]);
            html += `<div style="color:#46705a;margin-top:6px">A healthy 3DGS scene has scale p50 ÷ extent around 1e-4…1e-2. Far below means log-space scales reached the wire; far above means exp() was applied twice.</div>`;
        } else if (s) {
            html += rows([
                ['x min/max', `${num(s.x.min, 2)} / ${num(s.x.max, 2)}`],
                ['y min/max', `${num(s.y.min, 2)} / ${num(s.y.max, 2)}`],
                ['z min/max', `${num(s.z.min, 2)} / ${num(s.z.max, 2)}`],
                ['rgb max', num(Math.max(s.r.max, s.g.max, s.b.max), 3)]
            ]);
        }
        b.innerHTML = html;
    }

    function paintLoad() {
        const b = body('load');
        if (!state.marks.length) { b.innerHTML = '<div style="color:#5d8570">No load recorded.</div>'; return; }
        const total = state.marks[state.marks.length - 1].at - state.markStart;
        b.innerHTML = rows(state.marks.map(m => [
            m.label,
            `${m.delta.toFixed(1)} ms`,
            m.delta > 500 ? 'warn' : ''
        ]).concat([['<b>total</b>', `<b>${total.toFixed(1)} ms</b>`]]));
    }

    function paintFrame() {
        const b = body('frame');
        const r = ref.renderer;
        if (!r) { b.innerHTML = ''; return; }
        const info = r.info;
        b.innerHTML = rows([
            ['fps', num(state.fps, 1), state.fps < 20 ? 'warn' : 'ok'],
            ['frame time', `${num(1000 / Math.max(state.fps, 0.001), 1)} ms`],
            ['draw calls', info.render.calls],
            ['triangles', info.render.triangles.toLocaleString()],
            ['geometries', info.memory.geometries],
            ['textures', info.memory.textures],
            ['programs', info.programs ? info.programs.length : '—'],
            ['pixel ratio', num(r.getPixelRatio(), 2)],
            ['drawing buffer', `${r.getContext().drawingBufferWidth}×${r.getContext().drawingBufferHeight}`],
            ['context lost', state.contextLost ? 'YES' : 'no', state.contextLost ? 'bad' : 'ok'],
            ['last gl error', state.glErrors.length ? state.glErrors[state.glErrors.length - 1] : 'none',
                state.glErrors.length ? 'bad' : 'ok']
        ]);
    }

    function paintCamera() {
        const b = body('camera');
        const c = ref.camera, ctl = ref.controls;
        if (!c) { b.innerHTML = ''; return; }
        const d = ctl ? c.position.distanceTo(ctl.target) : NaN;
        const dir = new THREE.Vector3();
        c.getWorldDirection(dir);
        b.innerHTML = rows([
            ['position', `${num(c.position.x, 1)}, ${num(c.position.y, 1)}, ${num(c.position.z, 1)}`],
            ['target', ctl ? `${num(ctl.target.x, 1)}, ${num(ctl.target.y, 1)}, ${num(ctl.target.z, 1)}` : '—'],
            ['distance to target', num(d, 2)],
            ['view direction', `${num(dir.x, 2)}, ${num(dir.y, 2)}, ${num(dir.z, 2)}`],
            ['near / far', `${num(c.near, 3)} / ${num(c.far, 0)}`],
            ['far ÷ near', num(c.far / c.near, 0), (c.far / c.near) > 1e6 ? 'warn' : ''],
            ['fov / aspect', `${num(c.fov, 1)}° / ${num(c.aspect, 3)}`],
            ['sceneRoot scale', ref.sceneRoot ? num(ref.sceneRoot.scale.x, 5) : '—'],
            ['sceneRoot rot x', ref.sceneRoot ? `${num(THREE.MathUtils.radToDeg(ref.sceneRoot.rotation.x), 1)}°` : '—']
        ]);
    }

    function paintSort() {
        const b = body('sort');
        b.innerHTML = rows([
            ['generation', ref.sortGen],
            ['in flight', ref.sortBusy ? 'yes' : 'no'],
            ['sorts completed', state.sorts],
            ['last sort round trip', state.lastSortMs ? `${state.lastSortMs.toFixed(1)} ms` : '—',
                state.lastSortMs > 100 ? 'warn' : ''],
            ['— of which, worker time', state.lastWorkerMs ? `${state.lastWorkerMs.toFixed(1)} ms` : '—'],
            ['— postMessage/clone overhead', (state.lastSortMs && state.lastWorkerMs)
                ? `${Math.max(0, state.lastSortMs - state.lastWorkerMs).toFixed(1)} ms` : '—'],
            ['since last sort', state.lastSortAt ? `${((performance.now() - state.lastSortAt) / 1000).toFixed(1)} s` : '—']
        ]);
    }

    function paintGpu() {
        const b = body('gpu');
        const g = gpuInfo();
        const meta = ref.sceneMeta;
        const ceiling = Math.min(g.texelCeiling || 0, g.floatIndexCeiling || 0);
        b.innerHTML = rows([
            ['api', g.webgl2 ? 'WebGL 2' : 'WebGL 1'],
            ['renderer', g.renderer],
            ['vendor', g.vendor],
            ['float textures', g.floatTextures ? 'yes' : 'NO', g.floatTextures ? 'ok' : 'bad'],
            ['vertex texture units', g.maxVertexTextureUnits, g.maxVertexTextureUnits < 1 ? 'bad' : 'ok'],
            ['max texture size', g.maxTextureSize],
            ['max varyings', g.maxVaryingVectors],
            ['point size range', g.pointSizeRange],
            ['pixel ratio', num(g.pixelRatio, 2)],
            ['— splat ceiling, texture', (g.texelCeiling || 0).toLocaleString()],
            ['— splat ceiling, fp32 index', (g.floatIndexCeiling || 0).toLocaleString()],
            ['effective max splats', ceiling.toLocaleString()],
            ['headroom at current load', meta ? `${(ceiling / Math.max(meta.count, 1)).toFixed(1)}×` : '—']
        ]) + `<div style="color:#46705a;margin-top:6px">The texture ceiling is MAX_TEXTURE_SIZE × 512 splats per row. The index ceiling is 2²⁴ ÷ 4 texels, because aSplatIndex is a float attribute and the shader multiplies it by 4 before addressing. The lower of the two is the real limit.</div>`;
    }

    function paintLog() {
        if (!root) return;
        const b = body('log');
        if (!b) return;
        const wrap = b.querySelector('.adbg-log') || (() => {
            b.innerHTML = '<div class="adbg-log"></div>';
            return b.querySelector('.adbg-log');
        })();
        wrap.innerHTML = state.logs.slice(-120).reverse().map(l =>
            `<div class="lvl-${l.level}"><span class="t">${(l.t / 1000).toFixed(2)}s </span>${escapeHtml(l.msg)}</div>`
        ).join('');
    }

    function escapeHtml(s) {
        return String(s).replace(/[&<>]/g, c => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;' }[c]));
    }

    const PAINTERS = {
        probe: paintProbe, pixels: paintPixels, data: paintData, load: paintLoad,
        frame: paintFrame, camera: paintCamera, sort: paintSort, gpu: paintGpu, log: paintLog
    };

    function paintSection(id) {
        if (!root || !PAINTERS[id]) return;
        PAINTERS[id]();
    }

    function paintAll() {
        if (!root) return;
        paintVerdict();
        Object.keys(PAINTERS).forEach(paintSection);
    }

    function toggle(force) {
        state.open = force === undefined ? !state.open : force;
        if (root) root.style.display = state.open ? '' : 'none';
        if (tab) tab.style.display = state.open ? 'none' : '';
    }

    // ═══════════════════════════════════════════════════════════════
    //  Dump — one JSON blob with everything, for pasting into a bug
    //  report or a chat. Cheaper than screenshotting nine sections.
    // ═══════════════════════════════════════════════════════════════
    function dump() {
        return {
            when: new Date().toISOString(),
            url: location.href,
            userAgent: navigator.userAgent,
            gpu: gpuInfo(),
            verdict: state.verdict,
            meta: state.payload ? state.payload.meta : null,
            stats: state.payload ? state.payload.stats : null,
            probe: state.probe,
            pixelProbe: state.pixelProbe,
            marks: state.marks,
            camera: (() => {
                const c = ref.camera;
                if (!c) return null;
                return {
                    position: c.position.toArray(), near: c.near, far: c.far,
                    fov: c.fov, aspect: c.aspect,
                    target: ref.controls ? ref.controls.target.toArray() : null,
                    sceneRootScale: ref.sceneRoot ? ref.sceneRoot.scale.toArray() : null,
                    sceneRootRotation: ref.sceneRoot ? ref.sceneRoot.rotation.toArray().slice(0, 3) : null
                };
            })(),
            renderInfo: ref.renderer ? JSON.parse(JSON.stringify(ref.renderer.info)) : null,
            overrides: state.overridesActive,
            glErrors: state.glErrors,
            logs: state.logs.slice(-120)
        };
    }

    function copyDump() {
        const text = JSON.stringify(dump(), null, 2);
        if (navigator.clipboard) {
            navigator.clipboard.writeText(text)
                .then(() => log('info', `Diagnostic dump copied (${text.length} chars).`))
                .catch(() => { console.log(text); log('warn', 'Clipboard refused; dump written to console instead.'); });
        } else {
            console.log(text);
            log('warn', 'No clipboard API; dump written to console.');
        }
    }

    // ═══════════════════════════════════════════════════════════════
    //  Frame instrumentation
    // ═══════════════════════════════════════════════════════════════
    let uiTick = 0;
    function onFrame() {
        const now = performance.now();
        const dt = now - state.lastFrameT;
        state.lastFrameT = now;
        state.frames[state.frameIdx++ % FRAME_WINDOW] = dt;

        let sum = 0, n = 0;
        for (let i = 0; i < FRAME_WINDOW; i++) if (state.frames[i]) { sum += state.frames[i]; n++; }
        state.fps = n ? 1000 / (sum / n) : 0;

        // Repaint the cheap sections a few times a second, not per frame.
        if (state.open && ++uiTick % 20 === 0) {
            paintSection('frame');
            paintSection('camera');
            paintSection('sort');
            if (state.autoProbe) { runProbe(); paintSection('probe'); paintVerdict(); }
        }

        // Poll for GL errors sparsely — getError is a sync stall.
        if (uiTick % 240 === 0 && ref.renderer) {
            const gl = ref.renderer.getContext();
            const e = gl.getError();
            if (e !== gl.NO_ERROR) {
                const name = Object.keys(gl).find(k => gl[k] === e && /^[A-Z_]+$/.test(k)) || String(e);
                state.glErrors.push(name);
                log('error', `GL error: ${name}`);
            }
        }
    }

    // ═══════════════════════════════════════════════════════════════
    //  Public API
    // ═══════════════════════════════════════════════════════════════
    const AeroDebug = {
        // Optional: hand the panel explicit getters if the by-name lookup
        // fails (bundled build, module scope, renamed variables).
        bind(map) {
            Object.keys(map).forEach(k => { overrides[k] = typeof map[k] === 'function' ? map[k] : () => map[k]; });
            paintAll();
        },

        mark,
        resetMarks,
        log: (msg, data) => log('info', msg, data),
        warn: (msg, data) => log('warn', msg, data),
        error: (msg, data) => log('error', msg, data),

        // Call once per load, right after the ArrayBuffer is decoded.
        onPayload(floatArr, meta, mode) {
            let stats = null;
            try {
                stats = mode === 'splat' ? analyzeSplatPayload(floatArr, meta) : analyzeDensePayload(floatArr, meta);
            } catch (e) {
                log('error', `Payload analysis failed: ${e.message}`);
            }
            state.payload = { floatArr, meta, mode, stats };
            state.probe = null;
            deriveVerdict();
            paintAll();
            log('info', `Payload recorded: ${mode}, ${meta.count} records, ${(floatArr.byteLength / 1048576).toFixed(1)} MB.`);
        },

        onSortRequested() { state.sortRequestedAt = performance.now(); },
        onSortComplete(workerMs) {
            state.sorts++;
            state.lastSortAt = performance.now();
            if (state.sortRequestedAt) state.lastSortMs = state.lastSortAt - state.sortRequestedAt;
            if (typeof workerMs === 'number') state.lastWorkerMs = workerMs;
            paintSection('sort');
        },

        onFrame,
        probe: runProbe,
        pixelProbe: runPixelProbe,
        testScene: injectTestScene,
        frameScene,
        dump,
        copyDump,
        toggle,
        state
    };

    window.AeroDebug = AeroDebug;

    // ── Boot ───────────────────────────────────────────────────────
    function boot() {
        build();

        window.addEventListener('keydown', (e) => {
            if (e.key === 'F9' || (e.ctrlKey && e.shiftKey && e.key.toLowerCase() === 'd')) {
                e.preventDefault();
                toggle();
            }
        });

        // Context loss is the one failure that looks exactly like every
        // other black screen and produces no console output at all.
        const r = ref.renderer;
        if (r) {
            r.domElement.addEventListener('webglcontextlost', (e) => {
                e.preventDefault();
                state.contextLost = true;
                log('error', 'WebGL context lost.');
                deriveVerdict();
            });
            r.domElement.addEventListener('webglcontextrestored', () => {
                state.contextLost = false;
                log('info', 'WebGL context restored — reload the data.');
            });
        }

        window.addEventListener('unhandledrejection', (e) => {
            log('error', `Unhandled rejection: ${e.reason && e.reason.message ? e.reason.message : e.reason}`);
        });
        window.addEventListener('error', (e) => {
            log('error', `${e.message} (${e.filename}:${e.lineno})`);
        });

        log('info', 'Diagnostics ready. F9 toggles the panel.');
        deriveVerdict();   // catches the WebGL-failure case immediately, no payload needed
    }

    if (document.readyState === 'loading') {
        document.addEventListener('DOMContentLoaded', boot);
    } else {
        boot();
    }
})();
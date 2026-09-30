// ═══════════════════════════════════════════════════════════════════
//  AeroSplat-GIS — 3DGS WebGL Viewer
//  Справжній растерізатор еліпсів через GLSL
//
//  Rationale for the shader maths and failure handling is in docs/decisions/
//  (ADR-0004 shader correctness, ADR-0005 failure handling, ADR-0003 wire
//  format); the §-tags below are indexed in docs/decisions/README.md.
//
//  Splat data reaches the GPU as an RGBA32F texture, not as vertex
//  attributes. The server's 16-float record is exactly 4 RGBA texels
//  (see main.py's wire-format note), so the response body is uploaded
//  verbatim. Each instance carries one float — its splat index — which
//  is what the depth sort rewrites. Sorting therefore costs one float
//  per splat per update instead of reshuffling five attribute arrays.
// ═══════════════════════════════════════════════════════════════════

// review-2 §S3: this used to be `const container =
// document.getElementById('canvas-container');` with no guard. If the id
// is ever wrong, this throws at module top level and EVERY line below it
// — loadSpatialData, animate, the mode selector, showFatal itself — never
// runs. The symptom is indistinguishable from any other black screen:
// one console line and nothing else. Fall back to <body> so the rest of
// the script (and its own error reporting) survives.
// review-2 §S3/S4 follow-up: `let loadToken`/`lastFramedDataset` used to
// be declared much later, right before loadSpatialData's definition.
// `async function loadSpatialData(){}` is a hoisted function declaration
// — it's globally callable from the moment the script STARTS running,
// regardless of where its text sits. If anything earlier in the script
// throws (WebGL context creation, below, genuinely can), the script
// aborts and those later `let` lines never execute — but the function is
// already callable, and every call permanently throws "Cannot access
// 'loadToken' before initialization", because the declaration that would
// initialize it will never run. Declaring these two here, before
// anything that can throw, means loadSpatialData fails for the REAL
// reason (no renderer) instead of a confusing secondary TDZ error.
let loadToken = 0;
let lastFramedDataset;

// Set the moment ANY top-level statement below throws and aborts init.
// Checked by loadSpatialData/requestDepthSort/animate so a debug-panel
// click (or the automatic startup call) fails with one clear message
// instead of whatever half-initialized state happens to be lying around.
window.__AEROSPLAT_FATAL__ = false;

const container = document.getElementById('canvas-container') || document.body;
if (!document.getElementById('canvas-container')) {
    console.error('AeroSplat-GIS: #canvas-container not found in the DOM; falling back to <body>.');
}
const scene = new THREE.Scene();
scene.background = new THREE.Color(0x050a0f);

// ── Scene root: where the up-axis flip and scene scale live (§A3) ──
// COLMAP/trainer space is Z-up; WebGL is Y-up. The server used to apply
// (x, −z, y) to positions and leave the rotation quaternions in COLMAP
// space, misorienting every splat by 90° about X on top of §A1. Putting
// the same map here instead means mat3(modelViewMatrix) carries it into
// the covariance transform automatically.
const sceneRoot = new THREE.Group();
scene.add(sceneRoot);

const camera = new THREE.PerspectiveCamera(60, window.innerWidth / window.innerHeight, 0.1, 100000);
camera.position.set(0, 120, 180);

// MSAA resolves geometric edges; splat quads are alpha edges produced by
// discard, which MSAA does not touch. Supersampling the whole frame does
// help the soft Gaussian falloff, so render above native and let the
// browser downsample (§3.5).
const SUPERSAMPLE = 1.5;
const MAX_PIXEL_RATIO = 3.0;
const pixelRatioFor = () => Math.min(window.devicePixelRatio * SUPERSAMPLE, MAX_PIXEL_RATIO);

// review-2 follow-up: `new THREE.WebGLRenderer(...)` was completely
// unguarded. If the browser can't create a context — hardware
// acceleration disabled, an embedded/sandboxed preview without WebGL
// permission, a GPU driver on the blocklist, or simply too many WebGL
// contexts already open in this tab — Three.js throws synchronously and
// the ENTIRE script aborts mid-execution. Nothing below this line ever
// runs: no loadSpatialData, no animate, no mode selector, no fatal
// banner (showFatal itself is defined later in the file and never gets
// the chance to be called from its own error paths).
//
// This can't manufacture a WebGL context out of nothing — if the
// environment genuinely has none, the app cannot render, full stop. What
// this DOES do: try a couple of fallback context configurations before
// giving up (some environments reject the default options but accept a
// more conservative set), and on final failure, produce one clear,
// actionable diagnosis instead of a bare crash — and mark the failure so
// every other entry point (loadSpatialData, animate, the debug panel)
// can fail cleanly instead of cascading into confusing secondary errors.
function createRenderer() {
    const attempts = [
        { antialias: false },
        { antialias: false, powerPreference: 'default', failIfMajorPerformanceCaveat: false },
        { antialias: false, powerPreference: 'low-power', failIfMajorPerformanceCaveat: false },
    ];
    let lastError = null;
    for (const opts of attempts) {
        try {
            return new THREE.WebGLRenderer(opts);
        } catch (e) {
            lastError = e;
        }
    }
    throw lastError;
}

let renderer;
try {
    renderer = createRenderer();
} catch (err) {
    window.__AEROSPLAT_FATAL__ = true;
    console.error(
        'AeroSplat-GIS: could not create a WebGL context after multiple attempts ' +
        `(${err && err.message ? err.message : err}). This is a browser/environment ` +
        'issue, not a data or shader problem. Common causes, roughly in order of ' +
        'likelihood:\n' +
        '  • Chrome auto-blocklisted its GPU process after repeated software-GL ' +
        'crashes (chrome://gpu will say "GPU process was unable to boot: GPU ' +
        'process crashed too many times with software GL" if this is it — common ' +
        'in VMs/remote desktops/containers with no real GPU and a broken software ' +
        'rasterizer underneath). Fix: chrome://flags/#ignore-gpu-blocklist → ' +
        'Enabled → relaunch, or launch with ' +
        '--ignore-gpu-blocklist --use-gl=angle --use-angle=swiftshader-webgl. If ' +
        'that ALSO fails, the environment likely has no working OpenGL at all — on ' +
        'Linux check `glxinfo | grep "OpenGL renderer"` outside the browser.\n' +
        '  • Hardware acceleration disabled in browser settings.\n' +
        '  • Viewing this inside a sandboxed preview/iframe without WebGL permission.\n' +
        '  • Too many WebGL contexts already open in this tab/window (browsers cap ' +
        'this, typically ~16).\n' +
        'Try opening this page in a normal, non-embedded browser tab first, and try ' +
        'Firefox (about:support → Graphics) as a cross-check — if Firefox also has ' +
        'no WebGL, the problem is system-level, not Chrome-specific.'
    );
    showFatal(
        'Could not create a WebGL context — this is a browser/environment issue, ' +
        'not a bug in the scene data.\n\n' +
        'Try, in order:\n' +
        '  1. Open this page in a normal browser tab (not an embedded preview/iframe).\n' +
        '  2. Check chrome://gpu. If it says "GPU process crashed too many times with\n' +
        '     software GL", go to chrome://flags/#ignore-gpu-blocklist, set it to\n' +
        '     Enabled, and relaunch — or launch with:\n' +
        '     --ignore-gpu-blocklist --use-gl=angle --use-angle=swiftshader-webgl\n' +
        '  3. Try Firefox (about:support → Graphics) as a cross-check. If it also has\n' +
        '     no WebGL, this is system-level (no working OpenGL under the browser at\n' +
        '     all) — common in VMs/remote desktops/containers — not a Chrome setting.\n' +
        '  4. Enable "Use hardware acceleration" in browser settings, then restart.\n' +
        '  5. Close other tabs/apps using WebGL and reload — browsers cap concurrent contexts.\n\n' +
        'Full error: ' + (err && err.message ? err.message : err)
    );
    // Stop here. Everything below this line assumes a working renderer/
    // camera/scene and there is no meaningful degraded mode to fall back
    // to — a 3DGS viewer with no WebGL context cannot draw anything.
    throw err;
}
renderer.setSize(window.innerWidth, window.innerHeight);
renderer.setPixelRatio(pixelRatioFor());
container.appendChild(renderer.domElement);

// review-2 §S4: a lost WebGL context produces NO exception and NO
// console output — the canvas just stops updating. At 1.2M+ overdrawn
// splats with SUPERSAMPLE stacked on devicePixelRatio, this is a real
// failure mode, not a hypothetical one, and without this handler it is
// indistinguishable from every other silent black screen. This baseline
// handler works even without aero-debug.js loaded; the diagnostics panel
// additionally surfaces the state in its FRAME section.
renderer.domElement.addEventListener('webglcontextlost', (e) => {
    e.preventDefault();
    console.error('AeroSplat-GIS: WebGL context lost (likely a GPU driver timeout from overdraw). Reload, or lower SUPERSAMPLE / MAX_PIXEL_RATIO.');
    showFatal('WebGL context lost. The GPU driver dropped the context — usually caused by sustained overdraw. Try reloading, or add ?step=2 to the URL to render fewer splats.');
});
renderer.domElement.addEventListener('webglcontextrestored', () => {
    console.info('AeroSplat-GIS: WebGL context restored. Reloading scene data.');
    hasFramedCamera = false;
    loadSpatialData();
});

// ── Capability checks for the splat data texture ───────────────────
const glCtx = renderer.getContext();
const isWebGL2 = renderer.capabilities.isWebGL2;
const maxTextureSize = glCtx.getParameter(glCtx.MAX_TEXTURE_SIZE);
const maxVertexTextures = glCtx.getParameter(glCtx.MAX_VERTEX_TEXTURE_IMAGE_UNITS);

if (!isWebGL2 && !glCtx.getExtension('OES_texture_float')) {
    console.error('AeroSplat-GIS: float textures unavailable (no WebGL2, no OES_texture_float). Splat mode cannot run.');
}
if (maxVertexTextures < 1) {
    console.error('AeroSplat-GIS: this GPU exposes no vertex texture units. Splat mode cannot run.');
}

const controls = new THREE.OrbitControls(camera, renderer.domElement);
controls.enableDamping = true;
controls.dampingFactor = 0.05;

let activeMesh    = null;
let splatTexture  = null;
let currentMode   = new URLSearchParams(location.search).get('mode') || 'splat';
// Read once at load, same as apiUrl() already does for ?step= (see its
// comment: that param was previously write-only too -- "the advice was a
// dead end"). Without this, main.py's whole ?dataset= A/B-comparison
// design (see its own comments) was unreachable from the browser: nothing
// anywhere set currentDataset away from null, no matter what URL you used.
let currentDataset = new URLSearchParams(location.search).get('dataset') || null;
let sceneMeta     = null;
let hasFramedCamera = false;

// DEBUG: live-tunable splat-size multiplier. With §A1–§A4 fixed this
// should sit at 1.00 — it is no longer compensating for an orientation
// error, so a "correct" value other than 1.0 means something is still
// wrong upstream rather than something needing tuning.
let debugScaleMultiplier = 1.0;

const viewportPx = new THREE.Vector2();

// ═══════════════════════════════════════════════════════════════════
//  DEPTH SORT — back-to-front draw order for alpha-blended splats
//
//  Splats arrive in arbitrary order, so NormalBlending composites them
//  in whatever order the GPU rasterizes them — wrong wherever
//  translucent Gaussians overlap. A worker returns a view-space-depth
//  ordering which is written into the per-instance aSplatIndex
//  attribute.
//
//  §C3: requested on camera movement rather than every frame. The sort
//  is one frame stale at best, and during fast orbit the stale order
//  shows up as compositing shimmer on overlapping splats. Gating on
//  movement means the order is fresh whenever the camera settles, which
//  is when the user is actually looking.
// ═══════════════════════════════════════════════════════════════════
// review-2 §S3: `new Worker(...)` throws synchronously on a file:// origin
// or a blocked CSP path, which — unguarded — used to kill every line
// below it in the module, same failure class as the container lookup
// above. Guarded here, and given an onerror handler it never had: without
// one, a worker that fails partway through leaves sortInFlight stuck true
// forever and rendering silently stops sorting (the exact bug §C3 fixed
// for a different code path, reintroduced here for a different reason).
let sortWorker = null;
try {
    sortWorker = new Worker('depth-sort_worker.js');
    sortWorker.onerror = (e) => {
        console.error(`AeroSplat-GIS: depth-sort worker error (${e.message || e}). Rendering unsorted from here — expect compositing artifacts where splats overlap.`);
        sortInFlight = false;
        sortWorker = null;
    };
} catch (err) {
    console.error(`AeroSplat-GIS: could not start depth-sort worker (${err.message}). Rendering unsorted.`);
}
let   sortGeneration = 0;      // bumped each time splat data (re)loads
let   sortInFlight   = false;  // back-pressure: never queue a second job

const viewMatrixTemp = new THREE.Matrix4();
const modelViewTemp  = new THREE.Matrix4();
const currentViewDir = new THREE.Vector3();
const lastSortDir    = new THREE.Vector3();
const lastSortPos    = new THREE.Vector3();
let   haveSortRef    = false;

const SORT_ANGLE_EPS = 0.5 * Math.PI / 180;   // radians
const SORT_MOVE_FRAC = 0.005;                 // of scene radius

if (sortWorker) {
    sortWorker.onmessage = (e) => {
        sortInFlight = false;                     // cleared for 'sorted' and 'rejected' alike
        // Hook 5 (see aero-debug.js): reports the round trip, and
        // msg.workerMs (added to the worker) separates algorithm time
        // from postMessage/structured-clone overhead.
        window.AeroDebug && AeroDebug.onSortComplete(e.data.workerMs);
        const msg = e.data;
        if (msg.type !== 'sorted') return;
        if (msg.generation !== sortGeneration) return;   // stale reply from a previous dataset
        if (!activeMesh || currentMode !== 'splat') return;

        const orderAttr = activeMesh.geometry.getAttribute('aSplatIndex');
        if (!orderAttr || orderAttr.array.length !== msg.indices.length) return;

        orderAttr.array.set(msg.indices);
        orderAttr.needsUpdate = true;
    };
}

// ═══════════════════════════════════════════════════════════════════
//  GLSL: Vertex Shader — проєктує 3D Гаусіан в 2D еліпс
//
//  1. Читаємо параметри сплата з текстури за aSplatIndex
//  2. Трансформуємо центр в camera space
//  3. Будуємо 3D коваріацію з scale + quaternion
//  4. Проєктуємо через Якобіан → 2D матриця Σ'
//  5. Розкладаємо Σ' на власні осі й ставимо кут квада вздовж них
//  6. Передаємо Σ'⁻¹ і зміщення в пікселях у fragment shader
// ═══════════════════════════════════════════════════════════════════
const vertexShaderGS = `
    precision highp float;

    attribute vec2  aCorner;       // per-vertex quad corner, (±1, ±1)
    attribute float aSplatIndex;   // per-instance, rewritten by the depth sort

    uniform sampler2D uSplatTex;   // RGBA32F, 4 texels per splat
    uniform vec2  uTexSize;        // (width, height) in texels
    uniform vec2  uViewport;       // framebuffer size in pixels
    uniform float uScaleMultiplier;
    // review-2 §2.6: was a hardcoded -0.1 that matched camera.near only
    // by coincidence (frameCamera computes max(r*0.001, 0.01), which
    // happens to equal 0.1 at the default TARGET_SCENE_RADIUS=100).
    // Change the target radius and the two silently diverge. Now driven
    // by the same value the camera itself uses — set from frameCamera().
    uniform float uNearCull;

    varying vec3  vColor;
    varying float vOpacity;
    varying vec4  vCov2dInv;       // [a,b,c,d] → [[a,b],[c,d]]
    varying vec2  vDelta;          // offset from splat centre, pixels, y-up

    const float CUTOFF   = 3.5;    // sigma; matches CUTOFF_SQ in the fragment shader
    const float DILATION = 0.3;    // screen-space antialiasing filter, in px²

    // GLSL ES 1.00 has no transpose(); spell it out rather than rely on
    // a driver extending the language.
    mat3 transpose3(mat3 m) {
        return mat3(
            m[0][0], m[1][0], m[2][0],
            m[0][1], m[1][1], m[2][1],
            m[0][2], m[1][2], m[2][2]
        );
    }

    // ── Кватерніон → Матриця обертання 3×3 ───────────────────────
    // §A1: GLSL mat3(...) is COLUMN-major — each triple below is a
    // COLUMN of R. Laying the nine scalars out as rows (as this
    // previously did) silently builds Rᵀ: still a valid rotation, still
    // symmetric-positive-definite downstream, nothing NaNs, and every
    // anisotropic Gaussian points the wrong way.
    mat3 quatToMat(vec4 q) {
        q = normalize(q);
        float w = q.x, x = q.y, y = q.z, z = q.w;
        return mat3(
            // column 0
            1.0 - 2.0*(y*y + z*z),
            2.0*(x*y + w*z),
            2.0*(x*z - w*y),
            // column 1
            2.0*(x*y - w*z),
            1.0 - 2.0*(x*x + z*z),
            2.0*(y*z + w*x),
            // column 2
            2.0*(x*z + w*y),
            2.0*(y*z - w*x),
            1.0 - 2.0*(x*x + y*y)
        );
    }

    vec4 fetchTexel(float texelIndex) {
        float col = mod(texelIndex, uTexSize.x);
        float row = floor(texelIndex / uTexSize.x);
        return texture2D(uSplatTex, (vec2(col, row) + 0.5) / uTexSize);
    }

    void main() {
        // ── 0. Параметри сплата з текстури ───────────────────────
        float base = aSplatIndex * 4.0;
        vec4 t0 = fetchTexel(base);          // xyz, opacity
        vec4 t1 = fetchTexel(base + 1.0);    // scale.xyz, pad
        vec4 t2 = fetchTexel(base + 2.0);    // rotation (w,x,y,z)
        vec4 t3 = fetchTexel(base + 3.0);    // rgb, pad

        vColor = t3.rgb;

        // ── 1. Центр Гаусіана в camera space ─────────────────────
        vec4 camPos4 = modelViewMatrix * vec4(t0.xyz, 1.0);
        float z_c = camPos4.z;               // від'ємне перед камерою

        if (z_c >= -uNearCull) {              // за камерою — ховаємо
            vOpacity  = 0.0;
            vCov2dInv = vec4(0.0);
            vDelta    = vec2(0.0);
            gl_Position = vec4(0.0, 0.0, 2.0, 1.0);   // за far plane
            return;
        }

        // ── 2. 3D коваріація: Σ = R · S² · Rᵀ ───────────────────
        mat3 R = quatToMat(t2);
        vec3 s = t1.xyz * uScaleMultiplier;
        mat3 S2 = mat3(
            s.x*s.x, 0.0,     0.0,
            0.0,     s.y*s.y, 0.0,
            0.0,     0.0,     s.z*s.z
        );
        mat3 cov3d = R * S2 * transpose3(R);

        // Σ_cam = W · Σ · Wᵀ. W carries sceneRoot's up-axis rotation and
        // uniform scale, which is why neither needs to happen server-side
        // (§A3). Valid for any linear W without non-uniform scale/shear.
        mat3 W = mat3(modelViewMatrix);
        mat3 cov3d_cam = W * cov3d * transpose3(W);

        // ── 3. Якобіан перспективної проєкції ────────────────────
        // J = [[fx/z, 0, -fx·x/z²],
        //      [0, fy/z, -fy·y/z²]]
        float fx = projectionMatrix[0][0] * uViewport.x * 0.5;
        float fy = projectionMatrix[1][1] * uViewport.y * 0.5;

        // §B2: clamp the projected position before building J. Beyond
        // ~1.3× the frustum half-extent the x/z² and x²/z⁴ terms grow
        // without bound and smear enormous ellipses off the frame edges —
        // constant on an orbital aerial render, where a large fraction of
        // splats sit near the boundary at any moment. The clamp feeds the
        // Jacobian only; gl_Position still uses the true camPos4.
        float tan_fovx = 1.0 / projectionMatrix[0][0];
        float tan_fovy = 1.0 / projectionMatrix[1][1];
        float lim_x = 1.3 * tan_fovx * (-z_c);
        float lim_y = 1.3 * tan_fovy * (-z_c);
        float x_c = clamp(camPos4.x, -lim_x, lim_x);
        float y_c = clamp(camPos4.y, -lim_y, lim_y);

        float inv_z  = 1.0 / z_c;
        float inv_z2 = inv_z * inv_z;
        float fx2 = fx * fx;
        float fy2 = fy * fy;

        // Σ' = J · Σ_cam · Jᵀ (2×2), аналітично
        float cov2d_00 = fx2 * inv_z2 * cov3d_cam[0][0]
                       - 2.0 * fx2 * x_c * inv_z2 * inv_z * cov3d_cam[2][0]
                       + fx2 * x_c * x_c * inv_z2 * inv_z2 * cov3d_cam[2][2];

        float cov2d_11 = fy2 * inv_z2 * cov3d_cam[1][1]
                       - 2.0 * fy2 * y_c * inv_z2 * inv_z * cov3d_cam[2][1]
                       + fy2 * y_c * y_c * inv_z2 * inv_z2 * cov3d_cam[2][2];

        float cov2d_01 = fx * fy * inv_z2 * cov3d_cam[0][1]
                       - fx * fy * y_c * inv_z2 * inv_z * cov3d_cam[0][2]
                       - fx * fy * x_c * inv_z2 * inv_z * cov3d_cam[2][1]
                       + fx * fy * x_c * y_c * inv_z2 * inv_z2 * cov3d_cam[2][2];

        // ── 4. Дилатація + компенсація непрозорості ──────────────
        // The 0.3px dilation is the screen-space low-pass that stops
        // sub-pixel Gaussians aliasing into flicker, but it enlarges the
        // footprint without reducing peak alpha, making small/distant
        // splats systematically too opaque — read as haze and lost
        // contrast at altitude, where nearly every splat is small.
        // Compensate by the ratio of determinants so total integrated
        // alpha is preserved (Mip-Splatting, Yu et al. 2024) — §C2.
        float det_orig = cov2d_00 * cov2d_11 - cov2d_01 * cov2d_01;
        cov2d_00 += DILATION;
        cov2d_11 += DILATION;
        float det_dilated = cov2d_00 * cov2d_11 - cov2d_01 * cov2d_01;

        vOpacity = t0.w * sqrt(max(det_orig, 0.0) / max(det_dilated, 1e-9));

        // ── 5. Інверсія Σ' аналітично ─────────────────────────────
        // [a b]⁻¹ = 1/(ad-bc) · [d -b; -c a]
        float inv_det = 1.0 / max(det_dilated, 1e-6);
        vCov2dInv = vec4(
            cov2d_11 * inv_det,   // [0,0]
           -cov2d_01 * inv_det,   // [0,1]
           -cov2d_01 * inv_det,   // [1,0]
            cov2d_00 * inv_det    // [1,1]
        );

        // ── 6. Власні осі Σ' → орієнтований квад ─────────────────
        // §B3: the quad is fitted to the ellipse's own eigenbasis rather
        // than to its bounding square, so fill is spent on the ellipse.
        // That is what makes the 3.5σ cutoff essentially free.
        float mid  = 0.5 * (cov2d_00 + cov2d_11);
        float diff = 0.5 * (cov2d_00 - cov2d_11);
        float disc = sqrt(max(diff*diff + cov2d_01*cov2d_01, 0.0));
        float l1 = max(mid + disc, 1e-6);
        float l2 = max(mid - disc, 1e-6);

        // Eigenvector for l1. The (l1 − cov11, cov01) form is the one that
        // stays well-conditioned for an axis-aligned ellipse; guard the
        // circular case separately so normalize() never sees (0,0).
        vec2 major;
        if (abs(cov2d_01) < 1e-8) {
            major = (cov2d_00 >= cov2d_11) ? vec2(1.0, 0.0) : vec2(0.0, 1.0);
        } else {
            major = normalize(vec2(l1 - cov2d_11, cov2d_01));
        }
        vec2 minorAxis = vec2(-major.y, major.x);

        vec2 offsetPx = aCorner.x * major     * (CUTOFF * sqrt(l1))
                      + aCorner.y * minorAxis * (CUTOFF * sqrt(l2));

        // vDelta replaces gl_PointCoord and is already in the y-up pixel
        // convention Σ' was derived in, so §A2's sign flip is structurally
        // unnecessary here rather than merely fixed.
        vDelta = offsetPx;

        // ── 7. Проєкція центру + зміщення кута квада ─────────────
        // Offset in NDC, then multiply back by w so the GPU's own
        // perspective divide reproduces exactly the NDC we wanted and the
        // depth test still gets a correct z.
        vec4 clip = projectionMatrix * camPos4;
        float cw  = clip.w;
        vec3  ndc = clip.xyz / cw;
        ndc.xy += offsetPx / (uViewport * 0.5);
        gl_Position = vec4(ndc * cw, cw);
    }
`;

// ═══════════════════════════════════════════════════════════════════
//  GLSL: Fragment Shader — рендерить еліптичний Гаусіан
//
//  1. vDelta — зміщення від центру в пікселях (y-up)
//  2. d² = Δᵀ · Σ'⁻¹ · Δ  (Маланобіс)
//  3. G(x) = exp(-½ · d²)
//  4. Фінальний колір: RGB · opacity · G(x)
// ═══════════════════════════════════════════════════════════════════
const fragmentShaderGS = `
    precision highp float;

    varying vec3  vColor;
    varying float vOpacity;
    varying vec4  vCov2dInv;
    varying vec2  vDelta;

    // 3.5σ ≈ 0.5% of peak, versus 3σ's 1.1% — the old cutoff left a
    // visible hard circular edge where the Gaussian was truncated.
    const float CUTOFF_SQ = 12.25;

    // aero-debug.js debug modes: 0 = normal render. 1 = every placed quad
    // painted flat magenta with no cutoff and no alpha — if the screen
    // stays empty here, geometry never reached the rasterizer and the
    // fault is upstream of shading (wrong texture data, wrong transform,
    // culled instances); if it fills with magenta, placement is fine and
    // the image is being lost to size or opacity downstream. 2 = the
    // alpha that WOULD have been used, before the 1/255 discard, on a
    // green(opaque)-to-red(transparent) ramp — separates "too
    // transparent to see" from "correctly transparent".
    uniform int uDebugMode;

    void main() {
        vec2 d = vDelta;

        float mahal_sq = vCov2dInv.x * d.x * d.x
                       + (vCov2dInv.y + vCov2dInv.z) * d.x * d.y
                       + vCov2dInv.w * d.y * d.y;

        if (uDebugMode == 1) {
            gl_FragColor = vec4(1.0, 0.0, 1.0, 1.0);
            return;
        }

        if (mahal_sq > CUTOFF_SQ) discard;

        // Clamp at 0.99 as the reference does, so no single splat can
        // fully saturate transmittance on its own.
        float alpha = min(0.99, vOpacity * exp(-0.5 * mahal_sq));

        if (uDebugMode == 2) {
            gl_FragColor = vec4(1.0 - alpha, alpha, 0.0, 1.0);
            return;
        }

        // The old 0.02 floor bit hardest on the low-opacity splats that
        // carry soft detail — foliage, roof texture, atmospheric depth —
        // and with heavy aerial overdraw thousands of discarded 0.015
        // contributions add up to real missing density. 1/255 is below
        // 8-bit representability, so it is genuinely free.
        if (alpha < (1.0 / 255.0)) discard;

        gl_FragColor = vec4(vColor, alpha);
    }
`;

// ═══════════════════════════════════════════════════════════════════
//  DENSE CLOUD Шейдери (Варіант А — простий point cloud)
// ═══════════════════════════════════════════════════════════════════
const vertexShaderDense = `
    precision highp float;

    attribute vec3 aColor;
    varying vec3 vColor;

    uniform vec2  uViewport;
    uniform float uMaxPointSize;   // driver's real ALIASED_POINT_SIZE_RANGE max
    uniform float uPointSpacing;   // neighbour spacing, in *post-sceneRoot* units

    void main() {
        vColor = aColor;
        vec4 mvPosition = modelViewMatrix * vec4(position, 1.0);

        // Project one sample-spacing's worth of world-space size through
        // the camera — same fx-based projection the 3DGS shader uses for
        // covariance — so point size grows physically with proximity and
        // keeps covering the gaps between neighbouring samples.
        float fx = projectionMatrix[0][0] * uViewport.x * 0.5;
        float pixelSize = (uPointSpacing * fx) / max(-mvPosition.z, 0.001);

        gl_PointSize = clamp(pixelSize, 1.0, uMaxPointSize);
        gl_Position  = projectionMatrix * mvPosition;
    }
`;

const fragmentShaderDense = `
    precision highp float;   // §3.5: defaulted to mediump in WebGL1 → colour banding

    varying vec3 vColor;
    void main() {
        vec2 coord = gl_PointCoord - vec2(0.5);
        if (dot(coord, coord) > 0.25) discard;
        gl_FragColor = vec4(vColor, 1.0);
    }
`;

// ═══════════════════════════════════════════════════════════════════
//  UI
// ═══════════════════════════════════════════════════════════════════
function setHudText(id, text) {
    // §3.5: an absent HUD element used to throw here, and the throw was
    // swallowed by loadSpatialData's outer try — aborting the entire load
    // with nothing but a console message.
    const el = document.getElementById(id);
    if (el) el.innerText = text;
}

function showFatal(message) {
    // A failed load used to leave a black screen and a console line. The
    // HUD's own #render-info doesn't exist until injectModeSelector runs,
    // so errors need somewhere unconditional to go.
    let el = document.getElementById('fatal-banner');
    if (!el) {
        el = document.createElement('div');
        el.id = 'fatal-banner';
        el.style.cssText = `
            position: fixed; bottom: 10px; left: 10px; right: 10px;
            background: rgba(40, 0, 0, 0.92); color: #ff5555;
            border: 1px solid #ff3333; padding: 10px; z-index: 10000;
            font-family: monospace; font-size: 12px; white-space: pre-wrap;
        `;
        document.body.appendChild(el);
    }
    el.innerText = message;
}

function injectModeSelector() {
    if (document.getElementById('mode-selector-wrapper')) return;

    const wrapper = document.createElement('div');
    wrapper.id = 'mode-selector-wrapper';
    wrapper.style.cssText = `
        position: fixed; top: 155px; left: 10px;
        background: rgba(5, 10, 15, 0.9);
        border: 1px solid #00ff33; padding: 10px;
        border-radius: 4px; z-index: 9999; width: 322px;
        box-shadow: 0 0 15px rgba(0,255,51,0.15);
    `;
    wrapper.innerHTML = `
        <label style="color:#00ff33;font-family:monospace;display:block;margin-bottom:6px;font-size:11px;font-weight:bold;letter-spacing:1px;">
            РУШІЙ РЕНДЕРИНГУ:
        </label>
        <select id="mode-select-dropdown" style="background:#000;color:#00ff33;border:1px solid #00ff33;font-family:monospace;padding:6px;width:100%;cursor:pointer;font-size:12px;outline:none;">
            <option value="splat" ${currentMode === 'splat' ? 'selected' : ''}>Варіант Б: 3DGS (Еліптичні Гаусіани)</option>
            <option value="dense" ${currentMode === 'dense' ? 'selected' : ''}>Варіант А: Dense Cloud (MVS)</option>
        </select>
        <div id="render-info" style="color:#666;font-family:monospace;font-size:10px;margin-top:6px;">
            Рендер: еліптичний Гаусіан · Маланобіс · alpha blending
        </div>
        <div id="scale-debug-row" style="margin-top:10px; ${currentMode === 'splat' ? '' : 'display:none;'}">
            <label style="color:#00ff33;font-family:monospace;display:block;margin-bottom:4px;font-size:11px;font-weight:bold;letter-spacing:1px;">
                DEBUG: SCALE ×<span id="scale-debug-val">1.00</span>
            </label>
            <input id="scale-debug-slider" type="range" min="-2" max="2" step="0.02" value="0"
                style="width:100%; cursor:pointer;">
            <div style="color:#666;font-family:monospace;font-size:9px;margin-top:2px;">
                Має лишатись 1.00 — інакше проблема вище по конвеєру
            </div>
        </div>
    `;
    document.body.appendChild(wrapper);

    const scaleDebugRow    = document.getElementById('scale-debug-row');
    const scaleDebugSlider = document.getElementById('scale-debug-slider');

    // review-2 §10: linear 0.05-3.0 cannot express the factors that
    // actually matter here — an un-exponentiated log-scale or a double
    // exp() are errors of 10x, 100x, or more, not 2x. The slider now
    // spans an exponent from -2 to 2, i.e. x0.01 to x100, so those
    // failure modes are actually reachable by dragging it.
    scaleDebugSlider.addEventListener('input', (e) => {
        debugScaleMultiplier = Math.pow(10, parseFloat(e.target.value));
        setHudText('scale-debug-val', debugScaleMultiplier < 10 ? debugScaleMultiplier.toFixed(2) : debugScaleMultiplier.toFixed(0));
        if (activeMesh && activeMesh.material.uniforms && activeMesh.material.uniforms.uScaleMultiplier) {
            activeMesh.material.uniforms.uScaleMultiplier.value = debugScaleMultiplier;
        }
    });

    document.getElementById('mode-select-dropdown').addEventListener('change', (e) => {
        currentMode = e.target.value;
        setHudText('render-info', currentMode === 'splat'
            ? 'Рендер: еліптичний Гаусіан · Маланобіс · alpha blending'
            : 'Рендер: point cloud · MVS реконструкція');
        scaleDebugRow.style.display = currentMode === 'splat' ? '' : 'none';
        loadSpatialData();
    });
}

// ═══════════════════════════════════════════════════════════════════
//  SCENE TRANSFORM (§A3)
// ═══════════════════════════════════════════════════════════════════
function applySceneTransform(meta) {
    const rot = meta.up_axis_rotation;
    sceneRoot.rotation.set(0, 0, 0);
    if (rot) {
        const rad = THREE.MathUtils.degToRad(rot.degrees);
        if      (rot.axis === 'x') sceneRoot.rotation.x = rad;
        else if (rot.axis === 'y') sceneRoot.rotation.y = rad;
        else if (rot.axis === 'z') sceneRoot.rotation.z = rad;
    }

    // Uniform, and the same for every mode — which is what removes the
    // old dense-vs-splat 10× discrepancy (§B4) and leaves room for TRIPS
    // output to land in the same frame without a third magic constant.
    sceneRoot.scale.setScalar(meta.scene_scale);
    sceneRoot.updateMatrixWorld(true);

    // mat3(modelViewMatrix) is only a valid covariance transform when the
    // object matrix holds rotation and *uniform* scale — no shear, no
    // per-axis scaling. Assert it rather than discover it in the render.
    const sc = sceneRoot.scale;
    if (Math.abs(sc.x - sc.y) > 1e-9 || Math.abs(sc.y - sc.z) > 1e-9) {
        console.error('AeroSplat-GIS: sceneRoot scale is non-uniform; the shader\'s W·Σ·Wᵀ covariance transform is no longer valid.');
    }
}

function frameCamera(meta) {
    // The scene is centred on the origin by the server and scaled to a
    // fixed radius here, so framing is mode-independent — switching modes
    // deliberately does *not* move the camera any more.
    const r = meta.extent * meta.scene_scale;
    const nearCullFraction = meta.near_cull_fraction || 0.001;   // review-2 §2.6
    controls.target.set(0, 0, 0);
    camera.position.set(0, r * 1.2, r * 1.8);
    camera.near = Math.max(r * nearCullFraction, 0.01);
    camera.far  = r * 1000.0;
    camera.updateProjectionMatrix();
    camera.lookAt(0, 0, 0);
    controls.update();

    // Keep the shader's near-plane cull locked to the same value, in
    // case frameCamera runs again (dataset switch) after the mesh's
    // uniforms were already set at build time.
    if (activeMesh && activeMesh.material.uniforms && activeMesh.material.uniforms.uNearCull) {
        activeMesh.material.uniforms.uNearCull.value = camera.near;
    }
}

// ═══════════════════════════════════════════════════════════════════
//  MESH BUILDERS
// ═══════════════════════════════════════════════════════════════════
function buildSplatMesh(floatArr, meta) {
    const N       = meta.count;
    const stride  = meta.stride_floats;      // 16
    const texW    = meta.texture_width;      // 2048 texels
    const perRow  = meta.splats_per_row;     // 512 splats

    if (floatArr.length % stride !== 0) {
        throw new Error(`splat payload is ${floatArr.length} floats, not a multiple of ${stride}`);
    }
    const paddedCount = floatArr.length / stride;
    const texH = paddedCount / perRow;
    if (!Number.isInteger(texH)) {
        throw new Error(`splat payload is not padded to whole texture rows (${paddedCount} splats, ${perRow}/row)`);
    }
    if (texH > maxTextureSize) {
        throw new Error(`splat texture needs ${texH} rows but MAX_TEXTURE_SIZE is ${maxTextureSize} — reload with ?step=2`);
    }

    // The server's record layout is exactly 4 RGBA texels, so the response
    // body is uploaded verbatim — no de-interleaving pass (§3.2).
    splatTexture = new THREE.DataTexture(
        floatArr, texW, texH, THREE.RGBAFormat, THREE.FloatType
    );
    splatTexture.minFilter = THREE.NearestFilter;
    splatTexture.magFilter = THREE.NearestFilter;
    splatTexture.generateMipmaps = false;
    splatTexture.needsUpdate = true;

    // ── Instanced unit quad (§B3) ────────────────────────────────
    // Replaces THREE.Points. Three reasons, all fidelity rather than
    // speed: point sprites are clipped entirely once their centre leaves
    // the viewport; gl_PointSize is capped by the driver, truncating the
    // Gaussian tail on near splats; and a quad can be oriented to the
    // ellipse's own eigenbasis.
    const geometry = new THREE.InstancedBufferGeometry();
    geometry.setAttribute('aCorner', new THREE.BufferAttribute(
        new Float32Array([-1, -1,  1, -1,  1, 1,  -1, 1]), 2));
    geometry.setIndex([0, 1, 2, 0, 2, 3]);

    // Draw order. Identity until the first sort comes back, then
    // continuously rewritten in place — one float per splat, versus
    // reshuffling five attribute arrays.
    const order = new Float32Array(N);
    for (let i = 0; i < N; i++) order[i] = i;
    const orderAttr = new THREE.InstancedBufferAttribute(order, 1);
    orderAttr.setUsage(THREE.DynamicDrawUsage);
    geometry.setAttribute('aSplatIndex', orderAttr);
    geometry.instanceCount = N;

    // review-2 §2.1/§3: the vertex shader addresses TEXELS via
    // `aSplatIndex * 4.0`, so it's `base`, not `aSplatIndex`, that must
    // stay exactly representable in float32. The real ceiling on splat
    // COUNT is 2^24 / 4 = 4,194,304 — checking against 2^24 (as the
    // worker used to) would let a scene get 4x too big before warning,
    // at which point indices round and splats silently read each
    // other's parameters. Prefer the server's own count if scene-meta
    // supplied one, so client and server can't disagree.
    const MAX_ADDRESSABLE_SPLATS = meta.max_addressable_splats || ((2 ** 24) / 4);
    if (N > MAX_ADDRESSABLE_SPLATS) {
        console.error(
            `AeroSplat-GIS: ${N.toLocaleString()} splats exceeds the fp32 texel-` +
            `addressing ceiling (${MAX_ADDRESSABLE_SPLATS.toLocaleString()}). ` +
            `aSplatIndex * 4.0 will round and splats will read each other's data. ` +
            `Reload with a larger ?step= value.`
        );
    }

    // Hand the worker its own strided copy of the positions (transferred,
    // zero-copy) so it can sort without touching GPU-visible memory.
    const positionsForWorker = new Float32Array(N * 3);
    for (let i = 0; i < N; i++) {
        const s = i * stride;
        positionsForWorker[i * 3]     = floatArr[s];
        positionsForWorker[i * 3 + 1] = floatArr[s + 1];
        positionsForWorker[i * 3 + 2] = floatArr[s + 2];
    }
    sortGeneration++;
    sortInFlight = false;
    haveSortRef  = false;
    if (sortWorker) {
        sortWorker.postMessage(
            { type: 'init', positions: positionsForWorker, generation: sortGeneration },
            [positionsForWorker.buffer]
        );
    }

    // review-2 §2.6: near_cull_fraction comes from scene-meta so the
    // shader's cull plane and camera.near are derived from the same
    // number instead of matching by coincidence (frameCamera sets
    // camera.near the same way, from the same meta).
    const nearCullFraction = meta.near_cull_fraction || 0.001;
    const sceneRadius = meta.extent * meta.scene_scale;

    const material = new THREE.ShaderMaterial({
        vertexShader:   vertexShaderGS,
        fragmentShader: fragmentShaderGS,
        transparent:    true,
        blending:       THREE.NormalBlending,
        depthWrite:     false,
        depthTest:      true,
        side:           THREE.DoubleSide,   // quad winding depends on the ellipse basis
        uniforms: {
            uSplatTex:        { value: splatTexture },
            uTexSize:         { value: new THREE.Vector2(texW, texH) },
            uViewport:        { value: viewportPx.clone() },
            uScaleMultiplier: { value: debugScaleMultiplier },
            uNearCull:        { value: Math.max(sceneRadius * nearCullFraction, 0.01) },
            uDebugMode:       { value: 0 }   // aero-debug.js debug modes
        }
    });

    const mesh = new THREE.Mesh(geometry, material);
    // Positions live in the texture, so there is no 'position' attribute
    // to compute a bounding sphere from. Framing comes from scene-meta.
    mesh.frustumCulled = false;
    return mesh;
}

function buildDenseMesh(floatArr, meta) {
    const stride = 6;
    if (floatArr.length % stride !== 0) {
        throw new Error(`dense payload is ${floatArr.length} floats, not a multiple of ${stride}`);
    }
    const N = Math.floor(floatArr.length / stride);

    // §3.2: the server already sends the layout the GPU wants. Bind views
    // into the single ArrayBuffer instead of copying every point into two
    // fresh arrays.
    const geometry    = new THREE.BufferGeometry();
    const interleaved = new THREE.InterleavedBuffer(floatArr, stride);
    geometry.setAttribute('position', new THREE.InterleavedBufferAttribute(interleaved, 3, 0));
    geometry.setAttribute('aColor',   new THREE.InterleavedBufferAttribute(interleaved, 3, 3));

    // Point spacing, estimated as if N samples were spread over a sphere
    // of the scene's radius — MVS output is essentially a shell over
    // visible surfaces. Computed in *scaled* units, because mvPosition in
    // the shader has already been through sceneRoot: a size quantity has
    // to follow the position transform, which is the same class of bug
    // §A3 describes for rotation.
    const R = meta.extent * meta.scene_scale;
    const COVERAGE = 1.5;   // so neighbours overlap slightly rather than just touch
    const spacing = Math.sqrt((4.0 * Math.PI * R * R) / Math.max(N, 1)) * COVERAGE;

    const pointSizeRange = glCtx.getParameter(glCtx.ALIASED_POINT_SIZE_RANGE);

    const material = new THREE.ShaderMaterial({
        vertexShader:   vertexShaderDense,
        fragmentShader: fragmentShaderDense,
        transparent:    false,
        depthWrite:     true,
        depthTest:      true,
        uniforms: {
            uViewport:     { value: viewportPx.clone() },
            uMaxPointSize: { value: pointSizeRange[1] },
            uPointSpacing: { value: spacing }
        }
    });

    return new THREE.Points(geometry, material);
}

// ═══════════════════════════════════════════════════════════════════
//  ЗАВАНТАЖЕННЯ ДАНИХ
// ═══════════════════════════════════════════════════════════════════
function apiUrl(path) {
    const q = new URLSearchParams({ mode: currentMode });
    if (currentDataset) q.set('dataset', currentDataset);
    // review-2 §5: buildSplatMesh's texture-row error tells the user to
    // "reload with ?step=2", but nothing previously read that parameter
    // back out of the URL — the advice was a dead end. Also honours it
    // as a manual overdraw/fill-rate control (see review-2 §3).
    const step = new URLSearchParams(location.search).get('step');
    if (step) q.set('step', step);
    return `${path}?${q.toString()}`;
}

function disposeActiveMesh() {
    if (!activeMesh) return;
    sceneRoot.remove(activeMesh);
    activeMesh.geometry.dispose();
    activeMesh.material.dispose();
    activeMesh = null;
    if (splatTexture) {
        splatTexture.dispose();   // ShaderMaterial.dispose() does not free textures
        splatTexture = null;
    }
}

// review-2 §3: loadSpatialData is async and re-entrant (every dropdown
// change calls it), with no token guarding against overlap. Switch modes
// twice quickly and two loads race; whichever `await` resolves last wins
// disposeActiveMesh()/applySceneTransform() regardless of which one the
// user actually asked for last, and a splat mesh can end up built against
// dense meta (or vice versa) — the "not a multiple of stride" throw, with
// no mesh left standing afterward. (loadToken/lastFramedDataset are
// declared at the very top of the file now — see the note there.)

async function loadSpatialData() {
    if (window.__AEROSPLAT_FATAL__) {
        console.error('AeroSplat-GIS: cannot load — initialization failed earlier (see the WebGL error above). Fix that first; reloading data will not help.');
        return;
    }
    const token = ++loadToken;
    const stale = () => token !== loadToken;

    window.AeroDebug && AeroDebug.resetMarks();
    window.AeroDebug && AeroDebug.mark('start');

    try {
        // Meta first: it carries the transform, so the scene is placed
        // before any geometry exists rather than reframed afterwards.
        const metaResp = await fetch(apiUrl('/api/v1/scene-meta'));
        if (stale()) return;   // a newer load superseded this one
        if (!metaResp.ok) throw new Error(`scene-meta HTTP ${metaResp.status}: ${await metaResp.text()}`);
        const meta = await metaResp.json();
        if (stale()) return;
        sceneMeta = meta;
        window.AeroDebug && AeroDebug.mark('scene-meta');

        if (meta.dropped_nonfinite || meta.dropped_outliers) {
            console.warn(
                `AeroSplat-GIS: server dropped ${meta.dropped_nonfinite} non-finite and ` +
                `${meta.dropped_outliers} outlier points from ${meta.mode}.`
            );
        }

        const resp = await fetch(apiUrl('/api/v1/spatial-chunk'));
        if (stale()) return;
        if (!resp.ok) throw new Error(`spatial-chunk HTTP ${resp.status}`);

        setHudText('latency-val', (resp.headers.get('X-Processing-Time-Ms') || '?') + ' ms');
        setHudText('count-val', meta.count.toLocaleString());

        const floatArr = new Float32Array(await resp.arrayBuffer());
        if (stale()) return;   // check after every await, not just some
        window.AeroDebug && AeroDebug.mark('payload download');

        console.info('[AeroSplat] scene-meta', meta);
        console.info(
            `[AeroSplat] ${meta.mode}: ${floatArr.length} floats, count=${meta.count}, ` +
            `scene_scale=${meta.scene_scale}, extent=${meta.extent}, ` +
            `webgl2=${isWebGL2}, maxTextureSize=${maxTextureSize}`
        );

        // Hook 2 (see aero-debug.js): the real bytes, analyzed. Every
        // claim the panel makes about scale encoding, opacity, quaternion
        // norms and projected size is measured here, not taken from
        // scene-meta's say-so.
        window.AeroDebug && AeroDebug.onPayload(floatArr, meta, currentMode);

        disposeActiveMesh();
        applySceneTransform(meta);

        activeMesh = (currentMode === 'splat')
            ? buildSplatMesh(floatArr, meta)
            : buildDenseMesh(floatArr, meta);

        activeMesh.material.uniforms.uViewport.value.copy(viewportPx);
        sceneRoot.add(activeMesh);
        window.AeroDebug && AeroDebug.mark('mesh build');

        // review-2 §S5: `hasFramedCamera` used to be a one-shot flag that
        // was never reset, so if the FIRST load ever failed partway (meta
        // ok, spatial-chunk 500), the camera stayed at its constructor
        // position forever, framed for nothing. Reframe whenever this is
        // truly the first successful load, OR the dataset actually
        // changed — a mode toggle within the same dataset should NOT
        // reframe (that's the whole point of the shared scene frame from
        // §A3/§B4), but switching datasets legitimately needs a new view.
        const datasetChanged = lastFramedDataset !== undefined && lastFramedDataset !== (currentDataset || meta.dataset);
        if (!hasFramedCamera || datasetChanged) {
            frameCamera(meta);
            hasFramedCamera = true;
        }
        lastFramedDataset = currentDataset || meta.dataset;

        injectModeSelector();
        window.AeroDebug && AeroDebug.probe();

    } catch (err) {
        if (stale()) return;   // a newer load already superseded this failure
        console.error('Помилка завантаження:', err);
        showFatal(`AeroSplat-GIS load failed (${currentMode}):\n${err.message}`);
    }
}

// ═══════════════════════════════════════════════════════════════════
//  RENDER LOOP
// ═══════════════════════════════════════════════════════════════════
function requestDepthSort() {
    if (!sortWorker || !activeMesh || currentMode !== 'splat' || sortInFlight) return;

    camera.updateMatrixWorld();
    camera.getWorldDirection(currentViewDir);

    const radius   = sceneMeta ? sceneMeta.extent * sceneMeta.scene_scale : 100.0;
    const turned   = haveSortRef && lastSortDir.angleTo(currentViewDir) > SORT_ANGLE_EPS;
    const moved    = haveSortRef && lastSortPos.distanceTo(camera.position) > radius * SORT_MOVE_FRAC;

    if (haveSortRef && !turned && !moved) return;

    sceneRoot.updateMatrixWorld(true);
    viewMatrixTemp.copy(camera.matrixWorld).invert();
    modelViewTemp.multiplyMatrices(viewMatrixTemp, activeMesh.matrixWorld);

    lastSortDir.copy(currentViewDir);
    lastSortPos.copy(camera.position);
    haveSortRef = true;

    sortInFlight = true;
    window.AeroDebug && AeroDebug.onSortRequested();   // Hook 5
    sortWorker.postMessage({
        type: 'sort',
        generation: sortGeneration,
        modelView: modelViewTemp.elements
    });
}

function animate() {
    if (window.__AEROSPLAT_FATAL__) return;   // no renderer to render with
    requestAnimationFrame(animate);
    controls.update();
    requestDepthSort();
    renderer.render(scene, camera);
    window.AeroDebug && AeroDebug.onFrame();
}

function handleResize() {
    camera.aspect = window.innerWidth / window.innerHeight;
    camera.updateProjectionMatrix();

    // §3.5: pixel ratio has to be re-applied — dragging the window between
    // monitors changes devicePixelRatio and it was never picked up.
    renderer.setPixelRatio(pixelRatioFor());
    renderer.setSize(window.innerWidth, window.innerHeight);

    // uViewport is constant between resizes; recompute it here instead of
    // 60×/second inside animate().
    viewportPx.set(
        window.innerWidth  * renderer.getPixelRatio(),
        window.innerHeight * renderer.getPixelRatio()
    );
    if (activeMesh && activeMesh.material.uniforms && activeMesh.material.uniforms.uViewport) {
        activeMesh.material.uniforms.uViewport.value.copy(viewportPx);
    }
}

window.addEventListener('resize', handleResize);

handleResize();        // seed viewportPx before the first load
injectModeSelector();  // built up front, so its absence isn't a load-failure signal
loadSpatialData();
animate();

// Hook 6 (see aero-debug.js): the panel resolves most viewer bindings
// by name (they're reachable as top-level const/let in this classic
// script's declarative record), which covers reads. These four need
// explicit handles: injectTestScene calls buildSplatMesh and
// disposeActiveMesh directly, needs a way to trigger a real reload, and
// needs to write activeMesh back after inserting its synthetic scene.
if (window.AeroDebug) {
    AeroDebug.bind({
        buildSplatMesh:    () => buildSplatMesh,
        disposeActiveMesh: () => disposeActiveMesh,
        loadSpatialData:   () => loadSpatialData,
        setActiveMesh:     () => (m) => { activeMesh = m; }
    });
}
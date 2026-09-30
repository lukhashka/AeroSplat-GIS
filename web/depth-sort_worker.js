// ═══════════════════════════════════════════════════════════════════
//  AeroSplat-GIS — Depth-sort worker
//
//  Alpha-blended 3DGS splats must be drawn back-to-front or the
//  compositing is wrong wherever splats overlap. This worker keeps a
//  copy of the point positions and, on request, returns a draw-order
//  index array sorted by view-space depth — off the main thread, so
//  it never stalls rendering or input.
//
//  Algorithm: counting sort on quantized depth (O(N), not O(N log N)).
//  Points are bucketed into BUCKETS depth bins spanning the current
//  min/max view-space z, then scattered into a sorted output array
//  via a prefix sum over bucket counts. Order within a bucket is
//  arbitrary (original point order) — the resulting error is bounded
//  by one bucket's worth of depth, which at 65536 buckets is
//  negligible for anything but pathological point distributions.
//
//  Two changes against AeroSplat-viewer-review.md §3.1:
//
//  §C3  Every 'sort' message is now answered, including rejections.
//       A silent return left the main thread's sortInFlight
//       back-pressure flag stuck true, and no further sort was ever
//       requested — a latent hang whose only symptom is "the splats
//       stopped sorting".
//  §3.1 The scratch buffers are hoisted to module scope and sized
//       once on init, instead of ~20 MB of fresh allocation per sort
//       at 1.2M splats.
//
//  Output type change: the sorted order is emitted as a Float32Array,
//  not Uint32Array, because the viewer now consumes it as a per-instance
//  vertex attribute (aSplatIndex) rather than as an element index
//  buffer — GLSL ES 1.00 attributes are floats. fp32 represents
//  integers exactly up to 2^24 = 16,777,216 — but the shader multiplies
//  aSplatIndex by 4 to get a texel index, so the real ceiling on splat
//  COUNT is 2^24 / 4 = 4,194,304 (review-2 §2.1). We warn past that,
//  not past 2^24, which used to let scenes get 4x too big before the
//  warning fired.
// ═══════════════════════════════════════════════════════════════════

let positions  = null;  // Float32Array, length N*3, mesh-local space
let count      = 0;
let generation = -1;    // must match the data currently loaded on main thread

const BUCKETS = 65536;

// ── fp32 addressing ceiling (review-2 §2.1) ────────────────────────
// The vertex shader does not address splats directly — it addresses
// TEXELS, via `base = aSplatIndex * 4.0`. It is `base` that must stay
// exactly representable in float32, not `aSplatIndex` itself. Warning
// on `count > 2^24` (as this used to) is 4x too permissive: a scene at
// 4.2M-16.7M splats would pass this check clean and then silently
// corrupt, because `aSplatIndex * 4.0` rounds before it reaches the
// texture fetch and splats start reading each other's parameters.
const TEXELS_PER_SPLAT    = 4;
const MAX_EXACT_FLOAT_INT = 16777216;                              // 2^24
const MAX_ADDRESSABLE     = MAX_EXACT_FLOAT_INT / TEXELS_PER_SPLAT; // 4,194,304

// ── Scratch, allocated once per init (review §3.1) ────────────────
// `sorted` is deliberately NOT hoisted: it is transferred to the main
// thread on every reply, so the worker loses ownership of its buffer.
let depths  = null;     // Float32Array(count)
let buckets = null;     // Uint32Array(count)
let counts  = null;     // Uint32Array(BUCKETS + 1)
let cursor  = null;     // Uint32Array(BUCKETS)

self.onmessage = function (e) {
    const msg = e.data;

    // ── New splat data loaded on main thread ──────────────────────
    if (msg.type === 'init') {
        positions  = msg.positions;       // transferred, we now own it
        count      = positions.length / 3;
        generation = msg.generation;

        depths  = new Float32Array(count);
        buckets = new Uint32Array(count);
        counts  = new Uint32Array(BUCKETS + 1);
        cursor  = new Uint32Array(BUCKETS);

        if (count > MAX_ADDRESSABLE) {
            console.warn(
                `depth-sort: ${count} splats exceeds ${MAX_ADDRESSABLE.toLocaleString()} ` +
                `(2^24 / 4 texels-per-splat); the vertex shader's texel index ` +
                `(aSplatIndex * 4.0) is no longer exactly representable in float32 ` +
                `and splats will start reading each other's data.`
            );
        }
        return;
    }

    // ── Sort request for the current camera pose ──────────────────
    if (msg.type === 'sort') {
        if (!positions || msg.generation !== generation || count === 0) {
            // Always answer, even to reject. The main thread clears its
            // sortInFlight back-pressure flag on receipt; a silent return
            // leaves it stuck true and stops all further sorting.
            self.postMessage({ type: 'rejected', generation: msg.generation });
            return;
        }

        const sortStartMs = performance.now();

        // modelView is a 16-float column-major mat4 (THREE convention):
        // view-space z only needs row 2 → elements[2,6,10,14]
        const m  = msg.modelView;
        const m2 = m[2], m6 = m[6], m10 = m[10], m14 = m[14];

        // ── Pass 1: view-space depth per point + range ─────────────
        let zMin =  Infinity;
        let zMax = -Infinity;

        for (let i = 0; i < count; i++) {
            const p = i * 3;
            const z = m2 * positions[p] + m6 * positions[p + 1] + m10 * positions[p + 2] + m14;
            depths[i] = z;
            if (z < zMin) zMin = z;
            if (z > zMax) zMax = z;
        }

        const range = (zMax - zMin) || 1e-6;
        const scale = (BUCKETS - 1) / range;

        // ── Pass 2: bucket each point ───────────────────────────────
        // Ascending z = most-negative (farthest, camera looks down -Z)
        // first → naturally gives back-to-front order.
        counts.fill(0);

        for (let i = 0; i < count; i++) {
            let b = ((depths[i] - zMin) * scale) | 0;
            if (b < 0) b = 0; else if (b >= BUCKETS) b = BUCKETS - 1;
            buckets[i] = b;
            counts[b + 1]++;
        }

        // ── Prefix sum → per-bucket start offsets ───────────────────
        for (let b = 0; b < BUCKETS; b++) counts[b + 1] += counts[b];

        // ── Scatter into the sorted index array ─────────────────────
        cursor.set(counts.subarray(0, BUCKETS));
        const sorted = new Float32Array(count);
        for (let i = 0; i < count; i++) {
            const b = buckets[i];
            sorted[cursor[b]++] = i;
        }

        self.postMessage(
            // workerMs is the sort itself; the main thread's onSortComplete
            // subtracts this from the full round trip to see how much of
            // the latency is postMessage/structured-clone overhead versus
            // the algorithm.
            { type: 'sorted', generation, indices: sorted, workerMs: performance.now() - sortStartMs },
            [sorted.buffer]
        );
    }
};
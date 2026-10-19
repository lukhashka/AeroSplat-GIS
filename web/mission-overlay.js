// ═══════════════════════════════════════════════════════════════════
//  AeroSplat-GIS — mission overlay for the 3DGS viewer (web/mission.html)
//
//  Draws a mission and its flight over the splats: legs and the corridor
//  (tubes of the clearance radius the plan promised), the geofence, the
//  target access point with H, E and the heading at H, the true trajectory
//  by phase (failsafe stretches in their own colour), and the drone at the
//  replayed instant.
//
//  Frames. The parent page (web/app, "Місія / проліт") sends every point
//  already in the scene's *served* frame: Site ENU of the scene minus
//  scene-meta.center (aerosplat.store.scenes.metric_block: enu = xyz +
//  center), x east, y north, z up. The overlay group is a child of
//  sceneRoot, so the viewer's up-axis rotation and scale reach it exactly
//  as they reach the splats — there is no second transform here to get
//  wrong (ADR-0001).
//
//  Protocol: window.postMessage, same origin only,
//  {source: 'aerosplat-mission', type, ...}:
//    parent → here: place (no-scene mode), overlay, drone, view, layers
//    here → parent: ready {meta}, error {message}
// ═══════════════════════════════════════════════════════════════════

(function () {
    'use strict';

    const SOURCE = 'aerosplat-mission';
    const V = window.AeroViewer;
    const parentWin = window.parent !== window ? window.parent : null;

    function post(type, body) {
        if (parentWin) parentWin.postMessage(Object.assign({ source: SOURCE, type }, body || {}), location.origin);
    }

    if (!V) {
        // appGLSL.js stopped before the end (no WebGL context): it has said why in the console.
        post('error', { message: 'Переглядач не запустився (немає WebGL?). Подробиці — у консолі кадру.' });
        return;
    }

    const UP = new THREE.Vector3(0, 0, 1);      // ENU up, before sceneRoot's rotation
    const TRAIL_RADIUS_M = 0.18;
    const LABEL_FRAC = 0.032;                   // label height, fraction of the view height
    const LAYERS = ['corridor', 'fence', 'target', 'waypoints', 'trajectory', 'drone'];
    const ON_TOP = ['target', 'waypoints', 'trajectory', 'drone'];   // what "x-ray" lifts over the splats

    const root = new THREE.Group();
    root.name = 'mission-overlay';
    V.sceneRoot.add(root);
    const layers = {};
    for (const name of LAYERS) {
        layers[name] = new THREE.Group();
        layers[name].name = name;
        root.add(layers[name]);
    }

    let data = null;
    let xray = false;
    let follow = false;
    let trail = [];             // per trajectory segment: {bright, i0, i1, cum, total, tubular, radial}
    let drone = null;           // {group, body, drop, sprite}
    let lastDroneWorld = null;

    const v3 = (p) => new THREE.Vector3(p[0], p[1], p[2]);

    // ── helpers ────────────────────────────────────────────────────
    function disposeTree(obj) {
        obj.traverse((o) => {
            if (o.geometry) o.geometry.dispose();
            const mats = o.material ? (Array.isArray(o.material) ? o.material : [o.material]) : [];
            for (const m of mats) {
                if (m.map) m.map.dispose();
                m.dispose();
            }
        });
    }

    function clear(group) {
        while (group.children.length) {
            const c = group.children[group.children.length - 1];
            group.remove(c);
            disposeTree(c);
        }
    }

    function solid(color) {
        return new THREE.MeshBasicMaterial({ color });
    }

    function glass(color, opacity) {
        return new THREE.MeshBasicMaterial({
            color, transparent: true, opacity, depthWrite: false, side: THREE.DoubleSide,
        });
    }

    function lines(color, opacity) {
        return new THREE.LineBasicMaterial({ color, transparent: opacity < 1, opacity });
    }

    function polyline(points, material, closed) {
        const g = new THREE.BufferGeometry().setFromPoints(points.map(v3));
        return closed ? new THREE.LineLoop(g, material) : new THREE.Line(g, material);
    }

    // Constant size on screen: with sizeAttenuation off a sprite's scale is a fraction of the
    // view (≈ scale·0.87 of the height at 60° fov), times the scale of its parents — so divide
    // sceneRoot's back out.
    function label(text, color) {
        const fontPx = 40;
        const pad = 8;
        const canvas = document.createElement('canvas');
        const ctx = canvas.getContext('2d');
        const font = `600 ${fontPx}px system-ui, -apple-system, "Segoe UI", sans-serif`;
        ctx.font = font;
        canvas.width = Math.ceil(ctx.measureText(text).width) + pad * 2;
        canvas.height = fontPx + pad * 2;
        ctx.font = font;                                   // resizing the canvas reset it
        ctx.fillStyle = 'rgba(8, 12, 16, 0.80)';
        ctx.fillRect(0, 0, canvas.width, canvas.height);
        ctx.fillStyle = color;
        ctx.textBaseline = 'middle';
        ctx.fillText(text, pad, canvas.height / 2 + 1);
        const tex = new THREE.CanvasTexture(canvas);
        tex.minFilter = THREE.LinearFilter;
        const sprite = new THREE.Sprite(new THREE.SpriteMaterial({
            map: tex, depthTest: false, depthWrite: false, transparent: true, sizeAttenuation: false,
        }));
        const k = LABEL_FRAC / Math.max(V.sceneRoot.scale.x, 1e-9);
        sprite.scale.set(k * canvas.width / canvas.height, k, 1);
        sprite.center.set(0, 0);                           // anchored at its lower-left corner
        sprite.renderOrder = 30;
        return sprite;
    }

    function labelAt(text, color, p, liftM) {
        const s = label(text, color);
        s.position.copy(v3(p)).addScaledVector(UP, liftM || 0.8);
        return s;
    }

    function sphere(p, r, material) {
        const m = new THREE.Mesh(new THREE.SphereGeometry(r, 20, 12), material);
        m.position.copy(v3(p));
        return m;
    }

    // An open cylinder from a to b (the corridor around one straight piece of a leg).
    function cylinder(a, b, r, material) {
        const A = v3(a);
        const B = v3(b);
        const len = A.distanceTo(B);
        if (len < 1e-6) return null;
        const m = new THREE.Mesh(new THREE.CylinderGeometry(r, r, len, 40, 1, true), material);
        m.position.copy(A).add(B).multiplyScalar(0.5);
        m.quaternion.setFromUnitVectors(new THREE.Vector3(0, 1, 0), B.clone().sub(A).normalize());
        return m;
    }

    // A circle of radius r about c, perpendicular to `axis`.
    function ring(c, axis, r, material) {
        const n = v3(axis).normalize();
        const ref = Math.abs(n.z) < 0.9 ? UP : new THREE.Vector3(1, 0, 0);
        const u = new THREE.Vector3().crossVectors(n, ref).normalize();
        const w = new THREE.Vector3().crossVectors(n, u);
        const pts = [];
        for (let i = 0; i < 48; i++) {
            const a = (i / 48) * Math.PI * 2;
            pts.push(v3(c).addScaledVector(u, r * Math.cos(a)).addScaledVector(w, r * Math.sin(a)));
        }
        return new THREE.LineLoop(new THREE.BufferGeometry().setFromPoints(pts), material);
    }

    // ── layers ─────────────────────────────────────────────────────
    function buildCorridor(d) {
        const g = layers.corridor;
        for (const leg of d.legs) {
            const shell = glass(leg.color, 0.08);
            const edge = lines(leg.color, 0.35);
            for (let i = 0; i + 1 < leg.points.length; i++) {
                const a = leg.points[i];
                const b = leg.points[i + 1];
                const c = cylinder(a, b, leg.radius, shell);
                if (c) g.add(c);
                const axis = [b[0] - a[0], b[1] - a[1], b[2] - a[2]];
                const len = Math.hypot(axis[0], axis[1], axis[2]);
                if (len < 1e-6) continue;
                // Rings every ~10 m give the tube its shape where the shell alone is too faint.
                const n = Math.max(1, Math.round(len / 10));
                for (let k = 0; k <= n; k++) {
                    const t = k / n;
                    g.add(ring([a[0] + axis[0] * t, a[1] + axis[1] * t, a[2] + axis[2] * t], axis, leg.radius, edge));
                }
            }
            for (const p of [leg.points[0], leg.points[leg.points.length - 1]]) {
                g.add(sphere(p, leg.radius, glass(leg.color, 0.06)));
            }
        }
    }

    function buildFence(d) {
        const g = layers.fence;
        const f = d.fence;
        const n = f.floor.length;
        if (n < 3) return;
        const pos = [];
        for (let i = 0; i < n; i++) {
            const j = (i + 1) % n;
            const a = f.floor[i], b = f.floor[j], c = f.ceiling[j], e = f.ceiling[i];
            pos.push(...a, ...b, ...c, ...a, ...c, ...e);
        }
        const walls = new THREE.BufferGeometry();
        walls.setAttribute('position', new THREE.Float32BufferAttribute(pos, 3));
        g.add(new THREE.Mesh(walls, glass(d.palette.fence, 0.05)));
        const edge = lines(d.palette.fence, 0.7);
        g.add(polyline(f.floor, edge, true));
        g.add(polyline(f.ceiling, edge, true));
        const verts = [];
        for (let i = 0; i < n; i++) verts.push(...f.floor[i], ...f.ceiling[i]);
        const posts = new THREE.BufferGeometry();
        posts.setAttribute('position', new THREE.Float32BufferAttribute(verts, 3));
        g.add(new THREE.LineSegments(posts, lines(d.palette.fence, 0.35)));
    }

    function buildTarget(d) {
        const g = layers.target;
        const t = d.target;
        const pal = d.palette;
        if (t.outline.length >= 3) {
            const pos = [];
            for (let i = 1; i + 1 < t.outline.length; i++) pos.push(...t.outline[0], ...t.outline[i], ...t.outline[i + 1]);
            const fill = new THREE.BufferGeometry();
            fill.setAttribute('position', new THREE.Float32BufferAttribute(pos, 3));
            g.add(new THREE.Mesh(fill, glass(pal.target, 0.55)));
            g.add(polyline(t.outline, lines(pal.target, 1), true));
        }
        g.add(sphere(t.hover, 0.35, solid(pal.hover)));
        g.add(sphere(t.entry, 0.3, solid(pal.entry)));
        g.add(labelAt('H', pal.hover, t.hover, 0.6));
        g.add(labelAt('E', pal.entry, t.entry, 0.6));
        g.add(labelAt(t.label, pal.target, t.center, 1.0));
        if (t.headingDir) {
            const dir = v3(t.headingDir).normalize();
            g.add(new THREE.ArrowHelper(dir, v3(t.hover), 3.0, new THREE.Color(pal.hover).getHex(), 0.9, 0.55));
        }
    }

    function buildWaypoints(d) {
        const g = layers.waypoints;
        const pal = d.palette;
        g.add(polyline(d.plannedPath, new THREE.LineDashedMaterial({
            color: pal.plan, dashSize: 0.8, gapSize: 0.6, transparent: true, opacity: 0.9,
        }), false));
        g.children[g.children.length - 1].computeLineDistances();
        for (const w of d.waypoints) {
            g.add(sphere(w.p, 0.25, solid(pal.waypoint)));
            if (w.label) g.add(labelAt(w.label, pal.waypoint, w.p, 0.5));
        }
        g.add(ring(d.home, [0, 0, 1], 1.2, lines(pal.home, 1)));
        g.add(labelAt('дім', pal.home, d.home, 0.4));
    }

    function buildTrajectory(d) {
        const g = layers.trajectory;
        trail = [];
        const pts = d.trajectory.points.map(v3);
        for (const seg of d.trajectory.segments) {
            // Arc length at every sample of the segment; the tube samples its path by arc
            // length, so this is what maps "sample k" onto "how much of the tube to draw".
            const cum = [0];
            const path = new THREE.CurvePath();
            let prev = pts[seg.i0];
            for (let k = seg.i0 + 1; k <= seg.i1; k++) {
                const p = pts[k];
                const step = p.distanceTo(prev);
                cum.push(cum[cum.length - 1] + step);
                if (step > 0.01) {
                    path.add(new THREE.LineCurve3(prev.clone(), p.clone()));
                    prev = p;
                } else {
                    cum[cum.length - 1] -= step;          // too short to keep: not on the tube
                }
            }
            const total = cum[cum.length - 1];
            if (!path.curves.length || total < 0.05) continue;
            const tubular = Math.min(2400, Math.max(8, Math.ceil(total / 0.25)));
            const radial = 6;
            const geom = new THREE.TubeGeometry(path, tubular, TRAIL_RADIUS_M, radial, false);
            const faded = new THREE.Mesh(geom, glass(seg.color, 0.45));
            const bright = new THREE.Mesh(geom.clone(), solid(seg.color));
            g.add(faded);
            g.add(bright);
            trail.push({ bright, i0: seg.i0, i1: seg.i1, cum, total, tubular, radial });
        }
        setTrail(d.trajectory.points.length - 1);
    }

    function setTrail(k) {
        for (const s of trail) {
            let frac;
            if (k >= s.i1) frac = 1;
            else if (k <= s.i0) frac = 0;
            else frac = s.cum[k - s.i0] / s.total;
            s.bright.geometry.setDrawRange(0, Math.round(frac * s.tubular) * s.radial * 6);
        }
    }

    function buildDrone(d) {
        const g = layers.drone;
        const group = new THREE.Group();
        const body = new THREE.Mesh(new THREE.SphereGeometry(d.droneRadius, 24, 14), glass(d.palette.drone, 0.35));
        // A cone's apex is +y: north in the served frame, i.e. yaw 0 before the heading turns it.
        const nose = new THREE.Mesh(new THREE.ConeGeometry(0.28, 1.1, 16), solid(d.palette.drone));
        nose.position.set(0, 0.55, 0);
        const heading = new THREE.Group();                 // rotated about ENU up by the yaw
        heading.add(nose);
        group.add(body, heading);
        const dropGeom = new THREE.BufferGeometry().setFromPoints([new THREE.Vector3(), new THREE.Vector3()]);
        const drop = new THREE.Line(dropGeom, new THREE.LineDashedMaterial({
            color: d.palette.drone, dashSize: 0.5, gapSize: 0.4, transparent: true, opacity: 0.8,
        }));
        const sprite = label('●', d.palette.drone);
        g.add(group, drop);
        group.add(sprite);
        drone = { group, body, heading, nose, drop, sprite };
        group.visible = false;
        drop.visible = false;
    }

    function updateDrone(msg) {
        if (!drone || !data) return;
        const p = v3(msg.p);
        drone.group.visible = true;
        drone.drop.visible = true;
        drone.group.position.copy(p);
        // Compass yaw (clockwise from north) -> rotation about ENU up (counter-clockwise).
        drone.heading.rotation.set(0, 0, -THREE.MathUtils.degToRad(msg.yawDeg || 0));
        const color = msg.failsafe ? data.palette.droneFailsafe : data.palette.drone;
        drone.body.material.color.set(color);
        drone.nose.material.color.set(color);
        const attr = drone.drop.geometry.getAttribute('position');
        attr.setXYZ(0, p.x, p.y, p.z);
        attr.setXYZ(1, p.x, p.y, data.floorZ);
        attr.needsUpdate = true;
        drone.drop.geometry.computeBoundingSphere();
        drone.drop.computeLineDistances();
        setTrail(msg.index);
        root.updateMatrixWorld(true);
        const w = root.localToWorld(p.clone());
        if (follow && lastDroneWorld) {
            const dw = w.clone().sub(lastDroneWorld);
            V.camera.position.add(dw);
            V.controls.target.add(dw);
        }
        lastDroneWorld = w;
    }

    function applyXray() {
        for (const name of ON_TOP) {
            layers[name].traverse((o) => {
                if (!o.material || o.isSprite) return;
                const mats = Array.isArray(o.material) ? o.material : [o.material];
                for (const m of mats) {
                    if (m.userData.baseTransparent === undefined) m.userData.baseTransparent = m.transparent;
                    m.depthTest = !xray;
                    // In the transparent pass, after the splats (renderOrder 0), when lifted.
                    m.transparent = xray || m.userData.baseTransparent;
                    m.needsUpdate = true;
                }
                o.renderOrder = xray ? 10 : 0;
            });
        }
    }

    // ── views ──────────────────────────────────────────────────────
    function world(p) {
        root.updateMatrixWorld(true);
        return root.localToWorld(v3(p));
    }

    function look(eye, at) {
        V.controls.target.copy(world(at));
        V.camera.position.copy(world(eye));
        V.camera.lookAt(V.controls.target);
        V.controls.update();
    }

    function setView(name) {
        if (!data) return;
        const lo = data.bbox[0];
        const hi = data.bbox[1];
        const c = [(lo[0] + hi[0]) / 2, (lo[1] + hi[1]) / 2, (lo[2] + hi[2]) / 2];
        const span = Math.max(hi[0] - lo[0], hi[1] - lo[1], 20);
        if (name === 'top') {
            look([c[0], c[1] - 0.02 * span, c[2] + 1.35 * span], c);
        } else if (name === 'target' || name === 'side') {
            const t = data.target;
            const a = v3(t.direction).normalize();
            let side = new THREE.Vector3().crossVectors(a, UP);
            if (side.lengthSq() < 1e-6) side = new THREE.Vector3(1, 0, 0);
            side.normalize();
            // Stand on the side away from home: the transit legs (and their tubes) go that way.
            if (side.dot(v3(data.home).sub(v3(t.hover))) > 0) side.negate();
            const reach = v3(t.entry).distanceTo(v3(t.hover)) + 10;
            let eye;
            let at;
            if (name === 'target' && Math.abs(a.z) > 0.9) {
                // A pad is approached from above: look at it obliquely, from where the vehicle
                // arrives (home's side), so the descent and the surface are both in view.
                const back = v3(data.home).sub(v3(t.hover)).setZ(0);
                if (back.lengthSq() < 1e-6) back.set(0, -1, 0);
                back.normalize();
                eye = v3(t.hover).addScaledVector(back, 1.1 * reach).addScaledVector(UP, 0.7 * reach);
                at = v3(t.center).lerp(v3(t.entry), 0.3);
            } else if (name === 'target') {
                // From beyond E on the approach axis, off to the side and above: the wall, the
                // opening, H, E and the track between them in one view.
                eye = v3(t.hover).addScaledVector(a, reach).addScaledVector(side, 0.45 * reach).addScaledVector(UP, 0.25 * reach);
                at = v3(t.center).lerp(v3(t.entry), 0.3);
            } else {
                // In profile, along the target's plane: the surface is an edge and the standoff
                // of H (and of the track) from it is a distance on screen.
                const mid = v3(t.center).lerp(v3(t.entry), 0.5);
                eye = mid.clone().addScaledVector(side, 1.6 * reach).addScaledVector(UP, 0.08 * reach);
                at = mid;
            }
            look([eye.x, eye.y, eye.z], [at.x, at.y, at.z]);
        } else {
            look([c[0] - 0.5 * span, c[1] - 0.8 * span, c[2] + 0.55 * span], c);
        }
    }

    // ── messages ───────────────────────────────────────────────────
    function buildGround(d) {
        // No 3DGS scene under the mission (?noscene=1): a 5 m grid at home height for scale.
        const lo = d.bbox[0];
        const hi = d.bbox[1];
        const span = Math.ceil(Math.max(hi[0] - lo[0], hi[1] - lo[1], 20) * 1.4 / 10) * 10;
        const grid = new THREE.GridHelper(span, span / 5, 0x4a5560, 0x2a3138);
        grid.rotation.x = Math.PI / 2;                     // GridHelper lies in xz; ENU ground is xy
        grid.position.set((lo[0] + hi[0]) / 2, (lo[1] + hi[1]) / 2, d.floorZ);
        layers.fence.add(grid);
    }

    function build(d) {
        data = d;
        for (const name of LAYERS) clear(layers[name]);
        if (window.AEROSPLAT_NO_SCENE) buildGround(d);
        buildCorridor(d);
        buildFence(d);
        buildTarget(d);
        buildWaypoints(d);
        buildTrajectory(d);
        buildDrone(d);
        applyXray();
        setView('mission');
    }

    function publicMeta(meta) {
        return {
            dataset: meta.dataset, count: meta.count, center: meta.center, extent: meta.extent,
            scene_scale: meta.scene_scale, up_axis_rotation: meta.up_axis_rotation, metric: meta.metric,
        };
    }

    window.addEventListener('message', (e) => {
        if (e.origin !== location.origin || e.source !== parentWin) return;
        const msg = e.data;
        if (!msg || msg.source !== SOURCE) return;
        try {
            if (msg.type === 'place') {
                V.placeScene(msg.meta);
                post('ready', { meta: publicMeta(msg.meta) });
            } else if (msg.type === 'overlay') {
                build(msg.data);
            } else if (msg.type === 'drone') {
                updateDrone(msg);
            } else if (msg.type === 'view') {
                setView(msg.name);
            } else if (msg.type === 'layers') {
                for (const name of LAYERS) {
                    if (name in msg.visible) layers[name].visible = !!msg.visible[name];
                }
                if ('xray' in msg && !!msg.xray !== xray) {
                    xray = !!msg.xray;
                    applyXray();
                }
                if ('follow' in msg) follow = !!msg.follow;
            }
        } catch (err) {
            console.error('AeroSplat-GIS mission overlay:', err);
            post('error', { message: `Накладка місії: ${err.message}` });
        }
    });

    window.addEventListener('aerosplat:scene-loaded', (e) => post('ready', { meta: publicMeta(e.detail) }));
    window.addEventListener('aerosplat:scene-failed', (e) => post('error', { message: String(e.detail) }));
    // The scene loads asynchronously and may have finished while this file was still on its way.
    if (V.loaded() && V.meta()) post('ready', { meta: publicMeta(V.meta()) });
})();

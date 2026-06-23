const container = document.getElementById('canvas-container');
const scene = new THREE.Scene();
scene.background = new THREE.Color(0x0a0f1d); // Сучасний тактичний фон

const camera = new THREE.PerspectiveCamera(60, window.innerWidth / window.innerHeight, 0.1, 100000);
camera.position.set(0, 300, 600);

const renderer = new THREE.WebGLRenderer({ antialias: true });
renderer.setSize(window.innerWidth, window.innerHeight);
renderer.setPixelRatio(window.devicePixelRatio);
container.appendChild(renderer.domElement);

const controls = new THREE.OrbitControls(camera, renderer.domElement);
controls.enableDamping = true;
controls.dampingFactor = 0.05;

let activeMesh = null;
let currentMode = "splat"; // Дефолтний режим: Варіант Б

// Шейдери для Варіанта Б (3DGS)
const vertexShaderB = `
    attribute vec3 aSplatScale;
    attribute float aSplatOpacity;
    attribute vec3 aSplatColor;
    varying vec3 vColor;
    varying float vOpacity;

    void main() {
        vColor = aSplatColor;
        vOpacity = aSplatOpacity;
        float avgScale = (aSplatScale.x + aSplatScale.y + aSplatScale.z) * 1.0;
        vec4 mvPosition = modelViewMatrix * vec4(position, 1.0);
        float depthDistance = max(-mvPosition.z, 0.1);
        gl_PointSize = clamp(avgScale * (250.0 / depthDistance), 3.0, 18.0);    
        gl_Position = projectionMatrix * mvPosition;
    }
`;

const fragmentShaderB = `
    varying vec3 vColor;
    varying float vOpacity;

    void main() {
        vec2 coord = gl_PointCoord - vec2(0.5);
        float r2 = dot(coord, coord);
        if (r2 > 0.25) discard;
        float gaussianBlur = exp(-r2 * 12.0);
        if (gaussianBlur * vOpacity < 0.2) discard;
        gl_FragColor = vec4(vColor, vOpacity * gaussianBlur);
    }
`;

// 🔥 ЗАЛІЗОБЕТОННА ІНЖЕКЦІЯ ПАНЕЛІ КЕРУВАННЯ В BODY
function injectModeSelector() {
    // Якщо селектор вже є на екрані — нічого не робимо
    if (document.getElementById('mode-selector-wrapper')) return;

    const wrapper = document.createElement('div');
    wrapper.id = 'mode-selector-wrapper';
    
    // Абсолютне позиціонування поверх усіх елементів Three.js
    wrapper.style.position = 'fixed';
    wrapper.style.top = '150px'; // Розташування акурат під твоїм головним UI-вікном
    wrapper.style.left = '10px';
    wrapper.style.backgroundColor = 'rgba(0, 0, 0, 0.85)';
    wrapper.style.border = '1px solid #00ff00';
    wrapper.style.padding = '10px';
    wrapper.style.borderRadius = '4px';
    wrapper.style.zIndex = '9999';
    wrapper.style.width = '322px';
    wrapper.style.boxShadow = '0 0 15px rgba(0, 255, 0, 0.2)';

    wrapper.innerHTML = `
        <label style="color: #00ff00; font-family: monospace; display: block; margin-bottom: 6px; font-size: 11px; font-weight: bold; letter-spacing: 1px;">МЕРЕЖЕВИЙ РУШІЙ (ДИНАМІЧНИЙ ТОГЛ):</label>
        <select id="mode-select-dropdown" style="background: #000; color: #00ff00; border: 1px solid #00ff00; font-family: monospace; padding: 6px; width: 100%; cursor: pointer; font-size: 12px; outline: none;">
            <option value="splat" ${currentMode === 'splat' ? 'selected' : ''}>Варіант Б: 3D Gaussian Splatting (Нейромережа)</option>
            <option value="dense" ${currentMode === 'dense' ? 'selected' : ''}>Варіант А: Щільна хмара точок (MVS)</option>
        </select>
    `;
    
    // Примусово монтуємо в корінь сторінки
    document.body.appendChild(wrapper);

    // Вішаємо прослуховувач подій перемикання
    document.getElementById('mode-select-dropdown').addEventListener('change', (e) => {
        currentMode = e.target.value;
        console.log(`🔄 Перемикання рушія на режим: ${currentMode}`);
        loadSpatialData();
    });
}

async function loadSpatialData() {
    const url = `/api/v1/spatial-chunk?lat=47.0052&lon=34.9164&radius=0.005&mode=${currentMode}`;
    
    try {
        const response = await fetch(url);
        if (!response.ok) throw new Error(`HTTP error! status: ${response.status}`);
        
        const latency = response.headers.get('X-Processing-Time-Ms');
        const count = response.headers.get('X-Gaussians-Count');
        if (latency) {
            const latElem = document.getElementById('latency-val');
            if (latElem) latElem.innerText = latency + ' ms';
        }
        if (count) {
            const cntElem = document.getElementById('count-val');
            if (cntElem) cntElem.innerText = parseInt(count).toLocaleString();
        }

        const buffer = await response.arrayBuffer();
        const floatArray = new Float32Array(buffer);
        
        // Повне вивільнення GPU пам'яті перед заміною моделі
        if (activeMesh) {
            scene.remove(activeMesh);
            if (activeMesh.geometry) activeMesh.geometry.dispose();
            if (activeMesh.material) activeMesh.material.dispose();
            activeMesh = null;
        }

        let geometry = new THREE.BufferGeometry();
        let material;

        if (currentMode === "dense") {
            // --- ПАРСИНГ ВАРІАНТА А (6 флотів: XYZ + RGB) ---
            const numPoints = floatArray.length / 6;
            const positions = new Float32Array(numPoints * 3);
            const colors = new Float32Array(numPoints * 3);
            
            for (let i = 0; i < numPoints; i++) {
                const stride = i * 6;
                positions[i * 3]     = floatArray[stride];
                positions[i * 3 + 1] = floatArray[stride + 1];
                positions[i * 3 + 2] = floatArray[stride + 2];
                
                colors[i * 3]     = floatArray[stride + 3];
                colors[i * 3 + 1] = floatArray[stride + 4];
                colors[i * 3 + 2] = floatArray[stride + 5];
            }
            
            geometry.setAttribute('position', new THREE.BufferAttribute(positions, 3));
            geometry.setAttribute('color', new THREE.BufferAttribute(colors, 3));
            
            // 📊 Налаштування відображення Варіанта А (MVS)
            material = new THREE.PointsMaterial({
                size: 0.36,             // 🔥 Збільшуємо розмір часток з 1.2 до 3.5 для щільності текстури
                vertexColors: true,    // Вмикаємо зчитування переданих кольорів
                sizeAttenuation: true
            });
            
        } else {
            // --- ПАРСИНГ ВАРІАНТА Б (14 флотів: XYZ + Scale + Rot + Opacity + RGB) ---
            const numGaussians = floatArray.length / 14;
            const positions = new Float32Array(numGaussians * 3);
            const scales = new Float32Array(numGaussians * 3);
            const opacities = new Float32Array(numGaussians);
            const colors = new Float32Array(numGaussians * 3);
            
            for (let i = 0; i < numGaussians; i++) {
                const stride = i * 14;
                positions[i * 3]     = floatArray[stride];
                positions[i * 3 + 1] = floatArray[stride + 1];
                positions[i * 3 + 2] = floatArray[stride + 2];
                
                scales[i * 3]     = floatArray[stride + 3];
                scales[i * 3 + 1] = floatArray[stride + 4];
                scales[i * 3 + 2] = floatArray[stride + 5];
                
                opacities[i] = floatArray[stride + 10];
                
                colors[i * 3]     = floatArray[stride + 11];
                colors[i * 3 + 1] = floatArray[stride + 12];
                colors[i * 3 + 2] = floatArray[stride + 13];
            }
            
            geometry.setAttribute('position', new THREE.BufferAttribute(positions, 3));
            geometry.setAttribute('aSplatScale', new THREE.BufferAttribute(scales, 3));
            geometry.setAttribute('aSplatOpacity', new THREE.BufferAttribute(opacities, 1));
            geometry.setAttribute('aSplatColor', new THREE.BufferAttribute(colors, 3));
            
            material = new THREE.ShaderMaterial({
                vertexShader: vertexShaderB,
                fragmentShader: fragmentShaderB,
                transparent: false,
                blending: THREE.NormalBlending,
                depthWrite: true,
                depthTest: true
            });
        }

        activeMesh = new THREE.Points(geometry, material);
        scene.add(activeMesh);
        
        geometry.computeBoundingSphere();
        const boundingSphere = geometry.boundingSphere;
        if (boundingSphere) {
            const center = boundingSphere.center;
            const radius = boundingSphere.radius;
            controls.target.copy(center);
            camera.position.set(center.x, center.y + radius * 1.2, center.z + radius * 1.8);
            camera.lookAt(center);
            controls.update();
        }
        
        // Запускаємо гарантовану інжекцію меню в інтерфейс
        injectModeSelector();
        
    } catch (error) {
        console.error("❌ Помилка завантаження чанку:", error);
    }
}

function animate() {
    requestAnimationFrame(animate);
    controls.update();
    renderer.render(scene, camera);
}

window.addEventListener('resize', () => {
    camera.aspect = window.innerWidth / window.innerHeight;
    camera.updateProjectionMatrix();
    renderer.setSize(window.innerWidth, window.innerHeight);
});

loadSpatialData();
animate();
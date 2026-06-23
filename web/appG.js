const container = document.getElementById('canvas-container');
const scene = new THREE.Scene();
scene.background = new THREE.Color(0x050505);

const camera = new THREE.PerspectiveCamera(60, window.innerWidth / window.innerHeight, 0.1, 100000);
camera.position.set(0, 300, 600);

const renderer = new THREE.WebGLRenderer({ antialias: true });
renderer.setSize(window.innerWidth, window.innerHeight);
renderer.setPixelRatio(window.devicePixelRatio);
container.appendChild(renderer.domElement);

const controls = new THREE.OrbitControls(camera, renderer.domElement);
controls.enableDamping = true;
controls.dampingFactor = 0.05;

let splatMesh = null;

// 🔥 НАДІЙНІ ШЕЙДЕРИ З КВАНТОВИМ ЗАХИСТОМ НА GPU
const vertexShader = `
    attribute vec3 aSplatScale;
    attribute float aSplatOpacity;
    attribute vec3 aSplatColor;
    varying vec3 vColor;
    varying float vOpacity;

    void main() {
        vColor = aSplatColor;
        vOpacity = aSplatOpacity;
        
        // Розраховуємо середній об'ємний масштаб еліпсоїда
        float avgScale = (aSplatScale.x + aSplatScale.y + aSplatScale.z) * 15.0;
        
        vec4 mvPosition = modelViewMatrix * vec4(position, 1.0);
        
        // ЗАХИСТ: уникаємо ділення на 0, якщо камера підлітає впритул
        float depthDistance = max(-mvPosition.z, 0.1);
        
        // КЛЕМПІНГ: обмежуємо розмір від 1 до 64 пікселів, щоб точки не зникали і не крашили контекст
        gl_PointSize = clamp(avgScale * (400.0 / depthDistance), 1.0, 64.0);
        
        gl_Position = projectionMatrix * mvPosition;
    }
`;

const fragmentShader = `
    varying vec3 vColor;
    varying float vOpacity;

    void main() {
        // Перетворюємо квадратні плитки на м'які круглі диференційовані Гаусіани
        vec2 coord = gl_PointCoord - vec2(0.5);
        float r2 = dot(coord, coord);
        if (r2 > 0.25) discard; // Відсікаємо жорсткі кути
        
        // Формула розподілу Гаусса для м'якого згасання до країв точок
        float gaussianBlur = exp(-r2 * 7.0);
        
        gl_FragColor = vec4(vColor, vOpacity * gaussianBlur);
    }
`;

async function loadGaussianSplats() {
    const url = `/api/v1/spatial-chunk?lat=47.0052&lon=34.9164&radius=0.005`;
    
    try {
        const response = await fetch(url);
        if (!response.ok) throw new Error(`HTTP error! status: ${response.status}`);
        
        const latency = response.headers.get('X-Processing-Time-Ms');
        const count = response.headers.get('X-Gaussians-Count');
        if (latency) document.getElementById('latency-val').innerText = latency + ' ms';
        if (count) document.getElementById('count-val').innerText = parseInt(count).toLocaleString();

        const buffer = await response.arrayBuffer();
        const floatArray = new Float32Array(buffer);
        
        const numGaussians = floatArray.length / 14;
        console.log("🛸 Кількість Гаусіанів на фронтенді: " + numGaussians);
        
        if (numGaussians === 0) {
            console.warn("⚠️ Порожній чанк від сервера.");
            return;
        }
        
        const positions = new Float32Array(numGaussians * 3);
        const scales = new Float32Array(numGaussians * 3);
        const opacities = new Float32Array(numGaussians);
        const colors = new Float32Array(numGaussians * 3);
        
        for (let i = 0; i < numGaussians; i++) {
            const stride = i * 14;
            
            // XYZ координати центрів
            positions[i * 3]     = floatArray[stride];
            positions[i * 3 + 1] = floatArray[stride + 1];
            positions[i * 3 + 2] = floatArray[stride + 2];
            
            // Навчений 3D масштаб Гаусіана
            scales[i * 3]     = floatArray[stride + 3];
            scales[i * 3 + 1] = floatArray[stride + 4];
            scales[i * 3 + 2] = floatArray[stride + 5];
            
            // Коефіцієнт прозорості альфа-каналу
            opacities[i] = floatArray[stride + 10];
            
            // Фотометричні кольори RGB
            colors[i * 3]     = floatArray[stride + 11];
            colors[i * 3 + 1] = floatArray[stride + 12];
            colors[i * 3 + 2] = floatArray[stride + 13];
        }
        
        const geometry = new THREE.BufferGeometry();
        
        // Прив'язуємо масиви до кастомних безпечних атрибутів шейдера
        geometry.setAttribute('position', new THREE.BufferAttribute(positions, 3));
        geometry.setAttribute('aSplatScale', new THREE.BufferAttribute(scales, 3));
        geometry.setAttribute('aSplatOpacity', new THREE.BufferAttribute(opacities, 1));
        geometry.setAttribute('aSplatColor', new THREE.BufferAttribute(colors, 3));
        
        // Використовуємо стабільне NormalBlending для реалістичного накладання шарів поверхні
        const material = new THREE.ShaderMaterial({
            vertexShader: vertexShader,
            fragmentShader: fragmentShader,
            transparent: true,
            blending: THREE.NormalBlending,
            depthWrite: false,
            depthTest: true
        });
        
        if (splatMesh) scene.remove(splatMesh);
        
        splatMesh = new THREE.Points(geometry, material);
        scene.add(splatMesh);
        
        // Розрахунок ліній фокусу камери
        geometry.computeBoundingSphere();
        const boundingSphere = geometry.boundingSphere;
        if (boundingSphere) {
            const center = boundingSphere.center;
            const radius = boundingSphere.radius;
            controls.target.copy(center);
            camera.position.set(center.x, center.y + radius * 1.2, center.z + radius * 1.8);
            camera.lookAt(center);
            controls.update();
            console.log("🎯 Автофокус успішно зафіксовано на метричному центрі сцени.");
        }
        
    } catch (error) {
        console.error("❌ Помилка ініціалізації WebGL шейдерів:", error);
    }
}

// Цикл анімації сцени
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

loadGaussianSplats();
animate();
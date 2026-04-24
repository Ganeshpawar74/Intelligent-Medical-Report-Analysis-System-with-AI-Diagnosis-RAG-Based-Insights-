/* HealthVerse AI — 3D human-body visualizer.
 * Builds a stylized human silhouette from primitive geometries and lets us
 * highlight a specific organ with a pulsing glow. Uses Three.js (loaded
 * dynamically from a CDN on first use).
 */
(function () {
  const THREE_URL = 'https://unpkg.com/three@0.160.0/build/three.module.js';
  let THREE_PROMISE = null;
  function loadThree() {
    if (!THREE_PROMISE) {
      THREE_PROMISE = import(THREE_URL).catch(err => {
        console.error('Three.js failed to load', err);
        throw err;
      });
    }
    return THREE_PROMISE;
  }

  // Approximate organ positions inside a normalized body frame.
  // Origin (0,0,0) is roughly the navel. +Y is up, +X is the patient's left,
  // +Z is forward (towards the viewer). Units are arbitrary scene units.
  const ORGAN_LAYOUT = {
    brain:           { pos: [0,    2.55, 0.05], scale: [0.55, 0.45, 0.55], shape: 'sphere' },
    eye:             { pos: [0.18, 2.5,  0.45], scale: [0.10, 0.10, 0.10], shape: 'sphere', mirror: -0.36 },
    ear:             { pos: [0.45, 2.45, 0.0],  scale: [0.10, 0.18, 0.10], shape: 'sphere', mirror: -0.9  },
    nose_sinus:      { pos: [0,    2.35, 0.55], scale: [0.10, 0.18, 0.10], shape: 'cone'   },
    throat:          { pos: [0,    1.95, 0.20], scale: [0.18, 0.30, 0.18], shape: 'cylinder' },
    thyroid:         { pos: [0,    1.85, 0.30], scale: [0.30, 0.10, 0.18], shape: 'sphere'  },
    lungs:           { pos: [0,    1.10, 0.10], scale: [0.95, 0.85, 0.55], shape: 'lungs'   },
    heart:           { pos: [0.18, 1.15, 0.20], scale: [0.36, 0.42, 0.30], shape: 'sphere'  },
    esophagus:       { pos: [0,    1.30, 0.05], scale: [0.10, 0.90, 0.10], shape: 'cylinder' },
    stomach:         { pos: [-0.30, 0.45, 0.10], scale: [0.45, 0.40, 0.30], shape: 'sphere' },
    liver:           { pos: [0.35, 0.55, 0.10], scale: [0.65, 0.40, 0.40], shape: 'sphere'  },
    gallbladder:     { pos: [0.30, 0.30, 0.20], scale: [0.18, 0.18, 0.18], shape: 'sphere'  },
    pancreas:        { pos: [-0.05, 0.30, 0.0], scale: [0.55, 0.15, 0.20], shape: 'cylinder' },
    spleen:          { pos: [-0.50, 0.55, 0.05], scale: [0.30, 0.25, 0.25], shape: 'sphere' },
    kidney:          { pos: [0.45, 0.20, -0.20], scale: [0.22, 0.35, 0.20], shape: 'sphere', mirror: -0.9 },
    bladder:         { pos: [0,   -0.55, 0.15], scale: [0.35, 0.30, 0.30], shape: 'sphere'  },
    intestines:      { pos: [0,   -0.20, 0.10], scale: [0.85, 0.65, 0.45], shape: 'sphere'  },
    reproductive_f:  { pos: [0,   -0.55, 0.10], scale: [0.50, 0.30, 0.30], shape: 'sphere'  },
    reproductive_m:  { pos: [0,   -0.85, 0.20], scale: [0.30, 0.25, 0.25], shape: 'sphere'  },
    skin:            { pos: [0,    0,    0],    scale: [1,    1,    1   ], shape: 'fullbody'},
    joints:          { pos: [0.6, -1.2,  0],    scale: [0.18, 0.18, 0.18], shape: 'sphere', mirror: -1.2 },
    bones:           { pos: [0,    0.4,  -0.20], scale: [0.20, 2.4, 0.20], shape: 'cylinder' },
    blood:           { pos: [0,    0.5,  0],    scale: [1.1,  3.2,  0.7],  shape: 'fullbody' },
  };

  const SLATE_GRAY = 0x9aa6b8;

  function buildBody(THREE, group) {
    // A simple human-ish silhouette built from primitives.
    const skin = new THREE.MeshStandardMaterial({
      color: 0x90a4c0, transparent: true, opacity: 0.18,
      roughness: 0.85, metalness: 0.0,
    });
    const wire = new THREE.MeshBasicMaterial({
      color: 0x1a7df0, wireframe: true, transparent: true, opacity: 0.18,
    });

    const parts = [];
    function add(geom, x, y, z, sx, sy, sz, name) {
      const m = new THREE.Mesh(geom, skin);
      const w = new THREE.Mesh(geom, wire);
      m.position.set(x, y, z); m.scale.set(sx, sy, sz);
      w.position.copy(m.position); w.scale.copy(m.scale);
      m.userData.bodyName = name;
      group.add(m); group.add(w);
      parts.push(m);
      return m;
    }

    // Head
    add(new THREE.SphereGeometry(0.5, 32, 32), 0, 2.55, 0, 1, 1.05, 1, 'head');
    // Neck
    add(new THREE.CylinderGeometry(0.18, 0.22, 0.30, 24), 0, 2.0, 0, 1, 1, 1, 'neck');
    // Torso (capsule-like via two cylinders + sphere caps approximated as a single tapered box)
    add(new THREE.CylinderGeometry(0.78, 0.65, 1.6, 24), 0, 1.05, 0, 1, 1, 0.55, 'torso');
    // Hips
    add(new THREE.CylinderGeometry(0.65, 0.55, 0.50, 24), 0, 0.05, 0, 1, 1, 0.6, 'hips');
    // Arms (left and right)
    [[1.05, 0.95], [-1.05, 0.95]].forEach(([x, y], i) => {
      add(new THREE.CylinderGeometry(0.18, 0.16, 1.4, 16), x, y, 0, 1, 1, 1, 'upperArm' + i);
      add(new THREE.CylinderGeometry(0.16, 0.13, 1.3, 16), x, -0.3, 0, 1, 1, 1, 'forearm' + i);
      add(new THREE.SphereGeometry(0.16, 16, 16), x, -1.0, 0, 1, 1, 1, 'hand' + i);
    });
    // Legs
    [[0.32, -1.45], [-0.32, -1.45]].forEach(([x, y], i) => {
      add(new THREE.CylinderGeometry(0.28, 0.22, 1.5, 20), x, y, 0, 1, 1, 1, 'thigh' + i);
      add(new THREE.CylinderGeometry(0.22, 0.18, 1.4, 20), x, -2.95, 0, 1, 1, 1, 'shin' + i);
      add(new THREE.SphereGeometry(0.24, 16, 16), x, -3.7, 0.15, 1.4, 0.6, 1.6, 'foot' + i);
    });

    return parts;
  }

  function makeOrganMesh(THREE, layout, color) {
    const c = new THREE.Color(color);
    const mat = new THREE.MeshStandardMaterial({
      color: c,
      emissive: c,
      emissiveIntensity: 0.55,
      roughness: 0.4,
      metalness: 0.1,
      transparent: true,
      opacity: 0.95,
    });
    let geom;
    switch (layout.shape) {
      case 'sphere':
        geom = new THREE.SphereGeometry(0.5, 24, 24);
        break;
      case 'cylinder':
        geom = new THREE.CylinderGeometry(0.5, 0.5, 1, 24);
        break;
      case 'cone':
        geom = new THREE.ConeGeometry(0.5, 1, 16);
        break;
      case 'lungs':
        // Two lobes
        geom = null;
        break;
      case 'fullbody':
        geom = new THREE.SphereGeometry(0.5, 24, 24);
        break;
      default:
        geom = new THREE.SphereGeometry(0.5, 24, 24);
    }

    const meshes = [];
    if (layout.shape === 'lungs') {
      const lobeGeom = new THREE.SphereGeometry(0.5, 24, 24);
      [-0.45, 0.45].forEach(off => {
        const m = new THREE.Mesh(lobeGeom, mat.clone());
        m.position.set(layout.pos[0] + off, layout.pos[1], layout.pos[2]);
        m.scale.set(layout.scale[0] * 0.55, layout.scale[1], layout.scale[2]);
        meshes.push(m);
      });
    } else {
      const m = new THREE.Mesh(geom, mat);
      m.position.set(layout.pos[0], layout.pos[1], layout.pos[2]);
      m.scale.set(layout.scale[0], layout.scale[1], layout.scale[2]);
      meshes.push(m);
      if (layout.mirror !== undefined) {
        const m2 = new THREE.Mesh(geom, mat.clone());
        m2.position.set(layout.pos[0] + layout.mirror, layout.pos[1], layout.pos[2]);
        m2.scale.set(layout.scale[0], layout.scale[1], layout.scale[2]);
        meshes.push(m2);
      }
    }
    return meshes;
  }

  async function mount(container, opts) {
    const THREE = await loadThree();
    container.innerHTML = '';
    container.style.position = 'relative';
    const w = container.clientWidth || 320;
    const h = container.clientHeight || 480;

    const scene = new THREE.Scene();
    scene.background = null;

    const camera = new THREE.PerspectiveCamera(35, w / h, 0.1, 100);
    camera.position.set(0, 0.4, 8.5);

    const renderer = new THREE.WebGLRenderer({ antialias: true, alpha: true });
    renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
    renderer.setSize(w, h);
    container.appendChild(renderer.domElement);

    // Lights
    scene.add(new THREE.AmbientLight(0xe2e8f0, 0.6));
    const key = new THREE.DirectionalLight(0xffffff, 1.1);
    key.position.set(3, 5, 6); scene.add(key);
    const rim = new THREE.DirectionalLight(0x60a5fa, 0.7);
    rim.position.set(-4, 2, -3); scene.add(rim);

    // Body group
    const bodyGroup = new THREE.Group();
    scene.add(bodyGroup);
    buildBody(THREE, bodyGroup);

    // Floor disc for reference
    const floor = new THREE.Mesh(
      new THREE.CircleGeometry(2.5, 48),
      new THREE.MeshBasicMaterial({ color: 0x1a7df0, transparent: true, opacity: 0.10 })
    );
    floor.rotation.x = -Math.PI / 2;
    floor.position.y = -3.95;
    scene.add(floor);

    // Organ
    const organMeshes = [];
    if (opts.organKey && ORGAN_LAYOUT[opts.organKey]) {
      const layout = ORGAN_LAYOUT[opts.organKey];
      const meshes = makeOrganMesh(THREE, layout, opts.color || '#ef4444');
      meshes.forEach(m => { bodyGroup.add(m); organMeshes.push(m); });
    }

    // Pointer / drag rotation
    let dragging = false, lastX = 0, lastY = 0;
    let rotY = 0.0, rotX = 0.0, autoSpin = true;

    function pointerDown(e) {
      dragging = true; autoSpin = false;
      lastX = e.clientX || e.touches?.[0]?.clientX;
      lastY = e.clientY || e.touches?.[0]?.clientY;
    }
    function pointerMove(e) {
      if (!dragging) return;
      const x = e.clientX ?? e.touches?.[0]?.clientX;
      const y = e.clientY ?? e.touches?.[0]?.clientY;
      if (x == null || y == null) return;
      rotY += (x - lastX) * 0.008;
      rotX += (y - lastY) * 0.005;
      rotX = Math.max(-0.6, Math.min(0.6, rotX));
      lastX = x; lastY = y;
    }
    function pointerUp() { dragging = false; }
    renderer.domElement.addEventListener('mousedown', pointerDown);
    renderer.domElement.addEventListener('mousemove', pointerMove);
    window.addEventListener('mouseup', pointerUp);
    renderer.domElement.addEventListener('touchstart', pointerDown, { passive: true });
    renderer.domElement.addEventListener('touchmove', pointerMove, { passive: true });
    window.addEventListener('touchend', pointerUp);

    // Resize observer
    const ro = new ResizeObserver(() => {
      const W = container.clientWidth || w;
      const H = container.clientHeight || h;
      camera.aspect = W / H;
      camera.updateProjectionMatrix();
      renderer.setSize(W, H);
    });
    ro.observe(container);

    let alive = true;
    let t0 = performance.now();
    function tick() {
      if (!alive) return;
      const t = (performance.now() - t0) / 1000;
      if (autoSpin) rotY = t * 0.4;
      bodyGroup.rotation.y = rotY;
      bodyGroup.rotation.x = rotX;
      // Pulse organ
      const pulse = 1 + Math.sin(t * 3) * 0.08;
      organMeshes.forEach(m => {
        m.material.emissiveIntensity = 0.45 + 0.35 * (Math.sin(t * 3) * 0.5 + 0.5);
        const s0 = m.userData.baseScale || (m.userData.baseScale = m.scale.clone());
        m.scale.set(s0.x * pulse, s0.y * pulse, s0.z * pulse);
      });
      renderer.render(scene, camera);
      requestAnimationFrame(tick);
    }
    tick();

    return {
      destroy() {
        alive = false;
        ro.disconnect();
        renderer.dispose();
        if (renderer.domElement.parentNode) {
          renderer.domElement.parentNode.removeChild(renderer.domElement);
        }
      },
      resetView() { autoSpin = true; rotX = 0; }
    };
  }

  window.Body3D = { mount, ORGAN_LAYOUT };
})();

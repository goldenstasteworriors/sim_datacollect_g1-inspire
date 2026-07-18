import * as THREE from 'three';
import { GLTFLoader } from 'three/addons/loaders/GLTFLoader.js';
import { OrbitControls } from 'three/addons/controls/OrbitControls.js';
import { TransformControls } from 'three/addons/controls/TransformControls.js';

function applyOpacity(root, color, opacity) {
  root.traverse((node) => {
    if (!node.isMesh) return;
    node.material = new THREE.MeshStandardMaterial({
      color, transparent: true, opacity, roughness: 0.58, metalness: 0.04,
      depthWrite: false, side: THREE.DoubleSide,
    });
  });
}

export async function createHandScene(container, onTransform) {
  const scene = new THREE.Scene();
  scene.background = new THREE.Color(0x0d1112);
  scene.fog = new THREE.Fog(0x0d1112, 0.55, 1.2);
  const camera = new THREE.PerspectiveCamera(38, 1, 0.001, 10);
  camera.position.set(0.34, 0.28, 0.25);
  const renderer = new THREE.WebGLRenderer({ antialias: true });
  renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
  renderer.outputColorSpace = THREE.SRGBColorSpace;
  container.append(renderer.domElement);
  scene.add(new THREE.HemisphereLight(0xdcefee, 0x25302c, 2.4));
  const key = new THREE.DirectionalLight(0xffffff, 3.2);
  key.position.set(0.3, 0.4, 0.5);
  scene.add(key);
  const grid = new THREE.GridHelper(0.7, 28, 0x394342, 0x242b2b);
  grid.rotation.x = Math.PI / 2;
  grid.position.z = -0.07;
  scene.add(grid);
  scene.add(new THREE.AxesHelper(0.07));

  const loader = new GLTFLoader();
  const [manoGltf, inspireGltf] = await Promise.all([
    loader.loadAsync('/models/mano_open.glb'), loader.loadAsync('/models/inspire_open.glb'),
  ]);
  const mano = manoGltf.scene;
  const inspire = inspireGltf.scene;
  mano.name = 'MANO canonical open';
  inspire.name = 'Inspire open';
  applyOpacity(mano, 0x26c6da, 0.42);
  applyOpacity(inspire, 0xff8a3d, 0.48);
  scene.add(mano, inspire);

  const baseMarker = new THREE.Group();
  baseMarker.add(new THREE.AxesHelper(0.055));
  const marker = new THREE.Mesh(
    new THREE.SphereGeometry(0.004, 20, 12),
    new THREE.MeshBasicMaterial({ color: 0xffe0c2 }),
  );
  baseMarker.add(marker);
  inspire.add(baseMarker);

  const orbit = new OrbitControls(camera, renderer.domElement);
  orbit.enableDamping = true;
  orbit.target.set(-0.045, -0.03, 0);
  const transform = new TransformControls(camera, renderer.domElement);
  transform.setSize(0.45);
  transform.attach(inspire);
  transform.addEventListener('dragging-changed', (event) => { orbit.enabled = !event.value; });
  transform.addEventListener('objectChange', () => onTransform(inspire));
  scene.add(transform.getHelper());

  function resize() {
    const { clientWidth, clientHeight } = container;
    renderer.setSize(clientWidth, clientHeight, false);
    camera.aspect = clientWidth / clientHeight;
    camera.updateProjectionMatrix();
  }
  function fitView() {
    camera.position.set(0.34, 0.28, 0.25);
    orbit.target.set(-0.045, -0.03, 0);
    orbit.update();
  }
  function render() {
    resize(); orbit.update(); renderer.render(scene, camera); requestAnimationFrame(render);
  }
  render();
  return {
    inspire, mano, transform, fitView,
    setOpacity(target, opacity) {
      const root = target === 'mano' ? mano : inspire;
      root.traverse((node) => { if (node.isMesh) node.material.opacity = opacity; });
    },
  };
}

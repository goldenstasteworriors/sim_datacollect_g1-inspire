import * as THREE from 'three';
import './styles.css';
import { createHandScene } from './scene.js';

const axes = ['x', 'y', 'z'];
const state = { sceneApi: null, initialMatrix: null, syncing: false };

function makeControl(parent, spec, onInput) {
  const row = document.createElement('div');
  row.className = 'control-row';
  const label = document.createElement('label');
  label.className = `axis-label ${spec.axis || ''}`;
  label.textContent = spec.label;
  label.htmlFor = `${spec.group}-${spec.key}-number`;
  const range = document.createElement('input');
  range.type = 'range'; range.min = spec.min; range.max = spec.max; range.step = spec.step;
  range.id = `${spec.group}-${spec.key}-range`;
  range.setAttribute('aria-label', `${spec.label} ${spec.unit}`);
  const number = document.createElement('input');
  number.type = 'number'; number.min = spec.min; number.max = spec.max; number.step = spec.step;
  number.id = `${spec.group}-${spec.key}-number`;
  for (const input of [range, number]) {
    input.addEventListener('input', () => {
      const value = Number(input.value);
      (input === range ? number : range).value = String(value);
      onInput(value);
    });
  }
  row.append(label, range, number); parent.append(row);
  return { range, number, set(value) { range.value = value; number.value = Number(value).toFixed(spec.digits); } };
}

function matrixText(matrix) {
  const e = matrix.elements;
  const rows = matrixRows(matrix);
  return rows.map((row) => `[${row.map((v) => v.toFixed(6).padStart(10)).join(', ')}]`).join('\n');
}

function matrixRows(matrix) {
  const e = matrix.elements;
  return [0, 1, 2, 3].map((r) => [0, 1, 2, 3].map((c) => e[c * 4 + r]));
}

function poseJson() {
  const { inspire } = state.sceneApi;
  inspire.updateMatrix();
  return {
    convention: 'T_mano_wrist_inspire_base',
    translation_m: inspire.position.toArray(),
    rotation_rpy_xyz_rad: [inspire.rotation.x, inspire.rotation.y, inspire.rotation.z],
    matrix_row_major: matrixRows(inspire.matrix),
  };
}

const translationControls = {};
const rotationControls = {};

function syncFromModel() {
  if (state.syncing) return;
  const { inspire } = state.sceneApi;
  axes.forEach((axis) => {
    translationControls[axis].set(inspire.position[axis] * 1000);
    rotationControls[axis].set(THREE.MathUtils.radToDeg(inspire.rotation[axis]));
  });
  inspire.updateMatrix();
  document.querySelector('#matrix-output').textContent = matrixText(inspire.matrix);
}

function applyInputs() {
  const { inspire } = state.sceneApi;
  state.syncing = true;
  axes.forEach((axis) => {
    inspire.position[axis] = Number(translationControls[axis].number.value) / 1000;
    inspire.rotation[axis] = THREE.MathUtils.degToRad(Number(rotationControls[axis].number.value));
  });
  inspire.updateMatrix();
  state.syncing = false;
  syncFromModel();
}

async function initialize() {
  const transformData = await fetch('/models/current_transform.json').then((response) => response.json());
  const matrix = new THREE.Matrix4().fromArray(transformData.matrix_row_major.flat()).transpose();
  state.initialMatrix = matrix.clone();
  state.sceneApi = await createHandScene(document.querySelector('#scene'), syncFromModel);
  matrix.decompose(state.sceneApi.inspire.position, state.sceneApi.inspire.quaternion, state.sceneApi.inspire.scale);

  axes.forEach((axis) => {
    translationControls[axis] = makeControl(
      document.querySelector('#translation-controls'),
      { group: 'translation', key: axis, axis, label: axis.toUpperCase(), unit: 'mm', min: -160, max: 160, step: .1, digits: 1 },
      applyInputs,
    );
    rotationControls[axis] = makeControl(
      document.querySelector('#rotation-controls'),
      { group: 'rotation', key: axis, axis, label: axis.toUpperCase(), unit: 'degree', min: -180, max: 180, step: .1, digits: 1 },
      applyInputs,
    );
  });
  ['mano', 'inspire'].forEach((target) => makeControl(
    document.querySelector('#opacity-controls'),
    { group: 'opacity', key: target, label: target === 'mano' ? 'M' : 'I', unit: 'opacity', min: 0.05, max: 1, step: .01, digits: 2 },
    (value) => state.sceneApi.setOpacity(target, value),
  ).set(target === 'mano' ? .42 : .48));
  syncFromModel();

  document.querySelectorAll('[data-mode]').forEach((button) => button.addEventListener('click', () => {
    document.querySelectorAll('[data-mode]').forEach((item) => item.classList.remove('active'));
    button.classList.add('active'); state.sceneApi.transform.setMode(button.dataset.mode);
  }));
  document.querySelector('#fit-view').addEventListener('click', state.sceneApi.fitView);
  document.querySelector('#reset-pose').addEventListener('click', () => {
    state.initialMatrix.decompose(state.sceneApi.inspire.position, state.sceneApi.inspire.quaternion, state.sceneApi.inspire.scale);
    syncFromModel();
  });
  document.querySelector('#copy-pose').addEventListener('click', async () => {
    await navigator.clipboard.writeText(JSON.stringify(poseJson(), null, 2));
    const toast = document.querySelector('#toast'); toast.textContent = '参数已复制到剪贴板';
    window.setTimeout(() => { toast.textContent = ''; }, 2200);
  });
  window.addEventListener('keydown', (event) => {
    if (event.target.matches('input')) return;
    if (event.key.toLowerCase() === 'w') state.sceneApi.transform.setMode('translate');
    if (event.key.toLowerCase() === 'e') state.sceneApi.transform.setMode('rotate');
    if (event.key.toLowerCase() === 'f') state.sceneApi.fitView();
  });
}

initialize().catch((error) => {
  document.querySelector('#scene').innerHTML = `<p class="load-error">模型加载失败：${error.message}</p>`;
  console.error(error);
});

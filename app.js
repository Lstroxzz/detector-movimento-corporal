import {
  FilesetResolver,
  PoseLandmarker,
  DrawingUtils,
} from "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.14";

const video = document.querySelector("#camera");
const canvas = document.querySelector("#overlay");
const emptyState = document.querySelector("#emptyState");
const startButton = document.querySelector("#startButton");
const statusText = document.querySelector("#statusText");
const statusDot = document.querySelector("#statusDot");
const motionList = document.querySelector("#motionList");
const sensitivity = document.querySelector("#sensitivity");
const beautyBanner = ensureBeautyBanner();
const ctx = canvas.getContext("2d");

const bodyGroups = [
  { name: "Cabeca", points: [0, 2, 5, 7, 8], anchor: [0, 7, 8] },
  { name: "Pescoco", points: [7, 8, 11, 12], anchor: [7, 8, 11, 12] },
  { name: "Ombro esquerdo", points: [11], anchor: [11] },
  { name: "Ombro direito", points: [12], anchor: [12] },
  { name: "Braco esquerdo", points: [11, 13, 15], anchor: [13] },
  { name: "Braco direito", points: [12, 14, 16], anchor: [14] },
  { name: "Mao esquerda", points: [15, 17, 19, 21], anchor: [15, 19] },
  { name: "Mao direita", points: [16, 18, 20, 22], anchor: [16, 20] },
  { name: "Peito", points: [11, 12], anchor: [11, 12] },
  { name: "Tronco", points: [11, 12, 23, 24], anchor: [11, 12, 23, 24] },
  { name: "Quadril", points: [23, 24], anchor: [23, 24] },
  { name: "Perna esquerda", points: [23, 25, 27], anchor: [25] },
  { name: "Perna direita", points: [24, 26, 28], anchor: [26] },
  { name: "Pe esquerdo", points: [27, 29, 31], anchor: [29, 31] },
  { name: "Pe direito", points: [28, 30, 32], anchor: [30, 32] },
];

const motionState = new Map();

let poseLandmarker;
let drawingUtils;
let previousLandmarks;
let smoothedLandmarks;
let lastVideoTime = -1;
let animationId;
let cameraStartedAt = 0;

startButton.addEventListener("click", startApp);
window.addEventListener("resize", resizeCanvas);

async function startApp() {
  startButton.disabled = true;
  setStatus("Carregando detector corporal", "warn");

  try {
    await setupPoseDetector();
    await setupCamera();
    cameraStartedAt = performance.now();
    emptyState.classList.add("hidden");
    setStatus("Camera ligada", "live");
    detectLoop();
  } catch (error) {
    console.error(error);
    startButton.disabled = false;
    setStatus("Nao foi possivel iniciar a camera", "warn");
    emptyState.classList.remove("hidden");
    emptyState.querySelector("p").textContent =
      "Confira a permissao da camera e tente novamente";
  }
}

async function setupPoseDetector() {
  if (poseLandmarker) return;

  const vision = await FilesetResolver.forVisionTasks(
    "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.14/wasm",
  );

  poseLandmarker = await PoseLandmarker.createFromOptions(vision, {
    baseOptions: {
      modelAssetPath:
        "https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_lite/float16/latest/pose_landmarker_lite.task",
      delegate: "GPU",
    },
    runningMode: "VIDEO",
    numPoses: 1,
    minPoseDetectionConfidence: 0.55,
    minPosePresenceConfidence: 0.55,
    minTrackingConfidence: 0.55,
  });

  drawingUtils = new DrawingUtils(ctx);
}

async function setupCamera() {
  const stream = await navigator.mediaDevices.getUserMedia({
    audio: false,
    video: {
      facingMode: "user",
      width: { ideal: 1280 },
      height: { ideal: 720 },
    },

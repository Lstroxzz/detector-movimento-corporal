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
const ctx = canvas.getContext("2d");

const bodyGroups = [
  { name: "Cabeca", points: [0, 1, 2, 3, 4, 5, 6, 7, 8] },
  { name: "Ombro esquerdo", points: [11] },
  { name: "Ombro direito", points: [12] },
  { name: "Braco esquerdo", points: [11, 13, 15] },
  { name: "Braco direito", points: [12, 14, 16] },
  { name: "Mao esquerda", points: [15, 17, 19, 21] },
  { name: "Mao direita", points: [16, 18, 20, 22] },
  { name: "Tronco", points: [11, 12, 23, 24] },
  { name: "Quadril esquerdo", points: [23] },
  { name: "Quadril direito", points: [24] },
  { name: "Perna esquerda", points: [23, 25, 27] },
  { name: "Perna direita", points: [24, 26, 28] },
  { name: "Pe esquerdo", points: [27, 29, 31] },
  { name: "Pe direito", points: [28, 30, 32] },
];

let poseLandmarker;
let drawingUtils;
let previousLandmarks;
let lastVideoTime = -1;
let animationId;

startButton.addEventListener("click", startApp);
window.addEventListener("resize", resizeCanvas);

async function startApp() {
  startButton.disabled = true;
  setStatus("Carregando detector corporal", "warn");

  try {
    await setupPoseDetector();
    await setupCamera();
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
  });

  video.srcObject = stream;
  await video.play();
  resizeCanvas();
}

function detectLoop() {
  if (video.currentTime !== lastVideoTime) {
    lastVideoTime = video.currentTime;
    const result = poseLandmarker.detectForVideo(video, performance.now());
    render(result);
  }

  animationId = requestAnimationFrame(detectLoop);
}

function render(result) {
  resizeCanvas();
  ctx.clearRect(0, 0, canvas.width, canvas.height);

  const landmarks = result.landmarks?.[0];
  if (!landmarks) {
    previousLandmarks = undefined;
    renderMotionList([]);
    setStatus("Procurando uma pessoa na imagem", "warn");
    return;
  }

  setStatus("Detectando movimento", "live");
  drawPose(landmarks);

  const movingGroups = getMovingGroups(landmarks);
  drawMovingLabels(landmarks, movingGroups);
  renderMotionList(movingGroups);
  previousLandmarks = landmarks.map((point) => ({ ...point }));
}

function drawPose(landmarks) {
  ctx.save();
  ctx.scale(-1, 1);
  ctx.translate(-canvas.width, 0);

  drawingUtils.drawConnectors(landmarks, PoseLandmarker.POSE_CONNECTIONS, {
    color: "rgba(255, 255, 255, 0.62)",
    lineWidth: 3,
  });

  drawingUtils.drawLandmarks(landmarks, {
    color: "#34d399",
    fillColor: "#121416",
    lineWidth: 2,
    radius: 4,
  });

  ctx.restore();
}

function getMovingGroups(landmarks) {
  if (!previousLandmarks) return [];

  const threshold = Number(sensitivity.value) / 1000;

  return bodyGroups
    .map((group) => {
      const visiblePoints = group.points
        .map((index) => [landmarks[index], previousLandmarks[index]])
        .filter(([current, previous]) => current && previous && current.visibility > 0.45);

      if (!visiblePoints.length) return null;

      const motion =
        visiblePoints.reduce((total, [current, previous]) => {
          const dx = current.x - previous.x;
          const dy = current.y - previous.y;
          return total + Math.hypot(dx, dy);
        }, 0) / visiblePoints.length;

      return motion > threshold ? { ...group, motion } : null;
    })
    .filter(Boolean)
    .sort((a, b) => b.motion - a.motion)
    .slice(0, 7);
}

function drawMovingLabels(landmarks, movingGroups) {
  movingGroups.forEach((group) => {
    const anchor = averagePoint(group.points.map((index) => landmarks[index]).filter(Boolean));
    if (!anchor) return;

    const x = canvas.width - anchor.x * canvas.width;
    const y = anchor.y * canvas.height;
    const label = group.name;

    ctx.font = "700 14px Inter, system-ui, sans-serif";
    const metrics = ctx.measureText(label);
    const boxWidth = metrics.width + 22;
    const boxHeight = 30;
    const boxX = clamp(x - boxWidth / 2, 8, canvas.width - boxWidth - 8);
    const boxY = clamp(y - 44, 8, canvas.height - boxHeight - 8);

    ctx.fillStyle = "rgba(18, 20, 22, 0.88)";
    roundRect(ctx, boxX, boxY, boxWidth, boxHeight, 6);
    ctx.fill();
    ctx.strokeStyle = "#34d399";
    ctx.lineWidth = 1;
    ctx.stroke();
    ctx.fillStyle = "#f3f6f7";
    ctx.fillText(label, boxX + 11, boxY + 20);

    ctx.beginPath();
    ctx.arc(x, y, 9, 0, Math.PI * 2);
    ctx.fillStyle = "rgba(52, 211, 153, 0.34)";
    ctx.fill();
    ctx.strokeStyle = "#34d399";
    ctx.stroke();
  });
}

function renderMotionList(groups) {
  motionList.replaceChildren();

  if (!groups.length) {
    const item = document.createElement("li");
    item.textContent = "Nenhum movimento detectado";
    motionList.appendChild(item);
    return;
  }

  groups.forEach((group) => {
    const item = document.createElement("li");
    const name = document.createElement("strong");
    const amount = document.createElement("span");
    name.textContent = group.name;
    amount.textContent = `${Math.round(group.motion * 1000)}`;
    item.append(name, amount);
    motionList.appendChild(item);
  });
}

function averagePoint(points) {
  if (!points.length) return undefined;

  const total = points.reduce(
    (sum, point) => ({
      x: sum.x + point.x,
      y: sum.y + point.y,
    }),
    { x: 0, y: 0 },
  );

  return {
    x: total.x / points.length,
    y: total.y / points.length,
  };
}

function resizeCanvas() {
  const rect = canvas.getBoundingClientRect();
  const width = Math.round(rect.width);
  const height = Math.round(rect.height);

  if (canvas.width !== width || canvas.height !== height) {
    canvas.width = width;
    canvas.height = height;
  }
}

function setStatus(message, state) {
  statusText.textContent = message;
  statusDot.className = `status-dot ${state}`;
}

function clamp(value, min, max) {
  return Math.min(Math.max(value, min), max);
}

function roundRect(context, x, y, width, height, radius) {
  context.beginPath();
  context.moveTo(x + radius, y);
  context.lineTo(x + width - radius, y);
  context.quadraticCurveTo(x + width, y, x + width, y + radius);
  context.lineTo(x + width, y + height - radius);
  context.quadraticCurveTo(x + width, y + height, x + width - radius, y + height);
  context.lineTo(x + radius, y + height);
  context.quadraticCurveTo(x, y + height, x, y + height - radius);
  context.lineTo(x, y + radius);
  context.quadraticCurveTo(x, y, x + radius, y);
  context.closePath();
}

window.addEventListener("beforeunload", () => {
  cancelAnimationFrame(animationId);
  video.srcObject?.getTracks().forEach((track) => track.stop());
});

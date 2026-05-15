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

let FilesetResolver;
let PoseLandmarker;
let DrawingUtils;
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

  if (!FilesetResolver || !PoseLandmarker || !DrawingUtils) {
    ({ FilesetResolver, PoseLandmarker, DrawingUtils } = await import(
      "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.14"
    ));
  }

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
    smoothedLandmarks = undefined;
    motionState.clear();
    beautyBanner.classList.remove("visible");
    renderMotionList([]);
    setStatus("Procurando uma pessoa na imagem", "warn");
    return;
  }

  smoothedLandmarks = smoothLandmarks(landmarks, smoothedLandmarks);
  const readyForSpecialMessage = performance.now() - cameraStartedAt >= 4000;
  beautyBanner.classList.toggle("visible", readyForSpecialMessage);
  setStatus("Detectando movimento", "live");
  drawPose(smoothedLandmarks);

  const movingGroups = getMovingGroups(smoothedLandmarks);
  drawMovingLabels(smoothedLandmarks, movingGroups);
  renderMotionList(movingGroups);
  previousLandmarks = smoothedLandmarks.map((point) => ({ ...point }));
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
        .filter(([current, previous]) => current && previous && getVisibility(current) > 0.5);

      if (!visiblePoints.length) {
        motionState.set(group.name, (motionState.get(group.name) || 0) * 0.78);
        return null;
      }

      const rawMotion =
        visiblePoints.reduce((total, [current, previous]) => {
          const dx = current.x - previous.x;
          const dy = current.y - previous.y;
          return total + Math.hypot(dx, dy);
        }, 0) / visiblePoints.length;

      const previousMotion = motionState.get(group.name) || 0;
      const motion = previousMotion * 0.62 + rawMotion * 0.38;
      motionState.set(group.name, motion);

      return motion > threshold ? { ...group, motion } : null;
    })
    .filter(Boolean)
    .sort((a, b) => b.motion - a.motion)
    .slice(0, 8);
}

function drawMovingLabels(landmarks, movingGroups) {
  movingGroups.forEach((group) => {
    const anchor = averagePoint(group.anchor.map((index) => landmarks[index]).filter(Boolean));
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

function smoothLandmarks(current, previous) {
  const alpha = previous ? 0.36 : 1;

  return current.map((point, index) => {
    const oldPoint = previous?.[index];
    if (!oldPoint || getVisibility(point) < 0.35) return { ...point };

    return {
      ...point,
      x: oldPoint.x + (point.x - oldPoint.x) * alpha,
      y: oldPoint.y + (point.y - oldPoint.y) * alpha,
      z: oldPoint.z + (point.z - oldPoint.z) * alpha,
    };
  });
}

function getVisibility(point) {
  return point.visibility ?? point.presence ?? 1;
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

function ensureBeautyBanner() {
  const existingBanner = document.querySelector("#beautyBanner");
  if (existingBanner) return existingBanner;

  const cameraFrame = document.querySelector(".camera-frame");
  const banner = document.createElement("div");
  banner.id = "beautyBanner";
  banner.className = "beauty-banner";
  banner.setAttribute("aria-live", "polite");
  banner.textContent = "garota mais bonita do mundo detectada";
  cameraFrame.appendChild(banner);

  if (!document.querySelector("#beautyBannerStyles")) {
    const style = document.createElement("style");
    style.id = "beautyBannerStyles";
    style.textContent = `
      .beauty-banner {
        position: absolute;
        left: 50%;
        top: 18px;
        z-index: 2;
        width: min(560px, calc(100% - 32px));
        padding: 13px 18px;
        border: 1px solid rgba(52, 211, 153, 0.55);
        border-radius: 8px;
        background: rgba(18, 20, 22, 0.82);
        box-shadow: 0 14px 34px rgba(0, 0, 0, 0.28);
        color: #f3f6f7;
        font-weight: 900;
        text-align: center;
        text-transform: uppercase;
        transform: translate(-50%, -90px);
        opacity: 0;
        transition: opacity 180ms ease, transform 220ms ease;
      }

      .beauty-banner.visible {
        transform: translate(-50%, 0);
        opacity: 1;
      }
    `;
    document.head.appendChild(style);
  }

  return banner;
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

const video = document.querySelector("#camera");
const canvas = document.querySelector("#overlay");
const emptyState = document.querySelector("#emptyState");
const emptyStateText = document.querySelector("#emptyStateText");
const startButton = document.querySelector("#startButton");
const statusText = document.querySelector("#statusText");
const statusDot = document.querySelector("#statusDot");
const motionList = document.querySelector("#motionList");
const sensitivity = document.querySelector("#sensitivity");
const ctx = canvas.getContext("2d");

const bodyGroups = [
  { name: "Cabeca", source: "pose", points: [0, 2, 5, 7, 8], anchor: [0, 7, 8] },
  { name: "Pescoco", source: "pose", points: [7, 8, 11, 12], anchor: [7, 8, 11, 12] },
  { name: "Ombro esquerdo", source: "pose", points: [11], anchor: [11] },
  { name: "Ombro direito", source: "pose", points: [12], anchor: [12] },
  { name: "Braco esquerdo", source: "pose", points: [11, 13], anchor: [13] },
  { name: "Braco direito", source: "pose", points: [12, 14], anchor: [14] },
  { name: "Antebraco esquerdo", source: "pose", points: [13, 15], anchor: [13, 15] },
  { name: "Antebraco direito", source: "pose", points: [14, 16], anchor: [14, 16] },
  { name: "Peito", source: "pose", points: [11, 12], anchor: [11, 12] },
  { name: "Abdomen", source: "pose", points: [11, 12, 23, 24], anchor: [11, 12, 23, 24] },
  { name: "Quadril", source: "pose", points: [23, 24], anchor: [23, 24] },
  { name: "Coxa esquerda", source: "pose", points: [23, 25], anchor: [23, 25] },
  { name: "Coxa direita", source: "pose", points: [24, 26], anchor: [24, 26] },
  { name: "Canela esquerda", source: "pose", points: [25, 27], anchor: [25, 27] },
  { name: "Canela direita", source: "pose", points: [26, 28], anchor: [26, 28] },
  { name: "Pe esquerdo", source: "pose", points: [27, 29, 31], anchor: [29, 31] },
  { name: "Pe direito", source: "pose", points: [28, 30, 32], anchor: [30, 32] },
];

const handParts = [
  { name: "Palma", points: [0, 1, 5, 9, 13, 17], anchor: [0, 5, 9, 13, 17] },
  { name: "Polegar", points: [1, 2, 3, 4], anchor: [3, 4] },
  { name: "Ponta do polegar", points: [4], anchor: [4] },
  { name: "Indicador", points: [5, 6, 7, 8], anchor: [7, 8] },
  { name: "Ponta do indicador", points: [8], anchor: [8] },
  { name: "Dedo medio", points: [9, 10, 11, 12], anchor: [11, 12] },
  { name: "Ponta do dedo medio", points: [12], anchor: [12] },
  { name: "Anelar", points: [13, 14, 15, 16], anchor: [15, 16] },
  { name: "Ponta do anelar", points: [16], anchor: [16] },
  { name: "Mindinho", points: [17, 18, 19, 20], anchor: [19, 20] },
  { name: "Ponta do mindinho", points: [20], anchor: [20] },
];

const faceParts = [
  { name: "Rosto", points: [10, 152, 234, 454], anchor: [10, 152, 234, 454] },
  { name: "Testa", points: [10, 151, 9], anchor: [10] },
  { name: "Olho esquerdo", points: [33, 133, 159, 145], anchor: [33, 133] },
  { name: "Olho direito", points: [263, 362, 386, 374], anchor: [263, 362] },
  { name: "Nariz", points: [1, 2, 98, 327], anchor: [1] },
  { name: "Boca", points: [13, 14, 61, 291], anchor: [13, 14] },
  { name: "Queixo", points: [152, 199, 200], anchor: [152] },
  { name: "Bochecha esquerda", points: [234, 93, 132], anchor: [234] },
  { name: "Bochecha direita", points: [454, 323, 361], anchor: [454] },
];

const motionState = new Map();

let FilesetResolver;
let PoseLandmarker;
let HandLandmarker;
let FaceLandmarker;
let DrawingUtils;
let poseLandmarker;
let handLandmarker;
let faceLandmarker;
let drawingUtils;
let previousFrame;
let smoothedFrame;
let lastVideoTime = -1;
let animationId;
let cameraStream;
let vision;
let isStarting = false;
let isDetecting = false;

startButton.addEventListener("click", startApp);
window.addEventListener("resize", resizeCanvas);

async function startApp() {
  if (isStarting) return;
  isStarting = true;
  startButton.disabled = true;
  setStatus("Carregando detectores", "warn");

  try {
    await setupDetectors();
    await setupCamera();
    emptyState.classList.add("hidden");
    lastVideoTime = -1;
    isDetecting = true;
    setStatus("Camera ligada", "live");
    detectLoop();
  } catch (error) {
    console.error(error);
    stopCamera();
    setStatus("Nao foi possivel iniciar a camera", "warn");
    emptyState.classList.remove("hidden");
    emptyStateText.textContent = getCameraErrorMessage(error);
  } finally {
    isStarting = false;
    startButton.disabled = false;
  }
}

async function setupDetectors() {
  if (poseLandmarker && handLandmarker && faceLandmarker) return;

  if (!FilesetResolver) {
    ({ FilesetResolver, PoseLandmarker, HandLandmarker, FaceLandmarker, DrawingUtils } =
      await import("https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.14"));
  }

  vision ??= await FilesetResolver.forVisionTasks(
    "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.14/wasm",
  );

  poseLandmarker = await createWithFallback(PoseLandmarker, {
    modelAssetPath:
      "https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_lite/float16/latest/pose_landmarker_lite.task",
    runningMode: "VIDEO",
    numPoses: 1,
    minPoseDetectionConfidence: 0.55,
    minPosePresenceConfidence: 0.55,
    minTrackingConfidence: 0.55,
  });

  handLandmarker = await createWithFallback(HandLandmarker, {
    modelAssetPath:
      "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/latest/hand_landmarker.task",
    runningMode: "VIDEO",
    numHands: 2,
    minHandDetectionConfidence: 0.55,
    minHandPresenceConfidence: 0.55,
    minTrackingConfidence: 0.55,
  });

  faceLandmarker = await createWithFallback(FaceLandmarker, {
    modelAssetPath:
      "https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/latest/face_landmarker.task",
    runningMode: "VIDEO",
    numFaces: 1,
    minFaceDetectionConfidence: 0.55,
    minFacePresenceConfidence: 0.55,
    minTrackingConfidence: 0.55,
  });

  drawingUtils = new DrawingUtils(ctx);
}

async function createWithFallback(Landmarker, { modelAssetPath, ...options }) {
  const create = (delegate) =>
    Landmarker.createFromOptions(vision, {
      ...options,
      baseOptions: { modelAssetPath, delegate },
    });

  try {
    return await create("GPU");
  } catch (gpuError) {
    console.warn("GPU indisponível; usando CPU para detecção.", gpuError);
    return create("CPU");
  }
}

async function setupCamera() {
  if (!navigator.mediaDevices?.getUserMedia) {
    throw new Error("Este navegador não oferece acesso à câmera.");
  }

  stopCamera();
  cameraStream = await navigator.mediaDevices.getUserMedia({
    audio: false,
    video: {
      facingMode: "user",
      width: { ideal: 1280 },
      height: { ideal: 720 },
    },
  });

  video.srcObject = cameraStream;
  await video.play();
  resizeCanvas();
}

function detectLoop() {
  if (!isDetecting) return;

  try {
    if (video.currentTime !== lastVideoTime) {
      lastVideoTime = video.currentTime;
      const now = performance.now();
      const frame = {
        pose: poseLandmarker.detectForVideo(video, now).landmarks?.[0],
        hands: readHands(handLandmarker.detectForVideo(video, now)),
        face: faceLandmarker.detectForVideo(video, now).faceLandmarks?.[0],
      };
      render(frame);
    }
  } catch (error) {
    console.error(error);
    isDetecting = false;
    stopCamera();
    setStatus("Falha ao processar a câmera", "warn");
    emptyStateText.textContent = "O detector encontrou um erro. Tente ligar a câmera novamente.";
    emptyState.classList.remove("hidden");
  }

  if (isDetecting) animationId = requestAnimationFrame(detectLoop);
}

function render(frame) {
  resizeCanvas();
  ctx.clearRect(0, 0, canvas.width, canvas.height);

  if (!frame.pose && !frame.hands.length && !frame.face) {
    previousFrame = undefined;
    smoothedFrame = undefined;
    motionState.clear();
    renderMotionList([]);
    setStatus("Procurando uma pessoa na imagem", "warn");
    return;
  }

  smoothedFrame = smoothFrame(frame, smoothedFrame);
  setStatus("Detectando corpo, rosto e maos", "live");
  drawDetections(smoothedFrame);

  const movingGroups = getMovingGroups(smoothedFrame);
  drawMovingLabels(smoothedFrame, movingGroups);
  renderMotionList(movingGroups);
  previousFrame = cloneFrame(smoothedFrame);
}

function readHands(result) {
  return (result.landmarks || []).map((landmarks, index) => {
    const handedness = result.handednesses?.[index]?.[0]?.categoryName;
    return {
      id: handedness || `Mao ${index + 1}`,
      label: handedness === "Left" ? "esquerda" : handedness === "Right" ? "direita" : `${index + 1}`,
      landmarks,
    };
  });
}

function drawDetections(frame) {
  ctx.save();
  ctx.scale(-1, 1);
  ctx.translate(-canvas.width, 0);

  if (frame.pose) {
    drawingUtils.drawConnectors(frame.pose, PoseLandmarker.POSE_CONNECTIONS, {
      color: "rgba(255, 255, 255, 0.56)",
      lineWidth: 3,
    });
    drawingUtils.drawLandmarks(frame.pose, {
      color: "#34d399",
      fillColor: "#121416",
      lineWidth: 2,
      radius: 4,
    });
  }

  frame.hands.forEach((hand) => {
    drawingUtils.drawConnectors(hand.landmarks, HandLandmarker.HAND_CONNECTIONS, {
      color: "rgba(56, 189, 248, 0.92)",
      lineWidth: 3,
    });
    drawingUtils.drawLandmarks(hand.landmarks, {
      color: "#38bdf8",
      fillColor: "#081216",
      lineWidth: 2,
      radius: 3,
    });
  });

  if (frame.face) {
    drawingUtils.drawLandmarks(frame.face, {
      color: "rgba(245, 158, 11, 0.72)",
      fillColor: "rgba(18, 20, 22, 0.55)",
      lineWidth: 1,
      radius: 1.2,
    });
  }

  ctx.restore();
}

function getMovingGroups(frame) {
  if (!previousFrame) return [];

  const groups = [];
  const threshold = Number(sensitivity.value) / 1000;

  if (frame.pose && previousFrame.pose) {
    bodyGroups.forEach((group) => {
      groups.push(measureGroup(group.name, group.points, group.anchor, frame.pose, previousFrame.pose, threshold));
    });
  }

  if (frame.face && previousFrame.face) {
    faceParts.forEach((group) => {
      groups.push(measureGroup(group.name, group.points, group.anchor, frame.face, previousFrame.face, threshold * 0.7));
    });
  }

  frame.hands.forEach((hand) => {
    const previousHand = previousFrame.hands.find((item) => item.id === hand.id);
    if (!previousHand) return;

    handParts.forEach((group) => {
      groups.push(
        measureGroup(
          `${group.name} ${hand.label}`,
          group.points,
          group.anchor,
          hand.landmarks,
          previousHand.landmarks,
          threshold * 0.75,
        ),
      );
    });
  });

  return groups
    .filter(Boolean)
    .sort((a, b) => b.motion - a.motion)
    .slice(0, 12);
}

function measureGroup(name, pointIndexes, anchorIndexes, currentLandmarks, previousLandmarksForGroup, threshold) {
  const visiblePoints = pointIndexes
    .map((index) => [currentLandmarks[index], previousLandmarksForGroup[index]])
    .filter(([current, previous]) => current && previous && getVisibility(current) > 0.42);

  if (!visiblePoints.length) {
    motionState.set(name, (motionState.get(name) || 0) * 0.78);
    return null;
  }

  const rawMotion =
    visiblePoints.reduce((total, [current, previous]) => {
      const dx = current.x - previous.x;
      const dy = current.y - previous.y;
      return total + Math.hypot(dx, dy);
    }, 0) / visiblePoints.length;

  const previousMotion = motionState.get(name) || 0;
  const motion = previousMotion * 0.56 + rawMotion * 0.44;
  motionState.set(name, motion);

  return motion > threshold
    ? {
        name,
        motion,
        anchor: averagePoint(anchorIndexes.map((index) => currentLandmarks[index]).filter(Boolean)),
      }
    : null;
}

function drawMovingLabels(frame, movingGroups) {
  movingGroups.forEach((group) => {
    if (!group.anchor) return;

    const x = canvas.width - group.anchor.x * canvas.width;
    const y = group.anchor.y * canvas.height;
    const label = group.name;

    ctx.font = "700 13px Inter, system-ui, sans-serif";
    const metrics = ctx.measureText(label);
    const boxWidth = metrics.width + 20;
    const boxHeight = 28;
    const boxX = clamp(x - boxWidth / 2, 8, canvas.width - boxWidth - 8);
    const boxY = clamp(y - 42, 8, canvas.height - boxHeight - 8);

    ctx.fillStyle = "rgba(18, 20, 22, 0.88)";
    roundRect(ctx, boxX, boxY, boxWidth, boxHeight, 6);
    ctx.fill();
    ctx.strokeStyle = "#34d399";
    ctx.lineWidth = 1;
    ctx.stroke();
    ctx.fillStyle = "#f3f6f7";
    ctx.fillText(label, boxX + 10, boxY + 19);

    ctx.beginPath();
    ctx.arc(x, y, 8, 0, Math.PI * 2);
    ctx.fillStyle = "rgba(52, 211, 153, 0.34)";
    ctx.fill();
    ctx.strokeStyle = "#34d399";
    ctx.stroke();
  });
}

function smoothFrame(current, previous) {
  return {
    pose: smoothLandmarks(current.pose, previous?.pose, 0.38),
    face: smoothLandmarks(current.face, previous?.face, 0.46),
    hands: current.hands.map((hand) => {
      const previousHand = previous?.hands.find((item) => item.id === hand.id);
      return {
        ...hand,
        landmarks: smoothLandmarks(hand.landmarks, previousHand?.landmarks, 0.5),
      };
    }),
  };
}

function smoothLandmarks(current, previous, alpha) {
  if (!current) return undefined;

  return current.map((point, index) => {
    const oldPoint = previous?.[index];
    if (!oldPoint || getVisibility(point) < 0.28) return { ...point };

    return {
      ...point,
      x: oldPoint.x + (point.x - oldPoint.x) * alpha,
      y: oldPoint.y + (point.y - oldPoint.y) * alpha,
      z: oldPoint.z + (point.z - oldPoint.z) * alpha,
    };
  });
}

function cloneFrame(frame) {
  return {
    pose: frame.pose?.map((point) => ({ ...point })),
    face: frame.face?.map((point) => ({ ...point })),
    hands: frame.hands.map((hand) => ({
      ...hand,
      landmarks: hand.landmarks.map((point) => ({ ...point })),
    })),
  };
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
  stopCamera();
});

function stopCamera() {
  isDetecting = false;
  cancelAnimationFrame(animationId);
  cameraStream?.getTracks().forEach((track) => track.stop());
  cameraStream = undefined;
  video.srcObject = null;
}

function getCameraErrorMessage(error) {
  const errors = {
    NotAllowedError: "Permissão da câmera negada. Libere o acesso e tente novamente.",
    NotFoundError: "Nenhuma câmera foi encontrada neste dispositivo.",
    NotReadableError: "A câmera está sendo usada por outro aplicativo. Feche-o e tente novamente.",
  };
  return errors[error.name] || "Não foi possível iniciar. Confira a câmera e tente novamente.";
}

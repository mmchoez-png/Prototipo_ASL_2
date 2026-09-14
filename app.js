let modeloActivo = null;
let modoActual = 'abecedario';
let frameBuffer = [];
const FRAMES_TARGET = 60;

// CAMERA SWITCH CONTROL
let currentFacingMode = 'user'; // 'user' (front) or 'environment' (back)
let cameraInstance = null;
let handsInstance = null;

// ALPHABET LOCKS & TEXT
let letraRegistradaActual = "";
let sentenceText = "";
let letraCandidata = "";
let contadorEstabilidad = 0;
const FRAMES_REQUERIDOS_ABC = 8; 

// PHRASES LOCK & COOLDOWN
let fraseRegistradaActual = "";
let cooldownFrase = false;

// SPEECH RECOGNITION (LISTENER MIC)
let recognition = null;
let isListening = false;

if ('webkitSpeechRecognition' in window || 'SpeechRecognition' in window) {
  const SpeechRecognition = window.SpeechRecognition || window.webkitSpeechRecognition;
  recognition = new SpeechRecognition();
  recognition.continuous = true;
  recognition.interimResults = true;
  recognition.lang = 'en-US'; // SET TO ENGLISH

  recognition.onresult = (event) => {
    let transcript = '';
    for (let i = event.resultIndex; i < event.results.length; i++) {
      transcript += event.results[i][0].transcript;
    }
    const sttBox = document.getElementById('speech-output');
    if (sttBox) sttBox.textContent = transcript;
  };

  recognition.onerror = () => stopMicUI();
  recognition.onend = () => { if (isListening) recognition.start(); };
}

function toggleMic() {
  if (!recognition) return alert("Your browser does not support speech recognition.");
  const btn = document.getElementById('mic-btn');
  const txt = document.getElementById('mic-text');

  if (isListening) {
    recognition.stop();
    isListening = false;
    stopMicUI();
  } else {
    recognition.start();
    isListening = true;
    if (btn) btn.classList.add('recording');
    if (txt) txt.textContent = "Listening...";
  }
}

function stopMicUI() {
  const btn = document.getElementById('mic-btn');
  const txt = document.getElementById('mic-text');
  if (btn) btn.classList.remove('recording');
  if (txt) txt.textContent = "Speak (Listener)";
}

function limpiarVozOyente() {
  const sttBox = document.getElementById('speech-output');
  if (sttBox) sttBox.textContent = "Press \"Speak (Listener)\" to listen...";
}

// TOGGLE FRONT / BACK CAMERA
async function toggleCamera() {
  currentFacingMode = (currentFacingMode === 'user') ? 'environment' : 'user';
  
  const canvas = document.getElementById('output_canvas');
  if (canvas) {
    canvas.style.transform = (currentFacingMode === 'user') ? 'scaleX(-1)' : 'scaleX(1)';
  }

  if (cameraInstance) {
    await cameraInstance.stop();
  }

  const videoElement = document.getElementById('webcam');
  cameraInstance = new Camera(videoElement, {
    onFrame: async () => { await handsInstance.send({ image: videoElement }); },
    width: 640,
    height: 480,
    facingMode: currentFacingMode
  });
  cameraInstance.start();
}

// LOAD JSON MODELS
async function cargarModeloAuto(modo) {
  const ruta = modo === 'abecedario' ? './modelo_abecedario.json' : './modelo_frases.json';
  try {
    const res = await fetch(ruta);
    if (!res.ok) throw new Error(`Status ${res.status}`);
    modeloActivo = await res.json();
    document.getElementById('output-class').textContent = "Ready for signs 👋";
  } catch (e) {
    console.error(`Error loading ${ruta}:`, e);
    document.getElementById('output-class').textContent = `Error loading ${ruta}`;
  }
}

function reproducirVoz(texto) {
  window.speechSynthesis.cancel();
  const utterance = new SpeechSynthesisUtterance(texto);
  utterance.lang = 'en-US'; // SET TO ENGLISH
  window.speechSynthesis.speak(utterance);
}

function relu(arr) { return arr.map(v => Math.max(0, v)); }

function matMulAdd(input, W, b) {
  const M = b.length;
  const N = input.length;
  const output = new Array(M).fill(0);
  for (let j = 0; j < M; j++) {
    let sum = b[j];
    for (let i = 0; i < N; i++) sum += input[i] * W[i][j];
    output[j] = sum;
  }
  return output;
}

// PREDICTION ENGINE
function predict(inputVector, handsInFrame) {
  if (!modeloActivo) return;

  const confEl = document.getElementById('output-confidence');

  if (!handsInFrame) {
    document.getElementById('output-class').textContent = "Ready for signs 👋";
    if (!cooldownFrase && confEl) confEl.textContent = "Confidence: --";
    letraRegistradaActual = "";
    fraseRegistradaActual = "";
    letraCandidata = "";
    contadorEstabilidad = 0;
    return;
  }

  if (modeloActivo.weights && modeloActivo.weights[0]) {
    const expectedLen = modeloActivo.weights[0].length;
    if (inputVector.length !== expectedLen) {
      if (inputVector.length < expectedLen) {
        let filled = [];
        while (filled.length < expectedLen) filled.push(...inputVector);
        inputVector = filled.slice(0, expectedLen);
      } else {
        inputVector = inputVector.slice(0, expectedLen);
      }
    }
  }

  let h1 = relu(matMulAdd(inputVector, modeloActivo.weights[0], modeloActivo.biases[0]));
  let h2 = relu(matMulAdd(h1, modeloActivo.weights[1], modeloActivo.biases[1]));
  let logits = matMulAdd(h2, modeloActivo.weights[2], modeloActivo.biases[2]);

  const maxLogit = Math.max(...logits);
  const exps = logits.map(l => Math.exp(l - maxLogit));
  const sumExps = exps.reduce((a, b) => a + b, 0);
  const probs = exps.map(e => e / sumExps);

  let maxIdx = 0;
  for (let i = 1; i < probs.length; i++) {
    if (probs[i] > probs[maxIdx]) maxIdx = i;
  }

  const bestClass = modeloActivo.classes[maxIdx];
  const confPercent = Math.round(probs[maxIdx] * 100);

  document.getElementById('output-class').textContent = bestClass;
  if (!cooldownFrase && confEl) confEl.textContent = `Confidence: ${confPercent}%`;

  // ALPHABET RULE (100% Strict + Stability)
  if (modoActual === 'abecedario') {
    if (confPercent === 100) {
      if (bestClass === letraCandidata) {
        contadorEstabilidad++;
      } else {
        letraCandidata = bestClass;
        contadorEstabilidad = 1;
      }

      if (contadorEstabilidad >= FRAMES_REQUERIDOS_ABC && bestClass !== letraRegistradaActual) {
        reproducirVoz(bestClass);
        sentenceText += bestClass;
        const box = document.getElementById('sentence-display');
        if (box) box.textContent = sentenceText;
        letraRegistradaActual = bestClass;
      }
    } else {
      contadorEstabilidad = 0;
      letraCandidata = "";
      if (confPercent < 50) letraRegistradaActual = "";
    }

  // PHRASES RULE (Dynamic Threshold + Cooldown Buffer Clear)
  } else {
    const UMBRAL_FRASES = 65;

    if (confPercent >= UMBRAL_FRASES && !cooldownFrase) {
      reproducirVoz(bestClass);
      fraseRegistradaActual = bestClass;

      frameBuffer = [];
      cooldownFrase = true;

      if (confEl) {
        confEl.innerHTML = `Confidence: ${confPercent}% <span style="color:#ef4444; font-weight:bold;">⏳ (Ready in 1.8s)</span>`;
      }

      setTimeout(() => {
        cooldownFrase = false;
        if (confEl) confEl.textContent = `Confidence: --`;
      }, 1800);
    }
  }
}

function extractLandmarks(results) {
  let mano1 = [], mano2 = [];
  if (results.multiHandLandmarks && results.multiHandLandmarks.length > 0) {
    let lm1 = results.multiHandLandmarks[0], w1_x = lm1[0].x, w1_y = lm1[0].y;
    let dist1 = Math.hypot(lm1[9].x - w1_x, lm1[9].y - w1_y) || 1.0;
    lm1.forEach(pt => mano1.push((pt.x - w1_x) / dist1, (pt.y - w1_y) / dist1));

    if (results.multiHandLandmarks.length > 1) {
      let lm2 = results.multiHandLandmarks[1], w2_x = lm2[0].x, w2_y = lm2[0].y;
      let dist2 = Math.hypot(lm2[9].x - w2_x, lm2[9].y - w2_y) || 1.0;
      lm2.forEach(pt => mano2.push((pt.x - w2_x) / dist2, (pt.y - w2_y) / dist2));
    } else {
      mano2 = new Array(42).fill(0.0);
    }
  } else {
    mano1 = new Array(42).fill(0.0);
    mano2 = new Array(42).fill(0.0);
  }
  return [...mano1, ...mano2];
}

// CONSTRUCTOR BUTTON ACTIONS
function agregarEspacio() {
  sentenceText += " ";
  const box = document.getElementById('sentence-display');
  if (box) box.textContent = sentenceText;
}

function borrarPalabra() {
  let words = sentenceText.trim().split(" ");
  words.pop();
  sentenceText = words.join(" ") + (words.length > 0 ? " " : "");
  const box = document.getElementById('sentence-display');
  if (box) box.textContent = sentenceText || "...";
}

function limpiarTexto() {
  sentenceText = "";
  letraRegistradaActual = "";
  const box = document.getElementById('sentence-display');
  if (box) box.textContent = "...";
}

function leerOracion() {
  if (sentenceText.trim()) reproducirVoz(sentenceText);
}

// INITIALIZATION
function initModulo(tipo) {
  modoActual = tipo;
  cargarModeloAuto(tipo);

  const canvasElement = document.getElementById('output_canvas');
  const canvasCtx = canvasElement.getContext('2d');

  handsInstance = new Hands({
    locateFile: (file) => `https://cdn.jsdelivr.net/npm/@mediapipe/hands/${file}`
  });

  handsInstance.setOptions({
    maxNumHands: 2,
    modelComplexity: 1,
    minDetectionConfidence: 0.5,
    minTrackingConfidence: 0.5
  });

  handsInstance.onResults((results) => {
    canvasCtx.save();
    canvasCtx.clearRect(0, 0, 640, 480);
    canvasCtx.drawImage(results.image, 0, 0, 640, 480);

    const handsInFrame = results.multiHandLandmarks && results.multiHandLandmarks.length > 0;

    if (handsInFrame) {
      for (const landmarks of results.multiHandLandmarks) {
        drawConnectors(canvasCtx, landmarks, HAND_CONNECTIONS, { color: '#10b981', lineWidth: 3 });
        drawLandmarks(canvasCtx, landmarks, { color: '#6366f1', fillColor: '#ffffff', lineWidth: 1, radius: 4 });
      }
    }
    canvasCtx.restore();

    const points = extractLandmarks(results);

    if (modoActual === 'abecedario') {
      predict(points, handsInFrame);
    } else {
      frameBuffer.push(points);
      if (frameBuffer.length > FRAMES_TARGET) frameBuffer.shift();

      if (frameBuffer.length === FRAMES_TARGET) {
        predict(frameBuffer.flat(), handsInFrame);
      }
    }
  });

  const videoElement = document.getElementById('webcam');
  cameraInstance = new Camera(videoElement, {
    onFrame: async () => { await handsInstance.send({ image: videoElement }); },
    width: 640, height: 480,
    facingMode: currentFacingMode
  });
  cameraInstance.start();
}

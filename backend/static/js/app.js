// Dynamic Emoji mapping
const EMOTION_EMOJIS = {
  "neutral": "😐",
  "calm": "😌",
  "happy": "😊",
  "sad": "😢",
  "angry": "😡",
  "fearful": "😨",
  "disgust": "🤢",
  "surprised": "😲",
  "crying": "😭",
  "laughing": "😂"
};

// Emotion Colors Map matching backend
const EMOTION_COLORS = {
  "neutral": "#8D99AE",
  "calm": "#A8DADC",
  "happy": "#FFD166",
  "sad": "#457B9D",
  "angry": "#E63946",
  "fearful": "#9B5DE5",
  "disgust": "#4F772D",
  "surprised": "#F15BB5",
  "crying": "#0077B6",
  "laughing": "#FF9F1C"
};

// App State
let selectedFile = null;
let activeAudioUrl = null;
let currentAudioContext = null;
let history = [];
let darkMode = true;

// DOM Elements
const bodyEl = document.body;
const themeToggleBtn = document.getElementById('theme-toggle-btn');
const uploadDropzone = document.getElementById('upload-dropzone');
const audioFileInput = document.getElementById('audio-file-input');
const uploadPrompt = document.getElementById('upload-prompt');
const fileInfoContainer = document.getElementById('file-info-container');
const btnBrowse = document.getElementById('btn-browse');
const btnRemove = document.getElementById('btn-remove');
const btnAnalyze = document.getElementById('btn-analyze');
const fileNameEl = document.getElementById('file-name');
const fileSizeEl = document.getElementById('file-size');
const audioWidget = document.getElementById('audio-widget');
const actionsBox = document.getElementById('actions-box');
const loadingContainer = document.getElementById('loading-container');
const errorCard = document.getElementById('error-card');
const errorTextEl = document.getElementById('error-text');
const emptyState = document.getElementById('empty-state');
const resultsWrapper = document.getElementById('results-wrapper');
const resultEmoji = document.getElementById('result-emoji');
const resultEmotion = document.getElementById('result-emotion');
const resultConfidence = document.getElementById('result-confidence');
const resultMeterFill = document.getElementById('result-meter-fill');
const chartBarsContainer = document.getElementById('chart-bars-container');
const historySection = document.getElementById('history-section');
const historyList = document.getElementById('history-list');
const canvas = document.getElementById('waveform-canvas');

// --- Initialization ---
document.addEventListener('DOMContentLoaded', () => {
  loadThemePreference();
  loadHistory();
  setupEventHandlers();
});

// --- Theme Toggle ---
function loadThemePreference() {
  const savedTheme = localStorage.getItem('theme_preference');
  if (savedTheme === 'light') {
    darkMode = false;
    bodyEl.className = 'light-theme';
    themeToggleBtn.textContent = '🌙 Dark Mode';
  } else {
    darkMode = true;
    bodyEl.className = 'dark-theme';
    themeToggleBtn.textContent = '☀️ Light Mode';
  }
}

function toggleTheme() {
  darkMode = !darkMode;
  if (darkMode) {
    bodyEl.className = 'dark-theme';
    themeToggleBtn.textContent = '☀️ Light Mode';
    localStorage.setItem('theme_preference', 'dark');
  } else {
    bodyEl.className = 'light-theme';
    themeToggleBtn.textContent = '🌙 Dark Mode';
    localStorage.setItem('theme_preference', 'light');
  }
}

// --- Event Handlers Setup ---
function setupEventHandlers() {
  // Theme Toggle Button
  themeToggleBtn.addEventListener('click', toggleTheme);

  // File Upload Elements
  btnBrowse.addEventListener('click', (e) => {
    e.stopPropagation();
    audioFileInput.click();
  });
  
  uploadDropzone.addEventListener('click', () => {
    if (!selectedFile) audioFileInput.click();
  });

  audioFileInput.addEventListener('change', handleFileChange);

  // Drag and Drop
  ['dragenter', 'dragover'].forEach(eventName => {
    uploadDropzone.addEventListener(eventName, (e) => {
      e.preventDefault();
      e.stopPropagation();
      uploadDropzone.classList.add('drag-active');
    }, false);
  });

  ['dragleave', 'drop'].forEach(eventName => {
    uploadDropzone.addEventListener(eventName, (e) => {
      e.preventDefault();
      e.stopPropagation();
      uploadDropzone.classList.remove('drag-active');
    }, false);
  });

  uploadDropzone.addEventListener('drop', (e) => {
    const dt = e.dataTransfer;
    const files = dt.files;
    if (files.length) {
      validateAndSelectFile(files[0]);
    }
  });

  // Action Buttons
  btnRemove.addEventListener('click', (e) => {
    e.stopPropagation();
    clearSelectedFile();
  });

  btnAnalyze.addEventListener('click', runInference);
}

// --- File Handling and Verification ---
function handleFileChange(e) {
  if (e.target.files.length) {
    validateAndSelectFile(e.target.files[0]);
  }
}

function validateAndSelectFile(file) {
  // Validate extension
  const extension = file.name.split('.').pop().toLowerCase();
  if (extension !== 'wav' && extension !== 'mp3') {
    showError("Invalid file type. Only WAV and MP3 audio files are supported.");
    return;
  }

  // Validate size (50MB)
  const maxSize = 50 * 1024 * 1024;
  if (file.size > maxSize) {
    showError("File size is too large. Maximum limit is 50MB.");
    return;
  }

  hideError();
  selectedFile = file;

  // Setup metadata details
  fileNameEl.textContent = file.name;
  fileSizeEl.textContent = `${(file.size / (1024 * 1024)).toFixed(2)} MB`;

  // Create local object URL for playback
  if (activeAudioUrl) URL.revokeObjectURL(activeAudioUrl);
  activeAudioUrl = URL.createObjectURL(file);
  audioWidget.src = activeAudioUrl;

  // Toggle layout states
  uploadPrompt.style.display = 'none';
  fileInfoContainer.style.display = 'flex';
  uploadDropzone.classList.add('has-file');
  actionsBox.style.display = 'block';
  resultsWrapper.style.display = 'none';
  emptyState.style.display = 'flex';

  // Render static waveform visualization
  drawWaveform(file);
}

function clearSelectedFile() {
  selectedFile = null;
  audioFileInput.value = '';
  if (activeAudioUrl) {
    URL.revokeObjectURL(activeAudioUrl);
    activeAudioUrl = null;
  }
  audioWidget.src = '';
  
  uploadPrompt.style.display = 'block';
  fileInfoContainer.style.display = 'none';
  uploadDropzone.classList.remove('has-file');
  actionsBox.style.display = 'none';
  resultsWrapper.style.display = 'none';
  emptyState.style.display = 'flex';
  hideError();
}

// --- Waveform Decoder & Visualizer ---
function drawWaveform(file) {
  if (!canvas) return;
  const ctx = canvas.getContext('2d');
  const width = canvas.width;
  const height = canvas.height;

  // Clear canvas
  ctx.clearRect(0, 0, width, height);

  // Set up loading visual placeholder
  ctx.fillStyle = 'rgba(255, 107, 53, 0.2)';
  for (let i = 0; i < width; i += 4) {
    const h = 5 + Math.sin(i * 0.05) * 15;
    ctx.fillRect(i, (height - h) / 2, 2.5, h);
  }

  // Close active context if existing
  if (currentAudioContext && currentAudioContext.state !== 'closed') {
    currentAudioContext.close();
  }

  currentAudioContext = new (window.AudioContext || window.webkitAudioContext)();
  const reader = new FileReader();

  reader.onload = async (e) => {
    try {
      const arrayBuffer = e.target.result;
      const audioBuffer = await currentAudioContext.decodeAudioData(arrayBuffer);
      const data = audioBuffer.getChannelData(0);
      const step = Math.ceil(data.length / width);
      const amp = height / 2;

      // Draw real waveform
      ctx.clearRect(0, 0, width, height);
      ctx.fillStyle = '#FF6B35'; // Theme Orange

      for (let i = 0; i < width; i++) {
        let min = 1.0;
        let max = -1.0;
        for (let j = 0; j < step; j++) {
          const index = (i * step) + j;
          if (index >= data.length) break;
          const dat = data[index];
          if (dat < min) min = dat;
          if (dat > max) max = dat;
        }
        const barHeight = Math.max(2, (max - min) * amp * 1.5);
        ctx.fillRect(i * 1.5, amp - barHeight / 2, 1.2, barHeight);
      }
    } catch (err) {
      console.error("Local audio visualizer failed to decode file:", err);
    }
  };

  reader.readAsArrayBuffer(file);
}

// --- Inference Network Operations ---
async function runInference() {
  if (!selectedFile) return;

  // Toggle Loading states
  btnAnalyze.style.display = 'none';
  loadingContainer.style.display = 'flex';
  hideError();

  const formData = new FormData();
  formData.append('file', selectedFile);

  try {
    const response = await fetch('/predict', {
      method: 'POST',
      body: formData
    });

    if (!response.ok) {
      const errData = await response.json();
      throw new Error(errData.error || `Server responded with status ${response.status}`);
    }

    const data = await response.json();
    displayResults(data);
    saveToHistory(selectedFile.name, data);

  } catch (err) {
    showError(err.message || "Failed to reach backend server. Ensure Flask app.py is running.");
  } finally {
    btnAnalyze.style.display = 'block';
    loadingContainer.style.display = 'none';
  }
}

// --- Display Results & Charts ---
function displayResults(data) {
  emptyState.style.display = 'none';
  resultsWrapper.style.display = 'block';

  const emotion = data.emotion;
  const confidence = data.confidence;
  const probabilities = data.probabilities;

  // Fill Result Card elements
  const emoji = EMOTION_EMOJIS[emotion.toLowerCase()] || "🎙️";
  resultEmoji.textContent = emoji;
  resultEmoji.setAttribute('aria-label', emotion);
  resultEmotion.textContent = emotion.toUpperCase();
  resultConfidence.textContent = `${(confidence * 100).toFixed(1)}%`;
  
  const meterColor = confidence >= 0.70 ? '#2ECC71' : '#FF6B35';
  resultConfidence.style.color = meterColor;
  resultMeterFill.style.backgroundColor = meterColor;
  resultMeterFill.style.width = `${(confidence * 100).toFixed(1)}%`;

  // Render Horizontal Bar Chart
  renderChartBars(probabilities);
}

function renderChartBars(probabilities) {
  chartBarsContainer.innerHTML = '';

  // Sort entries descending
  const sorted = Object.entries(probabilities)
    .sort((a, b) => b[1] - a[1]);

  sorted.forEach(([emo, val]) => {
    const percent = (val * 100).toFixed(1);
    const color = EMOTION_COLORS[emo.toLowerCase()] || '#8D99AE';

    const row = document.createElement('div');
    row.className = 'chart-row';
    row.innerHTML = `
      <span class="chart-label">${emo.toUpperCase()}</span>
      <div class="chart-bar-bg">
        <div class="chart-bar-fill" style="width: ${percent}%; background-color: ${color};"></div>
      </div>
      <span class="chart-val">${percent}%</span>
    `;
    chartBarsContainer.appendChild(row);
  });
}

// --- Error Card Helper ---
function showError(msg) {
  errorTextEl.textContent = msg;
  errorCard.style.display = 'flex';
}

function hideError() {
  errorCard.style.display = 'none';
}

// --- Prediction History Operations ---
function loadHistory() {
  const saved = localStorage.getItem('voice_emotion_history');
  if (saved) {
    history = JSON.parse(saved);
    renderHistory();
  }
}

function saveToHistory(fileName, data) {
  const record = {
    id: Date.now(),
    fileName,
    emotion: data.emotion,
    confidence: data.confidence,
    probabilities: data.probabilities,
    time: new Date().toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' })
  };

  history = [record, ...history.slice(0, 4)];
  localStorage.setItem('voice_emotion_history', JSON.stringify(history));
  renderHistory();
}

function renderHistory() {
  if (history.length === 0) {
    historySection.style.display = 'none';
    return;
  }

  historySection.style.display = 'block';
  historyList.innerHTML = '';

  history.map(item => {
    const itemEl = document.createElement('div');
    itemEl.className = 'history-item fade-in';
    itemEl.innerHTML = `
      <div class="history-meta">
        <span class="history-file">${item.fileName}</span>
        <span class="history-time">${item.time}</span>
      </div>
      <div class="history-prediction">
        <span class="history-emotion">${item.emotion.toUpperCase()}</span>
        <span class="history-confidence" style="color: ${item.confidence >= 0.70 ? 'var(--accent-green)' : 'var(--accent-orange)'}">
          ${(item.confidence * 100).toFixed(1)}%
        </span>
      </div>
    `;
    itemEl.addEventListener('click', () => {
      displayResults(item);
      hideError();
    });
    historyList.appendChild(itemEl);
  });
}

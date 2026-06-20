import React, { useState, useEffect, useRef } from 'react';
import axios from 'axios';
import UploadArea from './components/UploadArea';
import ResultCard from './components/ResultCard';
import EmotionChart from './components/EmotionChart';
import LoadingSpinner from './components/LoadingSpinner';

// Local Audio Waveform Visualization Component
const AudioWaveform = ({ file }) => {
  const canvasRef = useRef(null);

  useEffect(() => {
    if (!file) return;

    const canvas = canvasRef.current;
    if (!canvas) return;
    const ctx = canvas.getContext('2d');
    const width = canvas.width;
    const height = canvas.height;

    // Clear canvas
    ctx.clearRect(0, 0, width, height);

    // Initial placeholder wave drawing
    ctx.fillStyle = 'rgba(255, 107, 53, 0.2)';
    for (let i = 0; i < width; i += 4) {
      const h = 5 + Math.sin(i * 0.05) * 15;
      ctx.fillRect(i, (height - h) / 2, 2.5, h);
    }

    const audioCtx = new (window.AudioContext || window.webkitAudioContext)();
    const reader = new FileReader();

    reader.onload = async (e) => {
      try {
        const arrayBuffer = e.target.result;
        const audioBuffer = await audioCtx.decodeAudioData(arrayBuffer);
        const data = audioBuffer.getChannelData(0);
        const step = Math.ceil(data.length / width);
        const amp = height / 2;

        // Clear placeholder
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
          // Draw neat vertical visualizer bars
          const barHeight = Math.max(2, (max - min) * amp * 1.5);
          ctx.fillRect(i * 1.5, amp - barHeight / 2, 1.2, barHeight);
        }
      } catch (err) {
        console.error("Local audio visualizer failed to decode file:", err);
      }
    };

    reader.readAsArrayBuffer(file);

    return () => {
      if (audioCtx.state !== 'closed') {
        audioCtx.close();
      }
    };
  }, [file]);

  return (
    <div className="waveform-visualizer">
      <span className="waveform-title">LOCAL WAVEFORM ANALYZER</span>
      <div className="canvas-wrapper">
        <canvas ref={canvasRef} width={800} height={120} className="waveform-canvas" />
      </div>
    </div>
  );
};

function App() {
  const [selectedFile, setSelectedFile] = useState(null);
  const [result, setResult] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);
  const [history, setHistory] = useState([]);
  const [darkMode, setDarkMode] = useState(true);

  // Load history from localStorage
  useEffect(() => {
    const savedHistory = localStorage.getItem('voice_emotion_history');
    if (savedHistory) {
      setHistory(JSON.parse(savedHistory));
    }
  }, []);

  const handleFileSelect = (file) => {
    setSelectedFile(file);
    setResult(null);
    setError(null);
  };

  const handleClear = () => {
    setSelectedFile(null);
    setResult(null);
    setError(null);
  };

  const analyzeEmotion = async () => {
    if (!selectedFile) return;

    setLoading(true);
    setError(null);
    setResult(null);

    const formData = new FormData();
    formData.append('file', selectedFile);

    try {
      // POST Request to Flask API
      const response = await axios.post('http://127.0.0.1:5000/predict', formData, {
        headers: {
          'Content-Type': 'multipart/form-data',
        },
      });

      const data = response.data;
      setResult(data);

      // Save to localStorage history
      const newHistoryItem = {
        id: Date.now(),
        fileName: selectedFile.name,
        emotion: data.emotion,
        confidence: data.confidence,
        probabilities: data.probabilities,
        time: new Date().toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' })
      };
      
      const updatedHistory = [newHistoryItem, ...history.slice(0, 4)];
      setHistory(updatedHistory);
      localStorage.setItem('voice_emotion_history', JSON.stringify(updatedHistory));

    } catch (err) {
      console.error(err);
      const errMsg = err.response?.data?.error || "Unable to reach prediction server. Make sure the backend is running.";
      setError(errMsg);
    } finally {
      setLoading(false);
    }
  };

  const loadHistoryItem = (item) => {
    setResult({
      emotion: item.emotion,
      confidence: item.confidence,
      probabilities: item.probabilities
    });
    setError(null);
  };

  const toggleTheme = () => {
    setDarkMode(!darkMode);
  };

  return (
    <div className={`app-root ${darkMode ? 'dark-theme' : 'light-theme'}`}>
      {/* Header / Navbar */}
      <header className="navbar">
        <div className="navbar-logo">
          <span className="logo-icon">🎙️</span>
          <span className="logo-text">VOICE EMOTION DETECTOR AI</span>
        </div>
        <div className="navbar-controls">
          <button className="btn-theme-toggle" onClick={toggleTheme}>
            {darkMode ? '☀️ Light Mode' : '🌙 Dark Mode'}
          </button>
        </div>
      </header>

      {/* Main Container */}
      <main className="main-content">
        
        {/* Hero Section */}
        <section className="hero-section">
          <h1 className="hero-title">
            Analyze Human Emotions Through Voice Using Artificial Intelligence
          </h1>
          <p className="hero-subtitle">
            Upload your voice recording and discover emotions locally using high‑precision neural networks.
          </p>
        </section>

        {/* Dashboard Grid */}
        <div className="dashboard-grid">
          
          {/* Left Column: Interactive Upload & Actions */}
          <div className="dashboard-card primary-column">
            <h2 className="section-title">Upload Audio Sample</h2>
            
            <UploadArea 
              onFileSelected={handleFileSelect}
              selectedFile={selectedFile}
              onClearFile={handleClear}
            />

            {selectedFile && !loading && (
              <div className="actions-box fade-in">
                {/* Waveform Visualization */}
                <AudioWaveform file={selectedFile} />
                
                <button 
                  className="btn-analyze" 
                  onClick={analyzeEmotion}
                >
                  🔍 Analyze Voice Sample
                </button>
              </div>
            )}

            {loading && <LoadingSpinner />}
            
            {error && (
              <div className="error-card fade-in">
                <span className="error-icon">⚠️</span>
                <span className="error-text">{error}</span>
              </div>
            )}
          </div>

          {/* Right Column: Results & Analytics */}
          <div className="dashboard-card secondary-column">
            <h2 className="section-title">Analysis & Insights</h2>
            
            {!result && !loading && !error && (
              <div className="empty-state">
                <div className="empty-icon">📊</div>
                <p className="empty-text">No audio sample analyzed yet.</p>
                <p className="empty-subtext">Upload a WAV/MP3 file and click Analyze to view predictions.</p>
              </div>
            )}

            {result && (
              <div className="results-wrapper">
                <ResultCard result={result} />
                <EmotionChart probabilities={result.probabilities} />
              </div>
            )}
          </div>

        </div>

        {/* Prediction History Log */}
        {history.length > 0 && (
          <section className="history-section dashboard-card fade-in">
            <h2 className="section-title">Recent Predictions</h2>
            <div className="history-list">
              {history.map((item) => (
                <div 
                  key={item.id} 
                  className="history-item"
                  onClick={() => loadHistoryItem(item)}
                >
                  <div className="history-meta">
                    <span className="history-file">{item.fileName}</span>
                    <span className="history-time">{item.time}</span>
                  </div>
                  <div className="history-prediction">
                    <span className="history-emotion">{item.emotion.toUpperCase()}</span>
                    <span className="history-confidence">{(item.confidence * 100).toFixed(1)}%</span>
                  </div>
                </div>
              ))}
            </div>
          </section>
        )}

      </main>

      {/* Footer */}
      <footer className="footer">
        <p className="footer-copyright">Built by Jivesh Gupta • BCA AI & ML</p>
        <p className="footer-sub">Voice Emotion Detector AI • 100% Local Inference</p>
      </footer>
    </div>
  );
}

export default App;

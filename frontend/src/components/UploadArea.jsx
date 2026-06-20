import React, { useState, useRef } from 'react';

const UploadArea = ({ onFileSelected, selectedFile, onClearFile }) => {
  const [isDragActive, setIsDragActive] = useState(false);
  const fileInputRef = useRef(null);

  const handleDrag = (e) => {
    e.preventDefault();
    e.stopPropagation();
    if (e.type === "dragenter" || e.type === "dragover") {
      setIsDragActive(true);
    } else if (e.type === "dragleave") {
      setIsDragActive(false);
    }
  };

  const validateAndSelectFile = (file) => {
    if (!file) return;

    // Validate extension
    const extension = file.name.split('.').pop().toLowerCase();
    if (extension !== 'wav' && extension !== 'mp3') {
      alert("Invalid file format. Only WAV and MP3 files are supported.");
      return;
    }

    // Validate size (50MB)
    const maxSize = 50 * 1024 * 1024;
    if (file.size > maxSize) {
      alert("File is too large. Maximum size allowed is 50MB.");
      return;
    }

    onFileSelected(file);
  };

  const handleDrop = (e) => {
    e.preventDefault();
    e.stopPropagation();
    setIsDragActive(false);

    if (e.dataTransfer.files && e.dataTransfer.files[0]) {
      validateAndSelectFile(e.dataTransfer.files[0]);
    }
  };

  const handleChange = (e) => {
    e.preventDefault();
    if (e.target.files && e.target.files[0]) {
      validateAndSelectFile(e.target.files[0]);
    }
  };

  const onButtonClick = () => {
    fileInputRef.current.click();
  };

  return (
    <div className="upload-container">
      <div 
        className={`upload-dropzone ${isDragActive ? 'drag-active' : ''} ${selectedFile ? 'has-file' : ''}`}
        onDragEnter={handleDrag}
        onDragLeave={handleDrag}
        onDragOver={handleDrag}
        onDrop={handleDrop}
      >
        <input 
          ref={fileInputRef}
          type="file" 
          accept=".wav,.mp3"
          onChange={handleChange}
          style={{ display: 'none' }}
        />

        {!selectedFile ? (
          <div className="upload-prompt">
            <div className="pulse-icon">🎙️</div>
            <h3 className="upload-title">Drag & drop your voice file here</h3>
            <p className="upload-subtitle">or click browse to find it on your computer</p>
            <button className="btn-browse" onClick={onButtonClick}>
              Browse Files
            </button>
            <div className="upload-meta">
              <span>WAV, MP3 formats supported</span>
              <span>•</span>
              <span>Max size 50MB</span>
            </div>
          </div>
        ) : (
          <div className="file-info-container">
            <div className="file-icon-box">🎵</div>
            <div className="file-meta">
              <span className="file-name">{selectedFile.name}</span>
              <span className="file-size">{(selectedFile.size / (1024 * 1024)).toFixed(2)} MB</span>
            </div>
            
            <div className="audio-preview">
              <audio 
                controls 
                src={URL.createObjectURL(selectedFile)} 
                className="audio-widget"
              />
            </div>

            <button className="btn-remove" onClick={onClearFile}>
              ✕ Remove File
            </button>
          </div>
        )}
      </div>
    </div>
  );
};

export default UploadArea;

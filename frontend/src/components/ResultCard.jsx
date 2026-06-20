import React from 'react';

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

const ResultCard = ({ result }) => {
  if (!result) return null;

  const { emotion, confidence } = result;
  const normalizedEmotion = emotion.toLowerCase();
  const emoji = EMOTION_EMOJIS[normalizedEmotion] || "🎙️";
  const confidencePercent = (confidence * 100).toFixed(1);

  // Color theme variables: Green for high confidence, Orange for lower
  const themeColor = confidence >= 0.70 ? '#2ECC71' : '#FF6B35';

  return (
    <div className="result-card fade-in">
      <div className="result-card-header">
        <span className="badge-success">ANALYSIS COMPLETE</span>
      </div>
      
      <div className="result-card-body">
        <div className="result-emotion-box">
          <span className="result-emoji-large" role="img" aria-label={emotion}>
            {emoji}
          </span>
          <span className="result-emotion-title">{emotion.toUpperCase()}</span>
        </div>

        <div className="result-meter-container">
          <div className="result-meter-labels">
            <span className="label-confidence">CONFIDENCE LEVEL</span>
            <span className="value-confidence" style={{ color: themeColor }}>
              {confidencePercent}%
            </span>
          </div>
          
          <div className="meter-bg">
            <div 
              className="meter-fill" 
              style={{ 
                width: `${confidencePercent}%`,
                backgroundColor: themeColor
              }}
            />
          </div>
        </div>
      </div>
    </div>
  );
};

export default ResultCard;

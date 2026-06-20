import React from 'react';

const LoadingSpinner = ({ message = "Processing audio file and extracting features..." }) => {
  return (
    <div className="loading-container">
      <div className="sharp-spinner">
        <div className="spinner-block block1"></div>
        <div className="spinner-block block2"></div>
        <div className="spinner-block block3"></div>
        <div className="spinner-block block4"></div>
      </div>
      <p className="loading-message">{message}</p>
    </div>
  );
};

export default LoadingSpinner;

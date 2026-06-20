import React from 'react';
import { Bar } from 'react-chartjs-2';
import {
  Chart as ChartJS,
  CategoryScale,
  LinearScale,
  BarElement,
  Title,
  Tooltip,
  Legend,
} from 'chart.js';

// Register Chart.js modules
ChartJS.register(
  CategoryScale,
  LinearScale,
  BarElement,
  Title,
  Tooltip,
  Legend
);

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

const EmotionChart = ({ probabilities }) => {
  if (!probabilities) return null;

  // Sort emotions by probability value in descending order
  const sortedEmotions = Object.entries(probabilities)
    .sort((a, b) => b[1] - a[1]);

  const labels = sortedEmotions.map(([emo]) => emo.toUpperCase());
  const dataValues = sortedEmotions.map(([, val]) => (val * 100).toFixed(1));
  const bgColors = sortedEmotions.map(([emo]) => EMOTION_COLORS[emo.toLowerCase()] || '#8D99AE');

  const data = {
    labels,
    datasets: [
      {
        data: dataValues,
        backgroundColor: bgColors,
        borderWidth: 0,
      }
    ]
  };

  const options = {
    indexAxis: 'y', // horizontal bar chart
    responsive: true,
    maintainAspectRatio: false,
    plugins: {
      legend: {
        display: false
      },
      tooltip: {
        backgroundColor: '#333333',
        titleColor: '#FFFFFF',
        bodyColor: '#FFFFFF',
        titleFont: { family: 'Outfit', size: 13, weight: 'bold' },
        bodyFont: { family: 'Outfit', size: 12 },
        borderWidth: 1,
        borderColor: '#FF6B35',
        cornerRadius: 0, // Sharp corners for tooltip matching design
        callbacks: {
          label: (context) => ` Probability: ${context.parsed.x}%`
        }
      }
    },
    scales: {
      x: {
        max: 100,
        grid: {
          display: false
        },
        ticks: {
          color: '#A0AEC0',
          font: { family: 'Outfit', size: 11 },
          callback: (val) => `${val}%`
        },
        border: {
          display: false
        }
      },
      y: {
        grid: {
          display: false
        },
        ticks: {
          color: '#FFFFFF',
          font: { family: 'Outfit', size: 12, weight: 'bold' }
        },
        border: {
          display: false
        }
      }
    }
  };

  return (
    <div className="chart-card fade-in">
      <h3 className="chart-card-title">Emotion Probability Analytics</h3>
      <div className="chart-holder">
        <Bar data={data} options={options} />
      </div>
    </div>
  );
};

export default EmotionChart;

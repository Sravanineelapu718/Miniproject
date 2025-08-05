// Sign Language to Text Editor using React, TensorFlow.js and MediaPipe

import React, { useRef, useEffect, useState } from 'react';
import Webcam from 'react-webcam';
import * as tf from '@tensorflow/tfjs';
import * as mpHands from '@mediapipe/hands';

const CLASS_LABELS = ['A', 'B', 'C', 'Hello', 'Thanks']; // Example classes

export default function App() {
  const webcamRef = useRef(null);
  const canvasRef = useRef(null);
  const [model, setModel] = useState(null);
  const [text, setText] = useState('');

  // Load ML model
  useEffect(() => {
    const loadModel = async () => {
      const loadedModel = await tf.loadLayersModel('/model/model.json');
      setModel(loadedModel);
    };
    loadModel();
  }, []);

  // Start hand detection
  useEffect(() => {
    const hands = new mpHands.Hands({
      locateFile: (file) => `https://cdn.jsdelivr.net/npm/@mediapipe/hands/${file}`,
    });
    hands.setOptions({
      maxNumHands: 1,
      modelComplexity: 1,
      minDetectionConfidence: 0.7,
      minTrackingConfidence: 0.7,
    });
    hands.onResults(onResults);

    const runMediapipe = async () => {
      const camera = new mpHands.Camera(webcamRef.current.video, {
        onFrame: async () => {
          await hands.send({ image: webcamRef.current.video });
        },
        width: 640,
        height: 480,
      });
      camera.start();
    };

    if (webcamRef.current) {
      runMediapipe();
    }
  }, [model]);

  const onResults = async (results) => {
    const canvasCtx = canvasRef.current.getContext('2d');
    canvasCtx.clearRect(0, 0, 640, 480);

    if (results.multiHandLandmarks && results.multiHandLandmarks.length > 0) {
      const landmarks = results.multiHandLandmarks[0];
      const flatLandmarks = landmarks.flatMap(p => [p.x, p.y, p.z]);
      const inputTensor = tf.tensor([flatLandmarks]);
      const prediction = model.predict(inputTensor);
      const predictedIndex = prediction.argMax(1).dataSync()[0];
      const predictedLabel = CLASS_LABELS[predictedIndex];
      setText(prev => prev + predictedLabel + ' ');
    }
  };

  return (
    <div className="p-4">
      <h1 className="text-xl font-bold mb-2">Sign Language to Text Editor</h1>
      <Webcam ref={webcamRef} width={640} height={480} className="rounded" />
      <canvas ref={canvasRef} width={640} height={480} className="absolute top-0 left-0" />
      <textarea
        value={text}
        rows={8}
        className="w-full mt-4 border rounded p-2"
        readOnly
      />
    </div>
  );
}

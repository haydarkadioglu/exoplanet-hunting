# NASA Exoplanet Classification Models - ONNX Deployment

## 🚀 Overview

This directory contains ONNX (Open Neural Network Exchange) versions of trained NASA exoplanet classification models from three missions:

- **K2 Mission Model**: Trained on K2 exoplanet candidates
- **Kepler Mission Model**: Trained on Kepler exoplanet candidates  
- **TESS Mission Model**: Trained on TESS Objects of Interest (TOI)

## 📁 Directory Structure

```
models/
├── k2/
│   ├── k2_model.onnx              # K2 classification model
│   ├── k2_scaler.onnx             # K2 feature scaler
│   ├── k2_deployment.json         # Deployment metadata
│   ├── k2_features.json           # Feature names
│   └── k2_original_metadata.json  # Original training metadata
├── kepler/
│   ├── kepler_model.onnx          # Kepler classification model
│   ├── kepler_scaler.onnx         # Kepler feature scaler
│   ├── kepler_deployment.json     # Deployment metadata
│   ├── kepler_features.json       # Feature names
│   └── kepler_original_metadata.json
└── tess/
    ├── tess_model.onnx            # TESS classification model
    ├── tess_scaler.onnx           # TESS feature scaler
    ├── tess_deployment.json       # Deployment metadata
    ├── tess_features.json         # Feature names
    └── tess_original_metadata.json
```

## 🎯 Classification Categories

All models classify exoplanet candidates into three categories:

- **Candidate**: Planet candidates requiring follow-up observation
- **Confirmed**: Confirmed exoplanets with high confidence
- **False Positive**: False positives and refuted planetary candidates

## 💻 Usage Examples

### Python with ONNX Runtime

```python
import onnxruntime as ort
import numpy as np
import json

# Load model and scaler
model_session = ort.InferenceSession('models/k2/k2_model.onnx')
scaler_session = ort.InferenceSession('models/k2/k2_scaler.onnx')

# Load feature names and metadata
with open('models/k2/k2_features.json', 'r') as f:
    features = json.load(f)

with open('models/k2/k2_deployment.json', 'r') as f:
    metadata = json.load(f)

# Prepare your data (example with random data)
# Your data should have the same features as in k2_features.json
input_data = np.random.random((1, len(features))).astype(np.float32)

# Scale the data
scaled_data = scaler_session.run(None, {'float_input': input_data})[0]

# Make prediction
prediction = model_session.run(None, {'float_input': scaled_data})
predicted_class = prediction[0][0]  # Class prediction
class_probabilities = prediction[1][0]  # Class probabilities

# Map to class names
class_names = metadata['deployment_info']['output_classes']
predicted_label = class_names[predicted_class]

print(f"Predicted class: {predicted_label}")
print(f"Probabilities: {dict(zip(class_names, class_probabilities))}")
```

### JavaScript with ONNX.js

```javascript
const ort = require('onnxruntime-web');

async function loadAndPredict() {
    // Load models
    const modelSession = await ort.InferenceSession.create('models/k2/k2_model.onnx');
    const scalerSession = await ort.InferenceSession.create('models/k2/k2_scaler.onnx');
    
    // Prepare input data
    const inputData = new Float32Array([/* your feature values */]);
    const tensor = new ort.Tensor('float32', inputData, [1, inputData.length]);
    
    // Scale data
    const scaledResult = await scalerSession.run({float_input: tensor});
    
    // Make prediction
    const prediction = await modelSession.run({float_input: scaledResult.variable});
    
    console.log('Prediction:', prediction);
}
```

### C# with ML.NET

```csharp
using Microsoft.ML.OnnxRuntime;
using System;

// Load model
var session = new InferenceSession("models/k2/k2_model.onnx");
var scalerSession = new InferenceSession("models/k2/k2_scaler.onnx");

// Prepare input data
var inputData = new float[] { /* your feature values */ };
var inputTensor = new DenseTensor<float>(inputData, new int[] { 1, inputData.Length });

// Scale data
var scaledResult = scalerSession.Run(new List<NamedOnnxValue> 
{
    NamedOnnxValue.CreateFromTensor("float_input", inputTensor)
});

// Make prediction
var prediction = session.Run(new List<NamedOnnxValue>
{
    NamedOnnxValue.CreateFromTensor("float_input", scaledResult.First().AsTensor<float>())
});
```

## 🔧 Model Performance

| Model | F1-Macro Score | Accuracy | Training Samples |
|-------|----------------|----------|------------------|
| K2    | [Check deployment.json] | [Check deployment.json] | [Check deployment.json] |
| Kepler| [Check deployment.json] | [Check deployment.json] | [Check deployment.json] |
| TESS  | [Check deployment.json] | [Check deployment.json] | [Check deployment.json] |

## 📋 Requirements

### Python
```bash
pip install onnxruntime numpy
```

### JavaScript/Node.js
```bash
npm install onnxruntime-web
```

### C#/.NET
```bash
dotnet add package Microsoft.ML.OnnxRuntime
```

## 🌐 Deployment Options

### Web Applications
- Use ONNX.js for browser-based inference
- WebAssembly support for high performance
- No server-side processing required

### Mobile Applications
- ONNX Runtime Mobile for iOS/Android
- Optimized for mobile hardware
- Offline inference capability

### Cloud Services
- Deploy with Azure ML, AWS SageMaker, or Google AI Platform
- Auto-scaling and managed inference
- REST API endpoints

### Edge Devices
- ONNX Runtime for IoT devices
- Raspberry Pi and embedded systems
- Low latency inference

## 📖 Additional Resources

- [ONNX Runtime Documentation](https://onnxruntime.ai/)
- [ONNX Format Specification](https://onnx.ai/)
- [NASA Exoplanet Archive](https://exoplanetarchive.ipac.caltech.edu/)

## 📄 License

See LICENSE file for details.

## 🤝 Contributing

Contributions welcome! Please see contributing guidelines.

---

Generated automatically from trained scikit-learn models.
Last updated: 2025-10-05 10:54:01

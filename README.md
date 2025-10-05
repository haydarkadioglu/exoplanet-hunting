# 🚀 NASA Exoplanet Hunter

**AI-Powered Exoplanet Classification System**

*Developed by **Kozmik Zihinler** (Cosmic Minds) Team*

**2025 NASA Space Apps Challenge**  
**Challenge: "A World Away: Hunting for Exoplanets with AI"**

---

## 🌌 Overview

The NASA Exoplanet Hunter is an advanced machine learning-powered web application designed to classify exoplanet candidates using real NASA mission data. Our system leverages state-of-the-art AI models trained on data from three major space missions: **Kepler**, **K2**, and **TESS**.

## 🎯 Features

- **Multi-Mission Support**: Classification models for Kepler, K2, and TESS datasets
- **Real-Time Predictions**: Interactive web interface for instant exoplanet classification
- **Advanced AI Models**: 
  - **K2**: Gradient Boosting (97% accuracy)
  - **Kepler**: LightGBM (85.36% accuracy) 
  - **TESS**: Random Forest (89.94% accuracy)
- **Manual & Sample Data Input**: Support for both custom parameter input and pre-loaded test samples
- **Confusion Matrix Visualization**: Detailed model performance analysis
- **ONNX Runtime Integration**: Optimized for web deployment

## 📊 Datasets Used

### 1. **Kepler Mission Dataset**
- **Source**: NASA Kepler Space Telescope Archive
- **Features**: 106 astronomical parameters
- **Classes**: Candidate, Confirmed, False Positive
- **Training Samples**: ~8,500 objects

### 2. **K2 Mission Dataset** 
- **Source**: NASA K2 Extended Mission Archive
- **Features**: 145 derived features from light curve analysis
- **Classes**: Candidate, Confirmed, False Positive  
- **Training Samples**: ~5,000 objects

### 3. **TESS Mission Dataset**
- **Source**: NASA TESS (Transiting Exoplanet Survey Satellite)
- **Features**: 43 TOI (TESS Objects of Interest) catalog parameters
- **Classes**: Candidate, Confirmed, False Positive
- **Training Samples**: ~7,703 objects

## 🛠️ Technology Stack

- **Frontend**: TypeScript, Vite, HTML5, CSS3
- **AI/ML**: ONNX Runtime Web, Python (training)
- **Models**: Gradient Boosting, LightGBM, Random Forest
- **Data Processing**: Scikit-learn, Pandas, NumPy
- **Visualization**: Custom confusion matrix generation

## 🚀 Quick Start

### Prerequisites
- Node.js (v16 or higher)
- Modern web browser with WebAssembly support

### Installation

1. Clone the repository:
```bash
git clone https://github.com/haydarkadioglu/exoplanet-hunting.git
cd exoplanet-hunting
```

2. Install dependencies:
```bash
npm install
```

3. Start the development server:
```bash
npm run dev
```

4. Open your browser and navigate to `http://localhost:3000/exoplanet-hunting/`

## 🎮 Usage

1. **Select a Model**: Choose between Kepler, K2, or TESS classification models
2. **Input Data**: 
   - Use **Sample Data** for quick testing with pre-loaded examples
   - Use **Manual Input** to enter custom astronomical parameters
3. **Get Predictions**: Click predict to see AI classification results
4. **Analyze Performance**: View confusion matrices to understand model accuracy

## 📈 Model Performance

| Mission | Model Type | Accuracy | Precision | Recall | F1-Score |
|---------|------------|----------|-----------|--------|----------|
| K2 | Gradient Boosting | 97.00% | 96.97% | 97.00% | 96.95% |
| Kepler | LightGBM | 85.36% | 85.13% | 85.36% | 85.13% |
| TESS | Random Forest | 89.94% | 85.64% | 89.27% | 89.27% |

## 🔬 Scientific Background

Our models classify astronomical objects into three categories:

- **🔍 Candidate Exoplanet**: Objects showing transit-like signals requiring further verification
- **✅ Confirmed Exoplanet**: High-confidence detections with validated planetary nature
- **❌ False Positive**: Signals caused by stellar activity, binary stars, or instrumental artifacts

## 🏆 Team: Kozmik Zihinler (Cosmic Minds)

We are a passionate team of developers and data scientists participating in the **2025 NASA Space Apps Challenge** with the challenge **"A World Away: Hunting for Exoplanets with AI"**, dedicated to advancing exoplanet discovery through artificial intelligence.

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🔗 Source Repository

This project is part of our NASA Space Apps Challenge 2025 submission. For the complete codebase, datasets, and additional resources, visit:

**🌟 [nasa-space-apps-25-workfield](https://github.com/haydarkadioglu/nasa-space-apps-25-workfield)**

## 🤝 Contributing

We welcome contributions! Please feel free to submit pull requests, report bugs, or suggest new features.

## 📧 Contact

For questions or collaboration opportunities, please reach out through our GitHub repository.

---

*"Exploring the cosmos, one exoplanet at a time." - Kozmik Zihinler Team*

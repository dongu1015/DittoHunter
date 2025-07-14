# 🔍 Advanced Deepfake Detection System

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://python.org)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red.svg)](https://pytorch.org)
[![License](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

> 🚀 **State-of-the-art deepfake detection using CNN-GAT hybrid architecture**

## ✨ Features

- 🎯 **High Accuracy**: CNN-GAT hybrid model architecture
- ⚡ **GPU Accelerated**: Fast training and inference with CUDA support
- 🔧 **Roop Integration**: Advanced face detection and quality assessment (optional)
- 📂 **Local Dataset**: Simple folder-based data organization (no database required)
- 🎨 **Data Augmentation**: Smart augmentation preserving face authenticity
- 🏗️ **Modular Design**: Clean, maintainable codebase

## 🏗️ Architecture

```
CNN Feature Extractor (MobileNetV2) → Graph Construction → GAT Layers → Classification
```

- **CNN Backbone**: MobileNetV2 for efficient feature extraction
- **Graph Neural Network**: GAT (Graph Attention Network) for spatial relationships
- **Face Validation**: Roop-based quality assessment and filtering
- **Optimization**: Optuna hyperparameter tuning

## 📁 Project Structure

```
deepfake-detection/
├── 🧠 Core System
│   ├── dataset.py                   # Model architectures & utilities
│   ├── train_model.py               # Model training pipeline
│   ├── test_model.py                # Model inference & testing
│   └── preprocess.py                # Image preprocessing utilities
├── 📁 Data (Bring Your Own)
│   ├── fake/                        # Your fake images
│   └── real/                        # Your real images
├── 🤖 Models
│   └── model.pth                    # Trained model
├── 🔧 Roop Integration
│   └── roop/                        # Face analysis framework
└── 📋 Configuration
    ├── requirements.txt             # Python dependencies
    └── config.py                    # Configuration settings
```

## 🚀 Quick Start

### 1. Environment Setup

```bash
# Clone repository
git clone https://github.com/your-username/deepfake-detection.git
cd deepfake-detection

# Install dependencies
pip install -r requirements.txt

# Setup check
python setup.py
```

### 2. Prepare Your Dataset

```bash
# Create data structure
mkdir -p data/fake data/real

# Add your images
# data/fake/ - Put your fake/manipulated images here
# data/real/ - Put your real/authentic images here
```

**Dataset Examples:**
- **Fake images**: DeepFake videos frames, face-swapped images, GAN-generated faces
- **Real images**: Original photos, authentic video frames, natural faces

### 3. Train Model

```bash
# Basic training
python train_model.py

# With custom settings
python train_model.py --epochs 50 --batch_size 32
```

### 4. Test Model

```bash
# Test single image
python test_model.py --image path/to/your/image.jpg

# Test dataset
python test_model.py --dataset data

# Test with visualization
python test_model.py --dataset data --visualize
```

### 5. Optional: Preprocess Images

```bash
# Organize mixed folder
python preprocess_simple.py --organize --input input_images --output data

# Process as fake images
python preprocess_simple.py --fake --input fake_images --output data

# Process as real images
python preprocess_simple.py --real --input real_images --output data
```

## 📊 Model Architecture

### CNN-GAT Hybrid Design
```
Input Image → CNN Feature Extractor → Graph Construction → 
GAT Layers → Global Pooling → Classification
```

### Key Components
- **CNN Backbone**: MobileNetV2 for efficient feature extraction
- **Graph Neural Network**: GAT (Graph Attention Network) for spatial relationships
- **Face Validation**: Optional roop-based quality assessment
- **Optimization**: Supports hyperparameter tuning with Optuna

## 🎯 Model Details

### Architecture Highlights
- **Feature Extractor**: MobileNetV2 (64-dim output)
- **GAT Layers**: 2 layers with 4 attention heads
- **Hidden Channels**: 32 (optimized by Optuna)
- **Dropout**: 0.021 (prevents overfitting)
- **Parameters**: 27,458 total

### Default Configuration
- **Optimizer**: Adam (configurable learning rate)
- **Batch Size**: 32 (configurable)
- **Data Augmentation**: Multiple augmentation techniques
- **Early Stopping**: Prevents overfitting

## 🔧 Advanced Usage

### Custom Dataset Training

```python
from train_model import SimpleDeepfakeDetectorTrainer

# Initialize trainer with your data directory
trainer = SimpleDeepfakeDetectorTrainer(data_dir="./your_data")

# Load and prepare data
fake_graphs, real_graphs = trainer.load_training_data()

# Train model
model, accuracy = trainer.train_model(
    fake_graphs=fake_graphs,
    real_graphs=real_graphs,
    epochs=50,
    learning_rate=0.001
)
```

### Single Image Inference

```python
from test_model import SimpleDeepfakeDetector

# Load trained model
detector = SimpleDeepfakeDetector("model.pth")

# Predict single image
prediction, confidence = detector.predict_single("path/to/image.jpg")
label = "REAL" if prediction == 1 else "FAKE"
print(f"Prediction: {label} (confidence: {confidence:.3f})")
```

### Dataset Organization

```python
from dataset_simple import SimpleDatasetLoader

# Load and analyze your dataset
loader = SimpleDatasetLoader("data")
stats = loader.load_dataset_stats()

# Create PyTorch data loaders
train_loader, test_loader = loader.create_data_loaders(batch_size=32)

# Visualize sample images
loader.visualize_samples(num_samples=8)
```

## 📋 Requirements

### System Requirements
- **Python**: 3.8+
- **CUDA**: 11.0+ (optional, for GPU acceleration)
- **RAM**: 4GB+ minimum, 8GB+ recommended
- **Storage**: 1GB+ for models and data

### Key Dependencies
- `torch>=2.0.0`
- `torch-geometric>=2.3.0`
- `torchvision>=0.15.0`
- `opencv-python>=4.8.0`
- `scikit-learn>=1.3.0`
- `matplotlib>=3.7.0`
- `optuna>=3.0.0` (optional, for hyperparameter tuning)
- `mediapipe>=0.10.0` (optional, for face detection)

## 🤝 Contributing

1. Fork the repository
2. Create feature branch (`git checkout -b feature/amazing-feature`)
3. Commit changes (`git commit -m 'Add amazing feature'`)
4. Push to branch (`git push origin feature/amazing-feature`)
5. Open Pull Request

## 📜 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- **Roop Framework**: For optional advanced face analysis capabilities
- **PyTorch Geometric**: For efficient graph neural network implementation
- **Optuna**: For automated hyperparameter optimization
- **OpenCV & MediaPipe**: For image processing and face detection

## 📝 Getting Started Tips

1. **Start Small**: Begin with a small dataset (100-200 images per class) to test the pipeline
2. **Quality over Quantity**: Focus on high-quality, diverse images rather than just large numbers
3. **Balanced Dataset**: Try to maintain roughly equal numbers of fake and real images
4. **GPU Recommended**: While not required, GPU acceleration significantly speeds up training
5. **Experiment**: Try different hyperparameters and augmentation strategies for your specific data

## 📞 Contact

For questions, issues, or collaboration opportunities, please open an issue on GitHub.

---

<div align="center">
  
**⭐ Star this repository if you found it helpful! ⭐**

*Built with ❤️ for the computer vision community*

</div>

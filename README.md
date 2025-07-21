# Hate Speech Detection System

A machine learning-based hate speech detection system built with TensorFlow and scikit-learn. This project provides an end-to-end pipeline for training and deploying hate speech classification models.

## 🏗️ Architecture Overview

```
┌─────────────────────────────────────────────────────────────┐
│                    Hate Speech Detection                    │
└─────────────────────────────────────────────────────────────┘
                                │
                                ▼
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   Data Layer    │    │  Pipeline Layer │    │   ML Layer      │
├─────────────────┤    ├─────────────────┤    ├─────────────────┤
│ • Data Ingestion│───▶│ • Training      │───▶│ • Model Trainer │
│ • Transformation│    │ • Prediction    │    │ • Evaluation    │
│                 │    │                 │    │ • Model Pusher  │
└─────────────────┘    └─────────────────┘    └─────────────────┘
                                │
                                ▼
┌─────────────────────────────────────────────────────────────┐
│              Configuration & Logging Layer                  │
├─────────────────┬─────────────────┬─────────────────────────┤
│   Constants     │   Configuration │      Exception          │
│   • Training    │   • Model       │      Handling           │
│   • Artifacts   │   • Data        │                         │
└─────────────────┴─────────────────┴─────────────────────────┘
```

## 🚀 Quick Start

### Prerequisites
- Python 3.8+
- pip package manager

### Installation

1. **Clone the repository**
   ```bash
   git clone <repository-url>
   cd HATE_SPEACH
   ```

2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Install the package**
   ```bash
   pip install -e .
   ```

### Usage

#### Training Pipeline
```bash
python app.py
```

#### Docker Deployment
```bash
# Build the image
docker build -t hate-speech-detector .

# Run the container
docker run hate-speech-detector
```

## 📁 Project Structure

```
hate_speech/
├── components/          # Core ML components
│   ├── data_ingestion.py
│   ├── data_transformation.py
│   ├── model_trainer.py
│   ├── model_evaluation.py
│   └── model_pusher.py
├── pipeline/           # Training and prediction pipelines
│   ├── training_pipeline.py
│   └── prediction_pipeline.py
├── configuration/      # Configuration management
├── constants/         # Project constants
├── entity/           # Data entities and schemas
├── exception/        # Custom exception handling
├── logger/          # Logging utilities
└── ml/             # ML utilities and models
```

## 🛠️ Key Components

- **Data Ingestion**: Handles data loading and preprocessing
- **Data Transformation**: Feature engineering and text processing
- **Model Trainer**: Trains the hate speech classification model
- **Model Evaluation**: Evaluates model performance metrics
- **Model Pusher**: Deploys trained models to production

## 📊 Dependencies

- **TensorFlow**: Deep learning framework
- **scikit-learn**: Machine learning utilities
- **NLTK**: Natural language processing
- **pandas/numpy**: Data manipulation
- **Google Cloud Storage**: Model storage and deployment

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/new-feature`)
3. Commit your changes (`git commit -am 'Add new feature'`)
4. Push to the branch (`git push origin feature/new-feature`)
5. Create a Pull Request


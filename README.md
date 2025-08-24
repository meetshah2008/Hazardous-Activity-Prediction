# Hazardous Activity Prediction

Hazardous Activity Prediction is a machine learning project that classifies short video clips as either **normal** or **hazardous** activities. The model combines Convolutional Neural Networks (CNN) and Long Short-Term Memory (LSTM) architectures to analyze spatial and temporal features in video data.

## Getting Started

Follow these steps to set up and run the project:

### 1. Clone the repository

```bash
git clone https://github.com/yourusername/hazardous-activity-prediction.git
cd hazardous-activity-prediction
```

### 2. Install requirements

Install all dependencies with:

```bash
pip install -r requirements.txt
```

The project uses the following Python libraries (listed in `requirements.txt`):

- torch
- torchvision
- numpy
- opencv-python
- Pillow
- scikit-learn
- tqdm
- requests

### 3. Prepare the dataset

Run the dataset preparation script:

```bash
python dataset.py
```

### 4. Train the model

Train the CNN + LSTM model with:

```bash
python train.py
```

### 5. Run inference

Classify new video clips using:

```bash
python inference.py
```

## Features

- Classification of video clips as "normal" or "hazardous"
- Deep learning architecture using CNN + LSTM
- Scripts for dataset preparation, training, and inference

## Model Architecture

Below is a high-level diagram of the pipeline used for hazardous activity prediction:

```markdown
![Model Architecture](architecture.png)
```

```
+-------------------+      +---------------------+      +---------------------+      +---------------------+      +----------------------+
|  Input Video Clip | ---> |  Frame Extraction   | ---> |   CNN (MobileNetV2) | ---> |  LSTM Sequence Model| ---> |  Classification Head |
+-------------------+      +---------------------+      +---------------------+      +---------------------+      +----------------------+
         |                        |                             |                           |                              |
         v                        v                             v                           v                              v
   Short MP4 file         12 frames per video           512-dim feature/frame      Temporal feature sequence      Output: Normal/Hazardous
```

#### Detailed Flow

1. **Input Video Clip**: Each MP4 video is split into 12 evenly sampled frames.
2. **Preprocessing**: Frames are resized, augmented, and normalized.
3. **CNN Feature Extraction**: Each frame passes through MobileNetV2 (pretrained), outputting a 1280-dim feature, then reduced to 512-dim.
4. **LSTM Sequence Model**: The sequence of frame features is processed by an LSTM to capture temporal dependencies.
5. **Classification Head**: The output from the last LSTM timestep is passed to a fully connected layer to predict `Normal` or `Hazardous`.

---





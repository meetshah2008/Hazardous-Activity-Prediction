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

## Contributing

Contributions are welcome! Please fork the repository and submit a pull request.

## License

This project is licensed under the MIT License.

## Contact

For questions or suggestions, please open an issue or contact [your email/contact here].

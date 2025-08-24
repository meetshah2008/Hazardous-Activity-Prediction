import torch
import torch.nn as nn
from torchvision import transforms
from torchvision.models import mobilenet_v2
import numpy as np
import cv2
from PIL import Image
import os

# Set device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# Configuration (must match training)
NUM_FRAMES = 12
IMG_SIZE = 224

# Define the model architecture (same as training)
class VideoSafetyClassifier(nn.Module):
    def __init__(self, num_classes=2, lstm_units=128):
        super().__init__()
        self.cnn = mobilenet_v2(weights=None)
        self.cnn.classifier = nn.Identity()
        self.feature_adjust = nn.Linear(1280, 512)
        self.lstm = nn.LSTM(512, lstm_units, batch_first=True)
        self.fc = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(lstm_units, num_classes)
        )
        
    def forward(self, x):
        batch_size, num_frames, c, h, w = x.shape
        x = x.view(batch_size * num_frames, c, h, w)
        cnn_features = self.cnn(x)
        cnn_features = self.feature_adjust(cnn_features)
        cnn_features = cnn_features.view(batch_size, num_frames, -1)
        lstm_out, _ = self.lstm(cnn_features)
        out = self.fc(lstm_out[:, -1, :])
        return out

# Load the trained model
def load_model(model_path):
    model = VideoSafetyClassifier().to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    print("Model loaded successfully!")
    return model

# Extract frames from video (same as training)
def extract_frames(video_path, num_frames=NUM_FRAMES, img_size=IMG_SIZE):
    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    # Select frame indices (same as training)
    if total_frames <= num_frames:
        frame_indices = list(range(total_frames))
        frame_indices += [total_frames - 1] * (num_frames - total_frames)
    else:
        stride = max(1, total_frames // num_frames)
        frame_indices = [i * stride for i in range(num_frames)]
        frame_indices = [min(idx, total_frames - 1) for idx in frame_indices]
    
    # Extract frames
    frames = []
    for idx in frame_indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = cap.read()
        if ret:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frames.append(frame)
        else:
            # Fallback: black frame
            frames.append(np.zeros((img_size, img_size, 3), dtype=np.uint8))
    
    cap.release()
    return frames

# Preprocess frames for model input
def preprocess_frames(frames, img_size=IMG_SIZE):
    transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    processed_frames = []
    for frame in frames:
        frame_pil = Image.fromarray(frame)
        frame_tensor = transform(frame_pil)
        processed_frames.append(frame_tensor)
    
    # Stack frames and add batch dimension
    video_tensor = torch.stack(processed_frames).unsqueeze(0)  # [1, NUM_FRAMES, 3, H, W]
    return video_tensor

# Classify a video
def classify_video(model, video_path):
    print(f"Processing video: {video_path}")
    
    # Extract and preprocess frames
    frames = extract_frames(video_path)
    video_tensor = preprocess_frames(frames).to(device)
    
    # Run inference
    with torch.no_grad():
        outputs = model(video_tensor)
        probabilities = torch.softmax(outputs, dim=1)
        predicted_class = torch.argmax(probabilities, dim=1).item()
        confidence = probabilities[0][predicted_class].item()
    
    # Return results
    class_names = ['Normal', 'Hazard']
    return {
        'prediction': class_names[predicted_class],
        'confidence': confidence,
        'normal_score': probabilities[0][0].item(),
        'hazard_score': probabilities[0][1].item()
    }

# Main function
def main(video_path):
    # Load the trained model
    model = load_model('best_model_24_08_2025.pth')
    
    # Ask user for video path
    # video_path = input("Enter the path to your video file: ").strip()

    # Remove quotes if user dragged and dropped the file
    # video_path = video_path.strip('"')
    
    try:
        # Classify the video
        results = classify_video(model, video_path)
        
        # Display results
        print("\n" + "="*50)
        print("CLASSIFICATION RESULTS")
        print("="*50)
        print(f"Video: {video_path}")
        print(f"Prediction: {results['prediction']}")
        print(f"Confidence: {results['confidence']:.2%}")
        print(f"Normal Score: {results['normal_score']:.4f}")
        print(f"Hazard Score: {results['hazard_score']:.4f}")
        print("="*50)
        
        # Interpretation
        if results['prediction'] == 'Hazard':
            print("🚨 SAFETY HAZARD DETECTED! 🚨")
        else:
            print("✅ No safety hazards detected")
            
    except Exception as e:
        print(f"Error processing video: {e}")
        print("Please check that:")
        print("1. The video path is correct")
        print("2. The video file is not corrupted")
        print("3. The video format is supported (mp4, avi, mov, etc.)")

if __name__ == "__main__":

    test_path = "Testing"
    files = os.listdir(test_path)
    for file in files:
        video = os.path.join(test_path, file)
        main(video)
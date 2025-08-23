import torch
import torch.nn as nn
from torch.utils.data import DataLoader, WeightedRandomSampler
from torchvision import transforms
from torchvision.models import mobilenet_v2
import numpy as np
from PIL import Image
import os
import cv2
from sklearn.model_selection import train_test_split
from tqdm import tqdm
import glob
import json

# Set device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# Configuration
BATCH_SIZE = 16  # Increased batch size
NUM_FRAMES = 12  # Reduced frames for speed
IMG_SIZE = 224
NUM_EPOCHS = 20
LEARNING_RATE = 1e-4
NUM_WORKERS = 4  # Parallel data loading

# Calculate class weights
hazard_count = 420
normal_count = 680
total = hazard_count + normal_count
weight_for_hazard = total / (2 * hazard_count)
weight_for_normal = total / (2 * normal_count)
class_weights = torch.tensor([weight_for_normal, weight_for_hazard])
print(f"Class weights: Normal={weight_for_normal:.3f}, Hazard={weight_for_hazard:.3f}")

# Define transformations
train_transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.RandomHorizontalFlip(),
    transforms.ColorJitter(brightness=0.2, contrast=0.2),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

val_transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Pre-extract frames to numpy arrays for faster loading
def pre_extract_frames(video_paths, output_dir):
    """Pre-extract frames from all videos and save as numpy arrays"""
    os.makedirs(output_dir, exist_ok=True)
    frame_paths = []

    for video_path in tqdm(video_paths, desc="Pre-extracting frames"):
        try:
            cap = cv2.VideoCapture(video_path)
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

            # Select frame indices
            if total_frames <= NUM_FRAMES:
                frame_indices = list(range(total_frames))
                frame_indices += [total_frames - 1] * (NUM_FRAMES - total_frames)
            else:
                stride = max(1, total_frames // NUM_FRAMES)
                frame_indices = [i * stride for i in range(NUM_FRAMES)]
                frame_indices = [min(idx, total_frames - 1) for idx in frame_indices]

            # Extract and resize frames
            frames = []
            for idx in frame_indices:
                cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
                ret, frame = cap.read()
                if ret:
                    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    frame = cv2.resize(frame, (IMG_SIZE, IMG_SIZE))
                    frames.append(frame)
                else:
                    frames.append(np.zeros((IMG_SIZE, IMG_SIZE, 3), dtype=np.uint8))

            cap.release()

            # Save as numpy array
            video_name = os.path.splitext(os.path.basename(video_path))[0]
            save_path = os.path.join(output_dir, f"{video_name}.npy")
            np.save(save_path, np.array(frames))
            frame_paths.append(save_path)

        except Exception as e:
            print(f"Error processing {video_path}: {e}")
            # Create empty array as fallback
            save_path = os.path.join(output_dir, f"error_{len(frame_paths)}.npy")
            np.save(save_path, np.zeros((NUM_FRAMES, IMG_SIZE, IMG_SIZE, 3), dtype=np.uint8))
            frame_paths.append(save_path)

    return frame_paths

# Fast Dataset that loads pre-extracted frames
class FastVideoDataset(torch.utils.data.Dataset):
    def __init__(self, frame_paths, labels, transform=None):
        self.frame_paths = frame_paths
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.frame_paths)

    def __getitem__(self, idx):
        # Load pre-extracted frames (much faster than video decoding)
        frames = np.load(self.frame_paths[idx])

        # Apply transformations
        if self.transform:
            processed_frames = []
            for frame in frames:
                frame_pil = Image.fromarray(frame)
                transformed = self.transform(frame_pil)
                processed_frames.append(transformed)
            frames = torch.stack(processed_frames)
        else:
            # Convert to tensor without augmentation for validation
            frames = torch.from_numpy(frames).float() / 255.0
            frames = frames.permute(0, 3, 1, 2)  # [T, H, W, C] -> [T, C, H, W]

        return frames, self.labels[idx]

# CNN + LSTM Model
class VideoSafetyClassifier(nn.Module):
    def __init__(self, num_classes=2, lstm_units=128):
        super().__init__()

        # Use pre-trained MobileNetV2 for feature extraction
        self.cnn = mobilenet_v2(weights='IMAGENET1K_V1')
        self.cnn.classifier = nn.Identity()

        # Freeze CNN parameters initially
        for param in self.cnn.parameters():
            param.requires_grad = False

        # MobileNetV2 outputs 1280 features
        self.feature_adjust = nn.Linear(1280, 512)

        # LSTM for temporal processing
        self.lstm = nn.LSTM(512, lstm_units, batch_first=True)

        # Classification head
        self.fc = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(lstm_units, num_classes)
        )

    def forward(self, x):
        batch_size, num_frames, c, h, w = x.shape

        # Process each frame through CNN
        x = x.view(batch_size * num_frames, c, h, w)
        cnn_features = self.cnn(x)

        # Adjust feature dimension
        cnn_features = self.feature_adjust(cnn_features)

        # Reshape for LSTM
        cnn_features = cnn_features.view(batch_size, num_frames, -1)

        # Process sequence through LSTM
        lstm_out, _ = self.lstm(cnn_features)

        # Use the output from the last time step
        out = self.fc(lstm_out[:, -1, :])

        return out

# Main execution
if __name__ == "__main__":
    # Prepare data paths
    hazard_videos = glob.glob('safety_videos/hazard/*.mp4')
    normal_videos = glob.glob('safety_videos/normal/*.mp4')
    all_videos = hazard_videos + normal_videos
    all_labels = [1] * len(hazard_videos) + [0] * len(normal_videos)

    # Pre-extract frames (only need to do this once!)
    frames_dir = 'pre_extracted_frames3'
    if not os.path.exists(frames_dir):
        print("Pre-extracting frames for faster training...")
        frame_paths = pre_extract_frames(all_videos, frames_dir)
    else:
        frame_paths = glob.glob(os.path.join(frames_dir, '*.npy'))
        print(f"Found {len(frame_paths)} pre-extracted frame files")

    # Split dataset
    train_frame_paths, val_frame_paths, train_labels, val_labels = train_test_split(
        frame_paths, all_labels, test_size=0.2, random_state=42, stratify=all_labels
    )

    print(f"Training videos: {len(train_frame_paths)}")
    print(f"Validation videos: {len(val_frame_paths)}")

    # Create datasets
    train_dataset = FastVideoDataset(train_frame_paths, train_labels, transform=train_transform)
    val_dataset = FastVideoDataset(val_frame_paths, val_labels, transform=val_transform)

    # Create weighted sampler
    def create_sampler(dataset):
        sample_weights = [weight_for_hazard if label == 1 else weight_for_normal
                         for label in dataset.labels]
        return WeightedRandomSampler(sample_weights, len(sample_weights), replacement=True)

    train_sampler = create_sampler(train_dataset)

    # Create data loaders with multiple workers
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, sampler=train_sampler,
                             num_workers=NUM_WORKERS, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False,
                           num_workers=2, pin_memory=True)

    # Initialize model
    model = VideoSafetyClassifier().to(device)
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")

    # Loss and optimizer
    criterion = nn.CrossEntropyLoss(weight=class_weights.to(device))
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

    # Training function
    def train_epoch(model, loader, criterion, optimizer, device):
        model.train()
        total_loss, correct, total = 0, 0, 0

        for data, target in tqdm(loader, desc="Training"):
            data, target = data.to(device), target.to(device)

            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            pred = output.argmax(dim=1)
            correct += (pred == target).sum().item()
            total += target.size(0)

        return total_loss / len(loader), correct / total

    # Validation function
    def validate_epoch(model, loader, criterion, device):
        model.eval()
        total_loss, correct, total = 0, 0, 0

        with torch.no_grad():
            for data, target in tqdm(loader, desc="Validation"):
                data, target = data.to(device), target.to(device)

                output = model(data)
                loss = criterion(output, target)

                total_loss += loss.item()
                pred = output.argmax(dim=1)
                correct += (pred == target).sum().item()
                total += target.size(0)

        return total_loss / len(loader), correct / total

    # Training loop
    print("Starting training...")
    best_val_acc = 0

    for epoch in range(20):
        print(f"\nEpoch {epoch+1}/{NUM_EPOCHS}")

        train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_acc = validate_epoch(model, val_loader, criterion, device)

        print(f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}")
        print(f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), 'best_model.pth')
            print("Saved best model!")

        # Unfreeze CNN after 5 epochs
        if epoch == 5:
            print("Unfreezing CNN for fine-tuning...")
            for param in model.cnn.parameters():
                param.requires_grad = True

    print(f"Training completed! Best validation accuracy: {best_val_acc:.4f}")
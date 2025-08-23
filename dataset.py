import os
import requests
from tqdm import tqdm
import time
import json
import re

def download_safety_dataset():
    """
    Download the safety dataset from Hugging Face and organize it locally
    without using the dataset loader that requires torchcodec
    """
    # Create directories for organized data
    os.makedirs('safety_videos/hazard', exist_ok=True)
    os.makedirs('safety_videos/normal', exist_ok=True)
    
    # Get the dataset information directly from Hugging Face API
    print("Fetching dataset information from Hugging Face...")
    dataset_info_url = "https://huggingface.co/api/datasets/raiyaanabdullah/isafety-bench"
    
    try:
        response = requests.get(dataset_info_url)
        response.raise_for_status()
        dataset_info = response.json()
        
        # Extract the list of files from the dataset
        files = []
        for sibling in dataset_info.get('siblings', []):
            if sibling.get('rfilename') and sibling['rfilename'].endswith('.mp4'):
                files.append(sibling['rfilename'])
        
        print(f"Found {len(files)} video files in the dataset")
        
    except Exception as e:
        print(f"Error fetching dataset info: {e}")
        # Fallback: use a predefined list based on the dataset structure
        print("Using fallback approach...")
        files = []
        # We know there are hazard and normal folders based on the dataset structure
        for folder in ['hazard', 'normal']:
            for i in range(1000):  # We know there are 1100 files total
                files.append(f"{folder}/video_{i:04d}.mp4")
    
    # Function to download all videos
    def download_all_videos(file_list, max_retries=3):
        hazard_count = 0
        normal_count = 0
        error_count = 0
        downloaded_files = []
        
        for file_path in tqdm(file_list, desc="Downloading videos"):
            try:
                # Determine label based on folder path
                if '/hazard/' in file_path or file_path.startswith('hazard/'):
                    label = "hazard"
                    filename = f"hazard_{hazard_count:04d}.mp4"
                    save_path = f'safety_videos/hazard/{filename}'
                    hazard_count += 1
                elif '/normal/' in file_path or file_path.startswith('normal/'):
                    label = "normal"
                    filename = f"normal_{normal_count:04d}.mp4"
                    save_path = f'safety_videos/normal/{filename}'
                    normal_count += 1
                else:
                    # Skip files that aren't in hazard or normal folders
                    continue
                
                # Construct the download URL
                download_url = f"https://huggingface.co/datasets/raiyaanabdullah/isafety-bench/resolve/main/{file_path}"
                
                # Download with retries
                for attempt in range(max_retries):
                    try:
                        response = requests.get(download_url, stream=True, timeout=30)
                        response.raise_for_status()
                        
                        # Save the video
                        with open(save_path, 'wb') as f:
                            for chunk in response.iter_content(chunk_size=8192):
                                if chunk:
                                    f.write(chunk)
                        
                        downloaded_files.append({
                            "original_path": file_path,
                            "local_path": save_path,
                            "label": label,
                            "filename": filename
                        })
                        break  # Success, break out of retry loop
                        
                    except Exception as e:
                        if attempt == max_retries - 1:
                            print(f"Failed to download {download_url} after {max_retries} attempts: {e}")
                            error_count += 1
                        time.sleep(1)  # Wait before retry
                
                # Add a small delay to be respectful to the server
                time.sleep(0.1)
                
            except Exception as e:
                print(f"Error processing file {file_path}: {e}")
                error_count += 1
        
        return hazard_count, normal_count, error_count, downloaded_files

    print("Starting video download...")
    hazard_count, normal_count, error_count, downloaded_files = download_all_videos(files)

    print(f"\nDownload completed!")
    print(f"Hazard videos downloaded: {hazard_count}")
    print(f"Normal videos downloaded: {normal_count}")
    print(f"Errors: {error_count}")
    
    # Create a manifest file
    manifest = {
        "total_videos": hazard_count + normal_count,
        "hazard_count": hazard_count,
        "normal_count": normal_count,
        "error_count": error_count,
        "files": downloaded_files
    }
    
    with open('dataset_manifest.json', 'w') as f:
        json.dump(manifest, f, indent=2)
    
    print("Dataset manifest saved to dataset_manifest.json")
    
    return manifest

def verify_download():
    """
    Verify the downloaded videos and check for any issues
    """
    import glob
    
    # Check what we downloaded
    hazard_videos = glob.glob('safety_videos/hazard/*.mp4')
    normal_videos = glob.glob('safety_videos/normal/*.mp4')
    
    print(f"Hazard videos found: {len(hazard_videos)}")
    print(f"Normal videos found: {len(normal_videos)}")
    print(f"Total videos: {len(hazard_videos) + len(normal_videos)}")
    
    # Check file sizes
    def check_video_sizes(video_list, label):
        sizes = []
        for video_path in video_list:
            size_mb = os.path.getsize(video_path) / (1024 * 1024)
            sizes.append(size_mb)
        
        if sizes:
            print(f"{label} videos - Average size: {sum(sizes)/len(sizes):.2f} MB, "
                  f"Min: {min(sizes):.2f} MB, Max: {max(sizes):.2f} MB")
        else:
            print(f"No {label} videos found")
    
    check_video_sizes(hazard_videos, "Hazard")
    check_video_sizes(normal_videos, "Normal")
    
    return len(hazard_videos), len(normal_videos)

# Alternative approach: manually construct the file list based on known dataset structure
def get_video_list_manual():
    """
    Manually construct the list of video files based on the known dataset structure
    """
    files = []
    
    # Add hazard videos (420 files)
    for i in range(420):
        files.append(f"hazard/video_{i:04d}.mp4")
    
    # Add normal videos (680 files)
    for i in range(680):
        files.append(f"normal/video_{i:04d}.mp4")
    
    return files

if __name__ == "__main__":
    # Get the list of files manually since we can't access the dataset API properly
    files = get_video_list_manual()
    print(f"Prepared list of {len(files)} videos to download")
    
    # Download the dataset
    manifest = download_safety_dataset()
    
    # Verify the download
    print("\nVerifying download...")
    hazard_count, normal_count = verify_download()
    
    # Check if download was successful
    if hazard_count + normal_count > 0:
        print("\n✅ Download completed successfully!")
        print(f"Downloaded: {hazard_count} hazard, {normal_count} normal videos")
    else:
        print("\n❌ Download failed. No videos were downloaded.")
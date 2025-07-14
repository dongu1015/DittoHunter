#!/usr/bin/env python3
"""
⚡ Simple Image Preprocessor - Local File Processing

This module provides simple preprocessing for local image datasets
for deepfake detection training. It validates image quality and 
organizes files for training.

Features:
- 📂 Local file processing
- 🔍 Basic image validation
- 📊 Progress monitoring
- 🎯 Quality assessment and organization

Author: Advanced Deepfake Detection Team
License: MIT
"""

# Standard Library
import os
import sys
import shutil
import glob
from pathlib import Path
from typing import List, Optional, Tuple

# Third Party - Core
import numpy as np
import cv2
from PIL import Image
from tqdm import tqdm

# roop 모듈 경로 추가 (선택적)
sys.path.append(os.path.join(os.path.dirname(__file__), '.'))

try:
    from roop.face_analyser import get_one_face, get_many_faces
    import roop.globals
    
    # GPU 우선 설정
    gpu_providers = ['CUDAExecutionProvider', 'CPUExecutionProvider']
    roop.globals.execution_providers = gpu_providers
    print("🚀 Roop face analyzer available (GPU enabled)")
    
    ROOP_AVAILABLE = True
except ImportError:
    print("⚠️  Roop not available - using basic image validation")
    ROOP_AVAILABLE = False


class SimpleImagePreprocessor:
    """Simple image preprocessor for local datasets"""
    
    def __init__(self, input_dir: str = "input_images", output_dir: str = "data"):
        """
        Initialize preprocessor
        
        Args:
            input_dir: Directory containing input images to process
            output_dir: Directory to save organized dataset (fake/real folders)
        """
        self.input_dir = input_dir
        self.output_dir = output_dir
        self.fake_dir = os.path.join(output_dir, "fake")
        self.real_dir = os.path.join(output_dir, "real")
        
        # Create directories
        os.makedirs(self.input_dir, exist_ok=True)
        os.makedirs(self.fake_dir, exist_ok=True)
        os.makedirs(self.real_dir, exist_ok=True)
        
        print(f"📁 Preprocessor directories:")
        print(f"   Input: {self.input_dir}")
        print(f"   Output Fake: {self.fake_dir}")
        print(f"   Output Real: {self.real_dir}")
        
        self.processed_count = 0
        self.valid_count = 0
        self.face_detected_count = 0
    
    def get_image_files(self, directory: str) -> List[str]:
        """Get all image files from a directory"""
        extensions = ['*.jpg', '*.jpeg', '*.png', '*.bmp', '*.tiff']
        files = []
        
        for ext in extensions:
            files.extend(glob.glob(os.path.join(directory, ext)))
            files.extend(glob.glob(os.path.join(directory, ext.upper())))
        
        return sorted(files)
    
    def validate_image(self, image_path: str) -> Tuple[bool, Optional[np.ndarray]]:
        """
        Validate image quality and readability
        
        Returns:
            (is_valid, image_array)
        """
        try:
            # Load image
            img = cv2.imread(image_path)
            if img is None:
                return False, None
            
            # Check minimum size
            height, width = img.shape[:2]
            if height < 64 or width < 64:
                return False, None
            
            # Check if image is not corrupted
            if img.size == 0:
                return False, None
            
            # Convert to RGB for consistency
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            
            return True, img_rgb
            
        except Exception as e:
            print(f"Error validating {image_path}: {e}")
            return False, None
    
    def detect_face(self, img: np.ndarray) -> Tuple[bool, float]:
        """
        Detect faces in image using roop (if available) or basic detection
        
        Returns:
            (face_detected, quality_score)
        """
        if not ROOP_AVAILABLE:
            # Basic face detection using OpenCV
            gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
            face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
            faces = face_cascade.detectMultiScale(gray, 1.1, 4)
            
            if len(faces) > 0:
                return True, 0.8  # Assume good quality if face detected
            return False, 0.0
        
        try:
            # Use roop for face detection
            face = get_one_face(img)
            if face is not None:
                # Calculate quality based on face properties
                quality = getattr(face, 'det_score', 0.8)  # Detection confidence
                return True, quality
            return False, 0.0
            
        except Exception as e:
            print(f"Face detection error: {e}")
            return False, 0.0
    
    def process_single_image(self, image_path: str, target_dir: str, 
                           min_quality: float = 0.5) -> bool:
        """
        Process a single image and copy to target directory if valid
        
        Args:
            image_path: Path to input image
            target_dir: Directory to copy processed image
            min_quality: Minimum face quality threshold
            
        Returns:
            Success status
        """
        # Validate image
        is_valid, img = self.validate_image(image_path)
        if not is_valid:
            return False
        
        self.valid_count += 1
        
        # Detect face (optional quality check)
        face_detected, quality = self.detect_face(img)
        
        if face_detected:
            self.face_detected_count += 1
            
            # Check quality threshold
            if quality < min_quality:
                return False
        
        # Copy to target directory
        try:
            filename = os.path.basename(image_path)
            target_path = os.path.join(target_dir, filename)
            
            # Avoid overwriting - add number suffix if needed
            counter = 1
            base_name, ext = os.path.splitext(filename)
            while os.path.exists(target_path):
                new_filename = f"{base_name}_{counter}{ext}"
                target_path = os.path.join(target_dir, new_filename)
                counter += 1
            
            shutil.copy2(image_path, target_path)
            return True
            
        except Exception as e:
            print(f"Error copying {image_path}: {e}")
            return False
    
    def process_directory(self, is_fake: bool = True, min_quality: float = 0.5):
        """
        Process all images in input directory
        
        Args:
            is_fake: Whether images are fake (True) or real (False)
            min_quality: Minimum face quality threshold
        """
        target_dir = self.fake_dir if is_fake else self.real_dir
        label = "FAKE" if is_fake else "REAL"
        
        print(f"\n🔄 Processing {label} images...")
        print(f"   Input: {self.input_dir}")
        print(f"   Output: {target_dir}")
        print(f"   Min quality: {min_quality}")
        
        # Get all images
        image_files = self.get_image_files(self.input_dir)
        
        if not image_files:
            print(f"⚠️  No images found in {self.input_dir}")
            return
        
        print(f"📊 Found {len(image_files)} images to process")
        
        # Reset counters
        self.processed_count = 0
        self.valid_count = 0
        self.face_detected_count = 0
        copied_count = 0
        
        # Process images with progress bar
        for image_path in tqdm(image_files, desc=f"Processing {label}"):
            self.processed_count += 1
            
            success = self.process_single_image(image_path, target_dir, min_quality)
            if success:
                copied_count += 1
        
        # Print summary
        print(f"\n✅ {label} processing complete!")
        print(f"   Processed: {self.processed_count}")
        print(f"   Valid images: {self.valid_count}")
        print(f"   Face detected: {self.face_detected_count}")
        print(f"   Copied to dataset: {copied_count}")
        print(f"   Success rate: {copied_count/self.processed_count*100:.1f}%")
    
    def organize_mixed_directory(self, fake_keywords: List[str] = None, 
                               real_keywords: List[str] = None):
        """
        Organize mixed directory into fake/real based on filename keywords
        
        Args:
            fake_keywords: Keywords in filename that indicate fake images
            real_keywords: Keywords in filename that indicate real images
        """
        if fake_keywords is None:
            fake_keywords = ['fake', 'generated', 'synthetic', 'deepfake', 'gan']
        
        if real_keywords is None:
            real_keywords = ['real', 'authentic', 'original', 'natural']
        
        print(f"\n🗂️  Organizing mixed directory...")
        print(f"   Fake keywords: {fake_keywords}")
        print(f"   Real keywords: {real_keywords}")
        
        image_files = self.get_image_files(self.input_dir)
        
        fake_count = 0
        real_count = 0
        unknown_count = 0
        
        for image_path in tqdm(image_files, desc="Organizing"):
            filename = os.path.basename(image_path).lower()
            
            # Check for fake keywords
            is_fake = any(keyword in filename for keyword in fake_keywords)
            is_real = any(keyword in filename for keyword in real_keywords)
            
            if is_fake and not is_real:
                success = self.process_single_image(image_path, self.fake_dir)
                if success:
                    fake_count += 1
            elif is_real and not is_fake:
                success = self.process_single_image(image_path, self.real_dir)
                if success:
                    real_count += 1
            else:
                unknown_count += 1
                print(f"⚠️  Unknown category: {filename}")
        
        print(f"\n✅ Organization complete!")
        print(f"   Fake images: {fake_count}")
        print(f"   Real images: {real_count}")
        print(f"   Unknown/Skipped: {unknown_count}")
    
    def get_dataset_stats(self) -> dict:
        """Get statistics about the processed dataset"""
        fake_files = self.get_image_files(self.fake_dir)
        real_files = self.get_image_files(self.real_dir)
        
        stats = {
            'fake_count': len(fake_files),
            'real_count': len(real_files),
            'total_count': len(fake_files) + len(real_files)
        }
        
        print(f"\n📊 Dataset Statistics:")
        print(f"   Fake images: {stats['fake_count']:,}")
        print(f"   Real images: {stats['real_count']:,}")
        print(f"   Total images: {stats['total_count']:,}")
        
        return stats


def main():
    """Demo function"""
    print("⚡ Simple Image Preprocessor Demo")
    print("=" * 50)
    
    # Initialize preprocessor
    preprocessor = SimpleImagePreprocessor(
        input_dir="input_images",
        output_dir="data"
    )
    
    # Check for input images
    input_images = preprocessor.get_image_files("input_images")
    
    if not input_images:
        print(f"\n📝 To use this preprocessor:")
        print(f"   1. Create 'input_images' directory")
        print(f"   2. Add images to process")
        print(f"   3. Run this script with options:")
        print(f"      - For fake images: python preprocess.py --fake")
        print(f"      - For real images: python preprocess.py --real")
        print(f"      - For mixed directory: python preprocess.py --organize")
        return
    
    print(f"Found {len(input_images)} images in input directory")
    
    # Simple demo - organize based on filename
    preprocessor.organize_mixed_directory()
    
    # Show final stats
    preprocessor.get_dataset_stats()


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Simple Image Preprocessor')
    parser.add_argument('--fake', action='store_true', help='Process as fake images')
    parser.add_argument('--real', action='store_true', help='Process as real images')
    parser.add_argument('--organize', action='store_true', help='Auto-organize mixed directory')
    parser.add_argument('--input', default='input_images', help='Input directory')
    parser.add_argument('--output', default='data', help='Output directory')
    parser.add_argument('--quality', type=float, default=0.5, help='Minimum face quality')
    
    args = parser.parse_args()
    
    # Initialize preprocessor
    preprocessor = SimpleImagePreprocessor(args.input, args.output)
    
    if args.fake:
        preprocessor.process_directory(is_fake=True, min_quality=args.quality)
    elif args.real:
        preprocessor.process_directory(is_fake=False, min_quality=args.quality)
    elif args.organize:
        preprocessor.organize_mixed_directory()
    else:
        main()
    
    # Show final stats
    preprocessor.get_dataset_stats()

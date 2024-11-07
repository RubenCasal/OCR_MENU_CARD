from ultralytics import YOLO
import albumentations as A
from albumentations.pytorch import ToTensorV2
import cv2
import torch
from torch.utils.data import Dataset
import os

# Define custom dataset with Albumentations
class CustomDataset(Dataset):
    def __init__(self, images_dir, labels_dir, transform=None):
        self.images_dir = images_dir
        self.labels_dir = labels_dir
        self.image_paths = [os.path.join(images_dir, img) for img in os.listdir(images_dir)]
        self.label_paths = [os.path.join(labels_dir, lbl) for lbl in os.listdir(labels_dir)]
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image_path = self.image_paths[idx]
        label_path = self.label_paths[idx]
        
        # Load image
        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # Convert to RGB
        
        # Load label (assumed to be in YOLO format)
        with open(label_path, 'r') as f:
            bboxes = [list(map(float, line.strip().split())) for line in f.readlines()]  # Read YOLO format bboxes

        if self.transform:
            transformed = self.transform(image=image, bboxes=bboxes, class_labels=[bbox[0] for bbox in bboxes])
            image = transformed['image']
            bboxes = transformed['bboxes']

        return image, bboxes

# Define your Albumentations augmentation pipeline
augmentations = A.Compose([
    A.ToGray(p=1.0),  # Convert to grayscale
    A.HorizontalFlip(p=0.5),
    A.RandomBrightnessContrast(p=0.2),
    A.ShiftScaleRotate(shift_limit=0.0625, scale_limit=0.1, rotate_limit=15, p=0.5),
    A.Blur(blur_limit=3, p=0.1),
    ToTensorV2()  # Convert image to PyTorch tensor
], bbox_params=A.BboxParams(format="yolo", label_fields=["class_labels"]))

if __name__ == '__main__':
    # Load the YOLOv8 model
    model = YOLO('yolov8n.pt')  # Use the appropriate model size

    # Create custom dataset with augmentations
    dataset = CustomDataset(
        images_dir='./yolov8_menu_card/menu_data/train/images',  # Path to your images directory
        labels_dir='./yolov8_menu_card/menu_data/train/labels',  # Path to your labels directory
        transform=augmentations
    )

    # Train the model using the custom dataset and transformations
    model.train(
        data='./yolov8_menu_card/menu_data/data.yaml',    # Path to your dataset YAML file
        epochs=1000,                      # Number of epochs
        batch=8,                         # Batch size
        imgsz=640,                       # Image size
        device=0,                        # Use GPU (0 for the first GPU)
        augment=True                      # Enable augmentations
    )

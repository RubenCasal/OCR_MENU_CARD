from paddleocr import PaddleOCR
import cv2
from collections import OrderedDict
import numpy as np
import os
ocr = PaddleOCR(use_angle_cls = True, lang='es')

def filter_items_with_price(organized_items):
    """
    Filters out items that do not have a 'price' component.

    Args:
        organized_items (dict): Dictionary of items with their components.

    Returns:
        dict: Filtered dictionary with only items that have a 'price' component.
    """
    # Filter items where 'price' is not None
    filtered_items = {item_bbox: components for item_bbox, components in organized_items.items() if components["price"] is not None}
    
    return filtered_items




def sort_item_bboxes_by_position(items):
    """
    Sorts item bounding boxes first by the left (x1) position and then by top (y1) position.

    Args:
        items (dict): Dictionary where each key is an item bounding box (tuple of x1, y1, x2, y2),
                      and the value is a dictionary of contained components.

    Returns:
        OrderedDict: Dictionary with item bounding boxes sorted by the left-to-right and top-to-bottom criteria.
    """
    # Define the sorting key with two criteria:
    # 1. Left (x1) position
    # 2. Top (y1) position within a tolerance of 10 pixels for x1
    sorted_items = OrderedDict(
        sorted(
            items.items(),
            key=lambda item: (round(item[0][0] / 10) * 10, item[0][1])
        )
    )
    
    return sorted_items

def extract_text_from_components(image_path, sorted_items, output_txt_path="extracted_text.txt", debug_dir="debug_images"):
    """
    Extracts text from each component in the sorted bounding boxes and saves it to a text file.
    
    Args:
        image_path (str): Path to the image file.
        sorted_items (OrderedDict): Ordered dictionary with item bounding boxes as keys and component bboxes as values.
        output_txt_path (str): Path to save the extracted text.
        debug_dir (str): Directory to save debug images of each component being processed.
    """
    # Load image
    image = cv2.imread(image_path)
   
    if image is None:
        print(f"Error: Could not load image from path {image_path}")
        return
    
  

    # Open file to write the extracted text
    with open(output_txt_path, "w", encoding="utf-8") as f:
        for i, (item_bbox, components) in enumerate(sorted_items.items(), start=1):
            # Initialize text variables for each component
            dish_name, description, price = "", "", ""

            # Extract text for each component, if present
            if components.get("title") is not None:
                title_bbox = components["title"]
                x1, y1, x2, y2 = map(int, title_bbox)
                title_img = image[max(0, y1 - 5):min(image.shape[0], y2 + 5), max(0, x1 - 5):min(image.shape[1], x2 + 5)]
                
                # Increase resolution before OCR
                title_img = increase_resolution(title_img)
                
                # Perform OCR
                title_text = ocr.ocr(title_img, cls=True)
                if title_text and title_text[0] and title_text[0][0] and title_text[0][0][1]:
                    dish_name = title_text[0][0][1][0]

            if components.get("description") is not None:
                description_bbox = components["description"]
                x1, y1, x2, y2 = map(int, description_bbox)
                description_img = image[y1:y2, x1:x2]
                
                # Increase resolution before OCR
                description_img = increase_resolution(description_img)
                
                # Perform OCR
                description_text = ocr.ocr(description_img, cls=True)
                if description_text and description_text[0] and description_text[0][0] and description_text[0][0][1]:
                    description = description_text[0][0][1][0]

            if components.get("price") is not None:
                price_bbox = components["price"]
                x1, y1, x2, y2 = map(int, price_bbox)
                price_img = image[y1:y2, x1:x2]
                
                # Increase resolution before OCR
                price_img = increase_resolution(price_img)
                
                # Perform OCR
                price_text = ocr.ocr(price_img, cls=True)
                if price_text and price_text[0] and price_text[0][0] and price_text[0][0][1]:
                    price = price_text[0][0][1][0]

            # Write to file
            f.write(f"{i}: {dish_name} {description} -> {price}€\n")
            print(f"Extracted text for item {i}: {dish_name} {description} -> {price}€")

    print(f"Extraction complete. Results saved to {output_txt_path}")


def increase_resolution(image, scale_factor=1.5):
    """
    Converts the image to grayscale and increases its resolution by the specified scale factor.
    
    Args:
        image (numpy array): The image to process and upscale.
        scale_factor (float): The factor by which to upscale the image.
    
    Returns:
        numpy array: The upscaled grayscale image.
    """
    # Convert the image to grayscale
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Upscale the grayscale image
    upscaled_image = cv2.resize(gray_image, None, fx=scale_factor, fy=scale_factor, interpolation=cv2.INTER_CUBIC)
    
    return upscaled_image
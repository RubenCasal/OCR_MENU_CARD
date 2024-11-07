import sys
import os
import cv2

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from yolov8_menu_card.model_detect_bbox import get_bounding_boxes, organize_items_with_contained_components
from image_to_text.utils_ocr import sort_item_bboxes_by_position, filter_items_with_price, extract_text_from_components

def process_menu_image(test_image_path, output_image_path="output_with_bboxes.jpg", output_txt_path="menu_text_output.txt"):
    """
    Processes a menu image, detects and organizes bounding boxes, sorts items, and extracts text.
    
    Args:
        test_image_path (str): Path to the test image.
        output_image_path (str): Path to save the image with drawn bounding boxes.
        output_txt_path (str): Path to save the extracted text output.
    """
    # Get bounding boxes
    boxes = get_bounding_boxes(test_image_path, draw=True, output_path=output_image_path)

    # Organize detected components by item
    organized_items = organize_items_with_contained_components(boxes)
    
    # Filter item bboxes with no price included
    filtered_bboxes = filter_items_with_price(organized_items)

    # Sort the item bounding boxes by left-to-right, top-to-bottom criteria
    sorted_bboxes = sort_item_bboxes_by_position(filtered_bboxes)
    
    # Extract text from components and save to a text file
    extract_text_from_components(test_image_path, sorted_bboxes, output_txt_path=output_txt_path)

# Testing block
if __name__ == "__main__":
    # Define the paths for testing
    test_image_path = "./real_menu_card_images/carta2.jpg"
    output_image_path = "output_with_bboxes.jpg"
    output_txt_path = "menu_text_output.txt"
    
    # Run the function
    process_menu_image(test_image_path, output_image_path, output_txt_path)

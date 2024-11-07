from ultralytics import YOLO
import cv2

# Load your trained YOLO model
model = YOLO('./yolov8_menu_card/menu_items_model/weights/best.pt')  # Update with the path to your trained model file

def get_bounding_boxes(image_path, draw=False, output_path="output_with_bboxes.jpg"):
    """
    Function to get bounding boxes from an image using YOLOv8.
    
    Args:
        image_path (str): Path to the input image.
        draw (bool): If True, draws the bounding boxes on the image and saves it.
        output_path (str): Path to save the image with drawn bounding boxes (if draw is True).
    
    Returns:
        list: A list of bounding boxes, where each bounding box is represented as
              [x1, y1, x2, y2, confidence, class_id].
    """
    # Load and preprocess the image
    image = cv2.imread(image_path)
    
    # Get predictions from the model
    results = model.predict(source=image, save=False)
    
    # Extract bounding boxes, confidence scores, and class IDs
    bounding_boxes = []
    for box in results[0].boxes.data.cpu().numpy():
        x1, y1, x2, y2, confidence, class_id = box
        bounding_boxes.append([x1, y1, x2, y2, confidence, int(class_id)])
    
    # Optionally draw and save the image with bounding boxes
    if draw:
        for box in bounding_boxes:
            x1, y1, x2, y2, confidence, class_id = box
            # Draw bounding box
            cv2.rectangle(image, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
            # Draw label with confidence
            label = f"Class {class_id}: {confidence:.2f}"
            cv2.putText(image, label, (int(x1), int(y1) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        
        # Save the image with bounding boxes
        cv2.imwrite(output_path, image)
        print(f"Image with bounding boxes saved to {output_path}")
    
    return bounding_boxes

def calculate_iou(box1, box2):
    """
    Calculate Intersection over Union (IoU) between two bounding boxes.
    
    Args:
        box1, box2: Each box is a list [x1, y1, x2, y2]
    
    Returns:
        float: IoU value between 0 and 1
    """
    x1, y1, x2, y2 = box1
    x1_p, y1_p, x2_p, y2_p = box2
    
    # Calculate intersection area
    xi1, yi1 = max(x1, x1_p), max(y1, y1_p)
    xi2, yi2 = min(x2, x2_p), min(y2, y2_p)
    inter_area = max(0, xi2 - xi1) * max(0, yi2 - yi1)
    
    # Calculate union area
    box1_area = (x2 - x1) * (y2 - y1)
    box2_area = (x2_p - x1_p) * (y2_p - y1_p)
    union_area = box1_area + box2_area - inter_area
    
    # Calculate IoU
    iou = inter_area / union_area if union_area > 0 else 0
    return iou

def organize_items_by_components(bounding_boxes, iou_threshold=0.3):
    """
    Organizes detected bounding boxes into items with their respective components.
    
    Args:
        bounding_boxes (list): List of bounding boxes with format:
                               [x1, y1, x2, y2, confidence, class_id]
        iou_threshold (float): Minimum IoU required to associate a component with an item.
    
    Returns:
        list: A list where each item contains its title, description, and price bounding boxes.
    """
    items = []
    titles = []
    descriptions = []
    prices = []
    
    # Separate bounding boxes by class ID
    for box in bounding_boxes:
        x1, y1, x2, y2, confidence, class_id = box
        if class_id == 0:  # Assuming 'item' is class ID 0
            items.append({"bbox": [x1, y1, x2, y2], "title": None, "description": None, "price": None})
        elif class_id == 1:  # Assuming 'title' is class ID 1
            titles.append({"bbox": [x1, y1, x2, y2], "confidence": confidence})
        elif class_id == 2:  # Assuming 'description' is class ID 2
            descriptions.append({"bbox": [x1, y1, x2, y2], "confidence": confidence})
        elif class_id == 3:  # Assuming 'price' is class ID 3
            prices.append({"bbox": [x1, y1, x2, y2], "confidence": confidence})
    
    # Assign titles, descriptions, and prices to their respective items based on IoU
    for item in items:
        item_bbox = item["bbox"]
        
        # Find titles with sufficient overlap inside the item bbox
        for title in titles:
            if calculate_iou(item_bbox, title["bbox"]) >= iou_threshold:
                item["title"] = title["bbox"]
        
        # Find descriptions with sufficient overlap inside the item bbox
        for description in descriptions:
            if calculate_iou(item_bbox, description["bbox"]) >= iou_threshold:
                item["description"] = description["bbox"]
        
        # Find prices with sufficient overlap inside the item bbox
        for price in prices:
            if calculate_iou(item_bbox, price["bbox"]) >= iou_threshold:
                item["price"] = price["bbox"]

    return items

# Example usage of the function
if __name__ == "__main__":
    # Define the path to your test image
    test_image_path = "./real_menu_card_images/carta5.jpg"
    output_image_path = "./image_to_text/output_with_bboxes.jpg"  # Path where you want to save the output image
    
    # Call the function with draw=True to save the image with bounding boxes
    boxes = get_bounding_boxes(test_image_path, draw=True, output_path=output_image_path)
    
    # Print the bounding boxes
    #print("Detected bounding boxes:", boxes)
    
    # Organize detected components by item
    organized_items = organize_items_by_components(boxes)
    print("Organized Items with Components:")
    for item in organized_items:
        print(item)

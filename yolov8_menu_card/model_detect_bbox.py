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
    
    # Define colors for each class_id
    colors = {
        0: (255, 0, 0),    
        1: (0, 255, 0),    
        2: (0, 0, 255),   
        3: (128, 0, 128)
    }
    
    # Extract bounding boxes, confidence scores, and class IDs
    bounding_boxes = []
    for box in results[0].boxes.data.cpu().numpy():
        x1, y1, x2, y2, confidence, class_id = box
        bounding_boxes.append([x1, y1, x2, y2, confidence, int(class_id)])
        
        # Draw the bounding box if 'draw' is set to True
        if draw:
            # Get color based on class_id, default to white if class_id is unknown
            color = colors.get(int(class_id), (255, 255, 255))
            label = f"Class {class_id}: {confidence:.2f}"
            
            # Draw the bounding box with the corresponding color
            cv2.rectangle(image, (int(x1), int(y1)), (int(x2), int(y2)), color, 2)
            # Add label text above the bounding box
            cv2.putText(image, label, (int(x1), int(y1) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

    # Save the image with bounding boxes if drawing is enabled
    if draw:
        cv2.imwrite(output_path, image)
        print(f"Image with bounding boxes saved to {output_path}")
    
    return bounding_boxes

def is_contained(inner, outer):
    """
    Check if 'inner' bbox is fully within 'outer' bbox.
    
    Args:
        inner (list): Bounding box [x1, y1, x2, y2] for the inner component.
        outer (list): Bounding box [x1, y1, x2, y2] for the outer item.
    
    Returns:
        bool: True if inner is fully contained within outer, else False.
    """
    return (
        inner[0] >= outer[0] and inner[1] >= outer[1] and
        inner[2] <= outer[2] and inner[3] <= outer[3]
    )

def organize_items_with_contained_components(bounding_boxes):
    """
    Organizes bounding boxes into items and their contained components (price, title, description).
    
    Args:
        bounding_boxes (list): List of bounding boxes with format [x1, y1, x2, y2, confidence, class_id].
    
    Returns:
        dict: A dictionary where each key is an item bbox and the value is a dictionary of contained components.
    """
    # Separate item bounding boxes and other bounding boxes
    item_bboxes = [box for box in bounding_boxes if box[5] == 1]  # assuming label '1' indicates item bounding boxes
    other_bboxes = {
        0: [],  # descriptions
        2: [],  # prices
        3: []   # titles
    }
    
    # Sort other boxes by class ID
    for box in bounding_boxes:
        if box[5] == 0:  # description
            other_bboxes[0].append(box)
        elif box[5] == 2:  # price
            other_bboxes[2].append(box)
        elif box[5] == 3:  # title
            other_bboxes[3].append(box)
    
    # Initialize dictionary to store the result
    result = {}

    # Loop over each item bounding box and find contained boxes
    for item_bbox in item_bboxes:
        contained_data = {"price": None, "title": None, "description": None}

        # Check for each type of component within the item bbox
        for label, bbox_list in other_bboxes.items():
            for bbox in bbox_list:
                if is_contained(bbox[:4], item_bbox[:4]):
                    # Map the contained bbox to the correct category
                    if label == 2:  # Assuming label '2' is for price
                        contained_data["price"] = bbox[:4]
                    elif label == 3:  # Assuming label '3' is for title
                        contained_data["title"] = bbox[:4]
                    elif label == 0:  # Assuming label '0' is for description
                        contained_data["description"] = bbox[:4]

        # Only add to the result if there are contained elements
        if any(contained_data.values()):
            result[tuple(item_bbox[:4])] = contained_data

    return result

# Example usage of the function
if __name__ == "__main__":
    # Define the path to your test image
    test_image_path = "./real_menu_card_images/carta5.jpg"
    output_image_path = "output_with_bboxes.jpg"  # Path where you want to save the output image
    
    # Get bounding boxes
    boxes = get_bounding_boxes(test_image_path, draw=True, output_path=output_image_path)
    
    # Organize detected components by item
    organized_items = organize_items_with_contained_components(boxes)
    print("Organized Items with Components:")
    for item_bbox, components in organized_items.items():
        print(f"Item BBox: {item_bbox} -> Components: {components}")

from image_to_text.image_to_text import process_menu_image


test_image_path = "./real_menu_card_images/carta1.jpg"
output_image_path = "output_with_bboxes.jpg"
output_txt_path = "menu_text_output.txt"

# Run the function
process_menu_image(test_image_path, output_image_path, output_txt_path)
import cv2
import json
import os

def display_images_with_boxes(json_path, image_dir):
    # Load bounding box annotations from the JSON file
    with open(json_path, 'r') as json_file:
        annotations = json.load(json_file)

    image_files = []
    image_ids = []
    for image in annotations['images']:
        image_files.append(image['file_name'])
        image_ids.append(image['id'])
    
    actual_annotations =  annotations['annotations']
    # bbox_dict = {}

    for i in range(0,len(image_ids)):
        bboxes = []
        for current_ann in actual_annotations:
            if current_ann['image_id'] in image_ids:
               bboxes.append(current_ann['bbox']) 
        # bbox_dict.update({f'{image_ids[i]}': bboxes})

        
    # # Iterate through images and display with bounding boxes
    # for image_filename, bounding_boxes in annotations.items():
        image_filename = image_files[i]
        # Construct the path to the image
        image_path = os.path.join(image_dir, image_filename)

        # Read the image
        image = cv2.imread(image_path)

        # Draw bounding boxes on the image
        for box in bboxes:
            x, y, w, h = box
            x = int(x)
            y = int(y)
            w = int(w)
            h = int(h)
            cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.rectangle(image, (x, y), (w, h), (255, 0, 0), 2)

        # Display the image with bounding boxes
        cv2.imshow('Image with Bounding Boxes', image)
        
        # Wait for a key press and close the window
        cv2.waitKey(0)
        cv2.destroyAllWindows()

if __name__ == "__main__":
    # Specify the path to the JSON file and the image directory
    json_file_path = '/home/adam.hawkins.net/workspace/vt/GOOD-Capstone/good-main/work_dirs/phase1_normal/bounding_box_output.pkl.json'
    image_directory = '/home/adam.hawkins.net/workspace/vt/GOOD-Capstone/good-main/dataset/coco/train_normal576_omni'

    # Call the function to display images with bounding boxes
    display_images_with_boxes(json_file_path, image_directory)

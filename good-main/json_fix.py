import os
import json
from tqdm import tqdm

def filter_annotation_file_data(json_path, image_dir):
    # Load the JSON file
    print("Getting annotation_file_data...")
    with open(json_path, 'r') as json_file:
        annotation_file_data = json.load(json_file)

    print("Getting image list...")
    # Get a list of image files in the specified directory
    image_files = [file.replace("_normal.png",".jpg") for file in os.listdir(image_dir) if file.lower().endswith(('.png', '.jpg', '.jpeg'))]

    print(f"Number of images found {len(image_files)}")

    print("Filtering annotation_file_data...")

    images = annotation_file_data['images']

    filtered_images = []
    filtered_ids = []
    for image in tqdm(images):
        file_name = image['file_name']
        id = image['id']
        if file_name in image_files:
            filtered_images.append(image)
            filtered_ids.append(id)
    

    for image in tqdm(filtered_images):
        image['file_name'] = image['file_name'].replace(".jpg", "_normal.png")

    annotation_file_data['images'] = filtered_images

    filtered_annotations = []

    annotations = annotation_file_data['annotations']
    for annotation in tqdm(annotations):
        if annotation['image_id'] in filtered_ids:
            filtered_annotations.append(annotation)

    annotation_file_data['annotations'] = filtered_annotations
 

    # Filter annotation_file_data to keep only information for images present in image_dir
    # filtered_annotation_file_data = {key: value for key, value in annotation_file_data.items() if key in image_files}

    print("Writing .json file...")

    # Save the filtered annotation_file_data back to the JSON file
    with open(json_path.replace(".json","_3.json"), 'w') as json_file:
        json.dump(annotation_file_data, json_file, indent=2)

if __name__ == "__main__":
    # Specify the path to the JSON file and the image directory
    json_file_path = '/good/dataset/coco/annotations/instances_train2017.json'
    image_directory = '/good/dataset/coco/train_normal576_omni'

    # Call the function to filter annotation_file_data
    filter_annotation_file_data(json_file_path, image_directory)

    print("Annotation filtering completed.")

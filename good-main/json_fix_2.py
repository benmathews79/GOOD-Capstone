import os
import json
from tqdm import tqdm

def filter_annotation_file_data(json_path, image_dir):
    # Load the JSON file
    print("Getting annotation_file_data...")
    with open(json_path, 'r') as json_file:
        annotation_file_data = json.load(json_file)





    images = annotation_file_data['images']

    filtered_images = []
    filtered_ids = []
    filtered_images.append(images[0])
    for image in filtered_images:
        id = image['id']
        filtered_ids.append(id)

    annotation_file_data['images'] = filtered_images

   

    annotations = annotation_file_data['annotations']
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
    with open(json_path.replace(".json","_2.json"), 'w') as json_file:
        json.dump(annotation_file_data, json_file, indent=2)

if __name__ == "__main__":
    # Specify the path to the JSON file and the image directory
    json_file_path = '/good/dataset/coco/annotations/instances_val2017.json'
    image_directory = '/good/dataset/coco/train_normal576_omni'

    # Call the function to filter annotation_file_data
    filter_annotation_file_data(json_file_path, image_directory)

    print("Annotation filtering completed.")

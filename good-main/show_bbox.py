import cv2
import json
import os
import numpy as np
from tqdm import tqdm
# import imagesize

modality = 'saliency'

# scale_factor = 512/384
scale_factor = 1

def nms_python(bboxes,psocres,threshold=0.5):
    '''
    NMS: first sort the bboxes by scores , 
        keep the bbox with highest score as reference,
        iterate through all other bboxes, 
        calculate Intersection Over Union (IOU) between reference bbox and other bbox
        if iou is greater than threshold,then discard the bbox and continue.
        
    Input:
        bboxes(numpy array of tuples) : Bounding Box Proposals in the format (x_min,y_min,x_max,y_max).
        pscores(numpy array of floats) : confidance scores for each bbox in bboxes.
        threshold(float): Overlapping threshold above which proposals will be discarded.
        
    Output:
        filtered_bboxes(numpy array) :selected bboxes for which IOU is less than threshold. 
    '''
    #Unstacking Bounding Box Coordinates
    bboxes = bboxes.astype('float')
    x_min = bboxes[:,0]
    y_min = bboxes[:,1]
    x_max = x_min + bboxes[:,2]
    y_max = y_min + bboxes[:,3]
    
    #Sorting the pscores in descending order and keeping respective indices.
    sorted_idx = psocres.argsort()[::-1]
    #Calculating areas of all bboxes.Adding 1 to the side values to avoid zero area bboxes.
    bbox_areas = (x_max-x_min+1)*(y_max-y_min+1)
    
    #list to keep filtered bboxes.
    filtered = []
    while len(sorted_idx) > 0:
        #Keeping highest pscore bbox as reference.
        rbbox_i = sorted_idx[0]
        #Appending the reference bbox index to filtered list.
        filtered.append(rbbox_i)
        
        #Calculating (xmin,ymin,xmax,ymax) coordinates of all bboxes w.r.t to reference bbox
        overlap_xmins = np.maximum(x_min[rbbox_i],x_min[sorted_idx[1:]])
        overlap_ymins = np.maximum(y_min[rbbox_i],y_min[sorted_idx[1:]])
        overlap_xmaxs = np.minimum(x_max[rbbox_i],x_max[sorted_idx[1:]])
        overlap_ymaxs = np.minimum(y_max[rbbox_i],y_max[sorted_idx[1:]])
        
        #Calculating overlap bbox widths,heights and there by areas.
        overlap_widths = np.maximum(0,(overlap_xmaxs-overlap_xmins+1))
        overlap_heights = np.maximum(0,(overlap_ymaxs-overlap_ymins+1))
        overlap_areas = overlap_widths*overlap_heights
        
        #Calculating IOUs for all bboxes except reference bbox
        ious = overlap_areas/(bbox_areas[rbbox_i]+bbox_areas[sorted_idx[1:]]-overlap_areas)
        
        #select indices for which IOU is greather than threshold
        delete_idx = np.where(ious > threshold)[0]+1
        # print(f"Removing {len(delete_idx)} boxes")
        delete_idx = np.concatenate(([0],delete_idx))
        
        #delete the above indices
        sorted_idx = np.delete(sorted_idx,delete_idx)
        
    # print(f'\nProcessed bounding boxes: {len(bboxes)} \nRemaining bounding boxes: {len(filtered)}\n')
    #Return filtered bboxes
    return bboxes[filtered].astype('int')



def display_images_with_boxes(json_path, image_dir):
    output_dir = image_directory + "_bbox_output"
    os.makedirs(output_dir, exist_ok=True)
    
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

    # br_corner = True

    for i in tqdm(range(0,len(image_ids))):
        bboxes = []
        scores = []
        for current_ann in actual_annotations:
            if current_ann['image_id'] == image_ids[i]:
               bboxes.append(current_ann['bbox']) 
               scores.append(current_ann['score']) 
        # bbox_dict.update({f'{image_ids[i]}': bboxes})

        

        
    # # Iterate through images and display with bounding boxes
    # for image_filename, bounding_boxes in annotations.items():
        # image_filename = image_files[i].replace(f'_{modality}','').replace('.png','.jpg')
        image_filename = image_files[i]
        # print(f'Image file name: {image_filename}')
        # exit()
        # Construct the path to the image
        # if "000000000036" in image_filename:
        image_path = os.path.join(image_dir, image_filename)

        # Read the image
        image = cv2.imread(image_path)
        
        if bboxes != []:
            nms_bboxes = nms_python(np.array(bboxes),np.array(scores))
            # Draw bounding boxes on the image
            # box = bboxes[np.argmax(scores)]
            # x, y, w, h = box
            # x = int(x)
            # y = int(y)
            # w = int(w)
            # h = int(h)

            # cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)

            for box in nms_bboxes:
            # box = nms_bboxes[0]
                x, y, w, h = box
                x = int(scale_factor*x)
                y = int(scale_factor*y)
                w = int(scale_factor*w)
                h = int(scale_factor*h)
                cv2.rectangle(image, (x, y), (x + w, y + h), (0, 0, 255), 2)

            x, y, w, h = nms_bboxes[0]
            x = int(scale_factor*x)
            y = int(scale_factor*y)
            w = int(scale_factor*w)
            h = int(scale_factor*h)
            cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)


        output_path = os.path.join(output_dir, image_filename.replace(".","_bbox."))

        cv2.imwrite(output_path, image)

        # Display the image with bounding boxes
        # cv2.imshow('Image with Bounding Boxes', image)
        
        # # Wait for a key press and close the window
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()
    # print(f'Bottom Right Corner Format = {br_corner}')

if __name__ == "__main__":
    # Specify the path to the JSON file and the image directory
    # json_file_path = '/home/adam.hawkins.net/workspace/vt/GOOD-Capstone/good-main/work_dirs/phase1_normal/bounding_box_output.pkl.json'
    # image_directory = '/home/adam.hawkins.net/workspace/vt/GOOD-Capstone/good-main/dataset/coco/train_normal576_omni'

    json_file_path = f'/good/work_dirs/phase1_vt_{modality}/bounding_box_output.pkl.json'
    # image_directory = '/good/dataset/coco/train_normal576_omni'
    image_directory = f'/good/dataset/coco/train_{modality}576_omni'


    # Call the function to display images with bounding boxes
    display_images_with_boxes(json_file_path, image_directory)

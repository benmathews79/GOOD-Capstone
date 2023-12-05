# Importing all the required modules
import cv2
import matplotlib.pyplot as plt
import numpy as np

saliency = cv2.saliency.StaticSaliencySpectralResidual_create()
f = open("train2000.csv")
textfile = f.read()

textfileList = textfile.split('\n')[:-1]

for name in textfileList:
    image = cv2.imread(name)
    
    # Compute saliency on the image
    (success, saliencyMap) = saliency.computeSaliency(image)
    saliencyMap = np.array(saliencyMap)

    #print(type(saliencyMap))
    saliencyMap = (saliencyMap * 255).astype("uint8")
    
    # Display saliency highlights image
    fig,ax = plt.subplots(1)
    fig.subplots_adjust(left=0,right=1,bottom=0,top=1)
    ax.axis('off')
    ax.imshow(saliencyMap)
    
    # name and save saliency image
    newName = name.replace('.jpg', '_saliency.jpg')
    fig.savefig(newName, transparent = True, bbox_inches= 'tight', pad_inches = 0.0)
print('Task complete.')

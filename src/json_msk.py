#mask from json file- for all tast data json file
import os
import glob
from resizeimage import resizeimage
import cv2
import skimage.transform
import numpy as np
import matplotlib.pyplot as plt
from numpy import nditer
import json
import re
import time


#creating masks from json files for trainig dataset
def json_to_mask(img_path, json_path, output):

    img_folders = sorted(os.listdir(img_path))

    #Reading json files    
    with open(json_path) as jf:    
        obj = json.load(jf)
        print(np.array(obj).shape)
        
        for jf in obj:
            print(os.path.join(img_path, "neurofinder."+jf.get("dataset")+'.png'))
            img_msk = cv2.imread(os.path.join(img_path, "neurofinder."+jf.get("dataset")+'.png'))
            plt.imshow(img_msk)
            mask = np.zeros_like(img_msk)
            contours = []
            
            for coors in jf.get("regions"):
                coor = coors.get("coordinates")
                contour = np.array([[np.array(c)] for c in coor], dtype="int32")
                contours.append(contour)
            
            #finding the drawing contours on mask    
            cv2.drawContours(mask, contours, -1, (255, 255, 255), thickness=-1)
            cv2.imwrite(img_path+"neurofinder."+jf.get("dataset")+'_msk.png', mask)

def mask_to_json(masks_path, output):

    msk_files = sorted(os.listdir(masks_path))
    json_dic = {}
    dataset_list = []
    for file in msk_files:

        head , tail = os.path.split(file)
        dataset_name = tail
        remove = ["neurofinder.","_msk.png", ".png"]
        dataset_name = re.sub(r'|'.join(map(re.escape, remove)), '', dataset_name)
        # json_dic["dataset"] = dataset_name
        print(dataset_name)
        
        mask = cv2.imread(os.path.join(masks_path, file),0)
        cnt,h=cv2.findContours(mask,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_NONE) 
        
        regions = []
        for c in cnt:
            c = np.reshape(c, (c.shape[0], c.shape[2])).tolist()
            regions.append({'coordinates': c})

        
        # json_dic["dataset", "regions"] = dataset_name, regions
        json_dic = {'dataset': dataset_name, 'regions': regions}

        dataset_list.append(json_dic)

    print(len( dataset_list ) , len(msk_files))
    print(dataset_list)
    time_stamp = time.strftime("%m-%d-%H-%M-%S", time.gmtime())
    with open(output + time_stamp + ".json", 'w') as fp:
        json.dump(dataset_list, fp)

def main():
    
    img_path = "/home/afarahani/Projects/project3/dataset/main_data/test_mod_single_filter/"
    out_path = "/home/afarahani/Projects/project3/dataset/test_comb/"
    json_path = "/home/afarahani/Projects/project3/dataset/jsonfiles/backup/outjson/03-29-03-38.json"
    #path1 = "/home/afarahani/Projects/project3/dataset/jsonfiles/testdata"
    out_json = "/home/afarahani/Projects/project3/dataset/json_mask/json/"
    tiramisu_masks = "/home/afarahani/Projects/team-rhodes-P3/src/tiramisu/.results/"
    tiramisu_jsons = "/home/afarahani/Projects/team-rhodes-P3/src/tiramisu/jsonfiles/"
    # json_to_mask(img_path, json_path, img_path)

    # mask_to_json(img_path, out_json)


    #tiramisu model 
    mask_to_json(tiramisu_masks, tiramisu_jsons)

if __name__ == "__main__":

    main()

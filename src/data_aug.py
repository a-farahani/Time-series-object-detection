
import numpy as np
import image_slicer
import cv2
import random
import os
import json
import random

def img_combine(input_path, output_path, seq):

    for common_path in input_path:
        
        head1, tail1 = os.path.split(common_path)
        #print(tail1)
        img_path = sorted(glob.glob(common_path+'/images/*'))

        sample1 = cv2.imread(img_path[0],0)
        img_combined = np.zeros_like(sample1)
        #img_combined = img_combined    
        
        # if tail1 == "neurofinder.03.00.test":
        print(output_path+tail1)


        average = cv2.imread(img_path[0]).astype(np.float)
        average_t = average
        for i,img_file in enumerate(img_path[1:]):
            head2, tail2 = os.path.split(img_file)  
            if tail1 == "neurofinder.03.00.test":
                img = cv2.imread(img_file)
                #print(os.path.join(output_path,tail1,tail2))
                
                average_t += img
                
                # img1 = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                # img1 = cv2.medianBlur(img1,13)
                # img_combined = np.maximum(img_combined,img1)
                
                if 'test' in tail1:
                    if not os.path.exists(output_path+tail1):
                        os.mkdir(output_path+tail1) 
                    average += img
            average_t /= len(img_path)
            #average_t -= average_noise
            output_t = cv2.normalize(average_t, None, 0, 255, cv2.NORM_MINMAX)
            try:
                if not os.path.exists(output_path):
                    os.makedirs(output_path)

            except OSError:
                print ('Error: Creating directory of data')
            cv2.imwrite(output_path+'/'+tail1+'.png', output_t)





#creating masks from json files for trainig dataset
def mask_gen(img_path, json_path, dataset_path):

    folders = sorted(os.listdir(json_path))
    folders = [f for f in folders if "test" not in f]

    data_path = img_path+'data/'
    masks_path = img_path+'masks/'

    for folder in folders:
        print(folder)
        coordinates = []
        print(data_path+folder+'.png')
        
        img_org = cv2.imread(data_path+folder+'.png', 0) 
        mask = np.zeros_like(img_org)
        mask = np.transpose(mask)
        mask = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)

        with open(json_path + folder + '/regions/regions.json') as fp:
            obj = json.load(fp)
            max_cnt = []
            for i,js in enumerate(obj):

                coor = js['coordinates']
                contour = np.asarray([[np.asarray(c)] for c in coor], dtype="int32")
                contour.astype("int32")
                coordinates.append(contour)
            
                cv2.drawContours(mask, contour, -1, (0,255,0), thickness=-1)


            mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
            mask = np.transpose(mask)            
            #ret, mask = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)
            mask = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
            mask = mask/255
            print(type(mask))
                
        if not os.path.exists(dataset_path+'/train/masks/'+folder+'.png'):
            cv2.imwrite(dataset_path+'/train/masks/'+folder+'.png', mask)
        if not os.path.exists(dataset_path+'/train/data/'+folder+'.png'):
            cv2.imwrite(dataset_path+'/train/data/'+folder+'.png', img_org)
                

def img_slice(dataset_path):

    img_path = dataset_path+'/train/data/'
    msk_path = dataset_path+'/train/masks/'

    img_files = sorted(os.listdir(img_path))
    img_files = [f for f in img_files if os.path.isfile(os.path.join(img_path,f)) and "test" not in f]

    mask_files = sorted(os.listdir(msk_path))
    mask_files = [f for f in mask_files if os.path.isfile(os.path.join(msk_path,f))]

    for img_f in img_files:
        image_slicer.slice(os.path.join(img_path,img_f), 4)

    for mask_f in mask_files:
        image_slicer.slice(os.path.join(msk_path,mask_f), 4)

def img_rotate(dataset_path):

    img_path = dataset_path+'/train/data/'
    msk_path = dataset_path+'/train/masks/'

    img_files = sorted(os.listdir(img_path))
    img_files = [f for f in img_files if os.path.isfile(os.path.join(img_path,f)) and "test" not in f]

    mask_files = sorted(os.listdir(msk_path))
    mask_files = [f for f in mask_files if os.path.isfile(os.path.join(msk_path,f))]

    for img_f in img_files:       
        img = cv2.imread(os.path.join(img_path,img_f))
        row, col,_ = img.shape
        
        for i in range (1,4):
            rot_mat = cv2.getRotationMatrix2D((col/2,row/2),90,1)
            img = cv2.warpAffine(img,rot_mat,(col,row))

            cv2.imwrite(img_path+img_f[:-4]+'_rot_'+str(i*90)+'.png',img)

    for mask_f in mask_files:
        mask = cv2.imread(msk_path+mask_f)
        row_m, col_m,_ = mask.shape
        
        for i in range (1,4):

            rotm_mat = cv2.getRotationMatrix2D((col_m/2,row_m/2),90,1)
            mask = cv2.warpAffine(mask,rotm_mat,(col_m,row_m))
            cv2.imwrite(msk_path+mask_f[:-4]+'_rot_'+str(i*90)+'.png',mask)

def img_trans(dataset_path):

    img_path = dataset_path+'/train/data/'
    msk_path = dataset_path+'/train/masks/'

    img_files = sorted(os.listdir(img_path))
    img_files = [f for f in img_files if os.path.isfile(os.path.join(img_path,f)) and "test" not in f]

    mask_files = sorted(os.listdir(msk_path))
    mask_files = [f for f in mask_files if os.path.isfile(os.path.join(msk_path,f))]

    for img_f in img_files:
        img = cv2.imread(os.path.join(img_path,img_f),0)    
        img= np.transpose(img)
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        cv2.imwrite(img_path+img_f[:-4]+'_tr.png',img)

    for mask_f in mask_files:
        mask = cv2.imread(msk_path+mask_f, 0)
        mask= np.transpose(mask)
        mask = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
        cv2.imwrite(msk_path+mask_f[:-4]+'_tr.png',mask)

def train_val_split(dataset_path):
    
    train_data = dataset_path+'/train/data/'
    train_msk = dataset_path+'/train/masks/'

    val_data = dataset_path+'/validate/data/'
    val_msk = dataset_path+'/validate/masks/'

    img_files = sorted(os.listdir(train_data))
    mask_files = sorted(os.listdir(train_msk))

    val_files = sorted(os.listdir(val_data))
    val_m_files = sorted(os.listdir(val_msk))

    data_size = len(img_files)
    portion = 0.2       
    val_size = int(0.2 * data_size)

    files = sorted(random.sample(img_files, val_size))

    for file in files:

        os.rename(os.path.join(train_data, file), os.path.join(val_data, file))
        os.rename(os.path.join(train_msk, file), os.path.join(val_msk, file))

def main():

    source_path = '/home/afarahani/Projects/project3/dataset/tiramisu_org_files/'
    json_path = "/media/hdd/dataset/project3/"
    dataset_path = "/home/afarahani/Projects/project3/dataset/data"

    if not os.path.exists(dataset_path+'/train/data'):
        os.makedirs(dataset_path+'/train/data')
    if not os.path.exists(dataset_path+'/validate/data'):
        os.makedirs(dataset_path+'/validate/data')
    if not os.path.exists(dataset_path+'/train/masks'):
        os.makedirs(dataset_path+'/train/masks')
    if not os.path.exists(dataset_path+'/validate/masks'):
        os.makedirs(dataset_path+'/validate/masks')


    img_combine(input_path, output_path)
    mask_gen(source_path, json_path, dataset_path)

    img_slice(dataset_path)

    img_rotate(dataset_path)

    img_trans(dataset_path)

    train_val_split(dataset_path)








if __name__ == "__main__":
    main()

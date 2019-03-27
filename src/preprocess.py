import cv2
import os
import glob
import numpy as np
from vidstab import VidStab

def img_combine(input_path, output_path):

    

    for common_path in input_path:
        
        head1, tail1 = os.path.split(common_path)
        print(tail1)
        img_path = sorted(glob.glob(common_path+'/images/*'))

        sample1 = cv2.imread(img_path[0],0)
        img_combined = np.zeros_like(sample1)    
        
        for i,img_file in enumerate(img_path):

            head2, tail2 = os.path.split(img_file)  
            img = cv2.imread(img_file)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            img_combined = np.maximum(img_combined,img)
        print(output_path)
        try:
            if not os.path.exists(output_path):
                os.makedirs(output_path)
        except OSError:
            print ('Error: Creating directory of data')
        cv2.imwrite(output_path+'/'+tail1+'.png', img_combined)




def processing_frame(input_path, output_path):

    for common_path in input_path:

        head1, tail1 = os.path.split(common_path)
        img_path = sorted(glob.glob(common_path+'/images/*'))
        for img in img_path:
            print(img)
            head2, tail2 = os.path.split(img)
            image = cv2.imread(img)
            image = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)

            #image[image < 2] = 0
            
            #clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))#auto illumination correction
            #image = clahe.apply(image)        
            #image = cv2.bilateralFilter(image,7,30,30) #smoothing the image
            
            image = cv2.medianBlur(image,5)
            out_path = os.path.join(output_path,tail1)            
            try:
                if not os.path.exists(out_path):
                    os.makedirs(out_path)
            except OSError:
                print ('Error: Creating directory of data')
            
            cv2.imwrite(out_path+'/'+tail2, image)

def main():

    input_path = sorted(glob.glob('/media/hdd/dataset/project3/*'))
    output_path = '/home/afarahani/Projects/project3/dataset/comb_img/'
    
    #processing_frame(input_path, output_path)
    img_combine(input_path, output_path)






if __name__ == "__main__":
    main()
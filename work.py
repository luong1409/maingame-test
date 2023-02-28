import pandas as pd
import os
import cv2
from numpy import save, load

from circle_hero import detect_circle_hero
from square_hero import detect_square_hero
import re

from argparse import ArgumentParser

from utils import gen_descriptor
import os

from tqdm import tqdm


parser = ArgumentParser()

parser.add_argument('-v', '--val', action='store_true', default=False)
parser.add_argument('-i', '--image', action='store', default=None)
parser.add_argument('-f', '--folder', action='store', default=None)

if not os.path.exists('circles'):
    print("generate descriptor..........")
    os.makedirs('circles')
    gen_descriptor(folder='circles')

if not os.path.exists('squares'):
    print("generate descriptor..........")
    os.makedirs('squares')
    gen_descriptor(folder='squares')

def get_label(img, flag='circles'):
    assert flag in ['circles', 'squares']
    
    best_sim_val = -100
    hero_name = None
    
    for feature in os.listdir(flag):
        # print(feature)
        feature1 = load(os.path.join(flag, feature))
        
        # img = cv2.imread(filename=img_path)
        
        # Convert the training image to RGB
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # Convert the training image to gray scale
        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        
        sift = cv2.SIFT_create()
        
        test_keypoints, test_descriptor = sift.detectAndCompute(gray, None)
        
        bf = cv2.BFMatcher()
        matches = bf.knnMatch(feature1, test_descriptor, k=2)
        
        # Apply ratio test
        good = []
        for m,n in matches:
            if m.distance < 0.73*n.distance:
                good.append([m])
        
        a=len(good)
        percent=(a*100)/len(test_keypoints)
        
        if percent > best_sim_val:
            best_sim_val = percent
            if 'crop' in feature:
                hero_name = re.findall('(\w+)_cropped', feature)[0]
            else:
                hero_name = os.path.splitext(feature)[0]
    return hero_name


if __name__ == "__main__":
    args = parser.parse_args()
    
    if args.val:
        df = pd.read_csv('test.txt', sep='\t', header=None)
        df = df.rename(columns={0:'path', 1:'label'})
        
        correct = 0

        for image, label in df.values:
            image_path = os.path.join('test_images', image)
            img = cv2.imread(image_path)
            
            detected_heros = detect_circle_hero(img_path=image_path)
            flag = 'circles'
            
            if detected_heros is None:
                detected_heros = detect_square_hero(img_path=image_path)
                # flag = 'squares'
                
                if len(detected_heros) == 0:
                    continue
                
                x, y, w, h = detected_heros[0]
                cropped_image = img[y:y+h, x:x+w]
            else:
                xc, yc, radius = detected_heros[0][0]
                cropped_image = img[yc-radius:yc+radius, xc-radius:xc+radius]
            
            pred = get_label(cropped_image, flag)
            # print(pred)
            
            if pred == label:
                correct += 1

        print(f"Percent of correct prediction: {(correct*100)/len(df)}")
    else:
        if args.folder is not None:
            with open('output.txt', 'w') as f:
                for file_name in tqdm(os.listdir(args.folder)):
                    file_path = os.path.join(args.folder, file_name)
                    img = cv2.imread(file_path)
                    
                    detected_heros = detect_circle_hero(img_path=file_path)
                    flag = 'circles'
                    
                    if detected_heros is None:
                        detected_heros = detect_square_hero(img_path=file_path)
                        flag = 'squares'
                        
                        if len(detected_heros) == 0 :
                            continue
                
                        x, y, w, h = detected_heros[0]
                        cropped_image = img[y:y+h, x:x+w]
                    else:
                        xc, yc, radius = detected_heros[0][0]
                        cropped_image = img[yc-radius:yc+radius, xc-radius:xc+radius]
                    
                    label = get_label(cropped_image, flag) or 'Unk'
                    
                    f.write(f"{file_path}\t{label}\n")
        else:
            img_path = args.image
            
            img = cv2.imread(img_path)
            
            label = get_label(img)
            
            with open('output.txt', 'w') as f:
                f.write(f"{img_path}\t{label}\n")
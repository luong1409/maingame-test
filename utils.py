import os
from numpy import load, save
import cv2
import imgaug.augmenters as iaa

def gen_descriptor(folder):
    assert folder in ['circles', 'squares']
    noise=iaa.AdditiveGaussianNoise(1,25)
    
    if folder == 'circles':
        
        for hero in os.listdir('heros'):
            image = cv2.imread(f'heros/{hero}')
            
            (h, w) = image.shape[:2]
            masked = cv2.circle(img=image.copy(), center=(w//2, h//2), radius=w//2, color=(0, 0, 0), thickness=-1)
            
            cropped_image = image - masked

            noise_cropped_image = noise.augment_image(cropped_image)
            
            # Convert the training image to RGB
            training_image = cv2.cvtColor(cropped_image, cv2.COLOR_BGR2RGB)
            
            noise_training_image = cv2.cvtColor(noise_cropped_image, cv2.COLOR_BGR2RGB)

            # Convert the training image to gray scale
            training_gray = cv2.cvtColor(training_image, cv2.COLOR_RGB2GRAY)
            
            noise_training_gray = cv2.cvtColor(noise_training_image, cv2.COLOR_RGB2GRAY)
            
            sift = cv2.SIFT_create()
            
            # find the keypoints and descriptors with SIFT
            train_keypoints, train_descriptor = sift.detectAndCompute(training_gray, None)
            _, noise_train_descriptor = sift.detectAndCompute(noise_training_gray, None)
            
            hero_name = os.path.splitext(hero)[0]
            
            save(file=f'features/{hero_name}.npy', arr=train_descriptor)
            save(file=f'features/{hero_name}_cropped.npy', arr=noise_train_descriptor)
    
    else:
        for hero in os.listdir('heros'):
            image = cv2.imread(f'heros/{hero}')
            
            # Convert the training image to RGB
            training_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

            # Convert the training image to gray scale
            training_gray = cv2.cvtColor(training_image, cv2.COLOR_RGB2GRAY)
            
            sift = cv2.SIFT_create()
            
            # find the keypoints and descriptors with SIFT
            train_keypoints, train_descriptor = sift.detectAndCompute(training_gray, None)
            
            hero_name = os.path.splitext(hero)[0]
            
            save(file=f'squares/{hero_name}.npy', arr=train_descriptor)
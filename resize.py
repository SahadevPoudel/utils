import os
import cv2


data_dir = '/home/poudelas/PycharmProjects/ColonCancer/training_images/original_training_images/train/UC' # Data
data_resized_dir = '/home/poudelas/PycharmProjects/ColonCancer/training_images/Augmentation/UC'

os.mkdir(data_resized_dir)

for each in os.listdir(data_dir):
        image = cv2.imread(os.path.join(data_dir, each))
        image = cv2.resize(image, (224, 224))
        cv2.imwrite(os.path.join(data_resized_dir, each), image)
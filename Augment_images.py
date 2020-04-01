import cv2
import glob
import imgaug as ia
from imgaug import augmenters as iaa
import numpy as np
import os
images_path = '/home/poudelas/Documents/Yolo-Annotation-Tool-New--master/brokenegg/augment/*.png'
output_images_path = '/home/poudelas/Documents/Yolo-Annotation-Tool-New--master/brokenegg/augment1/'
# aug_normalize = 'normalization'
# aug_multiply = 'Multiply'
aug_fliplr = 'fliplr'
aug_flipud = 'flipud'

aug_scale = 'scale'
# aug_gaussian_blue = 'gaussian_blur'
#aug_contrast='contrast'
#aug_HueSaturation = 'HueSaturation'

images_path = glob.glob(images_path)
images = []

for i in range(len(images_path)):
    img = cv2.imread(images_path[i],flags=cv2.IMREAD_COLOR)
    images.append(img)
# seq_fliplr = iaa.Sequential([
#         iaa.Fliplr(1)
# ])
# seq_flipud = iaa.Sequential([
#         iaa.Flipud(1)
# ])
seq_rotate90 = iaa.Sequential([
        iaa.Affine(scale=(1.0, 1.3))
])
# seq_rotate180 = iaa.Sequential([
#         iaa.Affine(rotate=(135,135)),
#         iaa.SomeOf(1,
#                    [
#                     iaa.ContrastNormalization((0.7, 1.3)),
#                     iaa.GaussianBlur((0, 2.0)),
#                     iaa.Affine(scale=(1.0, 1.3))
#
#                    ]
#                    )
# ])
# seq_rotate225 = iaa.Sequential([
#         iaa.Affine(rotate=(225, 225)),
#         iaa.SomeOf(1,
#                    [
#                     iaa.ContrastNormalization((0.7, 1.3)),
#                     iaa.GaussianBlur((0, 2.0)),
#                     iaa.Affine(scale=(1.0, 1.3))
#
#                    ]
#                    )
# ])
# seq_rotate315 = iaa.Sequential([
#         iaa.Affine(rotate=(315, 315)),
#         iaa.SomeOf(1,
#                    [
#                     iaa.ContrastNormalization((0.7, 1.3)),
#                     iaa.GaussianBlur((0, 2.0)),
#                     iaa.Affine(scale=(1.0, 1.3))
#
#                    ]
#                    )
# ])
# seq_aug_gaussian_blue = iaa.Sequential([
#         iaa.GaussianBlur((0, 3.0))
# ])
# seq_aug_contrast = iaa.Sequential([
#         iaa.ContrastNormalization((0.7, 1.3))
# ])


#
# for x in range(1):
#     seq_det = seq_aug_gaussian_blue.to_deterministic()
#     image_aug = seq_det.augment_images(images)
#     for i in range(len(image_aug)):
#         cv2.imwrite(output_images_path + str(i + 1) + '_' + aug_gaussian_blue + '_batch_' + str(x) + '.png',
#                     image_aug[i])
#         print ("Image saved. " + "batch" + str(x))
# for x in range(1):
#     seq_det = seq_aug_contrast.to_deterministic()
#     image_aug = seq_det.augment_images(images)
#     for i in range(len(image_aug)):
#         cv2.imwrite(output_images_path + str(i + 1) + '_' + aug_contrast + '_batch_' + str(x) + '.png',
#                     image_aug[i])
#         print ("Image saved. " + "batch" + str(x))
# for x in range(1):
#     seq_det = seq_fliplr.to_deterministic()
#     image_aug = seq_det.augment_images(images)
#     for i in range(len(image_aug)):
#         cv2.imwrite(output_images_path + str(i + 1) + '_' + aug_fliplr + '_batch_' + str(x) + '.png',
#                     image_aug[i])
#         print ("Image saved. " + "batch" + str(x))
# for x in range(1):
#     seq_det = seq_flipud.to_deterministic()
#     image_aug = seq_det.augment_images(images)
#     for i in range(len(image_aug)):
#         cv2.imwrite(output_images_path + str(i + 1) + '_' + aug_flipud + '_batch_' + str(x) + '.png',
#                     image_aug[i])
#         print ("Image saved. " + "batch" + str(x))

for x in range(1):
    seq_det = seq_rotate90.to_deterministic()
    image_aug = seq_det.augment_images(images)
    for i in range(len(image_aug)):
        cv2.imwrite(output_images_path + str(i + 1) + '_' + aug_scale + '_batch_' + str(x) + '.png',
                    image_aug[i])
        print ("Image saved. " + "batch" + str(x))
# for x in range(1):
#     seq_det = seq_rotate180.to_deterministic()
#     image_aug = seq_det.augment_images(images)
#     for i in range(len(image_aug)):
#         cv2.imwrite(output_images_path + str(i + 1) + '_' + aug_rotate135 + '_batch_' + str(x) + '.png',
#                     image_aug[i])
#         print ("Image saved. " + "batch" + str(x))
# for x in range(1):
#     seq_det = seq_rotate225.to_deterministic()
#     image_aug = seq_det.augment_images(images)
#     for i in range(len(image_aug)):
#         cv2.imwrite(output_images_path + str(i + 1) + '_' + aug_scale225 + '_batch_' + str(x) + '.png',
#                     image_aug[i])
#         print ("Image saved. " + "batch" + str(x))
# for x in range(1):
#     seq_det = seq_rotate315.to_deterministic()
#     image_aug = seq_det.augment_images(images)
#     for i in range(len(image_aug)):
#         cv2.imwrite(output_images_path + str(i + 1) + '_' + aug_scale315 + '_batch_' + str(x) + '.png',
#                     image_aug[i])
#         print ("Image saved. " + "batch" + str(x))
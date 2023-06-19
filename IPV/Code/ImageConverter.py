import os

import cv2

path = 'DATA/RESAMPLED_SAGITTAL_MID'

images = os.listdir(path)

for img in images:
    print(f'Converting: {img}.')

    cv_image = cv2.imread(f'{path}/{img}')

    cv2.imwrite(f'{path}/{img}', cv_image)

print(f'Finished converting {len(images)} images.')

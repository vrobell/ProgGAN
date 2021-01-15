import os
import cv2
import numpy as np
from PIL import Image
from mtcnn import MTCNN
import matplotlib.pyplot as plt


RAW_DATA_DIR = 'data/celeba/raw/img_align_celeba'
PROC_DATA_DIR = 'data/celeba/processed'
RES_LIST = [(4, 4), (8, 8), (16, 16), (32, 32), (64, 64)]
FACE_DETECTOR = MTCNN()


def process_raw_dara(verbose: int = 1000):
    num_processed = 0
    for file in os.listdir(RAW_DATA_DIR):
        img = Image.open(f'{RAW_DATA_DIR}/{file}')
        img = np.array(img)

        face_info = FACE_DETECTOR.detect_faces(img)
        if face_info:
            x1, y1, width, height = face_info[0]['box']
            img_trimmed = img[y1:y1 + height, x1:x1 + width, ...]

            if len(img_trimmed.shape) < 3:
                img_trimmed = np.repeat(np.expand_dims(img_trimmed, axis=-1), repeats=3, axis=-1)

            if img_trimmed.shape[-1] == 1:
                img_trimmed = np.repeat(img_trimmed, repeats=3, axis=-1)

            img_trimmed = ((img_trimmed / 255) * 2) - 1

            for res in RES_LIST:
                img_trimmed_res = cv2.resize(img_trimmed, dsize=res)
                img_trimmed_res = np.array(img_trimmed_res)
                np.save(f'{PROC_DATA_DIR}/{res[0]}x{res[1]}/{num_processed}.npy', img_trimmed_res)

            num_processed += 1

            if num_processed % verbose == 0:
                print('processed: ', num_processed)


def display_processed_data(res: tuple):
    data_dir = f'{PROC_DATA_DIR}/{res[0]}x{res[1]}'
    plt.figure()
    for file in os.listdir(data_dir):
        plt.clf()
        img = np.load(f'{data_dir}/{file}')
        plt.imshow(img)
        print(np.max(img))
        plt.pause(0.5)


if __name__ == '__main__':
    process_raw_dara()
    # display_processed_data(res=(32, 32))
    pass
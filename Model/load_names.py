import keras, os, shutil
import tensorflow as tf
from os.path import join
import numpy
import numpy as np
import glob
import json
import cv2
import string
import random


config = tf.compat.v1.ConfigProto()
config.gpu_options.allow_growth = True
sess = tf.compat.v1.Session(config=config)

dataset_dir = "Eyetracking"
dirs = glob.glob(os.path.join(dataset_dir, "*"))
train_dir = os.path.join(dataset_dir, "train")
validation_dir = os.path.join(dataset_dir, "validation")
test_dir = os.path.join(dataset_dir, "test")


# this function returtns a list of the paths for each the train and validtion
# sets, the amount is limited as we cannot have a too large of a shape
def load_data_names(path):
    seqs = sorted(glob.glob(join(path, "*")))
    seq_list = []

    for seq in seqs:
        file = open(join(seq,"img.txt"), "r")
        content = file.read().splitlines()
        for line in content:
            seq_list.append(line)
    print(len(seq_list))
    return seq_list

# load data is used to find convert the images into a uniformed numpy array
# for each input needed
def load_data(names, path, batch_size):

    left_eye_batch = np.zeros(shape=(batch_size, 240, 240, 3), dtype=np.float32)
    right_eye_batch = np.zeros(shape=(batch_size, 240, 240, 3), dtype=np.float32)
    face_batch = np.zeros(shape=(batch_size, 240, 240, 3), dtype=np.float32)
    face_grid_batch = np.zeros(shape=(batch_size, 1, 25, 25), dtype=np.float32)
    y_batch = np.zeros((batch_size, 2), dtype=np.float32)
    
    b = 0
    f = 0
    while b < batch_size:

        i = random.randrange(0,len(names)-1)
        
        
        img = names[i]

        # frame name
        frame = img[-9:]

        dir = img[:-9]
        # index of the frame inside the sequence

        idx = int(frame[:-4])
        
        face_file = open(join(path, dir, "appleFace.json"))
        left_file = open(join(path, dir, "appleLeftEye.json"))
        right_file = open(join(path, dir, "appleRightEye.json"))
        frames_file = open(join(path, dir, "frames.json"))
        info_file = open(join(path, dir, "info.json"))
        grid_file = open(join(path, dir, "faceGrid.json"))
        dot_file = open(join(path, dir, "dotInfo.json"))

        
        face_json = json.load(face_file)
        left_json = json.load(left_file)
        right_json = json.load(right_file)
        frames_json = json.load(frames_file)
        info_json = json.load(info_file)
        grid_json = json.load(grid_file)
        dot_json = json.load(dot_file)

        if face_json["IsValid"][idx] == 0 or left_json["IsValid"][idx] == 0 or right_json["IsValid"][idx] == 0:
            continue
        frames = cv2.imread(os.path.join(path, dir, "frames", frame))
#get face coordinates
        temp_x_face = int(face_json["X"][idx])
        temp_y_face = int(face_json["X"][idx])
        w = int(face_json["W"][idx])
        h = int(face_json["H"][idx])
        box_x_face = temp_x_face + w
        box_y_face = temp_y_face + h
        face = frames[temp_y_face:box_y_face,temp_x_face:box_y_face]

#get left eye coordinates
        temp_x = temp_x_face + int(left_json["X"][idx])
        temp_y = temp_y_face + int(left_json["Y"][idx])
        w = int(left_json["W"][idx])
        h = int(left_json["H"][idx])
        box_x = temp_x + w
        box_y = temp_y + h
        left_eye = frames[temp_y:box_y, temp_x:box_x]

#get right eye coordinates
        temp_x = temp_x_face + int(right_json["X"][idx])
        temp_y = temp_y_face + int(right_json["Y"][idx])
        w = int(right_json["W"][idx])
        h = int(right_json["H"][idx])
        box_x = temp_x + w
        box_y = temp_y + h
        right_eye = frames[temp_y:box_y, temp_x:box_y]

#get face grid coordinates
        face_grid = np.zeros(shape=(1, 25, 25), dtype=np.float32)
        temp_x = int(grid_json["X"][idx])
        temp_y = int(grid_json["Y"][idx])
        w = int(grid_json["W"][idx])
        h = int(grid_json["H"][idx])
        box_x = temp_x + w
        box_y = temp_y + h
        face_grid[0, temp_y:box_y, temp_x:box_x] = 1
        
        
        y_x = dot_json["XCam"][idx]
        y_y = dot_json["YCam"][idx]
        
        try:
            # resize each image and if it can't due to a shape issue skip this frame
            face = cv2.resize(face, (240, 240))
            left_eye = cv2.resize(left_eye,(240, 240))
            right_eye = cv2.resize(right_eye, (240, 240))
        except Exception as e:
            continue

        # normalise each peice of data
        face = face.astype('float32') / 255.
        face = face - np.mean(face)

        left_eye = left_eye.astype('float32') / 255.
        left_eye = left_eye - np.mean(left_eye)

        right_eye = right_eye.astype('float32') / 255.
        right_eye = right_eye - np.mean(right_eye)

        # add to the batch list
        right_eye_batch[b] = right_eye
        left_eye_batch[b]=left_eye
        face_batch[b] = face
        face_grid_batch[b] = face_grid
        y_batch[b][0] = y_x
        y_batch[b][1] = y_y
        b+=1
        
        
    return [right_eye_batch, left_eye_batch, face_batch, face_grid_batch], y_batch

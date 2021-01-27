import os, shutil
from os.path import join
import glob
import json
import cv2
import string


dataset_dir = "Eyetracking"
train_dir = os.path.join(dataset_dir, "train")
validation_dir = os.path.join(dataset_dir, "validation")
test_dir = os.path.join(dataset_dir, "test")

test_dirs = glob.glob(os.path.join(test_dir, "*"))
validation_dirs = glob.glob(os.path.join(validation_dir, "*"))
train_dirs = glob.glob(os.path.join(train_dir, "*"))

print(test_dirs)
for dir in train_dirs:
    print("analyzing {}".format(dir))

    # opening json files within each set
    face_file = open(join(dir, "appleFace.json"))
    left_file = open(join(dir, "appleLeftEye.json"))
    right_file = open(join(dir, "appleRightEye.json"))
    frames_file = open(join(dir, "frames.json"))
    info_file = open(join(dir, "info.json"))

    # read json content
    face_json = json.load(face_file)
    left_json = json.load(left_file)
    right_json = json.load(right_file)
    frames_json = json.load(frames_file)
    info_json = json.load(info_file)
    

    os.mkdir(join(dataset_dir, "tr", os.path.basename(dir)+"/"))
    output = open(join(dataset_dir, "tr", os.path.basename(dir), "img.txt"), "w+")

            # conducting a sanity check and removing any non valid images
    # and adding the path to a text file
    for i in range(0, int(info_json["TotalFrames"])):
        if left_json["IsValid"][i] and right_json["IsValid"][i] and face_json["IsValid"][i]:
            output.write(os.path.basename(dir) + "/" + frames_json[i])
            output.write("\n")
    print(output)
    # close the file
    output.close()

for dir in validation_dirs:
    print("analyzing {}".format(dir))

    # opening json files within each set
    face_file = open(join(dir, "appleFace.json"))
    left_file = open(join(dir, "appleLeftEye.json"))
    right_file = open(join(dir, "appleRightEye.json"))
    frames_file = open(join(dir, "frames.json"))
    info_file = open(join(dir, "info.json"))

    # read json content
    face_json = json.load(face_file)
    left_json = json.load(left_file)
    right_json = json.load(right_file)
    frames_json = json.load(frames_file)
    info_json = json.load(info_file)
    

    os.mkdir(join(dataset_dir, "val", os.path.basename(dir)+"/"))
    output = open(join(dataset_dir, "val", os.path.basename(dir), "img.txt"), "w+")

            # conducting a sanity check and removing any non valid images
    # and adding the path to a text file
    for i in range(0, int(info_json["TotalFrames"])):
        if left_json["IsValid"][i] and right_json["IsValid"][i] and face_json["IsValid"][i]:
            output.write(os.path.basename(dir) + "/" + frames_json[i])
            output.write("\n")
    print(output)
    # close the file
    output.close()

for dir in test_dirs:
    print("analyzing {}".format(dir))

    # opening json files within each set
    face_file = open(join(dir, "appleFace.json"))
    left_file = open(join(dir, "appleLeftEye.json"))
    right_file = open(join(dir, "appleRightEye.json"))
    frames_file = open(join(dir, "frames.json"))
    info_file = open(join(dir, "info.json"))
    # read json content
    face_json = json.load(face_file)
    left_json = json.load(left_file)
    right_json = json.load(right_file)
    frames_json = json.load(frames_file)
    info_json = json.load(info_file)
    


    os.mkdir(join(dataset_dir, "te", os.path.basename(dir)+"/"))
    output = open(join(dataset_dir, "te", os.path.basename(dir), "img.txt"), "w+")

            # conducting a sanity check and removing any non valid images
    # and adding the path to a text file
    for i in range(0, int(info_json["TotalFrames"])):
        if left_json["IsValid"][i] and right_json["IsValid"][i] and face_json["IsValid"][i]:
            output.write(os.path.basename(dir) + "/" + frames_json[i])
            output.write("\n")
    print(output)
    # close the file
    output.close()



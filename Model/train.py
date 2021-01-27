import h5py
import os
from keras import optimizers
from model import face_grid_model
import random
from load_names import load_data_names
from load_names import load_data
import tensorflow


def train():
    model = face_grid_model(3,240,240)
    model.compile(loss='mse',
                  optimizer='adam',
                  metrics=['acc'])
    model.summary()    
    train_dir = "Eyetracking/train"
    val_dir = "Eyetracking/validation"

    
    
    train = load_data_names("Eyetracking/tr")
    val = load_data_names("Eyetracking/val")
    t = len(train)
    v = len(val)
    history = model.fit(train_data_generator(train, train_dir), 
                    steps_per_epoch=len(train)/1000,
                    epochs=5,
                    validation_data=val_data_generator(val, val_dir),
                    validation_steps=len(val)/1000)

    model.save_weights('my_model_weights.h5')

def train_data_generator(names, path):
    while True:
        x,y = load_data(names, path, 100)
        yield x,y

def val_data_generator(names, path):
    while True:
        x,y = load_data(names, path, 100)
        yield x,y
        



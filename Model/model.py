import numpy as np
from keras.layers import Layer
from keras.layers import Input, Conv2D, Dense, Flatten, MaxPool2D, concatenate
from keras.models import Model

def eye_model(img_cols, img_rows, img_ch):
    eye_img_input = Input(shape=(img_cols, img_rows, img_ch))

    h = Conv2D(96, (11, 11), activation='relu')(eye_img_input)
    h = MaxPool2D(pool_size=(2, 2))(h)
    h = Conv2D(256, (5, 5), activation='relu')(h)
    h = MaxPool2D(pool_size=(2, 2))(h)
    h = Conv2D(384, (3, 3), activation='relu')(h)
    h = MaxPool2D(pool_size=(2, 2))(h)
    out = Conv2D(64, (1, 1), activation='relu')(h)

    model = Model(inputs=eye_img_input, outputs=out)

    return model

    
def face_model(img_cols, img_rows, img_ch):
    face_img_input = Input(shape=(img_cols, img_rows, img_ch))

    h = Conv2D(96, (11, 11), activation='relu')(face_img_input)
    h = MaxPool2D(pool_size=(2, 2))(h)
    h = Conv2D(256, (5, 5), activation='relu')(h)
    h = MaxPool2D(pool_size=(2, 2))(h)
    h = Conv2D(384, (3, 3), activation='relu')(h)
    h = MaxPool2D(pool_size=(2, 2))(h)
    out = Conv2D(64, (1, 1), activation='relu')(h)

    model = Model(inputs=face_img_input, outputs=out)

    return model

def face_grid_model(img_ch, img_cols, img_rows):
    eye_net = eye_model(img_cols, img_rows, img_ch)
    face_net = face_model(img_cols, img_rows, img_ch)
    
    left_model_input = Input(shape=(img_cols, img_rows, img_ch))
    left_eye_model = eye_net(left_model_input)
    
    right_model_input = Input(shape=(img_cols, img_rows, img_ch))
    right_eye_model = eye_net(right_model_input)
    
    face_model_input = Input(shape=(img_cols, img_rows, img_ch))
    faceModel = face_net(face_model_input)
    
    face_grid = Input(shape=(1,25,25))

    #dense layers for eyes
    e = concatenate([left_eye_model, right_eye_model])
    e = Flatten()(e)
    fc_e1 = Dense(128, activation='relu')(e)
    
    # dense layers for face
    f = Flatten()(faceModel)
    fc_f1 = Dense(128, activation='relu')(f)
    fc_f2 = Dense(64, activation='relu')(fc_f1)

    # dense layers for face grid
    fg = Flatten()(face_grid)
    fc_fg1 = Dense(256, activation='relu')(fg)
    fc_fg2 = Dense(128, activation='relu')(fc_fg1)

    # final dense layers
    h = concatenate([fc_e1, fc_f2, fc_fg2])
    fc1 = Dense(128, activation='relu')(h)
    fc2 = Dense(2, activation = 'linear')(fc1)


    final_model = Model(inputs=[right_model_input, left_model_input, face_model_input, face_grid],outputs=[fc2])

    return final_model


model = face_grid_model(3, 240, 240)
model.summary()

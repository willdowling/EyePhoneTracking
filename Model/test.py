import os
from load_names import load_data_names
from load_names import load_data
from model import face_grid_model
import numpy as np
import tensorflow



def test_generator(data):
        while True:
            x,y = load_data(data, "Eyetracking/test", 20)
            yield x,y

def test():
    dataset_dir = "Eyetracking"
    model = face_grid_model(3, 240, 240)
    model.summary()
    
    model.load_weights('my_model_weights.h5')
    
    test = load_data_names("Eyetracking/te")

    x_error = []
    y_error = []
    n=0
    
    for i in test_generator(test[:50000]):
        x = i[0]
        y=i[1]
        predictions = model.predict(x=x, batch_size=20, verbose=1)

        for i, prediction in enumerate(predictions):
            print("Predicited: {} {}".format(prediction[0], prediction[1]))
            print("Got: {} {} \n".format(y[i][0],y[i][1]))
            x_error.append(abs(prediction[0]-y[i][0]))
            y_error.append(abs(prediction[1]-y[i][1]))
        n+=1
        if n > 10000:
                break
    mae_x = np.mean(x_error)
    mae_y = np.mean(y_error)

    std_x = np.std(x_error)
    std_y = np.std(y_error)

    print("MAE: {} {} ( samples)".format(mae_x, mae_y))
    print("STD: {} {} ( samples)".format(std_x, std_y))

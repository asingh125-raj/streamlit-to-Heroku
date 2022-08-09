from fileinput import filename
import streamlit as st
from PIL import Image
import tensorflow as tf 
from tensorflow import keras
import numpy as np
from tensorflow.keras import preprocessing
from tensorflow.keras.models import load_model
from tensorflow.keras.activations import softmax
import matplotlib.pyplot as plt
#import tensorflow_hub as hub
import os 
import h5py
import pandas as pd
from cachetools import TTLCache
from tensorflow.keras.models import Sequential
from tensorflow.keras.models import  model_from_json

# filename = 'C:\Users\Arpana Singh\Videos\Learning\Final_Project\FineTuning.h5'
# f = h5py.File(filename,'r')

st.title("Tea leaf diseas detection")
st.header("Helopeltis")


def main():
    file_uploaded = st.file_uploader("Choose the file",type= ['jpg','png', 'jpeg'])
    if file_uploaded is not None:
        image = Image.open(file_uploaded)
        figure = plt.figure()
        plt.imshow(image)
        plt.axis('off')
        result = predict_class(image)
        st.write(result)
        st.pyplot(figure)
#predict_result = st.button('Predict')
def predict_class(image):
    json_file = open('model.json', 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    classifier_model = model_from_json(loaded_model_json)
    # load weights into new model
    classifier_model.load_weights("my_model.h5")
    classifier_model.compile(
    optimizer='adam',
    loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False),
    metrics=['accuracy']
)
    #classifier_model = tf.keras.models.load_model('model.json')
    #shape =((256,256,3))
    #model = tf.keras.Sequential(hub[hub.KerasLayer(classifier_model,input_shape =shape)])
    test_image = image.resize((256,256))
    test_image = preprocessing.image.img_to_array(test_image)
    test_image = test_image/255.0
    test_image = np.expand_dims(test_image, axis = 0)
    class_name = ['Healthy_leaf', 'Sick_leaf']
    predictions = classifier_model.predict(test_image)
    scores = tf.nn.softmax(predictions[0])
    scores = scores.numpy()
    image_class = class_name[np.argmax(scores)]
    result = "The image uploaded is: {}".format(image_class)
    return result

st.write("Is this tea leaves has Helopeltis or not.")
    #load_image()
if __name__ == '__main__':
    main()

if st.button('Drop picture here'):
     st.write('Not just yet. I am still training my model...')

# if predict_result:  
#     figure = plt.figure()
#     plt.imshow(image)
#     plt.axis('off')
#     result = predict_class(image)
#     st.write(result)
#     st.pyplot(figure)
# else:
#     st.write('')
    
    
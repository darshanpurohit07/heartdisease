# %%
# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 20GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session

# %% [markdown]
# ### Importing the packages/librairies:

# %%
import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
from keras.layers import *
from keras.models import Sequential
from keras.preprocessing import image
from glob import glob
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import os
import warnings
warnings.filterwarnings('ignore')

# %% [markdown]
# ### Preprocessing operations :

# %%
#glob function will create a list with all the paths, os.path.basename(x) : x will take the last portion of the path which is the name of the image :

path_to_train_test = {os.path.basename(x): x for x in 
                      glob(os.path.join('..', 'input', 'covid19-image-dataset', 'Covid19-dataset', '*'))}
print(path_to_train_test)

path_to_labels = {os.path.basename(x): x for x in 
                  glob(os.path.join('..', 'input', 'covid19-image-dataset', 'Covid19-dataset', 'test', '*'))}
print(path_to_labels)

#paths of data
 
path_train = 'D:/VS_CODE/python mini/Covid19-dataset/test/train'
path_test = 'D:/VS_CODE/python mini/Covid19-dataset/test/test'



# %%
#names of classes :
name_classes = os.listdir('Covid19-dataset/train')
 #os.listdir allow us to extract all the files in the specified path
print(name_classes)

# %% [markdown]
# ### Exploring Chest X-ray Data with Image Visualization

# %%
# Example for Covid X-rays:
print(path_train) # Path to train
image_path = path_train + '/' + 'Covid' # Acces to one of the classes, here we chose : 'Covid'
print(image_path) # Path to Covid
image_in_folder = os.listdir(image_path) # Using os.listdir to extract all the elements located in 'Covid'
print(image_in_folder) # Print contents inside 'Covid'
first_image = image_in_folder[0]
print('First Image :', first_image) #Choose an image
first_image_path = image_path + '/' + first_image
print(first_image_path)

img_input = '/kaggle/input/covid19-image-dataset/Covid19-dataset/test/Covid/0120.jpg' #insert image path
img = image.load_img(img_input)
plt.imshow(img)

# %%
#function to show one image each class
def plot_image(name_classes):
    plt.figure(figsize = (12,12))
    
    for i, category in enumerate(name_classes):
        image_path = path_train + '/' + category
        images_in_folder = os.listdir(image_path)
        
        first_image = images_in_folder[0]
        first_image_path = image_path + '/' + first_image
        
        img = image.load_img(first_image_path)
        img_array = image.img_to_array(img) / 255
        
        plt.subplot(1, 3, i + 1)
        plt.imshow(img_array)
        plt.title(category)
        plt.axis('off')
    plt.show()

# %%
plot_image(name_classes)

# %%
directory_train_for_Covid = {os.path.basename(x) : x for x in glob(os.path.join('..','input', 'covid19-image-dataset', 'Covid19-dataset', 'train', 'Covid', '*' ))}
directory_train_for_Normal = {os.path.basename(x) : x for x in glob(os.path.join('..','input', 'covid19-image-dataset', 'Covid19-dataset', 'train', 'Normal', '*' ))}
directory_train_for_VP = {os.path.basename(x) : x for x in glob(os.path.join('..','input', 'covid19-image-dataset', 'Covid19-dataset', 'train', 'Viral Pneumonia', '*' ))}

print(' \n Number of images for Covid : {} \n Number of images for Normal : {} \n Number of images for Viral Pneumonia : {}'
          .format(len(directory_train_for_Covid), len(directory_train_for_Normal), len(directory_train_for_VP)))
total_train_images = len(directory_train_for_Covid) + len(directory_train_for_Normal) + len(directory_train_for_VP)
print('Total Images for Train :', total_train_images)

# %%
directory_test_for_Covid = {os.path.basename(x) : x for x in glob(os.path.join('..','input', 'covid19-image-dataset', 'Covid19-dataset', 'test', 'Covid', '*' ))}
directory_test_for_Normal = {os.path.basename(x) : x for x in glob(os.path.join('..','input', 'covid19-image-dataset', 'Covid19-dataset', 'test', 'Normal', '*' ))}
directory_test_for_VP = {os.path.basename(x) : x for x in glob(os.path.join('..','input', 'covid19-image-dataset', 'Covid19-dataset', 'test', 'Viral Pneumonia', '*' ))}

print(' \n Number of images for Covid : {} \n Number of images for Normal : {} \n Number of images for Viral Pneumonia : {}'
          .format(len(directory_test_for_Covid), len(directory_test_for_Normal), len(directory_test_for_VP)))
total_test_images = len(directory_test_for_Covid) + len(directory_test_for_Normal) + len(directory_test_for_VP)
print('Total Images for Test :', total_test_images)

# %%
print('Total Images :', (total_train_images+total_test_images))

# %%
data_generator = ImageDataGenerator(rescale = 1/255)

# %%
#train data generator
train_data = data_generator.flow_from_directory(path_train,
                                                target_size = (224, 224),
                                                batch_size = 16
                                               )

# %%
#test data generator
test_data = data_generator.flow_from_directory(path_test,
                                                target_size = (224, 224),
                                                batch_size = 1
                                               )

# %% [markdown]
# ### CNN Keras/Tensorflow Librairies/Packages :

# %%
import random
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, BatchNormalization, Flatten, Dropout, Dense
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import EarlyStopping

# %%
def GetPixels(train_test_data):

    #train data generator
    train_data_example = data_generator.flow_from_directory(path_train,
                                                target_size = (28, 28),
                                                batch_size = 16
                                                   )
    random_image_selected = np.random.randint(0,11)
    images, labels = next(train_data_example) # Get a batch of images and labels
    COLOR2GRAY = np.mean(images[random_image_selected], axis=-1)
    
    plt.figure(figsize=(15,8))
    plt.imshow(COLOR2GRAY)
    print('Dimension of the Image:', str(COLOR2GRAY.shape))
    
    for i in range(28):
        for j in range(28):
            plt.text(i, j, round(COLOR2GRAY[i,j],1),fontsize=8, color='lime', ha='center', va='center')
            
    plt.show()

GetPixels(train_data)

# %% [markdown]
# ### Mathematics behind CNN :

# %%
#Medical Intelligence Assistance For Chest X-Ray Diagnostic (MAICXD):
model = tf.keras.models.Sequential(name='MAICXDNet', 
                layers= [
                        #Layer 1:
                        tf.keras.layers.Conv2D(filters=32, kernel_size=(3,3), padding='valid', data_format=None, 
                                              activation = 'relu', input_shape=(224,224,3)),
                        tf.keras.layers.MaxPooling2D(pool_size=(2,2), strides=None),
                        
                            #Layer 2:
                            tf.keras.layers.Conv2D(filters=68, kernel_size=(3,3), padding='valid', data_format=None, 
                                              activation = 'relu'),
                            tf.keras.layers.MaxPooling2D(pool_size=(2,2), strides=None),

                                #Flattened Layer:
                                tf.keras.layers.Flatten(),

                                    #Dense Layer (Fully-Connected):
                                    tf.keras.layers.Dense(units=3, activation='softmax')
                    ])

model.summary()

# %%
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# %%
model_fit = model.fit(train_data, epochs=3, validation_data=test_data, callbacks=None)

# %%
loss, accuracy = model.evaluate(test_data)
print('Loss : {:.2f}, Accuracy: {:.2f}'.format(loss, accuracy))

# %%
#Performance (History):
plt.figure(figsize=(10,8))
plt.style.use('bmh')
fig, (ax1, ax2) = plt.subplots(nrows=1, ncols=2, sharey=False, sharex=False)

#Loss & Validation Loss:
ax1.plot(model_fit.history['loss'], label='loss', color='red', marker='+', lw=1)
ax1.plot(model_fit.history['val_loss'], label='val_loss', color='grey', marker='*')
ax1.set_xlabel('Number of Epochs', fontsize=8, fontweight='bold')
ax1.set_ylabel('Loss & Val Loss', fontsize=8, fontweight='bold')
ax1.legend()
#Accuracy:
ax2.plot(model_fit.history['accuracy'], label='accuracy', color='blue', marker='o', lw=1)
ax2.set_xlabel('Number of Epochs', fontsize=8, fontweight='bold')
ax2.legend()

# %%
train_data.class_indices.items()

# %%
print(train_data.class_indices.items())
class_dict = dict([value, key] for key, value in train_data.class_indices.items())
class_dict

# %%
from tensorflow.keras.preprocessing import image

def prediction(Drop_XRAY_Here, actual_label, model, class_dict):
    """
    Function to make prediction on X-Ray image and display the result with the probability.
    
    Args:
    Drop_XRAY_Here (str): Path to the X-Ray image to be predicted.
    actual_label (str): Actual label of the image (used for comparison).
    model (keras.Model): The trained CNN model used for prediction.
    class_dict (dict): Dictionary mapping class indices to class labels.
    """
    # Charger et prétraiter l'image
    testing_img = image.load_img(Drop_XRAY_Here, target_size=(224, 224))  #Load the image
    test_img_array = image.img_to_array(testing_img) / 255  #Normalization
    
    #Reshape : (batch_size, height, width, channels))
    test_img_input = test_img_array.reshape(1, test_img_array.shape[0],
                                           test_img_array.shape[1],
                                           test_img_array.shape[2])
    
    #Prediction :
    prediction_array = model.predict(test_img_input)
    
    #Predict the class and the associated probability :
    predicted_class = np.argmax(prediction_array)  #Predicted Class
    predicted_img = class_dict[predicted_class]  
    predicted_prob = float(prediction_array[0][predicted_class] * 100)  #Probability of the Class in percentage
    
    plt.figure(figsize=(4, 4))
    plt.imshow(test_img_array)
    plt.title('Actual Label: {}, Predicted Label: {}\nProbability: {:.2f}%'.format(actual_label, predicted_img, predicted_prob), 
             fontsize=10)
    
    plt.grid(False)  # No Grid
    plt.axis('off')  # No Axes
    plt.show()

# %%
Drop_XRAY_Here = '/kaggle/input/test-images/00000011_004.png'  #Image Path
prediction(Drop_XRAY_Here, 
           actual_label='Covid', 
           model=model, 
           class_dict=class_dict
          ) # Predictor Function

# %%




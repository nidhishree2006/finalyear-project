import cv2 
import os
import gc
import numpy as np
from sklearn.model_selection import train_test_split
from tensorflow import keras
import pandas as pd
from IPython.display import display
import matplotlib.pyplot as plt

shape = 224
classes = []
class Dataset:
    def __init__(self, shape):
        self.path = "../Dataset/data"
        self.data_files = os.listdir(self.path)
        self.data_csv = 0
        self.x_data = []
        self.y_bbox_data = []
        self.y_label_data = []
        self.x_data_files = []
        self.y_data_files = []
        self.labels = []
        self.shape = shape
        
    def __preprocessing_image(self, path):
        image = cv2.imread(path)
        image = cv2.resize(image, (shape, shape))
        image = np.reshape(image, (shape, shape, 3))
        image = np.array(image, dtype = "float32")
        image /= 255.0
        
        return image
        
    def __seperating_images_labels(self):
        for data_file in self.data_files:
            if data_file.split(".")[-1] == "txt":
                data = open(f"{self.path}/{data_file}", "r").readline().split()
                df = pd.DataFrame()
                df["filename"] = [data_file]
                df["class"] =[float(data[0])]
                df["xmin"] = [float(data[1])]
                df["ymin"] = [float(data[2])]
                df["xmax"] = [float(data[3])]
                df["ymax"] = [float(data[4])]
                self.y_data_files.append(df)
    
        self.data_csv = pd.concat(self.y_data_files)
        for i in list(self.data_csv.columns[2:]):
            self.data_csv[i] = self.data_csv[i].astype(float)
            
    def __extract_images(self):
        self.images_paths = self.data_csv["filename"].values
        for image in self.images_paths:
            self.x_data.append(self.__preprocessing_image(self.path + "/" + image.split(".")[0] + ".jpeg"))
            
    def __extract_labels(self):
        global classes
        self.classes = self.data_csv["class"].unique().tolist()
        classes = self.classes
        self.y_label_data = self.data_csv["class"].values
        self.y_label_data = [self.classes.index(i) for i in self.y_label_data]
        self.y_label_data = keras.utils.to_categorical(self.y_label_data)
        self.y_label_data = np.array(self.y_label_data, dtype = "float32")
        
    def __extract_bboxes(self):
        for index, _ in enumerate(self.data_csv["xmin"].values):
            self.y_bbox_data.append(self.data_csv.iloc[[index]].values[0])
        self.y_bbox_data = np.array(self.y_bbox_data, dtype= "float32")
            
    def load_data(self):
        self.__seperating_images_labels()
        self.__extract_labels()
        self.__extract_images()
        self.data_csv = self.data_csv.drop('class', axis = 1)
        self.data_csv = self.data_csv.drop('filename', axis = 1)
        self.__extract_bboxes()
        
        return train_test_split(self.x_data, self.y_label_data, self.y_bbox_data, random_state = 42, shuffle = True)

dataset = Dataset(shape)
split = dataset.load_data()
(x_train, x_test) = split[:2]
(y_class_train, y_class_test) = split[2:4]
(y_bbox_train, y_bbox_test) = split[4:]
x_train = np.array(x_train, dtype = "float32")
x_test = np.array(x_test, dtype = "float32")
#y_bbox_train = np.reshape(y_bbox_train, (y_bbox_train.shape[0], 4))
#y_bbox_test =  np.reshape(y_bbox_test,  (y_bbox_test.shape[0], 4))
y_bbox_train = np.array(y_bbox_train, dtype = "float32")
y_bbox_test = np.array(y_bbox_test, dtype = "float32")
y_class_train = np.array(y_class_train, dtype = "float32")
y_class_test = np.array(y_class_test, dtype = "float32")

print(f"X_Train Shape : {x_train.shape}")
print(f"X_Test Shape : {x_test.shape}")
print(f"Y_class_Train Shape : {y_class_train.shape}")
print(f"Y_class_Test Shape : {y_class_test.shape}")
print(f"Y_bbox_Train Shape : {y_bbox_train.shape}")
print(f"Y_bbox_Test Shape : {y_bbox_test.shape}")

del dataset
del split
gc.collect()
class Model():
    def __init__(self, cnn):
        self.model_input1 = keras.layers.Input((shape, shape, 3))
        self.cnn = cnn(weights = "imagenet", include_top = False, input_tensor = self.model_input1)
        for layer in self.cnn.layers:
            layer.trainable = False
        self.__build_model()
        self.plot_loss_acc()
        
    def __build_model(self):
        
        flatten = keras.layers.Flatten()(self.cnn.output)
        x = keras.layers.Dense(1024, activation = "relu", kernel_initializer ="he_normal")(flatten)
        x = keras.layers.Dense(512, activation = "relu", kernel_initializer = "he_normal")(x)
        x = keras.layers.Dense(256, activation = "relu", kernel_initializer = "he_normal")(x)
        x = keras.layers.Dense(128, activation = "relu", kernel_initializer = "he_normal")(x)
        output_1 = keras.layers.Dense(len(classes), activation = "softmax", name = "Class")(x)
        
        x = keras.layers.Dense(256, activation = "relu", kernel_initializer ="he_normal")(flatten)
        x = keras.layers.Dense(128, activation = "relu", kernel_initializer ="he_normal")(x)
        x = keras.layers.Dense(64, activation = "relu", kernel_initializer = "he_normal")(x)
        x = keras.layers.Dense(32, activation = "relu", kernel_initializer = "he_normal")(x)
        output_2 = keras.layers.Dense(4, activation = "sigmoid", name = "BBOX")(x)
        
        optimizer = keras.optimizers.Adam(1e-4)
        self.model = keras.models.Model(inputs = [self.model_input1], outputs = [output_1, output_2])
        losses = {"Class": "categorical_crossentropy","BBOX": "mse"}
        lossWeights = {"Class": 1.0, "BBOX": 1.0}
        self.model.compile(optimizer = optimizer, loss = losses, metrics = ["accuracy"])
        
        #display(keras.utils.plot_model(self.model))
        checkpoint = keras.callbacks.ModelCheckpoint("../model/weeddetectionmodel.h5")
        
        self.model.fit(x_train, y = {"Class":y_class_train, "BBOX":y_bbox_train},
                       epochs = 10, validation_data = (x_test, {"Class":y_class_test, "BBOX":y_bbox_test}),
                       shuffle = True, callbacks = [checkpoint])
    
    def plot_loss_acc(self):
        #Loss vs Epochs
        plt.plot(self.model.history.history['Class_loss'], label='train')
        plt.plot(self.model.history.history['val_Class_loss'], label='test')
        plt.xlabel("Epochs")
        plt.ylabel("Loss")
        plt.title(f"Loss vs Epochs")
        plt.legend()
        plt.show()
        #Accuracy vs Epochs
        plt.plot(self.model.history.history['Class_accuracy'], label='train')
        plt.plot(self.model.history.history['val_Class_accuracy'], label='test')
        plt.xlabel("Epochs")
        plt.ylabel("Accuracy")
        plt.title(f"ACCURACY vs Epochs")
        plt.legend()
        plt.show()

model = Model(keras.applications.resnet.ResNet101)
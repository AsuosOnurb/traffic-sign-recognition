import tensorflow
from tensorflow.keras.layers.experimental.preprocessing import Rescaling, Resizing
from tensorflow.keras.layers import Conv2D, LeakyReLU, BatchNormalization, MaxPooling2D, Dropout, Dense, Flatten, Input
from tensorflow.keras.layers.experimental.preprocessing import  RandomRotation, RandomZoom, RandomHeight, RandomWidth, RandomContrast, RandomTranslation
from tensorflow.keras.models import  Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping,  ReduceLROnPlateau
from tensorflow.keras.utils import plot_model
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import numpy as np
from sklearn.metrics import accuracy_score
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import pandas as pd
from sklearn.metrics import classification_report
from matplotlib.pyplot import figure
from sklearn.metrics import log_loss


class CNN_C_RGB:

    def __init__(self, model_name: str, img_size=32, channels=1) -> None:

        self.training_history = None

        input_layer = Input(shape=(img_size, img_size, channels))

        hidden_layer = Resizing(img_size, img_size)(input_layer)
        
        hidden_layer = Conv2D(32, (3, 3), activation='relu')(hidden_layer)    

        hidden_layer = Conv2D(128, (3, 3), activation='relu')(hidden_layer)
        hidden_layer = MaxPooling2D(pool_size=(2, 2))(hidden_layer)
        hidden_layer = BatchNormalization() (hidden_layer)

        hidden_layer = Conv2D(128, (3, 3), activation='relu')(hidden_layer)
        hidden_layer = MaxPooling2D(pool_size=(2, 2))(hidden_layer)
        hidden_layer = BatchNormalization() (hidden_layer)

        hidden_layer = Dropout(0.25)(hidden_layer)

        hidden_layer = Conv2D(128, (3, 3), activation='relu')(hidden_layer)
        hidden_layer = MaxPooling2D(pool_size=(2, 2))(hidden_layer)
        hidden_layer = BatchNormalization() (hidden_layer)

        hidden_layer = Dropout(0.5)(hidden_layer)

        hidden_layer = Flatten()(hidden_layer)

        hidden_layer = Dense(128)(hidden_layer)
        hidden_layer = Dropout(0.5)(hidden_layer)

        output_layer = Dense(43, activation='softmax')(hidden_layer)

        model = Model(inputs=input_layer,
                      outputs=output_layer, name=model_name)

        self.input_layer = input_layer
        self.output_layer = output_layer

        self.model = model
        self.model_name = model_name

        self.model_save_path = f'saved_models/{self.model_name}'
        
        self.compile()
        plot_model(self.model, show_shapes=True, to_file=f"{model_name}.png")

      

        self.model.summary()

   

    def train(self, X_train, y_train, X_val, y_val, epochs=200, stop_early=True):

            
        checkpointer_callback = ModelCheckpoint(
            filepath=self.model_save_path,
            verbose=1,
            save_weights_only=False,
            save_best_only=True
        )

        reduceLR = ReduceLROnPlateau(
            monitor='val_loss', factor=0.2, patience=5, min_lr=0.000000001, verbose=1)
        callbacks = [checkpointer_callback, reduceLR]

        if stop_early:
            earlystopper_callback = EarlyStopping(
                monitor="val_loss", min_delta=0.0001, patience=15, verbose=1)
            callbacks.append(earlystopper_callback)
        
        '''
        self.training_history = self.model.fit(
            train_data,
            validation_data=validation_data,
            epochs=epochs,
            verbose=1,
            callbacks=callbacks
        )
        '''
        aug = ImageDataGenerator(
            shear_range=0.15,
            horizontal_flip=False,
            vertical_flip=False,
            fill_mode="nearest"
        )

        self.training_history = self.model.fit(
            aug.flow(
                X_train,
                y_train,
                batch_size=32
            ),
            epochs=epochs,
            validation_data=(X_val, y_val),
            callbacks=callbacks,
            verbose=1
        )

    def evaluate(self, X_test, y_test):
        # return self.model.evaluate(data)
        pred=self.model.predict(X_test) 
        pred=np.argmax(pred,axis=1)
        print('Test Data accuracy: ',accuracy_score(y_test, pred)*100)

        cf = confusion_matrix(y_test, pred)
        df_cm = pd.DataFrame(cf)
        plt.figure(figsize = (20,20))
        sns.heatmap(df_cm, annot=True, fmt='g')

        print(classification_report(y_test, pred))

    def evaluate_error(self, X_test, y_test):
        pred=self.model.predict(X_test) 
        #pred=np.argmax(pred,axis=1)
        print('Test Data error/loss: ', log_loss(y_test, pred, labels=y_test ))

    def load(self):
        self.model = tensorflow.keras.models.load_model(self.model_save_path)
        self.compile()

    def freeze(self):
        """
        Freezes all layers of the CNN model.
        """
        for layer in self.model.layers:
            layer.trainable = False
        self.compile()

    def unfreeze(self):
        """
        Unfreezes all layers of the CNN model.
        """
        for layer in self.model.layers:
            layer.trainable = True 
        self.compile()
        

    def compile(self):
        optimizer = Adam(learning_rate=0.001, decay=0.001 / (30 * 0.5))
        self.model.compile(optimizer=optimizer,
                      loss='categorical_crossentropy', metrics=['accuracy'])

    def plot_history(self):
        # summarize history for accuracy
        figure(figsize=(10, 10), dpi=100)
        plt.plot(self.training_history.history['accuracy'])
        plt.plot(self.training_history.history['val_accuracy'])
        plt.title('model accuracy')
        plt.ylabel('accuracy')
        plt.xlabel('epoch')
        plt.legend(['train', 'val'], loc='lower right')
        plt.show()

        # summarize history for loss
        figure(figsize=(10, 10), dpi=100)
        plt.plot(self.training_history.history['loss'])
        plt.plot(self.training_history.history['val_loss'])
        plt.title('model loss')
        plt.ylabel('loss')
        plt.xlabel('epoch')
        plt.legend(['train', 'val'], loc='upper right')
        plt.show()
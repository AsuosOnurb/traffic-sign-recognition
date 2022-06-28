import tensorflow
from tensorflow.keras.layers import  BatchNormalization, Dropout, Dense
from tensorflow.keras.layers import Concatenate
from tensorflow.keras.models import  Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping,  ReduceLROnPlateau
from tensorflow.keras.utils import plot_model
from vpc_cnn_a import CNN_A_Gray
import vpc_data
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import numpy as np
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
import pandas as pd
import seaborn as sns
from matplotlib.pyplot import figure
from sklearn.metrics import log_loss

gen = ImageDataGenerator(
        shear_range=0.15,
        horizontal_flip=False,
        vertical_flip=False,
        fill_mode="nearest"
)

BATCH_SIZE = 32

class EnsembleModel5AGray:

    def __init__(self, model_name: str, img_size: int) -> None:
        """
            Constructs and initializes an Ensemble Model.
        """
        
        self.training_history = None

    

        ################## Create each member model ##################
        self.members = []
        for i in range(5):
            # Add the model that uses grayscale normalized images
            modeli =  CNN_A_Gray(f"MemberModelA{i}", img_size=30, channels=1)
            self.members.append(modeli)


        # Concatenate the output of each sub model
        ensemble_output = [model.output_layer for model in self.members]
        merged_outputs = Concatenate(axis=1, name='ConcatLayer')(ensemble_output)

        hidden = Dense(32)(merged_outputs)
        hidden = BatchNormalization()(hidden)
        hidden = Dropout(0.4)(hidden)

        ensemble_output_layer = Dense(43, activation='softmax')(hidden)
        ensemble_input_layer = [model.model.input for model in self.members]

        self.stacked_model = Model(
            inputs=ensemble_input_layer,
            outputs=ensemble_output_layer,
            name=model_name
        )

        plot_model(self.stacked_model, show_shapes=True, to_file=f"saved_models/{model_name}.png")
        self.compile()

        self.stacked_model.summary()

        # We use this to write and read model weights for posterity
        self.model_save_path = f'saved_models/{model_name}'
        self.model_name = model_name

        self.IMG_SIZE = img_size




    def train_members(self, epochs, stop_early=True):
        """
            Trains each member model.
            Each member model trains on an individual set of (training, validation) data.
            Each member trains on an individual set of augmentations.
        """



        for member in self.members:
            print(f"================ Training member model {member.model_name} ====================")
            training_data, validation_data = vpc_data.load_training_data()
            training_data = vpc_data.config_for_performance(training_data)
            validation_data = vpc_data.config_for_performance(validation_data)


            member.train(
                training_data,
                validation_data,
                epochs=epochs,
                batch_size=BATCH_SIZE,
                stop_early=stop_early
            )
            print(f"================ Member model {member.model_name} trained! ====================")

    def freeze_members(self):
        for member in self.members:
            member.freeze()
        self.compile()

    def unfreeze_members(self):
        for member in self.members:
            member.unfreeze()
        self.compile()

    def freeze_all(self):
        for layer in self.stacked_model.layers:
            layer.trainable = False
        self.compile()

    def unfreeze_all(self):
        for layer in self.stacked_model.layers:
            layer.trainable = True
        self.compile()

    def train_ensemble(self, epochs, stop_early=True):

        X_train_rgb, y_train, X_val_rgb, y_val =  vpc_data.load_training_data(validation_split=0.2)


        checkpointer_callback = ModelCheckpoint(
            filepath=self.model_save_path,
            verbose=1,
            save_weights_only=False,
            save_best_only=True
        )

        reduceLR = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, min_lr=0.000000001, verbose=1)
        callbacks = [checkpointer_callback, reduceLR]

        if stop_early:
            earlystopper_callback = EarlyStopping(monitor="val_loss", min_delta=0.0001, patience=15, verbose=1)
            callbacks.append(earlystopper_callback)

    
        gen_flow = generate_data_generator(X_train_rgb, y_train)

        self.training_history = self.stacked_model.fit_generator(
            gen_flow,
            epochs=epochs,
            validation_data=([X_val_rgb, X_val_rgb, X_val_rgb, X_val_rgb, X_val_rgb], y_val),
            callbacks=callbacks,
            steps_per_epoch=len(X_train_rgb) / BATCH_SIZE,
            verbose=1
        )


    def load_members(self):
        """
            Loads the Ensemble configuration weights.
            Loading the Ensemble is done by going to each sub model and loading its weights and loading the classifier
        """
        
        for member in self.members:
            member.load()

    def load_full_model(self):
        self.stacked_model = tensorflow.keras.models.load_model(self.model_save_path)
        self.compile()
    



    def evaluate_ensemble(self, X_test_rgb, y_test):
        

        pred = self.stacked_model.predict([X_test_rgb, X_test_rgb, X_test_rgb, X_test_rgb, X_test_rgb])
        pred = np.argmax(pred, axis=1)
        print("Predictions: ", pred)
        print(pred.shape)
        print('Test Data accuracy: ',accuracy_score(y_test, pred)*100)

        cf = confusion_matrix(y_test, pred)
        df_cm = pd.DataFrame(cf)
        plt.figure(figsize = (22,22))
        sns.heatmap(df_cm, annot=True, fmt='g')

        print(classification_report(y_test, pred))

    def evaluate_error(self, X_test, y_test):
        pred=self.stacked_model.predict([X_test, X_test, X_test, X_test, X_test]) 
        #pred=np.argmax(pred,axis=1)
        print('Test Data error/loss: ', log_loss(y_test, pred, labels=y_test ))

    def compile(self):
        optimizer = Adam(learning_rate=0.001)
        self.stacked_model.compile(optimizer=optimizer,
                      loss='categorical_crossentropy', metrics=['accuracy'])



    def plot_history(self):

        figure(figsize=(10, 10), dpi=100)

        # summarize history for accuracy
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

def generate_data_generator(rgb_images, y):
    genX1 = gen.flow(rgb_images, y,  batch_size=BATCH_SIZE,seed=42)
    genX2 = gen.flow(rgb_images,rgb_images, batch_size=BATCH_SIZE,seed=42)
    genX3 = gen.flow(rgb_images,rgb_images, batch_size=BATCH_SIZE,seed=42)
    genX4 = gen.flow(rgb_images,rgb_images, batch_size=BATCH_SIZE,seed=42)
    genX5 = gen.flow(rgb_images,rgb_images, batch_size=BATCH_SIZE,seed=42)
    while True:
            X1i = genX1.next()
            X2i = genX2.next()
            X3i = genX3.next()
            X4i = genX4.next()
            X5i = genX5.next()
            #Assert arrays are equal - this was for peace of mind, but slows down training
            #np.testing.assert_array_equal(X1i[0],X2i[0])
            yield [X1i[0], X2i[1], X3i[1], X4i[1], X5i[1]], X1i[1]


    
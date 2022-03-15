from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras import Sequential
import pandas as pd
from src.class_loss import class_loss
from src.confusion_matrix import confusion_matrix
from src.get_weight_dict import get_weight_dict
from src.grid_search.grid_search import grid_search
from src.metrics import recall_m, precision_m, f1_m

class cnn:

    def __init__(self, load_model=True):
        self.load_model=load_model

    def train_model(self, X_train, y_train, X_val, y_val, epochs=20, batch_size=128):

        if len(y_train.shape) > 1:
            self.multi_target = True
        else:
            self.multi_target = False

        opt = keras.optimizers.SGD(learning_rate=0.007)
        loss = keras.losses.BinaryCrossentropy()

        input = layers.Input(shape=(X_train.shape[1:]))

        x = layers.Conv2D(64, (5, 5))(input)
        x = layers.Activation('relu')(x)
        x = layers.MaxPooling2D(pool_size=(4, 4))(x)

        x = layers.Conv2D(32, (5, 5))(x)
        x = layers.Activation('relu')(x)
        x = layers.MaxPooling2D(pool_size=(3, 3))(x)

        x = layers.Conv2D(16, (5, 5))(x)
        x = layers.Activation('relu')(x)
        x = layers.MaxPooling2D(pool_size=(2, 2))(x)

        x = layers.Flatten()(x)

        x = layers.Dense(16)(x)
        x = layers.Activation('relu')(x)

        if self.multi_target:
            outputs = []
            for i in range(y_train.shape[-1]):
                output = layers.Dense(1, activation='sigmoid')(x)

                outputs.append(output)
        else:
            outputs = layers.Dense(1, activation='sigmoid')(x)

        self.model = keras.Model(input, outputs)

        output_names = []
        for layer in self.model.layers:
            if type(layer) == layers.Dense:
                if layer.units == 1:
                    output_names.append(layer.name)

        {'batch size': 32, 'epochs': 25, 'loss': 'mean_squared_error', 'lr': 0.01, 'optimizer': 'adam'}

        search = grid_search()

        search.test_model(self.model, X_train, y_train, X_val, y_val, num_combs=12)

        opt = keras.optimizers.Adam(lr=0.01)
        self.model.compile(loss='mean_squared_error',
                optimizer=opt,
                metrics=['accuracy', f1_m, precision_m, recall_m])

        self.fit = self.model.fit(X_train, y_train, epochs=25, batch_size=32, validation_data=(X_val, y_val))

        try:
            self.model.save('data/saved_models/image_only/keras_cnn_model.h5')
        except:
            print("image only model could not be saved")

        return self.model

    def test_model(self, X_test, y_test):
        results = self.model.evaluate(X_test, y_test, batch_size=32)

        confusion_matrix(y_true=y_test, y_pred=self.model.predict(X_test), save_name="image_only_c_mat.png")

        return results

    def get_model(self, X_train=None, y_train=None, X_val=None, y_val=None, epochs=10, batch_size=32):

        if self.load_model:
            self.model = keras.models.load_model('data\\saved_models\\keras_cnn_model.h5')
        else:
            self.model = self.train_model(X_train, y_train, X_val, y_val, epochs, batch_size)

        return self.model

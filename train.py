from utils import build_data, vectorization
from model import build_model
from keras.callbacks import ModelCheckpoint
import io
import os


if __name__== "__main__":
    text = io.open('Model_data/shakespear.txt', encoding='utf-8').read().lower()

    Tx = 40
    chars = sorted(list(set(text)))
    char_indices = dict((c, i) for i, c in enumerate(chars))
    indices_char = dict((i, c) for i, c in enumerate(chars))

    print("Creating training set...")
    X, Y = build_data(text, Tx, stride = 1)
    print("Vectorizing training set...")
    x, y = vectorization(X, Y, n_x = len(chars), char_indices = char_indices)


    model=build_model(x,y)
    filepath = os.path.join(os.getcwd(),"Model_checkpoints/model.h5")
    checkpoint = ModelCheckpoint(filepath, monitor='loss', verbose=1, save_best_only=True, mode='min')
    model.fit(x, y, batch_size=128, epochs=20,callbacks= [checkpoint])

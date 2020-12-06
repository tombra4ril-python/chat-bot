#import dependencies
import random
import numpy as np
import os

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import (
  Dense,
  Activation,
  Dropout
)
from tensorflow.keras.optimizers import SGD # stockastic gradient descent

from .bag_of_words import create_bag_of_words

def train():
  print(f"Training model started...")
  X, Y = get_training_data()
  # start building neural network
  print(f"Building neural network...")
  # create model
  model = Sequential()
  model.add(Dense(128, input_shape=(len(X[0]),), activation="relu"))
  model.add(Dropout(0.5))
  model.add(Dense(64, activation="relu"))
  model.add(Dropout(0.5))
  model.add(Dense(len(Y[0]), activation="softmax"))
  #create optimizer
  sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
  model.compile(loss="categorical_crossentropy", optimizer=sgd, metrics=["accuracy"])
  # fit model
  fit_data = model.fit(np.array(X), np.array(Y), epochs=200, batch_size=5, verbose=1)
  # save model
  directory_name = "files"
  if os.path.dirname == "chat":
    file_name = os.path.join(directory_name, "chatbot_model.model")
    if not os.path.isdir(directory_name):
      os.mkdir(directory_name)
  else:
    file_name = os.path.abspath(os.path.join("chat", directory_name, "chatbot_model.model"))
    if not os.path.isdir(os.path.abspath(os.path.join("chat", directory_name))):
      os.mkdir(os.path.abspath(os.path.join("chat", directory_name)))

  model.save(file_name, fit_data)
  print(f"Successfully trained saved model...")

def get_training_data():
  training_data = create_bag_of_words()
  random.shuffle(training_data)
  # convert the training data to a numpy array to be used in tensorflow
  training_data = np.array(training_data)
  # create features and targets
  train_x = list(training_data[:, 0])
  train_y = list(training_data[:, 1])
  #
  return train_x, train_y

if __name__ == "__main__":
  train()
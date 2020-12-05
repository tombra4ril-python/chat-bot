# import dependencies
import random
import pickle
import numpy as np

from nltk.stem import WordNetLemmatizer

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import (
  Dense,
  Activation,
  Dropout
)
from tensorflow.keras.optimizers import SGD # imports stockastic gradient descent

from document import get_document

ignore_letters = ["?", "!", ".", ","]

def main():
  words, classes, documents = get_document()
  print(f"\nDocument is:")
  for doc in documents:
    print(f"{doc}")

if __name__ == "__main__":
  main()
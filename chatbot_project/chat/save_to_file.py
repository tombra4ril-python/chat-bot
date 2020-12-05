# import dependencies
import pickle
import os

from nltk.stem import WordNetLemmatizer
from document import get_documents

ignore_letters = ["?", "!", ".", ","]

def words_class_to_file():
  words, classes, documents = get_documents()
  # create a word lemmatizer
  lemmatizer = WordNetLemmatizer()
  # create a list of lemmatized words ignoring words in ignore_letters variable
  words = [lemmatizer.lemmatize(word) for word in words if word not in ignore_letters]
  words = sorted(set(words)) # remove duplicates
  # save to file
  directory_name = "files"
  words_file_name = os.path.join(directory_name, "words.pkl")
  classes_file_name = os.path.join(directory_name, "classes.pkl")
  if not os.path.isdir(directory_name):
    os.mkdir(directory_name)
  pickle.dump(words, open(words_file_name, "wb"))
  pickle.dump(classes, open(classes_file_name, "wb"))

if __name__ == "__main__":
  print(f"Writing pickle binaries to file...")
  words_class_to_file()
  print(f"Successfully wrote binaries to file...")
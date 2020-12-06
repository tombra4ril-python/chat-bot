# import dependencies
import pickle
import os

from nltk.stem import WordNetLemmatizer
from .document import get_documents

def words_classes_to_file():
  print(f"Writing pickle binaries to file...")
  words, classes, documents = get_documents()
  
  # save to file
  directory_name = "files"
  if os.path.dirname == "chat":
    if not os.path.isdir(directory_name):
      os.mkdir(directory_name)
    words_file_name = os.path.join(directory_name, "words.pkl")
    classes_file_name = os.path.join(directory_name, "classes.pkl")
  else:
    if not os.path.isdir(os.path.abspath(os.path.join("chat", directory_name))):
      os.mkdir(os.path.abspath(os.path.join("chat", directory_name)))
    words_file_name = os.path.abspath(os.path.join("chat", directory_name, "words.pkl"))
    classes_file_name = os.path.abspath(os.path.join("chat", directory_name, "classes.pkl"))

  pickle.dump(words, open(words_file_name, "wb"))
  pickle.dump(classes, open(classes_file_name, "wb"))
  print(f"Successfully wrote binaries to file...")

if __name__ == "__main__":
  words_class_to_file()
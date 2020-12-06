# import dependencies
import os
import pickle
import sys

def get_files():
  # check if the model exists first
  directory_name = "files" # get directory name
  if os.path.dirname == "chat":
    if os.path.isdir(os.path.abspath(os.path.join(os.path.dirname(__file__), directory_name))):
      # get the pickle objects
      words_name = os.path.abspath(os.path.join(os.path.dirname(__file__), directory_name, "words.pkl"))
      classes_name = os.path.abspath(os.path.join(os.path.dirname(__file__), directory_name, "classes.pkl"))
  else:
    if os.path.isdir(os.path.abspath(os.path.join(os.path.dirname(__file__), os.path.pardir, "chat", directory_name))):
      # get the pickle objects
      words_name = os.path.abspath(os.path.join(os.path.dirname(__file__), os.path.pardir, "chat", directory_name, "words.pkl"))
      classes_name = os.path.abspath(os.path.join(os.path.dirname(__file__), os.path.pardir, "chat", directory_name, "classes.pkl"))
  #  
  words = pickle.load(open(words_name, "rb"))
  classes = pickle.load(open(classes_name, "rb"))
  #
  return words, classes

if __name__ == "__main__":
  get_files()
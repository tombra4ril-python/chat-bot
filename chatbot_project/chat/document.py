# import dependencies
import json
import nltk
import os

from nltk.stem import WordNetLemmatizer

ignore_letters = ["?", "!", ".", ","]

def get_documents():
  print("Creating word document...")
  print(f"Reading json file...")
  file_name = os.path.abspath(os.path.join(os.path.dirname(__file__), os.path.pardir, "intent.json"))
  intents = json.loads(open(file_name).read())

  words = [] # contains tokenized words
  classes = [] # contains the class of the sentence or word
  documents = [] # holds tokenized words with corresponding class

  for intent in intents["intents"]:
    for pattern in intent["patterns"]:
      tokenized_words = nltk.word_tokenize(pattern)
      words.extend(tokenized_words)
      documents.append({"words": tokenized_words, "label": intent["tags"]})
      # check if the tag is not in the classes list
      if intent["tags"] not in classes:
        classes.append(intent["tags"])
  
  print("Successfully created words, classes and documents...")
  # create a word lemmatizer
  lemmatizer = WordNetLemmatizer()
  # create a list of lemmatized words ignoring words in ignore_letters variable
  words = [lemmatizer.lemmatize(word) for word in words if word not in ignore_letters]
  words = sorted(set(words)) # remove duplicates
  return words, classes, documents

if __name__ == "__main__":
  get_document()
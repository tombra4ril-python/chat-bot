# import dependencies
import json
import nltk

def get_document():
  intents = json.loads(open("../intent.json").read())

  words = [] # contains tokenized words
  classes = [] # contains the class of the sentence or word
  documents = [] # holds tokenized words with corresponding class

  for intent in intents["intents"]:
    for pattern in intent["patterns"]:
      tokenized_words = nltk.word_tokenize(pattern)
      words.extend(tokenized_words)
      documents.append((tokenized_words, intent["tags"]))
      # check if the tag is not in the classes list
      if intent["tags"] not in classes:
        classes.append(intent["tags"])
  
  return words, classes, documents

if __name__ == "__main__":
  get_document()
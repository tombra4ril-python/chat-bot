import random
import json
import pickle
import numpy as np
import os
from colorama import (
  init as coloramaInit,
  Fore, 
  Style
)

import nltk
from nltk.stem import WordNetLemmatizer

from tensorflow.keras.models import load_model

from chat.get_from_file import get_files

def main():
  model, words, classes, intents, lemmatizer = init()
  print(f"{Fore.GREEN}Chat bot running. Press Enter key to quit...")
  message = input(f">> ")
  while message:
    intent = predict_class(message, model, words, classes, lemmatizer)
    result = get_response(intent, intents)
    print(result)
    message = input(f">> ")
  print(f"Chat bot shut down!{Style.RESET_ALL}")

def init():
  # initialize colorama
  coloramaInit()
  # get the json intent file
  intents = json.loads(open("intent.json").read())
  # create an instance for lemmatization
  lemmatizer = WordNetLemmatizer()
  # check if the model exists first
  directory_name = os.path.join("chat", "files") # get directory name
  if not os.path.isdir(os.path.abspath(os.path.join(os.path.dirname(__file__), directory_name))):
    # import dependencies
    from chat.save_to_file import words_classes_to_file
    from chat.training import train
    #
    words_classes_to_file()
    train()
  else:
    print(f"Training model already exist!")
  # get the pickle objects
  words, classes = get_files()
  # get model
  model_name = os.path.join(directory_name, "chatbot_model.model")
  model = load_model(model_name)

  return model, words, classes, intents, lemmatizer

def clean_sentence(sentence, words, lemmatizer):
  # tokenize the sentence passed
  words = nltk.word_tokenize(sentence)
  # lemmatize the words
  lemma_words = [lemmatizer.lemmatize(word) for word in words]
  return lemma_words

def get_bag_of_words(sentence, words, lemmatizer):
  sentence = clean_sentence(sentence, words, lemmatizer)
  bag_of_words = [0] * len(words)
  #
  for sentence_word in sentence:
    for index, word in enumerate(words):
      if word == sentence_word:
        bag_of_words[index] = 1
  #
  return np.array(bag_of_words)

def predict_class(sentence, model, words, classes, lemmatizer):
  bag_of_words = get_bag_of_words(sentence, words, lemmatizer)
  result = model.predict(np.array([bag_of_words]))[0]
  error_threshold = 0.25
  results = [[i, r] for i, r in enumerate(result) if r > error_threshold]
  #sort the results in descending order
  results.sort(key = lambda _ : _[1], reverse = True)
  # create an empty prediction list to return
  prediction = []
  for result in results:
    prediction.append({"intent": classes[result[0]], "probability": str(result[1])})
  #
  return prediction

def get_response(prediction_intents, intents_json):
  tag = prediction_intents[0]["intent"]
  list_intents = intents_json["intents"]
  for index in list_intents:
    if index["tags"] == tag:
      result = random.choice(index["response"])
      break
  #
  return result

if __name__ == "__main__":
  main()
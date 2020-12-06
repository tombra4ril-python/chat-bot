# import dependencies
from nltk.stem import WordNetLemmatizer

from .document import get_documents

# get the document object
words, classes, documents = get_documents()
# create an instance of a lemmatizer
lemmatizer = WordNetLemmatizer()
# create a list equal in length to the number of classes with the assumption that label for a document is not present
labels = [0] * len(classes)
# create an empty training_data variable
training_data = []

def create_bag_of_words():
  print(f"Creating bag of words...")
  for document in documents:
    # create empty bag of words
    bag_of_words = []
    # get the lemmatized words for each document
    word_patterns = [lemmatizer.lemmatize(word.lower()) for word in document["words"]]
    # fill in the bag of words with 0s and 1s like one hot encoding if the word is present
    for word in words:
      bag_of_words.append(1) if word in word_patterns else bag_of_words.append(0)
    #
    # create a copy of the labels variable
    label = list(labels)
    # set the index of the label to 1 for which the current document corresponds to
    label[classes.index(document["label"])] = 1
    #
    # append the bag of words and label to the training variable
    training_data.append([bag_of_words, label])
  #
  print(f"Created training data Successfully...")
  return training_data

if __name__ == "__main__":
  create_bag_of_words()
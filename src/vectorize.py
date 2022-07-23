# Contains a one-hot vectorizer from genism.
# Directly returns pytorch sparse tensors. These
# can be converted when needed to dense tensors to
# reduce memory usage

import os
import pickle

import torch
import en_core_web_sm
from gensim.corpora import Dictionary

# The spacy pipeline instance we're going to use
spacy = en_core_web_sm.load()

# A class that contains the gensim dictionary and relevant
# utilities and config variables. Helps in enforcing a fixed
# vocab size
class Vectorizer:
    def __init__(self, vocabSize):
        self.dictionary = Dictionary()
        # Vocab size is vocabSize - 1 since the 0th
        # dimension is used for OOV tokens
        self.vocabSize = vocabSize - 1

    # Convenience function to fit the dictionary with new text
    # and limit vocab size
    def fit(self, listOfDocuments):
        listOfTokens = []
        # Tokenizing doc and adding all non-punctuation
        for i in range(len(listOfDocuments)):
            doc = spacy(listOfDocuments[i])
            for tok in doc:
                if not tok.is_punct:
                    listOfTokens.append(tok.lower_)
        
        # Passing to dictionary to fit
        self.dictionary.add_documents( [listOfTokens] )

        # We're also going to limit the size of the vocab right now
        self.dictionary.filter_extremes(keep_n=self.vocabSize)

    # Takes in a list of documents, and returns a
    # list-of-list-of one-hot encoded sparse pytorch tensors.
    # Each tensor corresponds to a token in the document, and each sublist
    # corresponds to one document.

    # NOTE: OUT OF V0CAB-TOKENS ALL MAP TO INDEX 0.
    def docs2Tensors(self, docList):
        listOfDocs = []
        for i in range(len(docList)):
            tokenList = []
            listOfTensors = []

            # Running doc through spacy so it can run its tokenization
            doc = spacy(docList[i])

            for tok in doc:
                tokenList.append(tok.lower_)
            
            # Getting list of indices from gensim
            doc2Indices = self.dictionary.doc2idx(tokenList)

            # Converting each index into a pytorch tensor, and adding to
            # list of tensors
            for index in doc2Indices:
                # Adding 1 to index so out-of-vocab goes from -1 to 0, and everthing
                # else is shifted one up, preserving the word that was at index 0
                denseTensor = torch.zeros( (len(self.dictionary) + 1, ) )
                denseTensor[index + 1] = 1

                sparseTensor = denseTensor.to_sparse()
                listOfTensors.append(sparseTensor)

            listOfDocs.append(listOfTensors)
        return listOfDocs

    def save(self, fName):
        with open(fName, "wb") as f:
            pickle.dump(self, f)
        
    @staticmethod
    def load(fName):
        with open(fName, "rb") as f:
            return pickle.load(f)
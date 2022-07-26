# Contains the actual model

import torch

class Word2VecModel(torch.nn.Module):
    def __init__(self, vocabSize, embedSize):
        super(Word2VecModel, self).__init__()
        # Storing architecture hyperparams
        self.vocabSize = vocabSize
        self.embedSize = embedSize

        # Defining word2vec architecture

        # Using a linear layer just since it makes life slightly easier,
        # since the embedding layer uses sparse inputs which are incompatible
        # with some optimizers. We can switch to an embedding bag later
        # if we want
        self.wordEmbedding = torch.nn.Linear(vocabSize, embedSize)

        self.endStack = torch.nn.Sequential(
            torch.nn.Linear(self.embedSize, self.vocabSize),
            torch.nn.LogSoftmax()
        )
    
    # Given an index, return the word embedding
    def index2Embedding(self, index):
        # Creating a sparse input vector to run through
        # to get embedding
        inVec = torch.zeros( (self.vocabSize,) )
        inVec[index] = 1

        return self.wordEmbedding(inVec)
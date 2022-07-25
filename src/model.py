# Contains the actual skipgram model

import torch

class SkipGramModel(torch.nn.Module):
    def __init__(self, vocabSize, embedSize):
        super(SkipGramModel, self).__init__()
        # Storing architecture hyperparams
        self.vocabSize = vocabSize
        self.embedSize = embedSize

        # Defining skipgram architecture

        # Using a linear layer just since it makes life slightly easier,
        # since the embedding layer uses sparse inputs which are incompatible
        # with some optimizers. We can switch to an embedding bag later
        # if we want
        self.wordEmbedding = torch.nn.Linear(vocabSize, embedSize)

        self.fullStack = torch.nn.Sequential(
            self.wordEmbedding,
            torch.nn.Linear(self.embedSize, self.vocabSize),
            torch.nn.Softmax()
        )
    
    def forward(self, inputSparseTensor):
        dense = inputSparseTensor.to_dense()
        return self.fullStack(dense)
    
    # Given an index, return the word embedding
    def index2Embedding(self, index):
        # Creating a sparse input vector to run through
        # to get embedding
        inVec = torch.zeros( (self.vocabSize,) )
        inVec[index] = 1

        return self.wordEmbedding(inVec.to_sparse_coo())
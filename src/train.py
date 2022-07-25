# Implements training loop for a word2vec model.
# Expects an iterable that yields (inputSparseVec, [outputSparseVecs]).
# The utilities in this codebase offer that, but you can write your
# own as well as long as they implement the above contract.

import torch

# Simply run this call multiple times to implement
# multi-epoch training
def train(skipGramModel, dataIterable):
    LR = 3e-4
    optim = torch.optim.Adam(skipGramModel.parameters(), lr=LR)
    lossFn = torch.nn.CrossEntropyLoss()

    # Iterate over each sample from the iterable, sum up loss
    # for each output, and then backprop
    for inSparse, outSparseList in dataIterable:
        inDense = inSparse.to_dense()
        modelOut = skipGramModel(inDense)
        
        # For consistency, we allow outputs to passed in as
        # sparse tensors. However, we actually want the indices,
        # so we'll just pull it out.
        lossList = []
        for outSparse in outSparseList:
            oneHotIndex = outSparse.indices()[0][0]
            lossList.append( lossFn(modelOut, oneHotIndex) )
        
        # Sum the losses and backprop
        meanLoss = torch.mean( torch.vstack(lossList) )

        print(meanLoss)

        optim.zero_grad()
        meanLoss.backward()
        optim.step()
# Implements training loop for a word2vec model.
# Expects an iterable that yields (inputSparseVec, [outputSparseVecs]) in
# skipgram mode, or reversed in cbow mode
# The utilities in this codebase offer that, but you can write your
# own as well as long as they implement the above contract.

import torch

# Simply run this call multiple times to implement
# multi-epoch training for skipgram models
def skipgram_train(skipGramModel, dataIterable):
    LR = 5e-3
    optim = torch.optim.SGD(skipGramModel.parameters(), lr=LR)
    lossFn = torch.nn.NLLLoss()

    # Iterate over each sample from the iterable, sum up loss
    # for each output, and then backprop
    for inSparse, outSparseList in dataIterable:
        inDense = inSparse.to_dense()

        embedOut = skipGramModel.wordEmbedding(inDense)
        modelOut = skipGramModel.endStack(embedOut)
        
        # For consistency, we allow outputs to passed in as
        # sparse tensors. However, we actually want the indices,
        # so we'll just pull it out.
        lossList = []
        for outSparse in outSparseList:
            oneHotIndex = outSparse.indices()[0][0]
            lossList.append( lossFn(modelOut, oneHotIndex) )
        
        # Mean the losses and backprop
        meanLoss = torch.mean( torch.vstack(lossList) )

        print(meanLoss)

        optim.zero_grad()
        meanLoss.backward()
        optim.step()

# This is the same but for cbow training
def cbow_train(cbowModel, dataIterable):
    LR = 5e-3
    optim = torch.optim.SGD(cbowModel.parameters(), lr=LR)
    lossFn = torch.nn.NLLLoss()

    for ins, out in dataIterable:
        # Running through embed layer as a batch
        denseInputs = [inp.to_dense() for inp in ins ]
        denseInBatch = torch.vstack(denseInputs)

        denseEmbedOuts = cbowModel.wordEmbedding(denseInBatch)
        # This matrix stores values that, when element-wise multiplied
        # with the denseEmbedOuts, normalizes each output in the batch
        normMatrix = []

        # L1 normalizing each dimension
        for i in range(denseEmbedOuts.shape[0]):
            l1Factor = torch.linalg.norm(denseEmbedOuts[i], ord=1)
            ones = torch.ones(denseEmbedOuts.shape[-1])
            ones *= l1Factor
            normMatrix.append(ones)

        normMatrix = torch.vstack(normMatrix)
        l1Normed = denseEmbedOuts * normMatrix

        normalizedEmbedMean = torch.mean(l1Normed, dim=-2)

        modelOut = cbowModel.endStack(normalizedEmbedMean)

        # Converting the output "label" to an index
        targetIndex = out.indices()[0][0]

        # Compute losses and backprop
        loss = lossFn(modelOut, targetIndex)
        print(loss)

        optim.zero_grad()
        loss.backward()
        optim.step()
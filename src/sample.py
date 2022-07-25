# Takes in a text document and returns a list of training samples from it;
# the number of samples is determined mostly by the window size paramater
# defined below

import torch.utils.data

import vectorize

# Takes in a list of documents, and yields tuples
# of training tensors: (OneHotInputTensor, [OneHotOutputTensorsInWindow])
class DocumentSampler(torch.utils.data.IterableDataset):
    # Expects a list of documents as inputs, and a window size.
    # Will iterate n times, where n is determined by the number of
    # samples it can return in order.

    # First we're going to vectorize, then we're going to create pairs
    # of tensors based on the window size
    def __init__(self, listOfDocs, vectorizer, windowSize):
        self.listOfSampleTuples = []
        
        vectorizedDocs = vectorizer.docs2Tensors(listOfDocs)

        for docNum in range(len(vectorizedDocs)):
            for tokIndex in range(len(vectorizedDocs[docNum])):
                currentTokenTensor = vectorizedDocs[docNum][tokIndex]
                # We're going to take all tensors within the window size
                # and return them, so here's that list
                listOfWindowTensors = []
                
                # Iterating over window, extract as many points as possible
                for windowIndex in range( max(tokIndex - windowSize, 0), min( len(vectorizedDocs[docNum]), tokIndex + windowSize )):
                    windowTokenTensor = vectorizedDocs[docNum][windowIndex]
                    listOfWindowTensors.append(windowTokenTensor)

                self.listOfSampleTuples.append(
                    ( currentTokenTensor,  listOfWindowTensors )
                )

        # This stores the next index we're going to return. When this index
        # points to an index after the list ends, raise StopIteration
        self.nextIndex = 0

    def __iter__(self):
        return self
    
    def __next__(self):
        # Break if we have no samples left
        if self.nextIndex >= len(self.listOfSampleTuples):
            raise StopIteration
        
        result = self.listOfSampleTuples[self.nextIndex]
        self.nextIndex += 1

        return result
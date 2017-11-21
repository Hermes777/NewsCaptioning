# By now, we should know that pytorch has a functional implementation (as opposed to class version)
# of many common layers, which is especially useful for layers that do not have any parameters.
# e.g. relu, sigmoid, softmax, etc.
from utils import *
from util_lstm import LSTM,entangledLSTMCell
from data_preprocessing import CocoCaptions,customBatchBuilder
import math
import torch.nn.functional as F
import torchvision.models as models


class ImageTextGeneratorModel(nn.Module):
    # The model has three layers: 
    #    1. An Embedding layer that turns a sequence of word ids into 
    #       a sequence of vectors of fixed size: embeddingSize.
    #    2. An RNN layer that turns the sequence of embedding vectors into 
    #       a sequence of hiddenStates.
    #    3. A classification layer that turns a sequence of hidden states into a 
    #       sequence of softmax outputs.
    def __init__(self, vocabularySize,imageFeatureSize,entangled_size):
        super(ImageTextGeneratorModel, self).__init__()
        # See documentation for nn.Embedding here:
        # http://pytorch.org/docs/master/nn.html#torch.nn.Embedding
        self.embedder = nn.Embedding(vocabularySize, 2048)

        self.Image2Embedding = nn.Linear(imageFeatureSize, 2048)
        self.rnn = LSTM(entangledLSTMCell, 2048, 1024 , factor_size=2048, entangled_size=entangled_size, batch_first = False)
        self.classifier = nn.Linear(1024, vocabularySize)
        self.vocabularySize = vocabularySize
        #self.resnet = models.vgg16(pretrained=True)
        #self.resnet = nn.Sequential(*list(self.resnet.children())[:-1])


    # The forward pass makes the sequences go through the three layers defined above.
    def forward(self, vggFeatures, tagFeatures, paddedSeqs):
        
        batchSequenceLength = paddedSeqs.size(0)  # 0-dim is sequence-length-dim.
        batchSize = paddedSeqs.size(1)  # 1-dim is batch dimension.
        
        # Transform word ids into an embedding vector.
        embeddingVectors = self.embedder(paddedSeqs)
        #print Images
        #print('vgg',vggFeatures)

        ImageEmbeddings=self.Image2Embedding(vggFeatures)
        #print('emb',ImageEmbeddings)
        ImageEmbeddings = ImageEmbeddings.view(1,64,2048)
        # Pass the sequence of word embeddings to the RNN.
        embeddingVectors=torch.cat((ImageEmbeddings,embeddingVectors[1:]), 0)
        rnnOutput, finalHiddenState = self.rnn(embeddingVectors,entangler=tagFeatures)
        
        # Collapse the batch and sequence-length dimensions in order to use nn.Linear.
        flatSeqOutput = rnnOutput.view(-1, 1024)
        predictions = self.classifier(flatSeqOutput)
        
        # Expand back the batch and sequence-length dimensions and return. 
        return predictions.view(batchSequenceLength, batchSize, self.vocabularySize), \
               finalHiddenState, embeddingVectors
        


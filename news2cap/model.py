# By now, we should know that pytorch has a functional implementation (as opposed to class version)
# of many common layers, which is especially useful for layers that do not have any parameters.
# e.g. relu, sigmoid, softmax, etc.
from utils import *
from util_lstm import LSTM,entangledLSTMCell,LSTMCell
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
        self.embedder = nn.Embedding(vocabularySize, 300)

        self.Image2Embedding = nn.Linear(imageFeatureSize, 300)
        self.rnn = LSTM(entangledLSTMCell, 300, 1024 , factor_size=1024, entangled_size=entangled_size,num_layers=1, batch_first = False)
        self.classifier = nn.Linear(1024, vocabularySize)
        self.vocabularySize = vocabularySize
        #self.resnet = models.vgg16(pretrained=True)
        #self.resnet = nn.Sequential(*list(self.resnet.children())[:-1])


    # The forward pass makes the sequences go through the three layers defined above.
    def forward(self, vggFeatures, tagFeatures, paddedSeqs, initialHiddenState=None, initialCellState=None, initialImage=0):
        
        batchSequenceLength = paddedSeqs.size(0)  # 0-dim is sequence-length-dim.
        batchSize = paddedSeqs.size(1)  # 1-dim is batch dimension.
        
        # Transform word ids into an embedding vector.
        embeddingVectors = self.embedder(paddedSeqs)
        #print Images
        #print('vgg',vggFeatures)

        ImageEmbeddings=self.Image2Embedding(vggFeatures)
        
        ImageEmbeddings = ImageEmbeddings.view(1,batchSize,300)
        # Pass the sequence of word embeddings to the RNN.
        if initialImage==0:
            embeddingVectors=torch.cat((ImageEmbeddings,embeddingVectors[1:]), 0)
        elif initialImage==1:
            embeddingVectors=ImageEmbeddings
        else:
            embeddingVectors=embeddingVectors
        #print(initialHiddenState.data.shape)
        if not (initialHiddenState is None):
            rnnOutput, (finalHiddenState, finalCellState) = self.rnn(embeddingVectors,hx=(initialHiddenState,initialCellState),entangler=tagFeatures)
        else:
            rnnOutput, (finalHiddenState, finalCellState) = self.rnn(embeddingVectors,entangler=tagFeatures)
        
        # Collapse the batch and sequence-length dimensions in order to use nn.Linear.
        #print(rnnOutput)
        flatSeqOutput = rnnOutput.view(-1, 1024)
        predictions = self.classifier(flatSeqOutput)
        #print('classifier',self.classifier.weight)
        
        # Expand back the batch and sequence-length dimensions and return. 
        return predictions.view(batchSequenceLength, batchSize, self.vocabularySize), \
               (finalHiddenState, finalCellState)
'''
# Let's test the model on some input batch.
f_aritcle=open('data/article_features/IND_dict.pickle',"rb")
tag_features=pickle.load(f_aritcle)
#print(tag_features)

f_image=open('data/image_features/vggfeatures-IND.pickle',"rb")
vgg_features=pickle.load(f_image)
print(len(vgg_features['6bd6ca984a47ea8f9a74e87e465cc50df155e77f']))

# Let's test the data class.
trainData = CocoCaptions(['data/captions/IND-JSON/IND_Partial_0.jsonld','data/captions/IND-JSON/IND_Partial_1.jsonld','data/captions/IND-JSON/IND_Partial_2.jsonld'],tag_features=tag_features,img_features=vgg_features)
print('Number of training examples: ', len(trainData))


# It would be a mistake to build a vocabulary using the validation set so we reuse.
valData = CocoCaptions(['data/captions/IND-JSON/IND_Partial_3.jsonld'],tag_features=tag_features,img_features=vgg_features, vocabulary = trainData.vocabulary)
print('Number of validation examples: ', len(valData))


# Data loaders in pytorch can use a custom batch builder, which we are using here.
trainLoader = data.DataLoader(trainData, batch_size = 1, 
                              shuffle = True, num_workers = 0,
                              collate_fn = customBatchBuilder)
valLoader = data.DataLoader(valData, batch_size = 1, 
                            shuffle = False, num_workers = 0,
                            collate_fn = customBatchBuilder)

# Now let's try using the data loader.
index, (imgIds, Tags, Imgs, paddedSeqs, seqLengths) = next(enumerate(trainLoader))



vocabularySize = len(trainData.vocabulary['word2id'])
model = ImageTextGeneratorModel(vocabularySize,4096,4096)
#print model
print("start")
model.eval()
# Create the initial hidden state for the RNN.
index, (imgIds, Tags, Imgs, paddedSeqs, seqLengths) = next(enumerate(trainLoader))
print(len(Tags))
Imgs=Variable(torch.from_numpy(np.array(Imgs)).float())
Tags=Variable(torch.from_numpy(np.array(Tags)).float())
predictions, _,embed = model(Imgs, Tags,torch.autograd.Variable(paddedSeqs))

print('outputs', predictions.size()) # 10 output softmax predictions over our vocabularySize outputs.

print('Here is the model:')
print(model)
'''

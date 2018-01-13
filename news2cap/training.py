
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator, FormatStrFormatter
from model import ImageTextGeneratorModel
from data_preprocessing import CocoCaptions,customBatchBuilder
from utils import *
from util_lstm import *
from tqdm import tqdm
from demo import sample_sentence
import pickle

def get_cfidf(trainData):
    fi=open('data/article_features/dict_document.pickle',"rb")
    documents=pickle.load(fi)
    
    cfidf={}
    df={}
    cap_freq=[0]*1000


    for i,seq in enumerate(trainData.annotations):
        if not (trainData.ids[i]+'.txt' in documents):
            continue
        for word in documents[trainData.ids[i]+'.txt'].split():
            if word.lower() in trainData.vocabulary['word2id'] and not (trainData.vocabulary['word2id'][word.lower()] in df):
                df[trainData.vocabulary['word2id'][word.lower()]]=1
            elif word.lower() in trainData.vocabulary['word2id']:
                df[trainData.vocabulary['word2id'][word.lower()]]+=1
        for word in list(set(documents[trainData.ids[i]+'.txt'].split())):
            if word in seq:
                if word.lower() in trainData.vocabulary['word2id'] and not (trainData.vocabulary['word2id'][word.lower()] in cfidf):
                    cfidf[trainData.vocabulary['word2id'][word.lower()]]=1
                elif word.lower() in trainData.vocabulary['word2id']:
                    cfidf[trainData.vocabulary['word2id'][word.lower()]]+=1
    for y in cfidf:
        cfidf[y]=cfidf[y]/df[y]
    for seq in trainData.annotations:
        cap_freq[len(seq.split())]+=1.0
    for i in range(200):
        cap_freq[i]=cap_freq[i]/len(trainData.annotations)+0.000001

                

    return documents,cfidf,cap_freq



def train_lstm_model(trainData, model, criterion, optimizer, trainLoader, valLoader, n_epochs = 1, use_gpu = True):
    min_loss = 10000

    trainLoss = open('train.csv', 'w')
    testLoss = open('test.csv', 'w')
    if use_gpu:
        model = model.cuda()
        #model = torch.nn.DataParallel(model)
        criterion = criterion.cuda()
        
    train_loss=[]
    
    _loss=[]

    documents,cfidf,cap_freq=get_cfidf(trainData)    
# Training loop.
    for epoch in range(0, n_epochs):
        correct = 0.0
        cum_loss = 0.0
        counter = 0
        
        # Make a pass over the training data.
        t = tqdm(trainLoader, desc = 'Training epoch %d' % epoch)
        model.train()  # This is important to call before training!
        for (i, (imgIds, Tags, Imgs, paddedSeqs, seqLengths)) in enumerate(t):
 
            # Wrap inputs, and targets into torch.autograd.Variable types.
            inputs = Variable(paddedSeqs[:-1])
            Imgs=Variable(torch.from_numpy(np.array(Imgs)).float())
            Tags=Variable(torch.from_numpy(np.array(Tags)).float())
            #labels = torch.Tensor(paddedSeqs.size(0)-1, paddedSeqs.size(1), 5003).zero_()
            #print labels
            labels = Variable(paddedSeqs[1:])

            # Forward pass:
            if use_gpu:
                
                inputs = inputs.cuda()
                labels = labels.cuda()
                Imgs = Imgs.cuda()
                Tags = Tags.cuda()

            
            outputs,(endhid,endc) = model(Imgs, Tags,inputs)
            #print('output',outputs)
            #print('endhid',endhid)
            #print('endc',endc)
            loss = Variable(torch.Tensor(1).zero_())
            if use_gpu:
                loss = loss.cuda()
            for (output,label) in zip(outputs,labels):
                loss = loss+criterion(output, label)
                if i%30==0: 
                    print(torch.exp(output[2][label[2]])/torch.sum(torch.exp(output[2])))
            if i%30==0:
                print(sample_sentence(i,model,trainData,documents,cfidf,cap_freq,use_cuda = True))

            
            # Backward pass:
            optimizer.zero_grad()
            # Loss is a variable, and calling backward on a Variable will
            # compute all the gradients that lead to that Variable taking on its
            # current value.
            loss.backward() 

            # Weight and bias updates.
            optimizer.step()

            # logging information.
            cum_loss += loss.data[0]
            #max_scores, max_labels = outputs.data.max(1)
            #correct += (max_labels == labels.data).sum()
            t.set_postfix(loss = cum_loss / (1 + i))
            
        train_loss.append(cum_loss/ (i + 1))
        
        trainLoss.write('{},{}\n'.format(epoch, cum_loss / (i+1)))
        trainLoss.flush()
        # Make a pass over the validation data.
        correct = 0.0
        cum_loss = 0.0
        counter = 0
        t = tqdm(valLoader, desc = 'Validation epoch %d' % epoch)
        test_loss=[]
        model.eval()  # This is important to call before evaluating!
        for (i, (imgIds, Tags, Imgs, paddedSeqs, seqLengths)) in enumerate(t):

            # Wrap inputs, and targets into torch.autograd.Variable types.
            inputs = Variable(paddedSeqs[:-1])
            labels = Variable(paddedSeqs[1:])
            Imgs=Variable(torch.from_numpy(np.array(Imgs)).float())
            Tags=Variable(torch.from_numpy(np.array(Tags)).float())

            if use_gpu:

                inputs = inputs.cuda()
                labels = labels.cuda()
                Imgs = Imgs.cuda()
                Tags = Tags.cuda()

            #net = torch.nn.DataParallel(model, device_ids=[3, 4, 5])
         
            outputs,(endhid,endc) = model(Imgs, Tags,inputs)
            loss = Variable(torch.Tensor(1).zero_())
            if use_gpu:
                loss = loss.cuda()
            for (output,label) in zip(outputs,labels):
                loss = loss+criterion(output, label)


            
            # logging information.
            cum_loss += loss.data[0]
            t.set_postfix(loss = cum_loss / (1 + i))
            
            #plt.figure(0)
        _loss.append(cum_loss/ (i + 1))
        print(cum_loss/ (i + 1))
        if((cum_loss/ (i + 1)) < min_loss):
            print("get new model")
            torch.save(model.state_dict(), 'checkpoint.pth.tar')
            """
            torch.save(the_model.state_dict(), PATH)
            the_model = TheModelClass(*args, **kwargs)
            the_model.load_state_dict(torch.load(PATH))
            """

            min_loss = cum_loss/ (i + 1)
        test_loss.append(cum_loss/ (i + 1))
        
        testLoss.write('{},{}\n'.format(epoch, cum_loss /(i+1)))
        testLoss.flush()
        
    
    majorLocator = MultipleLocator(5)
    
    trainLoss.close()
    testLoss.close()
    """
    

    plt.figure(0)
    fig, ax = plt.subplots()
    plt.title("trend of loss")
    plt.plot(range(len(train_loss)),train_loss,'r',label='train set')
    plt.plot(range(len(_loss)),_loss,'g',label='valid set')
    ax.xaxis.set_major_locator( majorLocator )
    ax.xaxis.set_label_text("iteration")
    ax.yaxis.set_label_text("loss")

    plt.legend(loc='best')
    plt.show()
    """
class MacOSFile(object):

    def __init__(self, f):
        self.f = f

    def __getattr__(self, item):
        return getattr(self.f, item)

    def read(self, n):
        # print("reading total_bytes=%s" % n, flush=True)
        if n >= (1 << 31):
            buffer = bytearray(n)
            idx = 0
            while idx < n:
                batch_size = min(n - idx, 1 << 31 - 1)
                # print("reading bytes [%s,%s)..." % (idx, idx + batch_size), end="", flush=True)
                buffer[idx:idx + batch_size] = self.f.read(batch_size)
                # print("done.", flush=True)
                idx += batch_size
            return buffer
        return self.f.read(n)

    def write(self, buffer):
        n = len(buffer)
        print("writing total_bytes=%s..." % n, flush=True)
        idx = 0
        while idx < n:
            batch_size = min(n - idx, 1 << 31 - 1)
            print("writing bytes [%s, %s)... " % (idx, idx + batch_size), end="", flush=True)
            self.f.write(buffer[idx:idx + batch_size])
            print("done.", flush=True)
            idx += batch_size


def pickle_dump(obj, file_path):
    with open(file_path, "wb") as f:
        return pickle.dump(obj, MacOSFile(f), protocol=pickle.HIGHEST_PROTOCOL)


def pickle_load(file_path):
    with open(file_path, "rb") as f:
        return pickle.load(MacOSFile(f))

def main():
    
    
    

    # Let's test the model on some input batch.
    #f_aritcle=open('data/article_features/dict.pickle',"rb")
    tag_features=pickle_load('data/article_features/1024_dimension_less_than_100_words_WP-articles_dict.pickle')
    #print(tag_features)

    f_image=open('data/image_features/vggfeatures-IND.pickle',"rb")
    vgg_features=pickle.load(f_image)
    f_image=open('data/image_features/vggfeatures-WP.pickle',"rb")
    vgg_features1=pickle.load(f_image)
    vgg_features=dict(vgg_features, **vgg_features1)
    print('Number of tag examples: ', len(tag_features))
    print('Number of vgg examples: ', len(vgg_features))

    # Let's test the data class.
    trainData = CocoCaptions(['data/captions/IND-JSON/IND_Partial_0.jsonld','data/captions/IND-JSON/IND_Partial_1.jsonld','data/captions/IND-JSON/IND_Partial_2.jsonld',\
        'data/captions/WP-JSON/WP_Partial_0.jsonld','data/captions/WP-JSON/WP_Partial_1.jsonld','data/captions/WP-JSON/WP_Partial_2.jsonld','data/captions/WP-JSON/WP_Partial_3.jsonld'\
        ,'data/captions/WP-JSON/WP_Partial_4.jsonld','data/captions/WP-JSON/WP_Partial_5.jsonld','data/captions/WP-JSON/WP_Partial_6.jsonld','data/captions/WP-JSON/WP_Partial_7.jsonld'\
        ,'data/captions/WP-JSON/WP_Partial_8.jsonld','data/captions/WP-JSON/WP_Partial_9.jsonld','data/captions/WP-JSON/WP_Partial_10.jsonld','data/captions/WP-JSON/WP_Partial_11.jsonld'\
        ,'data/captions/WP-JSON/WP_Partial_12.jsonld','data/captions/WP-JSON/WP_Partial_13.jsonld'],tag_features=tag_features,img_features=vgg_features)
    print('Number of training examples: ', len(trainData))



    # It would be a mistake to build a vocabulary using the validation set so we reuse.
    valData = CocoCaptions(['data/captions/IND-JSON/IND_Partial_3.jsonld'],tag_features=tag_features,img_features=vgg_features, vocabulary = trainData.vocabulary)
    print('Number of validation examples: ', len(valData))
    

    del tag_features
    del vgg_features
    # Data loaders in pytorch can use a custom batch builder, which we are using here.
    trainLoader = data.DataLoader(trainData, batch_size = 128, 
                                  shuffle = True, num_workers = 0,
                                  collate_fn = customBatchBuilder)
    valLoader = data.DataLoader(valData, batch_size = 128, 
                                shuffle = False, num_workers = 0,
                                collate_fn = customBatchBuilder)
    vocabularySize = len(trainData.vocabulary['word2id'])
    model = ImageTextGeneratorModel(vocabularySize,4096,1024)
    model = model.cuda()
    criterion = nn.CrossEntropyLoss().cuda()
    optimizer = torch.optim.Adam(model.parameters(), lr = 0.0005)

    # Train the previously defined model.
    train_lstm_model(trainData, model, criterion, optimizer, trainLoader, valLoader, n_epochs = 10, use_gpu = True)
    
    #loaded_model = pickle.load(open(filename, 'rb'))

if __name__ == '__main__':
    main()



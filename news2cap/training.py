import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator, FormatStrFormatter
from model import ImageTextGeneratorModel
from data_preprocessing import CocoCaptions,customBatchBuilder
from utils import *
from util_lstm import *
from tqdm import tqdm
import pickle




def train_lstm_model(model, criterion, optimizer, trainLoader, valLoader, n_epochs = 1, use_gpu = True):
    min_loss = 10000

    trainLoss = open('train.csv', 'w')
    testLoss = open('test.csv', 'w')
    if use_gpu:
        model = model.cuda()
        #model = torch.nn.DataParallel(model)
        criterion = criterion.cuda()
        
    train_loss=[]
    
    _loss=[]
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

            
            outputs,endhid,endc = model(Imgs, Tags,inputs)
            loss = Variable(torch.Tensor(1).zero_())
            if use_gpu:
                loss = loss.cuda()
            for (output,label) in zip(outputs,labels):
                loss = loss+criterion(output, label)

            
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
        
        trainLoss.write('{},{}\n'.format(epoch, cum_loss / (n_epochs)))
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
         
            outputs,endhid,endc = model(Imgs, Tags,inputs)
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
        
        testLoss.write('{},{}\n'.format(epoch, cum_loss / (n_epochs)))
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

def main():
    
    
    

    # Let's test the model on some input batch.
    f_aritcle=open('data/article_features/IND_dict.pickle',"rb")
    tag_features=pickle.load(f_aritcle)
    #print(tag_features)

    f_image=open('data/image_features/vggfeatures-IND.pickle',"rb")
    vgg_features=pickle.load(f_image)

    # Let's test the data class.
    trainData = CocoCaptions(['data/captions/IND-JSON/IND_Partial_0.jsonld','data/captions/IND-JSON/IND_Partial_1.jsonld','data/captions/IND-JSON/IND_Partial_2.jsonld'],tag_features=tag_features,img_features=vgg_features)
    print('Number of training examples: ', len(trainData))


    # It would be a mistake to build a vocabulary using the validation set so we reuse.
    valData = CocoCaptions(['data/captions/IND-JSON/IND_Partial_3.jsonld'],tag_features=tag_features,img_features=vgg_features, vocabulary = trainData.vocabulary)
    print('Number of validation examples: ', len(valData))
    

    # Data loaders in pytorch can use a custom batch builder, which we are using here.
    trainLoader = data.DataLoader(trainData, batch_size = 64, 
                                  shuffle = True, num_workers = 0,
                                  collate_fn = customBatchBuilder)
    valLoader = data.DataLoader(valData, batch_size = 64, 
                                shuffle = False, num_workers = 0,
                                collate_fn = customBatchBuilder)
    vocabularySize = len(trainData.vocabulary['word2id'])
    model = ImageTextGeneratorModel(vocabularySize,4096,4096)
    model = model.cuda()
    criterion = nn.CrossEntropyLoss().cuda()
    optimizer = torch.optim.Adam(model.parameters(), lr = 0.0005)

    # Train the previously defined model.
    train_lstm_model(model, criterion, optimizer, trainLoader, valLoader, n_epochs = 10, use_gpu = True)
    
    #loaded_model = pickle.load(open(filename, 'rb'))

if __name__ == '__main__':
    main()


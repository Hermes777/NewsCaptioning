from utils import *

class CocoCaptions(data.Dataset):
    # Load annotations in the initialization of the object.
    def __init__(self, captionsFiles, tag_features, img_features, batchsize = 64, vocabulary = None):
        self.ids=[]
        self.Imgs=[]
        self.Tags=[]
        self.annotations=[]
        for captionsFile in captionsFiles:
            self.data = json.load(open(captionsFile))
            for x in self.data['@reverse']['publisher']:
                if 'image' not in x:
                    continue
                #print(x['image'])
                if x['@id']==None or (x['@id'] not in img_features) or (x['@id']+".txt" not in tag_features):
                    continue
                if (x['headline'] is None) or len(x['headline'])<2:
                    continue
                #if '/' in x['image']['caption']:
                #    continue
                self.annotations.append(x['headline'])
                self.Imgs.append(img_features[x['@id']])
                self.Tags.append(tag_features[x['@id']+".txt"])
                self.ids.append(x['@id'])
        #print(self.annotations)
        
        # Build a vocabulary if not provided.
        _len=len(self.annotations)//batchsize*batchsize
        self.annotations=self.annotations[:_len]
        self.Imgs=self.Imgs[:_len]
        self.Tags=self.Tags[:_len]
        self.ids=self.ids[:_len]
        if not vocabulary:
            self.build_vocabulary()
        else:
            self.vocabulary = vocabulary
        
    # Build a vocabulary using the top 50000 words.
    def build_vocabulary(self, vocabularySize = 12000):
        # Count words, this will take a while.
        word_counter = dict()
        for annotation in self.annotations:
            if annotation[0]=='\"' or annotation[-1]=='\"':
                annotation = annotation[1:-1];
            words = word_tokenize(annotation.lower())
            if "\'" in words:
                words.remove("\'")
            if "\"" in words:
                words.remove("\"")
            if "\\" in words:
                words.remove("\\")
            if "/" in words:
                words.remove("/")
            if "\"\"" in words:
                words.remove("\"\"")
            if "\'\'" in words:
                words.remove("\'\'")
            if "``" in words:
                words.remove("``")
            for word in words:
                word_counter[word] = word_counter.get(word, 0) + 1
                
        # Sort the words and find keep only the most frequent words.
        sorted_words = sorted(list(word_counter.items()), 
                              key = lambda x: -x[1])
        most_frequent_words = [w for (w, c) in sorted_words[:vocabularySize]]
        word2id = {w: (index + 1) for (index, w) in enumerate(most_frequent_words)}#a dict that key is word and value is index.
        
        # Add a special characters for START, END sentence, and UNKnown words.
        word2id['[END]'] = 0
        word2id['[START]'] = len(word2id)
        word2id['UNK'] = len(word2id)
        id2word = {index: w for (w, index) in word2id.items()}#a dict that key is index and value is word.
        self.vocabulary = {'word2id': word2id, 'id2word': id2word}
    
    # Transform a caption into a list of word ids.
    def caption2ids(self, caption):
        word2id = self.vocabulary['word2id']
        words=word_tokenize(caption.lower())
        if "\'" in words:
            words.remove("\'")
        if "\"" in words:
            words.remove("\"")
        if "\"\"" in words:
            words.remove("\"\"")
        if "\'\'" in words:
            words.remove("\'\'")
        if "``" in words:
            words.remove("``")
            
        caption_ids = [word2id.get(w, word2id['UNK']) for w in words]
        #print("UNK id: ",word2id['UNK'])
        
        #print("words:", words)
        #if word2id['UNK'] in caption_ids: 
            
        #    caption_ids = [x for x in caption_ids if x != word2id['UNK']]
            
            
        caption_ids.insert(0, word2id['[START]'])
        caption_ids.append(word2id['[END]'])
        #print("after",caption_ids)
        return torch.LongTensor(caption_ids)
    
    # Transform a list of word ids into a caption.
    def ids2caption(self, caption_ids):
        id2word = self.vocabulary['id2word']
        return " ".join([id2word[w] for w in caption_ids])
    
    # Return imgId, and a random caption for that image.
    def __getitem__(self, index):
        annotation = self.annotations[index]
        return  self.ids[index],self.Tags[index],self.Imgs[index],self.caption2ids(annotation)
    
    # Return the number of elements of the dataset.
    def __len__(self):
        return len(self.annotations)
'''
# Let's test the data class.
trainData = CocoCaptions('data/IND-JSON/IND_Partial_0.jsonld')
print('Number of training examples: ', len(trainData))

# It would be a mistake to build a vocabulary using the validation set so we reuse.
valData = CocoCaptions('data/IND-JSON/IND_Partial_1.jsonld', vocabulary = trainData.vocabulary)
print('Number of validation examples: ', len(valData))

# Print a sample from the training data.
_id,caption = trainData[0]
print('caption', caption.tolist())
print('captionString', trainData.ids2caption(caption))
'''
# The batch builder will pack all sequences of different length into a single tensor by 
# padding shorter sequences with a padding token.
def customBatchBuilder(samples):
    imgIds, Tags, Imgs ,captionSeqs = zip(*samples)
    
    # Sort sequences based on length.
    seqLengths = [len(seq) for seq in captionSeqs]
    maxSeqLength = max(seqLengths)
    sorted_list = sorted(zip(list(imgIds),np.array(Tags).tolist(),np.array(Imgs).tolist(), captionSeqs, seqLengths), key = lambda x: -x[4])
    imgIds, Tags,Imgs, captionSeqs, seqLengths = zip(*sorted_list)
    
    # Create tensor with padded sequences.
    paddedSeqs = torch.LongTensor(len(imgIds), maxSeqLength)
    paddedSeqs.fill_(0)
    for (i, seq) in enumerate(captionSeqs):
        paddedSeqs[i, :len(seq)] = seq
    return imgIds,Tags,Imgs, paddedSeqs.t(), seqLengths
'''
# Data loaders in pytorch can use a custom batch builder, which we are using here.
trainLoader = data.DataLoader(trainData, batch_size = 128, 
                              shuffle = True, num_workers = 0,
                              collate_fn = customBatchBuilder)
valLoader = data.DataLoader(valData, batch_size = 128, 
                            shuffle = False, num_workers = 0,
                            collate_fn = customBatchBuilder)

# Now let's try using the data loader.
index, (imgIds, paddedSeqs, seqLengths) = next(enumerate(trainLoader))
print('imgIds', imgIds)
print('paddedSequences', paddedSeqs.size())
print('seqLengths', seqLengths)
'''

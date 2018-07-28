
# coding: utf-8

# # Kommasetzung by Till Nöllgen
# 
# With this little program we try to predict at which positions in any kind of german text there is a comma. 
# For this we use a LSTM network that we train with random free available german books online. In the hope that they properly use the comma rules. For better readablility I wrote the program in this jupyter notebook, because it lets me combine text and code in the same sheet as well print out preliminary statuses. The project was inspired from previouse code from Udacity's Deep Learning Nanodegree and variouse scientific papers about sentiment analysis and word2vec - research.

# In[1]:


import os
import io
import csv
import sys
import time
import re
import string
from collections import Counter
import numpy as np
import tensorflow as tf
import nltk
nltk.download('punkt')


# Make sure to use Tensorflow 1.0.x

# In[2]:


print(tf.VERSION)


# # Load dataset
# Data set is made out of 12 online free available books (Like for example "Harry Potter - Der Stein der Weisen")

# In[3]:


with open('wikiextractor/text/books/untitled.txt', "r", encoding="utf-8") as text:
    text = text.read()
    
### Posibility to shrink down dataset for faster processing if wanted!!!
text = text[:]

#Minimal dataset cleaning by removing the page numbers as well single standing words 
text = re.sub(r'\n[0-9]+ \n', '', text)
text = re.sub(r'\n\n[a-zA-Z] \n\n', '', text)

print(text[:100])


# # Tokenizing into words

# In[4]:


#TOKENIZING INTO WORDS
from nltk.tokenize import word_tokenize

woerter = word_tokenize(text)
print(woerter[:50])


# # Stemming

# In[5]:


#STEMMING
x = (len(set(woerter)))
sno = nltk.stem.SnowballStemmer('german')
woerter_stem = [sno.stem(wort) for wort in woerter]
y = (len(set(woerter_stem)))

print(str((x-y)*100/(x)) +" percent of the words saved by using stemming")


# In[6]:


counts = Counter(woerter_stem)
print(counts.most_common(20))


# Actually here you can see how many commas are in the training data. Normally it is around every 10th word...

# # Part of speech 
# 
# In this project the "TIGER Corpus Release 2.1" of the university Stuttgart was used. Make sure to adjust the root of the corpus depending where it is safed (unzipped 60 MB)
# 
# http://www.ims.uni-stuttgart.de/forschung/ressourcen/korpora/TIGERCorpus/download/start.html
# 
# https://datascience.blog.wzb.eu/2016/07/13/accurate-part-of-speech-tagging-of-german-texts-with-nltk/
# 
# 

# In[7]:


# GERMAN WORD CLASSES / LEXICAL CATEGORIES
from ClassifierBasedGermanTagger.ClassifierBasedGermanTagger import ClassifierBasedGermanTagger
#from nltk.tag import UnigramTagger

'''
    WORDS = 'words'   #: column type for words
    POS = 'pos'       #: column type for part-of-speech tags
    TREE = 'tree'     #: column type for parse trees
    CHUNK = 'chunk'   #: column type for chunk structures
    NE = 'ne'         #: column type for named entities
    SRL = 'srl'       #: column type for semantic role labels
    IGNORE = 'ignore' #: column type for column that should be ignored

    #: A list of all column types supported by the conll corpus reader.
    COLUMN_TYPES = (WORDS, POS, TREE, CHUNK, NE, SRL, IGNORE)
'''
#Choose the root 
root = '/output/' #For a server like FloydHub
#root = '/Users/Till/Dropbox/Deep Learning Udacity/deep-learning/KommasetzungAusweichordner'
fileid = 'tiger_release_aug07.corrected.16012013.conll09'
columntypes = ['ignore', 'words', 'ignore', 'ignore', 'pos']

#Load corpus
corp = nltk.corpus.ConllCorpusReader(root, fileid, columntypes, encoding='utf8')

#Train on whole corpus (normally it has a accuracy around 94% - 96% depending on text)
tagger = ClassifierBasedGermanTagger(train=corp.tagged_sents())
#tagger = UnigramTagger(corp.tagged_sents())


# In[8]:


#Part of speech tagging on data set
idx = int(len(woerter)*1.0)
w_classes = tagger.tag(woerter[:idx])


# In[9]:


print(w_classes[:10])


# In[10]:


woerter_classes = []

for word in w_classes:
    if "$" in str(word[1]):
        word_class = word[0] + "c"
    else:
        word_class = word[1]
    woerter_classes.append(word_class)


# In[11]:


print(woerter_classes[:30])


# # Transform words into numbers

# In[12]:


vocab = sorted(counts, key=counts.get, reverse=True)

counts_classes = Counter(woerter_classes)
vocab_classes = sorted(counts_classes, key=counts_classes.get, reverse=True)

vocabs = vocab + vocab_classes
print(str(len(vocabs)) +" unique words/lemmas in the lookup table")

vocabs_to_int = {wort: i for i, wort in enumerate(vocabs, 1)}
int_to_vocabs = {wort: i for i, wort in vocabs_to_int.items()}

# Add "None" key for later unknown words
i = len(int_to_vocabs)+1
w = int_to_vocabs[i]

del vocabs_to_int[w]
vocabs_to_int["None"] = i

int_to_vocabs[i] = "None"


# In[13]:


print(len(vocabs_to_int))
print(int_to_vocabs[62341])
print(int_to_vocabs[62340])

print(vocabs_to_int["None"])


# # Setting labels
# 
# If word is a comma, then the label is "1" otherwise "0"

# In[14]:


labels = []
#for idx, wort in enumerate(woerter,0):
for wort in woerter:
    if wort == ",":
        labels.append(1)
        #labels.append([1, wort])
    else:
        labels.append(0)
        #labels.append([0, wort])    

print(labels[:40])
print(woerter[:40])


# # Removing Commas
# Let's remove comma placeholder in the labels and the text: That means a word after a comma now will become the label "1" instead of "0" and one "0" will be removed as well as all comma and comma synonyms will be removed.

# In[15]:


labels_indicator = np.roll(labels, 1)
labels = [labels[i] for i in range(0,len(labels)) if labels_indicator[i] != 1]
labels = np.asarray(labels)

woerter = [woerter[i] if labels_indicator[i] != 1 else (", " + woerter[i]) for i in range(0,len(woerter))]
woerter = [wort for wort in woerter if wort!= ","]
woerter_stem = [wort for wort in woerter_stem if wort!= ","]
woerter_classes = [wort for wort in woerter_classes if wort!= ",c"]


# # Transform text to numbers

# In[16]:


woerter_ints = [vocabs_to_int[wort] if wort in vocabs_to_int else vocabs_to_int["None"] for wort in woerter_stem]
woerter_classes_ints= [vocabs_to_int[wort] if wort in vocabs_to_int else vocabs_to_int["None"] for wort in woerter_classes]

print(woerter_ints[:10])
print(woerter_classes_ints[:10])


# # Making features
# 
# ### So what the function actually does:
# 
# You feed in the index of the word and get out a list of 20 digits (depends on the window size). The first ten digits are the ten stemmed words behind the index-word/comma. The second ten digits are the same ten words behind the index-word/comma but representing the word classification (also know as Lemmatization). In the beginning I also experimented by feeding in words that stand in front of the index-word/comma. But actually the network already has "seen" those words and "interpreted" the most important informations of it (by the way a LSTM-network works). I also experimented with randomizing the window size based on a idea of [Mikolov et al.](https://arxiv.org/pdf/1301.3781.pdf) (word to vector research). But no positive effect could be seen, so I simplified it again.

# In[17]:


def get_batch(woerter_ints, idx, window_size=10): # woerter_ints should be woerter_ints or woerter_classes_ints 
                                  
    
    #r = np.random.randint(5, window_size+1)
    r = window_size
    
    
    #lower Boundaries for the beginning of the data set
    if idx - r < 0:
        minus_idx = 0
    else:
        minus_idx = idx - r
        
    #upper Boundaries for the ending of the data set   
    if idx + r > len(woerter_ints):
        plus_idx = len(woerter_ints)
    else:
        plus_idx = idx + r
        
        
    davor = woerter_ints[minus_idx:idx]
    danach = woerter_ints[idx:plus_idx]
    
    
    #Cut of at a fullstop (Isolate sentences)
    for ii in range(0,len(davor)):
        if davor[ii] == vocabs_to_int["."] or davor[ii] == vocabs_to_int[".c"]:
            davor = davor[ii+1:]
            break
    
    for ii in range(0,len(danach)):
        if danach[ii] == vocabs_to_int["."] or danach[ii] == vocabs_to_int[".c"]:
            danach = danach[:ii+1]
            break
    
    
    x_davor = ([0]*(window_size-len(davor))) + davor
    x_danach = danach + ([0]*(window_size-len(danach)))
    
    x = x_davor + x_danach
    #x = x_danach
    
    return x
        


# In[18]:


features = []
feature_woerter = [get_batch(woerter_ints, idx, window_size=10) for idx in range(0,len(labels))]
feature_classes =  [get_batch(woerter_classes_ints, idx, window_size=10) for idx in range(0,len(labels))]
features = list(zip(feature_woerter, feature_classes))
features = np.asarray(features).reshape(len(labels), -1)

print(features.shape)


# In[19]:


print(woerter[20:30])
print(features[20:30])


# # Splitting into training, validation & test set
# 95% of the dataset is getting used for training. 2.5% will be used for validation in between and 2.5% for testing in the end

# In[20]:


split_idx = int(len(features)*0.95)

print(features.shape)
print(labels.shape)

train_x, val_x = features[:split_idx], features[split_idx:]
train_y, val_y = labels[:split_idx], labels[split_idx:]
train_woerter, val_woerter = woerter_stem[:split_idx], woerter_stem[split_idx:]

test_idx = int(len(val_x)*0.5)
val_x, test_x = val_x[:test_idx], val_x[test_idx:]
val_y, test_y = val_y[:test_idx], val_y[test_idx:]
val_woerter, test_woerter = val_woerter[:test_idx], val_woerter[test_idx:]

# For a better detection if we run into overfitting we want to compare the training accuracy (and not the training loss) 
# directly with the with the validation accuracy
# For that we need a smaller training subset with preferably the same length as the validation set

train_acc_x = features[-test_idx:]
train_acc_y = labels[-test_idx:]

print("\t\t\tFeature Shapes:")
print("Train set: \t\t{}".format(train_x.shape), 
      "\nValidation set: \t{}".format(val_x.shape),
      "\nTest set: \t\t{}".format(test_x.shape))


# # Build the graph
# Here, we'll build the graph. First up, defining the hyperparameters.
# 
# lstm_size: Number of units in the hidden layers in the LSTM cells. Started with 128 but scaled it up to 256 and then 512, because of better performance
# 
# lstm_layers: Number of LSTM layers in the network. I'm using 3 because it seems to be a reasonable number for such a task
# 
# batch_size: The number of reviews to feed the network in one training pass. Higher means faster, but also you can get stuck in local minima. 128 & 256 seemed to be a good trade off
# 
# learning_rate: Learning rate was in the beginning 0.0005 (inspired by word-to-vec) but I lowered it to 0.0003 because of better performance

# In[21]:


lstm_size = 512 # 512
lstm_layers = 3 # 3
batch_size = 512 # 256
learning_rate = 0.0003 #0.0003


# We'll also be using dropout on the LSTM layer, so we'll make a placeholder for the keep probability.

# In[22]:


n_words = (len(vocabs_to_int)) + 1 # Adding 1 because we use 0's for padding, dictionary started at 1


# Create the graph object
graph = tf.Graph()
# Add nodes to the graph
with graph.as_default():
    inputs_ = tf.placeholder(tf.int32, [None, None], name='inputs')
    labels_ = tf.placeholder(tf.int32, [None, None], name='labels')
    keep_prob = tf.placeholder(tf.float32, name='keep_prob')


# # Embedding
# 
# Now we'll add an embedding layer. We need to do this because there are thousands of words in our vocabulary. It is massively inefficient to one-hot encode our classes here (learned from word2vec) Instead of one-hot encoding, we can have an embedding layer and use that layer as a lookup table.

# In[23]:


# Size of the embedding vectors (number of units in the embedding layer)
embed_size = 300 #300

with graph.as_default():
    embedding = tf.Variable(tf.truncated_normal((n_words, embed_size), stddev = 0.3))
    #embedding = tf.Variable(tf.random_uniform((n_words, embed_size), -1, 1))
    embed = tf.nn.embedding_lookup(embedding, inputs_)


# In[24]:


with graph.as_default():
    # The basic LSTM cell
    lstm = tf.contrib.rnn.BasicLSTMCell(lstm_size)
    
    # Add dropout to the cell
    drop = tf.contrib.rnn.DropoutWrapper(lstm, output_keep_prob=keep_prob)
    
    # Stack up multiple LSTM layers, for deep learning
    cell = tf.contrib.rnn.MultiRNNCell([drop] * lstm_layers)
    
    # Getting an initial state of all zeros
    initial_state = cell.zero_state(batch_size, tf.float32)


# In[25]:


with graph.as_default():
    outputs, final_state = tf.nn.dynamic_rnn(cell, embed, initial_state=initial_state)


# # Output
# 
# We only care about the final output, we'll be using that as our comma prediction. So we need to grab the last output with `outputs[:, -1]`, the calculate the cost from that and `labels_`.

# In[26]:


with graph.as_default():
    predictions = tf.contrib.layers.fully_connected(outputs[:, -1], 1, activation_fn=tf.sigmoid)
    cost = tf.losses.mean_squared_error(labels_, predictions)
    
    optimizer = tf.train.AdamOptimizer(learning_rate).minimize(cost)


# Here we can add a few nodes to calculate the accuracy which we'll use in the validation pass.

# In[27]:


with graph.as_default():
    correct_pred = tf.equal(tf.cast(tf.round(predictions), tf.int32), labels_)
    accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))


# # Batching
# 
# This is a simple function for returning batches from our data. First it removes data such that we only have full batches. Then it iterates through the `x` and `y` arrays and returns slices out of those arrays with size `[batch_size]`.

# In[28]:


def get_batches(x, y, woerter, batch_size=100):
    
    n_batches = len(x)//batch_size
    x, y, woerter = x[:n_batches*batch_size], y[:n_batches*batch_size], woerter[:n_batches*batch_size]
    for ii in range(0, len(x), batch_size):
        yield x[ii:ii+batch_size], y[ii:ii+batch_size], woerter[ii:ii+batch_size]### Batching


# # Training
# 
# Training of the LSTM. Taking time measurements all the way. Saving checkpoints when the training accuracy exceeds prior results. 
# 
# num_epochs: because of time and money constraints the network never was trained more than 10 hours (8 epochs) but normally there werent that huge improvments seen anymore after 2 epochs

# In[29]:


num_epochs = 7

file_output = "data.txt"

cathegory = ["epochs", "iteration", "val_accuracies", "train_accuracies"]

with graph.as_default():
    saver = tf.train.Saver(max_to_keep=1) # keep last best iteration
    


# In[ ]:


from time import time
import matplotlib.pyplot as plt
import matplotlib.animation as animation
get_ipython().magic('matplotlib inline')


print("START")
t0 = time()



with tf.Session(graph=graph) as sess:
    sess.run(tf.global_variables_initializer())
    iteration = 0
    
    with open(file_output, "r+") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(cathegory)
    
        epochs_stat = []
        iteration_stat = []
        train_accs_stat = []
        val_accs_stat = []
        best_val_accs = 0
        
        
        for e in range(num_epochs):
            state = sess.run(initial_state)
            

            for ii, (x, y, wort) in enumerate(get_batches(train_x, train_y, train_woerter, batch_size), 1):
                
                #Training the network (after every batch the weights gets updated)
                t1 = time()
                
                feed = {inputs_: x,
                        labels_: y[:,None],
                        keep_prob: 0.85,
                        initial_state: state}  # normally 0.5 (0.6 - 0.7 for best results so far)
                loss, state, _ = sess.run([cost, final_state, optimizer], feed_dict=feed)

                #Print out status
                if iteration%10==0:
                    print("Epoch: {}/{}".format(e, num_epochs),
                          "Iteration: {}".format(iteration),
                          "Train loss: {:.3f}".format(loss))
                    print("10 Epochs training time:", round(time()-t1, 3), "s")
                

                if iteration%1000==0:
                    
                    t2 = time()
                    
                    train_acc = []
                    val_acc = []
                    
                    #Training set accuracy
                    train_acc_state = sess.run(cell.zero_state(batch_size, tf.float32))
                    for x, y, wort in get_batches(train_acc_x, train_acc_y, woerter, batch_size):
                        feed = {inputs_: x,
                                labels_: y[:,None],
                                keep_prob: 1,
                                initial_state: state}
                        batch_acc, train_acc_state = sess.run([accuracy, final_state], feed_dict=feed)
                        train_acc.append(batch_acc)
                    print("Train acc: {:.3f}".format(np.mean(train_acc)))
                    
                    
                    #Validation set accuracy
                    val_state = sess.run(cell.zero_state(batch_size, tf.float32))
                    for x, y, wort in get_batches(val_x, val_y, val_woerter, batch_size):
                        feed = {inputs_: x,
                                labels_: y[:,None],
                                keep_prob: 1,
                                initial_state: val_state}
                        batch_acc, val_state = sess.run([accuracy, final_state], feed_dict=feed)
                        val_acc.append(batch_acc)
                    print("Val acc: {:.3f}".format(np.mean(val_acc)))
                    
                    val_accs = np.mean(val_acc)
                    train_accs = np.mean(train_acc)
                    
                    to_write = [e] + [iteration] + [val_accs] + [train_accs]
                    writer.writerow(to_write)
                    
                    epochs_stat.append(e)
                    iteration_stat.append(iteration)
                    train_accs_stat.append(train_accs)
                    val_accs_stat.append(val_accs)
                    
                    print("1 Validation time:", round(time()-t2, 3), "s")
                    
                    
                    #Safe checkpoint if it is better than prior results
                    t4 = time()
                    
                    if best_val_accs < val_accs:
                        best_val_accs = val_accs
                        saver.save(sess, "checkpoints/best.ckpt")
                        print("New best saving time:", round(time()-t4, 3), "s")
                        
                        
                if iteration%1000==0:
                    
                    #Plot a graph with the validation and training accuracies
                    t3 = time()
                    
                    plt.figure()
                    plt.plot(iteration_stat, train_accs_stat, label="train_acc")
                    plt.plot(iteration_stat, val_accs_stat, label="val_acc")
                    plt.legend()
                    _ = plt.ylim()
                    plt.show()
                    plt.pause(0.0001) #Just short necessary stop to show the plot
                    
                    print("Ploting time:", round(time()-t3, 3), "s")
                       
                
                iteration +=1
                
                        
    #saver.save(sess, "checkpoints/last.ckpt")
    print("Total time:", round(time()-t0, 3), "s")
    
    print("FINISH !!!!")


# In[31]:


print("Kernel still alive :)")


# # Valididation accuracy graph

# In[32]:


import matplotlib.pyplot as plt
get_ipython().magic('matplotlib inline')

plt.figure()
plt.plot(iteration_stat, val_accs_stat, label="val_acc")
plt.legend()
_ = plt.ylim()
plt.show()


# # Testing
# 

# In[34]:


test_acc = []
with tf.Session(graph=graph) as sess:
    saver.restore(sess, tf.train.latest_checkpoint('checkpoints'))
    test_state = sess.run(cell.zero_state(batch_size, tf.float32))
    for ii, (x, y, wort) in enumerate(get_batches(test_x, test_y, test_woerter, batch_size), 1):
        feed = {inputs_: x,
                labels_: y[:, None],
                keep_prob: 1,
                initial_state: test_state}
        batch_acc, test_state = sess.run([accuracy, final_state], feed_dict=feed)
        test_acc.append(batch_acc)
    print("Test accuracy: {:.3f}".format(np.mean(test_acc)))


# ### Show samples of testing batch
# Gives really good insights by showing:
# prediction, label, text

# In[35]:


predictions_list = []
batch_size = 512
counter_max = 3

with tf.Session(graph=graph) as sess:
    saver.restore(sess, tf.train.latest_checkpoint('checkpoints'))
    val_state = sess.run(cell.zero_state(batch_size, tf.float32))
    
    counter = 0
    
    for ii, (x, y, wort) in enumerate(get_batches(test_x, test_y, test_woerter, batch_size), 1):
        feed = {inputs_: x,
                labels_: y[:, None],
                keep_prob: 1,
                initial_state: val_state}
        prediction = sess.run(predictions, feed_dict=feed)
    
        for i in range(0,len(prediction)):
            predictions_list.append([round(float(prediction[i]),4), y[i], wort[i]])
        
        ###    
        if counter == counter_max:
            break
        counter += 1
        ###
        
        
        
prediction_list = list(predictions_list)
for iii in range(0, len(prediction_list)):
    print(prediction_list[iii])
  
    


# # USE TRAINED NEURONAL NET ON NEW TEXT
# 
# ### Preprocess the input text

# In[54]:


import time
import numpy as np
import tensorflow as tf
import re
from collections import Counter

import nltk
from nltk.tokenize import word_tokenize

nltk.download('punkt')

def preprocess(text_name, vocabs_to_int=vocabs_to_int, int_to_vocabs=int_to_vocabs):
    
    ## Open text file
    with open(text_name, "r", encoding="utf-8") as text:
        text = text.read()
    
    ## Filtering out page numbers
    text = re.sub(r'\n[0-9]+ \n', '', text)
    text = re.sub(r'\n\n[a-zA-Z] \n\n', '', text)
    
    ## Tokenizing
    woerter = word_tokenize(text)
    
    ## Stemming
    sno = nltk.stem.SnowballStemmer('german')
    woerter_stem = [sno.stem(wort) for wort in woerter]
    
    ## POS
    root = '/output/'
    #root = '/Users/Till/Dropbox/Deep Learning Udacity/deep-learning/KommasetzungAusweichordner'
    fileid = 'tiger_release_aug07.corrected.16012013.conll09'
    columntypes = ['ignore', 'words', 'ignore', 'ignore', 'pos']
    
    #corp = nltk.corpus.ConllCorpusReader(root, fileid, columntypes, encoding='utf8')
    #tagged_sents = corp.tagged_sents()
    #tagger = ClassifierBasedGermanTagger(train=corp.tagged_sents())
    w_classes = tagger.tag(woerter)
    
    ## Mark punctuation with a "c"
    woerter_classes = []
    for word in w_classes:
        if "$" in str(word[1]):
            word_class = word[0] + "c"
        else:
            word_class = word[1]
        woerter_classes.append(word_class)
        
    ## Sample doesn't have any commas so they also don't have to be removed
    
    ## Generate features with the use of look up table (if not in lookup table -> None)   
    woerter_ints = [vocabs_to_int[wort] if wort in vocabs_to_int else vocabs_to_int["None"] for wort in woerter_stem]
    woerter_classes_ints = [vocabs_to_int[wort] if wort in vocabs_to_int else vocabs_to_int["None"] for wort in woerter_classes]
    
    
    ## Generate features
    features = []
    feature_woerter = [get_batch(woerter_ints, idx, window_size=10) for idx in range(0,len(woerter_ints))]
    feature_classes =  [get_batch(woerter_classes_ints, idx, window_size=10) for idx in range(0,len(woerter_ints))]
    features = list(zip(feature_woerter, feature_classes))
    features = np.asarray(features).reshape(len(woerter_ints), -1)
    
    return features, woerter


# ### Predicting location of commas

# In[74]:


predictions_list = []
batch_size = 512

with tf.Session(graph=graph) as sess:
    saver.restore(sess, tf.train.latest_checkpoint("checkpoints"))
    val_state = sess.run(cell.zero_state(batch_size, tf.float32))
    x, woerter = preprocess("test.txt")
    
    for ii, (x, y, wort) in enumerate(get_batches(x, test_y, woerter, batch_size), 1):
        
        feed = {inputs_: x,
                labels_: y[:, None],
                keep_prob: 1,
                initial_state: val_state}
        prediction = sess.run(predictions, feed_dict=feed)
    
        for i in range(0,len(prediction)):
            predictions_list.append([round(float(prediction[i]),4), wort[i]])
        
        
predictions_list = list(predictions_list)


for iii in predictions_list:
    print(iii)


# ## Putting it back in a textual format
# Commas with a prediction higher than the upper_gate will be added with ",". Predictions just higher that the lower_gate will be added with "(,)"

# In[56]:


import string

def retext(upper_gate, lower_gate, pred = predictions_list):
    text = [t[1] for t in pred]
    
    for i in range(1, len(pred)):
        if pred[i][0] >= upper_gate:
            text[i-1] = str(text[i-1]) + ","
        elif  upper_gate > pred[i][0] >= lower_gate:
            text[i-1] = str(text[i-1]) + "(,)"
    
        if str(pred[i][1]) in str(string.punctuation):
            text[i-1] = text[i-1] + text[i]
            text[i] = "XXX"
            
    text = [te for te in text if "XXX" not in te]
    ready_text = " ".join(text)
    return ready_text


# In[57]:


#prediction_list = [[pred, wort] for pred, lab, wort in prediction_list]
retext(0.9, 0.8)


# ### This is just a preversion!!! The time wasn't sufficient to increase performance to a generell knowledge. There are still tones of possible improvments and ideas. To get the best possible performance it is still really important to train it on the right sort of text and tune the decision bounderies (from which prediction possiblity level on it will really set the comma) by hand . For more formal texts, later on a different data set was used. Because of time and money constraints the training resources were really limited. Also the excess to good and clean data sets, without having access to online language corpuses (often just accessibly if enrolled at a university), was not possible. 
# 
# ### But overall the programm is learning and is getting some basic understanding of some rules. Until the begin of my study I hopefully can implement  some  more of my ideas, that should increase the performance a lot.

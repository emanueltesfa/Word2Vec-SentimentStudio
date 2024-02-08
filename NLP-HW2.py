# %%
import torch
import torch.nn.functional as F
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import re
from bs4 import BeautifulSoup
import warnings 
warnings.filterwarnings("ignore")
from gensim.test.utils import datapath
from gensim import utils
import gensim.models
import gensim.downloader as api
from nltk.corpus import stopwords
import nltk
from nltk.stem import WordNetLemmatizer
from sklearn.linear_model import Perceptron
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import torch.nn as nn 
import torch.nn.functional as f 
from torch.utils.data import Dataset, DataLoader
from gensim.downloader import load
from sklearn.svm import SVC
from tqdm import tqdm 

tqdm.pandas()

if torch.cuda.is_available():       
    device = torch.device("cuda")
    print(torch.cuda.get_device_name(0))

else:
    print('No GPU available, using the CPU instead.')
    device = torch.device("cpu")


# %% [markdown]
# ## 1) Dataset Generation 

# %%
df = pd.read_csv("amazon_reviews_us_Office_Products_v1_00.tsv", sep='\t', on_bad_lines='skip')#, usecols=['review_body','star_rating']) #lineterminator='\r'
df.drop(df.columns[0], axis=1, inplace=True)
df = df[['review_body', 'star_rating']]
df.columns

# %% [markdown]
# ### Preprocess/Cleaning take 15 minutes

# %%
contraction_mapping = {
    "ain't": "am not",
    "aren't": "are not",
    "can't": "cannot",
    "can't've": "cannot have",
    "'cause": "because",
    "could've": "could have",
    "couldn't": "could not",
    "couldn't've": "could not have",
    "didn't": "did not",
    "doesn't": "does not",
    "don't": "do not",
    "hadn't": "had not",
    "hadn't've": "had not have",
    "hasn't": "has not",
    "haven't": "have not",
    "he'd": "he would",
    "he'd've": "he would have",
    "he'll": "he will",
    "he'll've": "he will have",
    "he's": "he is",
    "how'd": "how did",
    "how'd'y": "how do you",
    "how'll": "how will",
    "how's": "how is",
    "I'd": "I would",
    "I'd've": "I would have",
    "I'll": "I will",
    "I'll've": "I will have",
    "I'm": "I am",
    "I've": "I have",
    "isn't": "is not",
    "it'd": "it would",
    "it'd've": "it would have",
    "it'll": "it will",
    "it'll've": "it will have",
    "it's": "it is",
    "let's": "let us",
    "ma'am": "madam",
    "mayn't": "may not",
    "might've": "might have",
    "mightn't": "might not",
    "mightn't've": "might not have",
    "must've": "must have",
    "mustn't": "must not",
    "mustn't've": "must not have",
    "needn't": "need not",
    "needn't've": "need not have",
    "o'clock": "of the clock",
    "oughtn't": "ought not",
    "oughtn't've": "ought not have",
    "shan't": "shall not",
    "sha'n't": "shall not",
    "shan't've": "shall not have",
    "she'd": "she would",
    "she'd've": "she would have",
    "she'll": "she will",
    "she'll've": "she will have",
    "she's": "she is",
    "should've": "should have",
    "shouldn't": "should not",
    "shouldn't've": "should not have",
    "so've": "so have",
    "so's": "so is",
    "that'd": "that would",
    "that'd've": "that would have",
    "that's": "that is",
    "there'd": "there would",
    "there'd've": "there would have",
    "there's": "there is",
    "they'd": "they would",
    "they'd've": "they would have",
    "they'll": "they will",
    "they'll've": "they will have",
    "they're": "they are",
    "they've": "they have",
    "to've": "to have",
    "wasn't": "was not",
    "we'd": "we would",
    "we'd've": "we would have",
    "we'll": "we will",
    "we'll've": "we will have",
    "we're": "we are",
    "we've": "we have",
    "weren't": "were not",
    "what'll": "what will",
    "what'll've": "what will have",
    "what're": "what are",
    "what's": "what is",
    "what've": "what have",
    "when's": "when is",
    "when've": "when have",
    "where'd": "where did",
    "where's": "where is",
    "where've": "where have",
    "who'll": "who will",
    "who'll've": "who will have",
    "who's": "who is",
    "who've": "who have",
    "why's": "why is",
    "why've": "why have",
    "will've": "will have",
    "won't": "will not",
    "won't've": "will not have",
    "would've": "would have",
    "wouldn't": "would not",
    "wouldn't've": "would not have",
    "y'all": "you all",
    "y'all'd": "you all would",
    "y'all'd've": "you all would have",
    "y'all're": "you all are",
    "y'all've": "you all have",
    "you'd": "you would",
    "you'd've": "you would have",
    "you'll": "you will",
    "you'll've": "you will have",
    "you're": "you are",
    "you've": "you have"
}


pattern_contractions = re.compile('(%s)' % '|'.join(contraction_mapping.keys()))
lemmatizer = WordNetLemmatizer()
nltk.download('stopwords', 'punkt')
stop_words = set(stopwords.words('english'))


def expand_contractions(text, contraction_map=contraction_mapping):
    return pattern_contractions.sub(lambda occurrence: contraction_map[occurrence.group(0)], text)


def rem_stopwords(review,stp):
    words = review.split()
    filtered_words = [word for word in words if word not in stp]
    filtered_sentence = ' '.join(filtered_words)
    return filtered_sentence


def lemmazation(review):
    words = review.split()
    lemmatized_words = [lemmatizer.lemmatize(word) for word in words]
    lemmatized_review = ' '.join(lemmatized_words)
    return lemmatized_review


def clean_preproc_reviews(reviews, stp):
    ### CLEANING
    reviews = reviews.str.lower()
    reviews = reviews.progress_apply(lambda x: BeautifulSoup(x, "html.parser").get_text())
    reviews = reviews.replace(r'http\S+', '', regex = True)
    reviews = reviews.replace("[^a-zA-Z]", " ", regex = True)
    reviews = reviews.replace('\s+', ' ', regex = True).str.strip()
    reviews = reviews.progress_apply(lambda x: expand_contractions(x))

    ### PREPROCESSING
    reviews = reviews.progress_apply(lambda x : rem_stopwords(x, stp))
    reviews = reviews.progress_apply(lemmazation)

    return reviews

# Clean the reviews
df['review_body'] = df['review_body'].astype(str)
df.dropna(subset=['review_body'], inplace = True)
df['review_body'] =  clean_preproc_reviews(df['review_body'], stop_words) #df['review_body'].progress_apply(lambda x: clean_preproc_reviews(x, stop_words) )

df.dropna(subset = ['review_body'], inplace = True)


# %%
df['label'] = df['star_rating'].progress_apply(lambda x: 0 if x in [4, 5] else (1 if x in [1, 2] else 2))

star_ratings = [5, 4, 3, 2, 1]
samples = [ df[df['star_rating'] == rating].sample(n = 50000, random_state = 42) for rating in star_ratings]
merged_dataset = pd.concat(samples)

# %% [markdown]
# ## 2) Word Embedding

# %% [markdown]
# ### (a)
# Load the pretrained “word2vec-google-news-300” Word2Vec model and learn
# how to extract word embeddings for your dataset. Try to check semantic
# similarities of the generated vectors using two examples of your own, e.g.,
# King − M an + W oman = Queen or excellent ∼ outstanding.
# 

# %%
pretrained_model = api.load('word2vec-google-news-300')

print("Similarity between 'hey' and 'hello': ", pretrained_model.similarity('ocean', 'sea'))

# %% [markdown]
# ### (b)
# Check the semantic similarities for the same two examples
# in part (a). What do you conclude from comparing vectors generated by
# yourself and the pretrained model? Which of the Word2Vec models seems
# to encode semantic similarities between words better?
# 
# In the pretrained model, the similarity score between Outstanding and Excellent was lower than the custom model I trained, thus showing that similarities between vectors were stronger in my model. But the pretrained model does a better join of building relationships between words

# %%
# Reference https://radimrehurek.com/gensim/auto_examples/tutorials/run_word2vec.html
class MyCorpus:
    def __init__(self, df, col):
        self.df = df
        self.col = col

    def __iter__(self):
        for line in self.df[self.col]:
            yield utils.simple_preprocess(line)


my_model = gensim.models.Word2Vec(sentences = MyCorpus(merged_dataset, 'review_body'), vector_size = 300, window = 11, min_count = 10, workers = 4)

# %%
word_vectors = my_model.wv

print("Similarity between 'ocean' and 'sea':", word_vectors.similarity('ocean', 'sea'))

# %%
def document_vector(word2vec_model, doc_review):
    doc_review = [word for word in doc_review if word in word2vec_model.key_to_index]
    
    if len(doc_review) == 0:
        return np.zeros(word2vec_model.vector_size)
        
    return np.mean(word2vec_model[doc_review], axis=0)


def gen_concat_feature_vector(word2vec_model, doc_review, vector_size=300, max_words=10):
    concatenated_vector = np.zeros(vector_size * max_words) # Initialize an empty array for the concatenated vectors.

    for i, word in enumerate(doc_review[:max_words]):
        if word in word2vec_model.key_to_index:
            concatenated_vector[i*vector_size:(i+1)*vector_size] = word2vec_model[word]
            
    return concatenated_vector


merged_dataset['processed_text'] = merged_dataset['review_body'].progress_apply(gensim.utils.simple_preprocess)
merged_dataset['pretrained_vector'] = merged_dataset['processed_text'].progress_apply(lambda doc_review: document_vector(pretrained_model, doc_review))
merged_dataset['custom_vector'] = merged_dataset['processed_text'].progress_apply(lambda doc_review: document_vector(my_model.wv, doc_review))
#####
merged_dataset['pre_concatenated_vector'] = merged_dataset['processed_text'].progress_apply(lambda row_indx: gen_concat_feature_vector(pretrained_model, row_indx))
merged_dataset['custom_concatenated_vector'] = merged_dataset['processed_text'].progress_apply(lambda row_indx: gen_concat_feature_vector(pretrained_model, row_indx))

X = np.vstack( merged_dataset['custom_vector'] )
X_pre = np.vstack( merged_dataset['pretrained_vector']) 
Y = np.vstack( merged_dataset['label'] ) 


# %%
# merged_dataset['binary_label'] = merged_dataset['label'].progress_apply(lambda x: pd.NA if x == 2 else (0 if x == 0 else 1))
filtered_dataset = merged_dataset[merged_dataset['label'] != 2]
filtered_dataset['binary_label'] = filtered_dataset['label'].astype(int)

# %% [markdown]
# ## 3) Simple models 
# 
# What do you conclude from comparing performances for the models
# trained using the three different feature types (TF-IDF, pretrained Word2Vec,
# your trained Word2Vec)?
# 
# It seems tha pretrained Word2Vec embeddings marginally perform better than the custom models embeddings and better than TF-IDF 
# most likely due to the moderately large window size that is being used in the Word2Vec model.
# 

# %%
def evaulate(y_label, y_predicted):
    accuracy = accuracy_score(y_label, y_predicted)
    precision = precision_score(y_label, y_predicted, average = 'weighted')
    recall = recall_score(y_label, y_predicted, average = 'weighted')
    f1 = f1_score(y_label, y_predicted, average = 'weighted')

    return accuracy, precision, recall, f1


def sklearn_model_train(data, model_type, prefix):
    for name, model_name in model_type:
        X_train, X_test, y_train, y_test = data 
        model_name.fit(X_train, y_train)
        y_pred_test = model_name.predict(X_test)
        te_acc, _, _, _ = evaulate(y_test, y_pred_test)

        print(prefix,name, "Testing: Accuracy: {:.4f}".format(te_acc) )


# %%
split_data_custom = train_test_split(X, Y, test_size = 0.2, random_state = 42)
pretrain_split_data = train_test_split(X_pre, Y, test_size = 0.2, random_state = 42)
pretrain_bin = train_test_split(np.vstack(filtered_dataset['pretrained_vector'].values), np.vstack( filtered_dataset['binary_label'].values ), test_size = 0.2, random_state = 42)
custom_bin = train_test_split(np.vstack( filtered_dataset['custom_vector'].values), np.vstack(filtered_dataset['binary_label'].values), test_size = 0.2, random_state = 42)


model_names = [
    ("Perceptron Model", Perceptron()), 
    ("SVM Model", SVC(max_iter = 500))
]

sklearn_model_train(pretrain_split_data, model_names, prefix = 'Ternary Pretrained Embedding')
sklearn_model_train(split_data_custom, model_names, prefix = 'Ternary Custom Embedding')
sklearn_model_train(pretrain_bin, model_names, prefix = 'Binary Pretrained Embedding')
sklearn_model_train(custom_bin, model_names, prefix = 'Binary Custom Embedding')

# %% [markdown]
# ## 4) Feedforward Neural Networks

# %% [markdown]
# ### a)

# %%
class Net(nn.Module): 
    def __init__(self, n_classes, n_dim):
        super(Net, self).__init__()

        hidden_1 = 50
        hidden_2 = 10

        self.n_classes = n_classes
        self.n_dim = n_dim

        self.fc1 = nn.Linear(n_dim, hidden_1)
        self.fc2 = nn.Linear(hidden_1, hidden_2)
        self.fc3 = nn.Linear(hidden_2, n_classes)
        self.dropout = nn.Dropout(0.5)

    def forward(self, x):
        x = f.gelu(self.fc1(x))
        x = self.dropout(x)
        x = f.gelu(self.fc2(x))
        x = self.dropout(x)
        x = f.softmax(self.fc3(x)) # REMOVE SOFTMAX?
        return x 


ternary_model = Net(n_classes = 3, n_dim = 300)
binary_model = Net(n_classes = 2, n_dim = 300)
concat_ternary_model = Net(n_classes = 3, n_dim = 3000)
concat_binary_model = Net(n_classes = 2, n_dim = 3000)

print(ternary_model)

# %%
class TextDataset(Dataset):
    def __init__(self, features, labels):
        self.features = features
        self.labels = labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.features[idx], self.labels[idx]

# %%
def model_preprocess(x, y, model, cnn_bit:int = 0, optimi = None):
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2)
    x_train = torch.tensor(np.array(x_train.tolist(), dtype=np.float32), dtype=torch.float32)
    y_train = torch.tensor(np.array(y_train), dtype=torch.long)
    x_test = torch.tensor(np.array(x_test.tolist(), dtype=np.float32), dtype=torch.float32)
    y_test = torch.tensor(np.array(y_test), dtype=torch.long)

    train_loader = DataLoader(TextDataset(x_train, y_train), batch_size = 64, shuffle=True)
    test_loader = DataLoader(TextDataset(x_test, y_test), batch_size = 64, shuffle=False)
    
    if optimi is None:
        optimi = torch.optim.Adadelta(model.parameters(), lr=0.25, rho=0.95) if cnn_bit == 1 else torch.optim.Adam(model.parameters(), lr = 0.0001)
        # print(optimi)

    return (train_loader, test_loader, nn.CrossEntropyLoss(), optimi)


# %%
#### https://www.kaggle.com/code/mishra1993/pytorch-multi-layer-perceptron-mnist/notebook
def train_model(hyperparams, model, debug_mode = 0, n_epochs = 50):
    train_loader, test_loader, criterion, optimizer = hyperparams
    valid_loss_min = np.Inf
    
    for epoch in range(n_epochs):
        train_loss = 0.0
        valid_loss = 0.0
        correct_train = 0
        correct_valid = 0

        # Set model to training mode
        model.train()

        for data, target in train_loader:
            optimizer.zero_grad()
            output = model(data)
            target = target.squeeze()
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()

            # For debug mode, calculate detailed training metrics
            if debug_mode == 1:
                train_loss += loss.item() * data.size(0)
                _, predicted = torch.max(output.data, 1)
                correct_train += (predicted == target).sum().item()

        # Validation phase
        model.eval()
        with torch.no_grad():
            for data, target in test_loader:
                output = model(data)
                target = target.squeeze()
                loss = criterion(output, target)
                valid_loss += loss.item() * data.size(0)
                _, predicted = torch.max(output.data, 1)
                correct_valid += (predicted == target).sum().item()

        # Always calculate validation accuracy
        valid_accuracy = correct_valid / len(test_loader.dataset)

        # For debug mode, print detailed metrics and check for model improvement
        if debug_mode == 1:
            train_loss = train_loss / len(train_loader.dataset)
            valid_loss = valid_loss / len(test_loader.dataset)
            train_accuracy = correct_train / len(train_loader.dataset)
            print(f'Epoch: {epoch+1} \tTraining Loss: {train_loss:.6f} \tTraining Accuracy: {train_accuracy * 100:.2f}% \tValidation Loss: {valid_loss:.6f} \tValidation Accuracy: {valid_accuracy * 100:.2f}%')

            if valid_loss < valid_loss_min:
                print(f'Validation loss decreased ({valid_loss_min:.6f} --> {valid_loss:.6f}).  Saving model ...')
                torch.save(model.state_dict(), 'model.pt')
                valid_loss_min = valid_loss
    print("Test Accuracy: ")
    return valid_accuracy


# %% [markdown]
# ### Custom Embeddings and 3 Class MLP

# %%
model_hyperparameters_tern_custom = model_preprocess(merged_dataset['custom_vector'].values, merged_dataset['label'].values, ternary_model )
train_model(model_hyperparameters_tern_custom, ternary_model)

# %% [markdown]
# ### Pretrained Embeddings and 3 Class MLP

# %%
model_hyperparameters_tern_pre = model_preprocess(merged_dataset['pretrained_vector'].values, merged_dataset['label'].values, ternary_model)
train_model(model_hyperparameters_tern_pre, ternary_model)

# %% [markdown]
# ### Custom Embeddings and 2 Class MLP

# %%
model_hyperparameters_bin_custom = model_preprocess(filtered_dataset['custom_vector'].values, filtered_dataset['binary_label'].values, binary_model)
train_model(model_hyperparameters_bin_custom, binary_model)

# %% [markdown]
# ### Pretrained Embeddings and 2 Class MLP

# %%
model_hyperparameters_bin_pre = model_preprocess(filtered_dataset['pretrained_vector'].values, filtered_dataset['binary_label'].values, binary_model)
train_model(model_hyperparameters_bin_pre, binary_model)

# %% [markdown]
# (b) (15 points)
# What do you conclude by comparing accuracy values you obtain with
# those obtained in the “’Simple Models” section (note you can compare the
# accuracy values for binary classification)
# 
# In the ternary classifiation, simple models perform worse by an average of 10% in their validation accuracies. This is due to the unique choice of hyperparamters (hidden dimensions, nonlinerity and learning etc) that pytorch allows us to customize. 

# %% [markdown]
# ### Pretrained Concatenated Embeddings and 2 Class MLP

# %%
model_hyperp_concat_bin_pre = model_preprocess(filtered_dataset['pre_concatenated_vector'].values, filtered_dataset['binary_label'].values, concat_binary_model)
train_model(model_hyperp_concat_bin_pre, concat_binary_model)

# %% [markdown]
# ### Custom Concatenated Embeddings and 2 Class MLP

# %%
model_hyperp_concat_bin_custom = model_preprocess(filtered_dataset['custom_concatenated_vector'].values, filtered_dataset['binary_label'].values, concat_binary_model, optimi = torch.optim.SGD(concat_binary_model.parameters(), lr = 0.001))
train_model(model_hyperp_concat_bin_custom, concat_binary_model)

# %% [markdown]
# ### Pretrained Concatenated Embeddings and 3 Class MLP

# %%
model_hyperp_concat_ter_pre = model_preprocess(merged_dataset['pre_concatenated_vector'].values, merged_dataset['label'].values, concat_ternary_model, optimi = torch.optim.Adam(concat_ternary_model.parameters(), lr = 0.001))
train_model(model_hyperp_concat_ter_pre, concat_ternary_model, n_epochs = 1, debug_mode = 0)

# %% [markdown]
# ### Custom Concatenated Embeddings and 3 Class MLP

# %%
model_hyperp_concat_ter_custom = model_preprocess(merged_dataset['custom_concatenated_vector'].values, merged_dataset['label'].values, concat_ternary_model)
train_model(model_hyperp_concat_ter_custom, concat_ternary_model, n_epochs = 10)

# %% [markdown]
# ## 5) Convolutional Neural Networks

# %%
class SimpleCNN(nn.Module):
    def __init__(self, num_classes):
        super(SimpleCNN, self).__init__()
        self.num_classes = num_classes
        self.conv1 = nn.Conv1d(1, 50, kernel_size = 5, padding = 2) ## 50 words store as 300 dim wv's 
        self.conv2 = nn.Conv1d(50, 10, kernel_size = 5, padding = 2) 
        self.fc = nn.Linear(3000, self.num_classes)
        
    def forward(self, x):
        x = x.squeeze()  # Ensure the input indices are of type Long
        # x = self.embedding(x).permute(0, 2, 1)
        x = x.reshape(x.shape[0], 1 , x.shape[1])
        x = F.gelu(self.conv1(x))
        x = F.gelu(self.conv2(x))
        x = x.view(x.size(0), -1)
        x = self.fc(x)

        return x

# %% [markdown]
# ### Custom embeddings & 3 Class CNN

# %%
CNNnet_ter_custom_complex = SimpleCNN(num_classes = 3)
cnn_hyperp_ter_cus = model_preprocess(x = merged_dataset['custom_vector'].values, y = merged_dataset['label'].values, model = CNNnet_ter_custom_complex, cnn_bit = 1)
train_model(hyperparams = cnn_hyperp_ter_cus, model = CNNnet_ter_custom_complex, n_epochs = 20 )

# %% [markdown]
# ### Custom embeddings & 2 Class CNN

# %%
CNNnet_bin_custom = SimpleCNN(num_classes = 2)
cnn_hyperp_bin_cust = model_preprocess(x = filtered_dataset['custom_vector'].values, y = filtered_dataset['label'].values, model = CNNnet_bin_custom, cnn_bit = 1)
train_model(hyperparams = cnn_hyperp_bin_cust, model = CNNnet_bin_custom, n_epochs = 15) 

# %% [markdown]
# ### Pretrained embeddings & 2 Class CNN

# %%
CNNnet_bin_pre = SimpleCNN(num_classes = 2)
cnn_hyperp_bin_pre = model_preprocess(x = filtered_dataset['pretrained_vector'].values, y = filtered_dataset['label'].values, model = CNNnet_bin_pre, cnn_bit = 1)
train_model(hyperparams = cnn_hyperp_bin_pre, model = CNNnet_bin_pre, n_epochs = 42)

# %% [markdown]
# ### Pretrained embeddings & 3 Class CNN

# %%
CNNnet_ter_pre = SimpleCNN(num_classes = 3)
cnn_hyperp_ter_pre = model_preprocess(x = merged_dataset['pretrained_vector'].values, y = merged_dataset['label'].values, model = CNNnet_ter_pre, cnn_bit = 1)
train_model(hyperparams = cnn_hyperp_ter_pre, model = CNNnet_ter_pre, n_epochs = 35) # 66.71

# %%




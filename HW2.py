df = pd.read_csv("data/amazon_reviews_us_Office_Products_v1_00.tsv", sep='\t', on_bad_lines='skip')#, usecols=['review_body','star_rating']) #lineterminator='\r'
df.drop(df.columns[0], axis=1, inplace=True)
df = df[['review_body', 'star_rating']]

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
    reviews = reviews.apply(lambda x: BeautifulSoup(x, "html.parser").get_text())
    reviews = reviews.replace(r'http\S+', '', regex=True)
    reviews = reviews.replace("[^a-zA-Z]", " ", regex=True)
    reviews = reviews.replace('\s+', ' ', regex=True).str.strip()
    reviews = reviews.apply(lambda x: expand_contractions(x))

    # ### PREPROCESSING
    reviews = reviews.apply(lambda x : rem_stopwords(x, stp))
    reviews = reviews.apply(lemmazation)

    return reviews

# Clean the reviews
df['review_body'] =df['review_body'].astype(str)
df.dropna(subset=['review_body'], inplace=True)
df['review_body'] = clean_preproc_reviews(df['review_body'], stop_words)
df.dropna(subset=['review_body'], inplace=True)


df['label'] = df['star_rating'].apply(lambda x: 0 if x in [4, 5] else (1 if x in [1, 2] else 2))

star_ratings = [5, 4, 3, 2, 1]
samples = [ df[df['star_rating'] == rating].sample(n = 50000, random_state = 42) for rating in star_ratings]
merged_dataset = pd.concat(samples)

class MyCorpus:
    def __init__(self, df, col):
        self.df = df
        self.col = col

    def __iter__(self):
        for line in self.df[self.col]:
            yield utils.simple_preprocess(line)


sentences = MyCorpus(merged_dataset, 'review_body')
my_model = gensim.models.Word2Vec(sentences=sentences, vector_size=300, window=11, min_count=10, workers=4)

def document_vector(word2vec_model, doc):
    doc = [word for word in doc if word in word2vec_model.key_to_index]

    if len(doc) == 0:
        return np.zeros(word2vec_model.vector_size)
        
    return np.mean(word2vec_model[doc], axis=0)

merged_dataset['processed_text'] = merged_dataset['review_body'].apply(gensim.utils.simple_preprocess)
merged_dataset['pretrained_vector'] = merged_dataset['processed_text'].apply(lambda doc: document_vector(pretrained_model, doc))
merged_dataset['doc_vector'] = merged_dataset['processed_text'].apply(lambda doc: document_vector(my_model.wv, doc))

Y = merged_dataset['label']
X = merged_dataset['doc_vector']
X_pre = merged_dataset['pretrained_vector']


class Net(nn.Module): 
    def __init__(self):
        super(Net, self).__init__()
        n_dim = 300
        hidden_1 = 50
        hidden_2 = 10
        n_classes = 3

        self.fc1 = nn.Linear(n_dim, hidden_1)
        self.fc2 = nn.Linear(hidden_1, hidden_2)
        self.fc3 = nn.Linear(hidden_2, n_classes)
        self.dropout = nn.Dropout(0.4)

    def forward(self, x):
        x = f.gelu(self.fc1(x))
        x = self.dropout(x)
        x = f.gelu(self.fc2(x))
        x = self.dropout(x)
        x = f.softmax(self.fc3(x))
        return x 


ff_model = Net()

class TextDataset(Dataset):
    def __init__(self, features, labels):
        self.features = features
        self.labels = labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.features[idx], self.labels[idx]


X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
y_train_tensor = torch.tensor(y_train, dtype=torch.long)
y_test_tensor = torch.tensor(y_test, dtype=torch.long)

# Create Dataset
train_dataset = TextDataset(X_train_tensor, y_train_tensor)
test_dataset = TextDataset(X_test_tensor, y_test_tensor)

# Create DataLoader
batch_size = 64  # Adjust the batch size as necessary
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(ff_model.parameters(), lr=0.01)

for epoch in range(n_epochs):
    train_loss = 0.0
    valid_loss = 0.0
    correct_train = 0
    correct_valid = 0
    
    # Training phase
    ff_model.train()
    for data, target in train_loader:
        optimizer.zero_grad()
        output = ff_model(data)
        target = target.squeeze()
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
        train_loss += loss.item() * data.size(0)
        _, predicted = torch.max(output.data, 1)
        correct_train += (predicted == target).sum().item()

    # Validation phase
    ff_model.eval()
    with torch.no_grad():
        for data, target in test_loader:
            output = ff_model(data)
            target = target.squeeze()
            loss = criterion(output, target)
            valid_loss += loss.item() * data.size(0)
            _, predicted = torch.max(output.data, 1)
            correct_valid += (predicted == target).sum().item()

    # Calculate average losses
    train_loss = train_loss / len(train_loader.dataset)
    valid_loss = valid_loss / len(test_loader.dataset)
    
    # Calculate accuracy
    train_accuracy = correct_train / len(train_loader.dataset)
    valid_accuracy = correct_valid / len(test_loader.dataset)

    print('Epoch: {} \tTraining Loss: {:.6f} \tTraining Accuracy: {:.2f}% \tValidation Loss: {:.6f} \tValidation Accuracy: {:.2f}%'.format(
        epoch+1, 
        train_loss,
        train_accuracy * 100,
        valid_loss,
        valid_accuracy * 100
    ))

    # Save the model if validation loss has decreased
    if valid_loss <= valid_loss_min:
        print('Validation loss decreased ({:.6f} --> {:.6f}).  Saving model ...'.format(
        valid_loss_min,
        valid_loss))
        torch.save(ff_model.state_dict(), 'ff_model.pt')
        valid_loss_min = valid_loss
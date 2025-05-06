import pandas as pd
import re
import numpy as np
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential, load_model
from keras.layers import Dense, Embedding, LSTM
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from keras.utils.np_utils import to_categorical
import pickle

# Load dataset
data = pd.read_csv("Data.csv")
data = data[['text','sentiment']]
data['text'] = data['text'].apply(lambda x: x.lower())
data['text'] = data['text'].apply(lambda x: re.sub('[^a-zA-Z0-9\s]', '', x))
data['text'] = data['text'].apply(lambda x: x.replace('rt', ' '))

# Tokenization
max_fatures = 2000
tokenizer = Tokenizer(num_words=max_fatures, split=' ')
tokenizer.fit_on_texts(data['text'].values)
X = tokenizer.texts_to_sequences(data['text'].values)
X = pad_sequences(X)

# Encode labels
labelencoder = LabelEncoder()
integer_encoded = labelencoder.fit_transform(data['sentiment'])
y = to_categorical(integer_encoded)

# Split data
X_train, X_test, Y_train, Y_test = train_test_split(X, y, test_size=0.33, random_state=42)

# Define and compile model
embed_dim = 128
lstm_out = 196
model = Sequential()
model.add(Embedding(max_fatures, embed_dim, input_length=X.shape[1]))
model.add(LSTM(lstm_out, dropout=0.2, recurrent_dropout=0.2))
model.add(Dense(3, activation='softmax'))
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

# Train model
model.fit(X_train, Y_train, epochs=1, batch_size=32, verbose=2)

# Save model and tokenizer
model.save("sentiment_model.h5")
with open('tokenizer.pkl', 'wb') as handle:
    pickle.dump(tokenizer, handle, protocol=pickle.HIGHEST_PROTOCOL)
with open('labelencoder.pkl', 'wb') as handle:
    pickle.dump(labelencoder, handle, protocol=pickle.HIGHEST_PROTOCOL)

# Load for prediction
model = load_model("sentiment_model.h5")
with open('tokenizer.pkl', 'rb') as handle:
    tokenizer = pickle.load(handle)
with open('labelencoder.pkl', 'rb') as handle:
    labelencoder = pickle.load(handle)

# Predict on new text
new_text = ["A lot of good things are happening. We are respected again throughout the world, and that's a great thing .@realDonaldTrump"]
new_text = [re.sub('[^a-zA-Z0-9\s]', '', x.lower().replace('rt', '')) for x in new_text]
seq = tokenizer.texts_to_sequences(new_text)
padded = pad_sequences(seq, maxlen=X.shape[1])
pred = model.predict(padded)
print("Predicted probabilities:", pred)
print("Predicted sentiment:", labelencoder.inverse_transform([np.argmax(pred)]))


# Grid search
import pandas as pd
import re
import numpy as np
import pickle
from keras.models import Sequential
from keras.layers import Dense, Embedding, LSTM
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.utils.np_utils import to_categorical
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split, GridSearchCV
from scikeras.wrappers import KerasClassifier

# Load and preprocess dataset
data = pd.read_csv("Data.csv")
data = data[['text', 'sentiment']]
data['text'] = data['text'].apply(lambda x: x.lower())
data['text'] = data['text'].apply(lambda x: re.sub('[^a-zA-Z0-9\\s]', '', x))
data['text'] = data['text'].apply(lambda x: x.replace('rt', ' '))

# Tokenization
max_features = 2000
tokenizer = Tokenizer(num_words=max_features, split=' ')
tokenizer.fit_on_texts(data['text'].values)
X = tokenizer.texts_to_sequences(data['text'].values)
X = pad_sequences(X)

# Encode labels
labelencoder = LabelEncoder()
integer_encoded = labelencoder.fit_transform(data['sentiment'])
y = to_categorical(integer_encoded)

# Train/test split
X_train, X_test, Y_train, Y_test = train_test_split(X, y, test_size=0.33, random_state=42)

# Define model creation function
def create_model(embed_dim=128, lstm_out=196):
    model = Sequential()
    model.add(Embedding(max_features, embed_dim, input_length=X.shape[1]))
    model.add(LSTM(lstm_out, dropout=0.2, recurrent_dropout=0.2))
    model.add(Dense(3, activation='softmax'))
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model

# Wrap the model using KerasClassifier
model = KerasClassifier(model=create_model, verbose=0)

# Define the parameter grid
param_grid = {
    'model__embed_dim': [64, 128],
    'model__lstm_out': [100, 196],
    'batch_size': [32, 64],
    'epochs': [1]  # Keep low for quick search; increase for real training
}

# Apply GridSearchCV
grid = GridSearchCV(estimator=model, param_grid=param_grid, cv=3, verbose=2)
grid_result = grid.fit(X_train, Y_train)

# Print results
print("Best Score: {:.4f} using {}".format(grid_result.best_score_, grid_result.best_params_))




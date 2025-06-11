import os
import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense, SpatialDropout1D
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import joblib


# 1. Load and Prepare Data

real_df = pd.read_csv(r"C:\Users\PC\Downloads\individual_project\data\real_songs.csv")
fake_df = pd.read_csv(r"C:\Users\PC\Downloads\individual_project\data\fake_songs.csv")

real_df['label'] = 0
fake_df['label'] = 1

df = pd.concat([real_df, fake_df], ignore_index=True)
df = df.dropna(subset=['lyrics'])  # Ensure 'lyrics' column is not null

texts = df['lyrics'].astype(str).values
labels = df['label'].values

# 2. Text Preprocessing

tokenizer = Tokenizer(num_words=10000, oov_token="<OOV>")
tokenizer.fit_on_texts(texts)
sequences = tokenizer.texts_to_sequences(texts)

# Remove empty sequences
filtered = [(seq, label) for seq, label in zip(sequences, labels) if len(seq) > 0]
if not filtered:
    raise ValueError("All sequences are empty after tokenization. Check your data!")

sequences, labels = zip(*filtered)
X = pad_sequences(sequences, maxlen=300)
y = np.array(labels)


# 3. Define and Train Model

model = Sequential([
    Embedding(input_dim=10000, output_dim=128),
    SpatialDropout1D(0.2),
    LSTM(128, dropout=0.2, recurrent_dropout=0.2),
    Dense(1, activation='sigmoid')
])

model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

model.fit(X, y, batch_size=32, epochs=5, validation_split=0.2)


# 4. Save Model and Tokenizer

os.makedirs("model", exist_ok=True)
model.save("model/lstm_model.h5")
joblib.dump(tokenizer, "model/tokenizer.pkl")

print(" Training complete. Model and tokenizer saved.")

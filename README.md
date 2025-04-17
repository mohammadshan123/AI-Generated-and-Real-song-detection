# AI-Generated Song vs. Real Song Detection Using Deep Learning

This project aims to classify whether a song's lyrics are AI-generated or written by a human using a Long Short-Term Memory (LSTM) deep learning model. With the rise of generative AI models, distinguishing between machine-generated and real creative content is becoming increasingly important.

## 📌 Project Description

We built a binary classification model to detect whether a given song's lyrics are real (written by a human) or generated by an AI. The project uses natural language processing techniques and an LSTM network to analyze lyrical structure, patterns, and semantics.

---

## 📂 Dataset

The dataset contains two categories:
- **Real Songs**: Lyrics of songs written by humans.
- **AI-Generated Songs**: Lyrics generated using AI models like GPT, T5, etc.

### Dataset Sources:
- Real song lyrics: Publicly available datasets from Kaggle or other sources.
- AI-generated lyrics: Created using pretrained generative models.

The data was preprocessed to clean text, remove noise, and tokenize the lyrics.

---

## 🧠 Model: LSTM

The core of the project is an LSTM-based deep learning model built using TensorFlow/Keras. LSTM is well-suited for sequential text data and helps capture temporal dependencies in lyrics.

### Model Architecture:
- Embedding Layer
- LSTM Layer
- Dense Output Layer (Sigmoid activation for binary classification)

---

## ⚙️ Technologies Used

- Python 🐍
- TensorFlow / Keras
- NumPy, Pandas
- Matplotlib / Seaborn (for visualization)
- Scikit-learn (for preprocessing and evaluation)
- Jupyter Notebook
- Flask (for deployment - optional)

---

## 🚀 How to Run

1. **Clone the repository**
   git clone https://github.com/Mohammad Shan. N. S/song-lyric-detector.git
   cd song-lyric-detector
   
2.**Install dependencies**
  pip install -r requirements.txt

2.**Train the model**
  python train_lstm.py

3.**Test / Predict**
  python predict.py

4.**Run Flask app**
  python app.py

### Requirements
Make sure you have the following libraries installed

Flask
pandas
joblib
SpeechRecognition
tensorflow


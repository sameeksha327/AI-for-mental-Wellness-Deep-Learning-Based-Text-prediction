
 🧠 AI for Mental Wellness — Deep Learning-Based Text Prediction and Assessment

 📌 Overview

This project uses **Natural Language Processing (NLP)** and **Deep Learning** to analyze text data and predict the mental wellness state of an individual.
It identifies patterns and emotions like **Anxiety, Stress, Depression, Suicidal thoughts, Bipolar disorder, Personality disorder,** and **Normal** mental state based on user statements.

 🎯 Objective

To build a deep learning model that can automatically detect a person’s mental health condition from written text and assist in early emotional assessment.

🧾 Dataset

* Total Records:~53,000
* Columns:

  * `statement` — Text input expressing a person’s thoughts or feelings
  * `status` — Label of the mental wellness class
* Preprocessing Steps:

  * Removed null and noisy records
  * Cleaned unwanted symbols and special characters
  * Retained only 7 valid emotion classes
  * Tokenized, padded, and lowercased all text


🔍 Data Visualization & Charts

1. Class Distribution Chart**
   Shows the number of samples for each emotion class. The dataset was slightly imbalanced, with some emotions (like “Normal”) having more samples than others.

2. **Word Frequency Chart**
   Highlights the most common words in each class — for example, “sad”, “alone”, and “hurt” were frequent in the Depression class.

3. **Confusion Matrix**
   Displays how well the model predicted each class.
   More diagonal values = higher accuracy for those categories.

4. Accuracy & Loss Curves
   The training and validation accuracy gradually improved, showing the model learned effectively without overfitting.


⚙️ Model Building

Two models were experimented with:

1. DistilBERT (Benchmark Model)

* Pre-trained transformer with ~66M parameters
* Captures bidirectional context from sentences
* Accuracy: ~81–90%
* Drawback: Slow inference and needs GPU

 2. BiLSTM (Final Model)

* Architecture:
  `Embedding (100 dim) → BiLSTM (128) → BiLSTM (64) → Dense (128, ReLU) → Dropout → Dense (7, Softmax)`
* Optimizer: Adam
* Loss Function: Categorical Crossentropy
* Used weighted training to handle data imbalance
* Randomly initialized embeddings were learned during training

✅ Final Accuracy: Around **88–90%**
✅ Model chosen: BiLSTM (for efficiency and good accuracy)

 📊 Results Summary

| Model      | Accuracy | Key Feature                     | Drawback                    |
| ---------- | -------- | ------------------------------- | --------------------------- |
| DistilBERT | 81–90%   | Strong contextual understanding | Slower inference            |
| BiLSTM     | 88–90%   | Lightweight, fast, efficient    | Requires more preprocessing |

 💬 Examples of Each Class

| Class                | Example Statement                                             |
| -------------------- | ------------------------------------------------------------- |
| Anxiety              | “oh my gosh”                                                  |
| Normal               | “Dreaming of my ex crush, life feels fine.”                   |
| Depression           | “I just want the pain to stop; I feel empty inside.”          |
| Suicidal             | “I feel like life isn’t worth living anymore.”                |
| Stress               | “I’m worried about my health and can’t relax.”                |
| Bipolar              | “One moment I’m happy, next I’m angry for no reason.”         |
| Personality Disorder | “I want to connect with people, but I’m scared of rejection.” |

🧩 Conclusion

The BiLSTM model effectively identifies emotional states from text and can support early mental health awareness and intervention.
It shows how AI and NLP can be used to **analyze emotions, detect distress**, and **support mental wellness** through text-based communication.

🚀 Future Work

* Improve dataset balance using advanced data augmentation
* Implement real-time text emotion detection API
* Deploy as a web app or chatbot for mental wellness support

 *Tech Stack*

* Python
* TensorFlow / Keras
* Pandas, NumPy
* Matplotlib / Seaborn
* NLP techniques (tokenization, padding, embeddings)





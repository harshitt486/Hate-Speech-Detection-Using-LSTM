# 🧠 Hate Speech Detection Using LSTM

A Deep Learning based NLP project that detects whether a given text contains hate speech or not using an LSTM (Long Short-Term Memory) network.  
This project focuses on automated content moderation and online safety.

---

## 📌 Problem Statement

With the rapid growth of social media platforms, detecting and filtering hateful content has become essential.  
This project builds a deep learning model that classifies text into:

Classes:
0 → Non-hate
1 → Hate


using Natural Language Processing and LSTM.

---

## 📊 Dataset

- Labeled text dataset for hate speech detection  
- Total test samples: **5000**

*(Dataset source link here:- https://www.kaggle.com/datasets/pankaazshah/code-mixed-text-dataset-hinglish)*

---

## ⚙️ Tech Stack

- Python  
- TensorFlow / Keras  
- NumPy  
- Pandas  
- Matplotlib / Seaborn  
- Scikit-learn  
- NLP (Tokenization, Padding)

---

## 🧠 Model Architecture

- Text Preprocessing  
- Tokenization & Sequence Padding  
- Embedding Layer  
- LSTM Layer  
- Dropout Layer  
- Dense Output Layer  

---

## 📈 Training Performance

| Metric | Value |
|--------|-------|
Final Training Loss | **0.5034**  
Accuracy | **74%**  
Macro Precision | **0.80**  
Macro Recall | **0.78**  
Macro F1-score | **0.74**  
Weighted F1-score | **0.73**  

---

## 📉 Classification Report

| Class | Precision | Recall | F1-score | Support |
|-------|-----------|--------|----------|----------|
| 0 (Non-Hate) | 0.60 | 1.00 | 0.75 | 2000 |
| 1 (Hate) | 1.00 | 0.56 | 0.72 | 3000 |

### ✅ Overall Accuracy: **74%**

---

## 🔢 Confusion Matrix

[[2000 0]
[1316 1684]]


### Interpretation

- Class **0** is perfectly recalled (no false negatives)
- Some hate speech instances are misclassified as non-hate
- Model shows scope for improving recall for hate speech detection

---

## 📊 Model Learning Curve

### Training Loss per Epoch

Epoch 1 → 0.5610
Epoch 2 → 0.5541
Epoch 3 → 0.5485
Epoch 4 → 0.5445
Epoch 5 → 0.5397
Epoch 6 → 0.5357
Epoch 7 → 0.5320
Epoch 8 → 0.5270
Epoch 9 → 0.5218
Epoch 10 → 0.5165
Epoch 11 → 0.5106
Epoch 12 → 0.5034


✅ Loss consistently decreases → model is learning effectively.

---

## 🚀 How to Run the Project

### 1️⃣ Clone the repository

```bash
git clone https://github.com/harshitt486/Hate-Speech-Detection-Using-LSTM.git
cd Hate-Speech-Detection-Using-LSTM
2️⃣ Install dependencies
pip install -r requirements.txt
3️⃣ Run the model
python Hate_Speech_Detection.py
🔍 Sample Predictions
Input Text	Prediction
I hate you	Hate Speech
Have a nice day	Non-Hate Speech
🌍 Real-World Applications
Social media content moderation

Cyberbullying detection

Online community monitoring

AI-based safety systems

🔮 Future Improvements
Use BiLSTM / GRU

Hyperparameter tuning

Pretrained embeddings (GloVe / Word2Vec)

Transformer-based models (BERT)

Deploy using Streamlit / Flask

📂 Project Structure
Hate-Speech-Detection-Using-LSTM
│── Hate_Speech_Detection.py
│── requirements.txt
│── README.md
👨‍💻 Author
Harshit Kumar Tiwari
🎓 B.Tech CSE (Cyber Security)
📍 India

🔗 GitHub: https://github.com/harshitt486
🔗 LinkedIn: https://www.linkedin.com/in/harshit-tiwari-8206b1329

⭐ Support
If you found this project useful, give it a ⭐ on GitHub!

📄 README.md
# Hate Speech Detection Using LSTM (Hinglish Text)

## 📌 Project Overview

This project implements a **Hate Speech Detection system** for Hinglish (Hindi + English mixed) text using a **Long Short-Term Memory (LSTM)** deep learning model.

The model classifies text into:

- **0 → Non-Hate Speech**
- **1 → Hate Speech**

This project demonstrates an end-to-end NLP pipeline including:

- Text preprocessing
- Train-test split
- Vocabulary creation
- Sequence encoding
- LSTM model training
- Performance evaluation

---

## 📊 Dataset

- Total samples: **25,000**
- Hate speech: **15,000**
- Non-hate speech: **10,000**
- Language: **Hinglish**

⚠️ Dataset is not uploaded due to size limitations.

Update the dataset path in the code before running:

```python
pd.read_csv(r"YOUR_DATASET_PATH")

🧠 Model Architecture

Text Cleaning using Regex

Train-Test Split (80/20)

Vocabulary built using training data only

Sequence padding

LSTM Neural Network:

Embedding Layer

LSTM Layer

Dropout Layer

Fully Connected Layer

Class imbalance handled using:

BCEWithLogitsLoss(pos_weight)

⚙️ Technologies Used

Python

PyTorch

Pandas

Scikit-learn

Regex (re)

📂 Project Structure
├── Hate_Speech_Detection.py
├── README.md
└── requirements.txt

▶️ How To Run The Project
1️⃣ Clone the Repository
git clone https://github.com/YOUR_USERNAME/YOUR_REPO_NAME.git
cd YOUR_REPO_NAME

2️⃣ Install Dependencies
pip install torch pandas scikit-learn


OR

pip install -r requirements.txt

3️⃣ Update Dataset Path

In the Python file, update:

pd.read_csv(r"YOUR_DATASET_PATH")

4️⃣ Run the Script
python Hate_Speech_Detection.py

📈 Model Performance

Accuracy: ~75%

Macro F1-score: ~0.74

Confusion Matrix
[[2000    0]
 [1259 1741]]

Classification Report
Class	Precision	Recall	F1-score
Non-Hate	0.61	1.00	0.76
Hate	1.00	0.58	0.73
🎯 Key Features

✔ Handles Hinglish text
✔ Prevents data leakage (vocab built after split)
✔ Handles class imbalance
✔ Uses LSTM for sequential learning
✔ Realistic evaluation metrics

🚀 Future Improvements

Bidirectional LSTM

Transformer models (BERT)

Web app deployment (Flask/Streamlit)

Real-time hate speech detection

🎓 Academic Use

This project was developed as a Mini Project for B.Tech (Computer Science – Cyber Security).

👨‍💻 Author

Harshit Kumar Tiwari

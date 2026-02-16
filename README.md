📄 README.md
# Hate Speech Detection Using LSTM (Hinglish Text)

## 📌 Project Overview

This project implements a **Hate Speech Detection system for Hinglish (Hindi + English code-mixed) text** using a **Long Short-Term Memory (LSTM)** deep learning model.

The model classifies text into:

- **0 → Non-Hate Speech**
- **1 → Hate Speech**

This project demonstrates a complete NLP pipeline from preprocessing to evaluation.
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

⚙️ Technologies Used:

-Python
-PyTorch
-Pandas
-Scikit-learn
-Regex (re) 

## 🚀 Project Highlights

✔ Handles Hinglish code-mixed text  
✔ Prevents data leakage (vocabulary built after train-test split)  
✔ Handles class imbalance using weighted loss  
✔ Uses LSTM for sequential learning  
✔ Realistic evaluation using multiple metrics  

---

## 🧠 How It Works

Raw Text
↓
Text Cleaning
↓
Tokenization
↓
Sequence Padding
↓
LSTM Model
↓
Prediction (Hate / Non-Hate)


---

## 📂 Project Structure

Hate-Speech-Detection-Using-LSTM
│
├── src/
│ └── Hate_Speech_Detection.py
│
├── dataset/
│ └── hinglish_cyberbullying_dataset_25000.csv ← place dataset here
│
├── requirements.txt
├── README.md
└── sample_output.png


---

## ⚙️ Installation & Setup

### 1️⃣ Clone the Repository

```bash
git clone https://github.com/harshitt486/Hate-Speech-Detection-Using-LSTM.git
cd Hate-Speech-Detection-Using-LSTM
2️⃣ Install Dependencies
pip install -r requirements.txt
This makes the project runnable in one command.

3️⃣ Run the Project
python src/Hate_Speech_Detection.py
📈 Results
Accuracy: 75%
Macro F1-score: 0.74

Confusion Matrix
[[2000    0]
 [1259 1741]]

🔮 Future Work
Bidirectional LSTM

BERT fine-tuning

Streamlit web app deployment

Real-time hate speech detection

🎓 Academic Use
Developed as a Mini Project for B.Tech – Computer Science (Cyber Security).

👨‍💻 Author
Harshit Kumar Tiwari

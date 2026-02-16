import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import re
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix

# 1️⃣ Load Dataset (FULL PATH FIXED)
data = pd.read_csv("dataset/hinglish_cyberbullying_dataset_25000.csv")

texts = data['Text'].astype(str).tolist()
labels = data['Label'].tolist()

# 2️⃣ Clean Text
def clean_text(text):
    text = re.sub(r"http\S+", "", text)
    text = re.sub(r"@\w+", "", text)
    text = re.sub(r"[^a-zA-Z\s]", "", text)
    return text.lower()

texts = [clean_text(t) for t in texts]

# 3️⃣ Train-Test Split FIRST
texts_train, texts_test, y_train, y_test = train_test_split(
    texts, labels, test_size=0.2, random_state=42, stratify=labels
)

# 4️⃣ Build Vocabulary ONLY from training data
vocab = {"<PAD>": 0, "<UNK>": 1}
word_idx = 2

def encode_texts(text_list):
    global word_idx
    encoded = []
    for text in text_list:
        tokens = text.split()
        token_indices = []
        for token in tokens:
            if token not in vocab:
                vocab[token] = word_idx
                word_idx += 1
            token_indices.append(vocab[token])
        encoded.append(token_indices)
    return encoded

X_train = encode_texts(texts_train)

# Encode test using same vocab
X_test = []
for text in texts_test:
    tokens = text.split()
    token_indices = [vocab.get(token, vocab["<UNK>"]) for token in tokens]
    X_test.append(token_indices)

max_seq_length = max(len(seq) for seq in X_train)

X_train = [seq + [0] * (max_seq_length - len(seq)) for seq in X_train]
X_test = [seq + [0] * (max_seq_length - len(seq)) for seq in X_test]

X_train = torch.tensor(X_train)
X_test = torch.tensor(X_test)
y_train = torch.tensor(y_train)
y_test = torch.tensor(y_test)

# 5️⃣ LSTM Model
class LSTMClassifier(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim):
        super(LSTMClassifier, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, batch_first=True)
        self.dropout = nn.Dropout(0.5)
        self.fc = nn.Linear(hidden_dim, 1)

    def forward(self, x):
        embedded = self.embedding(x)
        lstm_out, (hidden, cell) = self.lstm(embedded)
        out = self.dropout(hidden[-1])
        out = self.fc(out)
        return out

model = LSTMClassifier(len(vocab), 64, 64)

# 6️⃣ Handle Class Imbalance
num_positive = sum(y_train).item()
num_negative = len(y_train) - num_positive
pos_weight = torch.tensor(num_negative / num_positive)

criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
optimizer = optim.Adam(model.parameters(), lr=0.001)

# 7️⃣ Training
epochs = 12

for epoch in range(epochs):
    model.train()
    optimizer.zero_grad()
    outputs = model(X_train).view(-1)
    loss = criterion(outputs, y_train.float())
    loss.backward()
    optimizer.step()
    print(f"Epoch {epoch+1}, Loss: {loss.item()}")

# 8️⃣ Evaluation
model.eval()
with torch.no_grad():
    outputs = torch.sigmoid(model(X_test))
    predictions = (outputs > 0.5).int().view(-1)

print("\nClassification Report:\n")
print(classification_report(y_test.numpy(), predictions.numpy()))

print("\nConfusion Matrix:\n")
print(confusion_matrix(y_test.numpy(), predictions.numpy()))


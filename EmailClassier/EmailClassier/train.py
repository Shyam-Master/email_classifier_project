# train.py
from models import EmailClassifier

classifier = EmailClassifier()
classifier.train("data/emails.csv")
print("✅ Model trained and saved as model.pkl")

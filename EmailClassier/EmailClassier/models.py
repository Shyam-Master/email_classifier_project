import pandas as pd
import torch
from sklearn.preprocessing import LabelEncoder
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer, BertForSequenceClassification
from torch.optim import AdamW
import spacy
import json
from tqdm import tqdm

# Load SpaCy NER
nlp = spacy.load("en_core_web_sm")

ENTITY_MAP = {
    "PERSON": "full_name",
    "EMAIL": "email",
    "PHONE_NUMBER": "phone_number",
    "DATE": "dob"
}

def mask_pii(text):
    doc = nlp(text)
    masked_text = text
    entities = []
    offset = 0

    for ent in doc.ents:
        if ent.label_ in ENTITY_MAP:
            classification = ENTITY_MAP[ent.label_]
            start = ent.start_char + offset
            end = ent.end_char + offset
            entity_val = ent.text
            masked = f"[{classification}]"

            masked_text = masked_text[:start] + masked + masked_text[end:]
            offset += len(masked) - (end - start)
            entities.append({
                "position": [start, start + len(masked)],
                "classification": classification,
                "entity": entity_val
            })
    return masked_text, entities

# Load and preprocess dataset
df = pd.read_csv("emails.csv")
label_encoder = LabelEncoder()
df['label'] = label_encoder.fit_transform(df['type'])

# Tokenizer and Model
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=len(df['label'].unique()))

# Custom Dataset
class EmailDataset(Dataset):
    def __init__(self, texts, labels):
        self.encodings = tokenizer(texts.tolist(), truncation=True, padding=True, max_length=64)
        self.labels = labels.tolist()

    def __getitem__(self, idx):
        item = {k: torch.tensor(v[idx]) for k, v in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels[idx])
        return item

    def __len__(self):
        return len(self.labels)

dataset = EmailDataset(df['email'], df['label'])
loader = DataLoader(dataset, batch_size=8, shuffle=True)

# Training
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
model.train()
optimizer = AdamW(model.parameters(), lr=5e-5)

for epoch in range(4):
    print(f"Epoch {epoch+1}/4")
    for batch in tqdm(loader):
        inputs = {k: v.to(device) for k, v in batch.items()}
        outputs = model(**inputs)
        loss = outputs.loss
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

# Inference function
def classify_and_mask(email_text):
    model.eval()
    inputs = tokenizer(email_text, return_tensors="pt", truncation=True, padding=True, max_length=64).to(device)
    with torch.no_grad():
        outputs = model(**inputs)
    pred = torch.argmax(outputs.logits, dim=1).item()
    predicted_label = label_encoder.inverse_transform([pred])[0]

    masked_email, entities = mask_pii(email_text)

    return {
        "input_email_body": email_text,
        "list_of_masked_entities": entities,
        "masked_email": masked_email,
        "category_of_the_email": predicted_label
    }

# Example usage
email = "Hi, I'm Meera. My email is meera@gmail.com. Please help with server access."
result = classify_and_mask(email)
print(json.dumps(result, indent=2))

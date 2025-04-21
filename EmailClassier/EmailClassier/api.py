from fastapi import FastAPI
from pydantic import BaseModel
from utils import mask_pii
from models import EmailClassifier

app = FastAPI()
classifier = EmailClassifier()
classifier.load()

class EmailInput(BaseModel):
    email_body: str

@app.post("/")
def classify_email(input: EmailInput):
    original_email = input.email_body
    masked_email, entities = mask_pii(original_email)
    category = classifier.predict(masked_email)
    
    return {
        "input_email_body": original_email,
        "list_of_masked_entities": entities,
        "masked_email": masked_email,
        "category_of_the_email": category
    }

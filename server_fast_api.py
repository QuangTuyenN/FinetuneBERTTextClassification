# Import library for build API
from fastapi import FastAPI
from pydantic import BaseModel
from typing import List
from fastapi.middleware.cors import CORSMiddleware

# Import lib for bert
import torch
from transformers import AutoTokenizer, XLMRobertaForSequenceClassification

# Load model đã finetune và tokenizer
model_name = "quangtuyennguyen/vi_xlmroberta_text_classification"  # Thay bằng đường dẫn tới model đã finetune

tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=False)
model = XLMRobertaForSequenceClassification.from_pretrained(model_name)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

id2label = {3: '4 sao', 1: '2 sao', 0: '1 sao', 4: '5 sao', 2: '3 sao'}


def get_prediction(text):
    input_encoded = tokenizer(text, return_tensors='pt').to(device)
    with torch.no_grad():
        outputs = model(**input_encoded)
    logits = outputs.logits
    pred = torch.argmax(logits, dim=1).item()
    return id2label[pred]


# Khởi tạo FastAPI app
app_text_classification = FastAPI()


# khởi tạo model dữ liệu đầu vào api
class InputData(BaseModel):
    text: str


# Cấu hình CORS
app_text_classification.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Cho phép tất cả các nguồn (nếu muốn giới hạn, thay "*" bằng danh sách các URL cụ thể)
    allow_credentials=True,
    allow_methods=["*"],  # Cho phép tất cả các phương thức (GET, POST, PUT, DELETE, v.v.)
    allow_headers=["*"],  # Cho phép tất cả các headers
)


@app_text_classification.post("/process")
def process_data(input_data: InputData):
    input_to_rank = input_data.text
    rank = get_prediction(input_to_rank)
    return {
        "rank": rank
    }






















import torch
from transformers import AutoTokenizer, XLMRobertaForSequenceClassification

# Load model đã finetune và tokenizer
model_name = "quangtuyennguyen/bert_base_train_dir"  # Thay bằng đường dẫn tới model đã finetune

tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=False)
model = XLMRobertaForSequenceClassification.from_pretrained(model_name)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

example = "Máy cao cấp nhưng tần số quét chỉ có 60Hz"

id2label = {3: '4 sao', 1: '2 sao', 0: '1 sao', 4: '5 sao', 2: '3 sao'}


def get_prediction(text):
    input_encoded = tokenizer(text, return_tensors='pt').to(device)
    with torch.no_grad():
        outputs = model(**input_encoded)
    logits = outputs.logits
    pred = torch.argmax(logits, dim=1).item()
    return id2label[pred]


while True:
    input_data = input("Nhập câu comment bạn muốn phân loại sao: ")
    print("Kết quả phân loại: ", get_prediction(input_data))

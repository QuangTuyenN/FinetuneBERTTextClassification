from huggingface_hub import login
import pandas as pd
import os
import torch
import json
from datasets import load_dataset, Dataset, DatasetDict
from transformers import AutoTokenizer, Automodel, XLMRobertaForSequenceClassification, AutoConfig, TrainingArguments, Trainer
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, classification_report
import evaluate
import numpy as np

# Đặt các token vào biến môi trường hoặc gán trực tiếp
os.environ["HUGGINGFACE_WRITE_TOKEN"] = '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'

login('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')

# khai báo pre train:
model_ckpt = "5CD-AI/Vietnamese-Sentiment-visobert"

# download dataset
dataset_org = load_dataset("thanhchauns2/vietnamese-sentiment-analysis")
df = pd.DataFrame(dataset_org['train'])
df2 = pd.DataFrame(dataset_org['test'])

# Nối df và df2 lại với nhau thành 1 DataFrame duy nhất
df = pd.concat([df, df2], axis=0)
# Reset lại index
df = df.reset_index(drop=True)
# Hiển thị df
print(df)

# Đổi tên cột 'comment' thành 'text'
df = df.rename(columns={'comment': 'text'})

# Thêm cột 'label_name' dựa trên giá trị của cột 'label' 1 tương ứng với 1 sao, ...
df['label_name'] = df['label'].apply(lambda x: f"{x} sao")

# Kiểm tra lại DataFrame sau khi thêm cột 0 tương ứng với 1 sao, ...
df['label'] = df['label'] - 1

# kiểm tra dataset
# thêm cột tính số lượng từ của câu
df['Words per Tweet'] = df['text'].str.split().apply(len)
# vẽ ra đồ thị tham khảo
df.boxplot("Words per Tweet", by="label_name")

# load tokenizer
tokenizer = AutoTokenizer.from_pretrained(model_ckpt)

# train test val
train, test = train_test_split(df, test_size=0.3, stratify=df['label_name'])
test, validation = train_test_split(test, test_size=1/3, stratify=test['label_name'])

dataset = DatasetDict(
    {'train': Dataset.from_pandas(train, preserve_index=False),
     'test': Dataset.from_pandas(test, preserve_index=False),
     'validation': Dataset.from_pandas(validation, preserve_index=False)
     }
)


def tokenize(batch):
    temp = tokenizer(batch['text'], padding=True, truncation=True)
    return temp


# label2id, id2label
label2id = {x['label_name']: x['label'] for x in dataset['train']} # {'4 sao': 3, '2 sao': 1, '1 sao': 0, '5 sao': 4, '3 sao': 2}
id2label = {v: k for k, v in label2id.items()} # {3: '4 sao', 1: '2 sao', 0: '1 sao', 4: '5 sao', 2: '3 sao'}

# load pretrain model
model = AutoModel.from_pretrained(model_ckpt)

num_labels = len(label2id)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
config = AutoConfig.from_pretrained(model_ckpt, label2id=label2id, id2label=id2label)
model = XLMRobertaForSequenceClassification.from_pretrained(model_ckpt, config=config, ignore_mismatched_sizes=True).to(device)

batch_size = 64
training_dir = "quangtuyen_xlm_roberta_text_classification"

training_args = TrainingArguments( output_dir=training_dir,
                                  overwrite_output_dir = True,
                                  num_train_epochs = 2,
                                  learning_rate = 2e-5,
                                  per_device_train_batch_size = batch_size,
                                  per_device_eval_batch_size = batch_size,
                                  weight_decay = 0.01,
                                  evaluation_strategy = 'epoch',
                                  disable_tqdm = False)

# build compute metric function
accuracy = evaluate.load("accuracy")


def compute_metrics_evaluate(eval_pred):
    predictions, labels = eval_pred
    predictions = np.argmax(predictions, axis=1)
    return accuracy.compute(predictions=predictions, references=labels)


def compute_metrics(pred):
    labels = pred.label_ids
    preds = pred.predictions.argmax(-1)
    f1 = f1_score(labels, preds, average="weighted")
    acc = accuracy_score(labels, preds)
    return {"accuracy": acc, "f1": f1}


# xử lý dataset qua hàm tokenize trước khi train
emotion_encoded = dataset.map(tokenize, batched=True, batch_size=None)

# train model
trainer = Trainer(model=model, args=training_args,
                  compute_metrics=compute_metrics,
                  train_dataset = emotion_encoded['train'],
                  eval_dataset = emotion_encoded['validation'],
                  tokenizer = tokenizer)
trainer.train()

# val
preds_output = trainer.predict(emotion_encoded['test'])

y_pred = np.argmax(preds_output.predictions, axis=1)
y_true = emotion_encoded['test'][:]['label']

# so sánh giá trị true và predict
print(classification_report(y_true, y_pred))

# lưu model
trainer.save_model(training_dir)

# push to hub
trainer.push_to_hub()
















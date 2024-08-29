This project finetune BERT for TextClassification in Vietnamese language

It will classify customer's comments in electrical product such as mobilephone, laptop, headphone,... and the service. Comments will be rank in 5 level: 1 sao, 2 sao, 3 sao, 4 sao, 5 sao.

Pretrain model: 5CD-AI/Vietnamese-Sentiment-visobert it base on XLM Roberta

Dataset: vietnamese-sentiment-analysis

Train model, eval, push to Huggingface hub by train.py

Deploy model using fastapi by use_model.py

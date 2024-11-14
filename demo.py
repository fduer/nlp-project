# Use a pipeline as a high-level helper
from transformers import pipeline

model_name = 'D:/llama3.1-8b-instruct/LLM-Research/Meta-Llama-3___1-8B-Instruct'
messages = [
    {"role": "system", "content": '''You are Qwen, created by Alibaba Cloud. You are a helpful assistant.'''},
    {"role": "system", "content": '''Your task is to read the header information in Markdown format below and generate a Python code function called 'answer (table_name)' to answer the question. Each column in the header is given in the form of a 'column name (column type)' cell, and cells in different columns are separated by a '|'.
     '''},
    {"role": "user", "content": '''Here, the table information is:
```
SCIENTIFIC NAME(string) | GROUP(string) | SUB GROUP(string) | FOOD NAME(string) | FOOD NAME_gx_text_length(integer) | graphext_cluster(list) | graphext_cluster(list) | FOOD NAME_gx_lang(string) | FOOD NAME_gx_cardiff_nlp_sentiment_aux(string) | FOOD NAME_gx_cardiff_nlp_sentiment(string)
```
The question is: What are the top 3 most common food groups?
'''}
]
pipe = pipeline("text-generation", model=model_name, max_length=4096, device=0)
pipe(messages)
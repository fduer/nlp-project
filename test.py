import os
import dashscope
from sympy.polys.polyconfig import query
import transformers
import torch


class MODEL():
    def __init__(self):
        self.pipeline = ''

    def load_model(self):
        model_id = "/root/nlp_project/llama3.1-8b-adapted"
        pipeline = transformers.pipeline(
            "text-generation",
            model=model_id,
            model_kwargs={"torch_dtype": torch.bfloat16},
            device_map="auto",
            max_length=4096
        )
        self.pipeline = pipeline

    def run_model(self, user_question: str, table_info: str):
        messages = [
            {'role': 'system',
             'content': "Your task is to read the header information in Markdown format below and generate a Python code function called 'answer (table_name)' to answer the question. Each column in the header is given in the form of a 'column name (column type)' cell, and cells in different columns are separated by a '|'."
             },
            {'role': 'user', 'content': '''
        The table information is:
        ```
        SCIENTIFIC NAME(string) | GROUP(string) | SUB GROUP(string) | FOOD NAME(string)
        ```
        What are the bottom 4 least common food name lengths?

        Please output the string format of the Python code only.
        '''}, {
                'role': 'assistant',
                'content': '''def answer(data):\n  data['length'] = data['FOOD NAME'].apply(lambda x: len(x))\n  return data['length'].value_counts().index[-4:].tolist()'''
            },
            {'role': 'user', 'content': '''
        The table information is:
        ```
        SCIENTIFIC NAME(string) | GROUP(string) | SUB GROUP(string) | FOOD NAME(string)
        ```
        What are the top 2 most common sub groups?

        Please output the string format of the Python code only.
        '''
             },
            {
                'role': 'assistant',
                'content': '''def answer(data):\n  return data['SUB GROUP'].value_counts().index[:2].tolist()'''
            },
            {'role': 'user', 'content': '''
        The table information is:
        ```
        SCIENTIFIC NAME(string) | GROUP(string) | SUB GROUP(string) | FOOD NAME(string)
        ```
        What is the scientific name of the food named 'Kiwi'?

        Please output the string format of the Python code only.
        '''
             },
            {
                'role': 'assistant',
                'content': '''def answer(data):\n    return data[data['FOOD NAME'] == 'Kiwi']['SCIENTIFIC NAME'].values[0]'''
            }
        ]
        question = {'role': 'user', 'content': f'''
            The table information is:
        ```
        {table_info}
        ```

        The question is: {user_question}

        Please output the string format of the Python code only.
            '''}
        messages.append(question)
        response = self.pipeline(
            messages,
            max_new_tokens=1024,
        )
        return response[0]["generated_text"][-1]["content"]
    # response = dashscope.Generation.call(
    #     # 若没有配置环境变量，请用百炼API Key将下行替换为：api_key="sk-xxx",
    #     api_key="sk-47392a0b4923461eabd021351f98f063",
    #     model="qwen2.5-72b-instruct",  # 模型列表：https://help.aliyun.com/zh/model-studio/getting-started/models
    #     messages=messages,
    #     result_format='message'
    # )

    # return str(response["output"]["choices"][0]["message"]["content"])
# print(response["output"]["choices"][0]["message"]["content"])

import pandas as pd
import numpy as np
from datasets import load_dataset
class Dataset:
    def __init__(self):
        self.data = {}

    def load_data(self):

        # Load all QA pairs
        all_qa = load_dataset("cardiffnlp/databench", name="qa", split="train")

        # Load SemEval 2025 task 8 Question-Answer splits
        semeval_train_qa = load_dataset("cardiffnlp/databench", name="semeval", split="train")
        semeval_dev_qa = load_dataset("cardiffnlp/databench", name="semeval", split="dev")
        Staff_Satisfaction = pd.read_parquet(f"hf://datasets/cardiffnlp/databench/data/018_Staff/all.parquet")
        Staff_Satisfaction_dict = {}
        Staff_Satisfaction_dict['table'] = Staff_Satisfaction
        Staff_Satisfaction_dict['info'] = "Satisfaction Level(float64) | Work Accident(object) | Average Monthly Hours(int64) | Last Evaluation(float64) | Years in the Company(int64) | salary(object) | Department(object) | Number of Projects(int64) | Promoted in the last 5 years?(object) | Date Hired(object) | Left(object)"
        self.data['Staff Satisfaction'] = Staff_Satisfaction_dict

        Food_Names_dict = {}
        Food_Names = pd.read_parquet(f"hf://datasets/cardiffnlp/databench/data/015_Food/all.parquet")
        Food_Names_dict['table'] = Food_Names
        Food_Names_dict['info'] = "SCIENTIFIC NAME(string) | GROUP(string) | SUB GROUP(string) | FOOD NAME(string)"
        self.data['Food Names'] = Food_Names_dict

    def get_data(self, name: str):
        return self.data[name]

    def run_query(self, table_name, ans):
        print(ans)
        exec (ans, globals())

        return answer(self.data[table_name]['table'])
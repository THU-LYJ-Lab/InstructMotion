import json
import pandas as pd
from datasets import Dataset


jsonl_path = "preference_data/preference_labels.jsonl"
data = list()
with open(jsonl_path, "r") as file:
    for line in file:
        data.append(json.loads(line))

df = pd.DataFrame(data)
dataset = Dataset.from_pandas(df)

dataset = dataset.filter(lambda x: x["chosen"][-1]["choice"] != "skipped")
dataset = dataset.train_test_split(test_size=0.1, shuffle=True, seed=2222)
train_dataset, eval_dataset = dataset["train"], dataset["test"]

# randomly select 100 from eval_dataset
import random
random.seed(2222)
eval_dataset = eval_dataset.shuffle(seed=2222)
eval_dataset = eval_dataset.select(range(100))

selected_path = "preference_data/selected_prompts_test.txt"
with open(selected_path, "w") as file:
    for i in range(len(eval_dataset)):
        file.write(f"{eval_dataset[i]['id']}#{eval_dataset[i]['prompt']}\n")
        print(f"{eval_dataset[i]['id']}#{eval_dataset[i]['prompt']}")

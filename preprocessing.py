import pandas as pd
import json


with open("math_train.json") as json_file:
    json_data = json.load(json_file)
    print(json_data['data'])

dict_one_hot_answer = {
    0: "A",
    1: "B",
    2: "C",
    3: "D"
}

list_question = []
list_answer = []
list_A = []
list_B = []
list_C = []
list_D = []
list_explanation = []

for record in json_data['data']:
  question = record['question']
  choices = record['choices']
  try:
    explanation = record['explanation']
  except KeyError:
    explanation = "None"
  answer = record['answer']

  list_A.append(choices[0])
  list_B.append(choices[1])
  list_C.append(choices[2])
  try:
    list_D.append(choices[3])
  except IndexError:
    list_D.append("None")
  list_question.append(question)
  one_hot_answer = choices.index(answer)
  list_answer.append(dict_one_hot_answer[one_hot_answer])
  list_explanation.append(explanation)

data_df = pd.DataFrame(list(zip(list_question, list_explanation, list_A, list_B, list_C, list_D, list_answer)),
                       columns=['question', 'explanation', 'A', 'B', 'C', 'D', 'answer'])
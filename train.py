import json
import torch
from datasets import Dataset
from transformers import AutoModelForMultipleChoice, TrainingArguments, Trainer
from peft import LoraConfig, get_peft_model, TaskType

######
model = AutoModelForMultipleChoice.from_pretrained('MODEL')

######
if USE_PEFT:
    print('We are using PEFT.')
    from peft import LoraConfig, get_peft_model, TaskType
    peft_config = LoraConfig(
        r=8, lora_alpha=4, task_type=TaskType.SEQ_CLS, lora_dropout=0.1,
        bias="none", inference_mode=False,
        target_modules=["query_proj", "value_proj"],
        modules_to_save=['classifier','pooler'],
    )
    model = get_peft_model(model, peft_config)
    model.print_trainable_parameters()


######
if FREEZE_EMBEDDINGS:
    print('Freezing embeddings.')
    for param in model.deberta.embeddings.parameters():
        param.requires_grad = False
if FREEZE_LAYERS>0:
    print(f'Freezing {FREEZE_LAYERS} layers.')
    for layer in model.deberta.encoder.layer[:FREEZE_LAYERS]:
        for param in layer.parameters():
            param.requires_grad = False


######
from sklearn.metrics import accuracy_score
import numpy as np

def compute_accuracy(predictions, labels):
    # Chọn lớp có xác suất cao nhất
    pred = np.argmax(predictions, axis=1)
    # Tính độ chính xác
    accuracy = accuracy_score(labels, pred)
    return accuracy

def compute_metrics(p):
    predictions = p.predictions.tolist()
    labels = p.label_ids.tolist()
    return {"accuracy": compute_accuracy(predictions, labels)}


######
training_args = TrainingArguments(
    warmup_ratio=0.1,
    learning_rate=2e-5,
    per_device_train_batch_size=2,
    per_device_eval_batch_size=2,
    num_train_epochs=15,
    report_to='none',
    output_dir = f'./checkpoints_{VER}',
    overwrite_output_dir=True,
    fp16=True,
    gradient_accumulation_steps=8,
    logging_steps=25,
    evaluation_strategy='steps',
    eval_steps=25,
    save_strategy="steps",
    save_steps=25,
    load_best_model_at_end=False,
    metric_for_best_model='accuracy',
    lr_scheduler_type='cosine',
    weight_decay=0.01,
    save_total_limit=2,
)


######
trainer = Trainer(
    model=model,
    args=training_args,
    tokenizer=tokenizer,
    data_collator=DataCollatorForMultipleChoice(tokenizer=tokenizer),
    train_dataset=tokenized_dataset,
    eval_dataset=tokenized_dataset_valid,
    compute_metrics = compute_metrics,
    # callbacks=[EarlyStoppingCallback(early_stopping_patience=5)],
)

trainer.train()
trainer.save_model(f'model_v{VER}')


del model, trainer
if USE_PEFT:
    model = AutoModelForMultipleChoice.from_pretrained(MODEL)
    model = get_peft_model(model, peft_config)
    checkpoint = torch.load(f'model_v{VER}/pytorch_model.bin')
    model.load_state_dict(checkpoint)
else:
    model = AutoModelForMultipleChoice.from_pretrained(f'model_v{VER}')
trainer = Trainer(model=model)


######
with open("math_test.json") as json_file:
    json_test = json.load(json_file)
    print(json_test['data'])


######
dict_one_hot_answer = {
    0: "A",
    1: "B",
    2: "C",
    3: "D"
}

list_id = []
list_question = []
list_explanation = []
list_A = []
list_B = []
list_C = []
list_D = []

for record in json_test['data']:
  id = record['id']
  question = record['question']
  choices = record['choices']
  explanation = "None"

  list_A.append(choices[0])
  list_B.append(choices[1])
  list_C.append(choices[2])
  try:
    list_D.append(choices[3])
  except IndexError:
    list_D.append("None")

  list_id.append(id)
  list_question.append(question)
  list_explanation.append(explanation)


######
df_test = pd.DataFrame(list(zip(list_id, list_question, list_explanation, list_A, list_B, list_C, list_D)),
                       columns=['id', 'question', 'explanation', 'A', 'B', 'C', 'D'])


######
def test_preprocess(example):
    first_sentence = [ "[CLS] " + example['explanation'] ] * 4
    second_sentences = [" #### " + example['question'] + " [SEP] " + example[option] + " [SEP]" for option in 'ABCD']
    tokenized_example = tokenizer(first_sentence, second_sentences, truncation=True,
                                  max_length=MAX_INPUT, add_special_tokens=False, padding='max_length')

    return tokenized_example


######
tokenized_test_dataset = Dataset.from_pandas(df_test).map(
        test_preprocess, remove_columns=['id', 'question', 'explanation', 'A', 'B', 'C', 'D'])
# tokenized_dataset_valid
test_predictions = trainer.predict(tokenized_test_dataset).predictions
predictions_as_ids = np.argsort(-test_predictions, 1)
predictions_as_answer_letters = np.array(list('ABCD'))[predictions_as_ids]
predictions_as_string = df_test['prediction'] = [
    ' '.join(row) for row in predictions_as_answer_letters[:, :1]
]


######
# Tạo dictionary ánh xạ từ chữ cái đến câu trả lời
answer_dict = {'A': 'A', 'B': 'B', 'C': 'C', 'D': 'D'}

# Chuyển đổi các chữ cái thành câu trả lời
def map_prediction_to_answer(row):
    return row[answer_dict[row['prediction']]]

df_test['answer'] = df_test.apply(map_prediction_to_answer, axis=1)


######
df_test
submission = df_test[['id', 'answer']]
submission
submission.to_csv('output/submission_2.csv', index=False)

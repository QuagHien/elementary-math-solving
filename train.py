from transformers import AutoModelForMultipleChoice, TrainingArguments, Trainer

######
model = AutoModelForMultipleChoice.from_pretrained('microsoft/deberta-v3-large')

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
# %%
import pandas as pd
import numpy as np

# %%
import torch
import random
from transformers.file_utils import is_tf_available, is_torch_available, is_torch_tpu_available
from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments
from sklearn.model_selection import train_test_split
import transformers
from sklearn.metrics import f1_score, accuracy_score, precision_score, recall_score

# %%
df=pd.read_csv('input9.csv')

# %%
df['class']=df['class'].apply(lambda x: 0 if x=='WITHOUT_CLASSIFICATION' else 1)

# %%
X=df["comment"].copy()
y=df["class"].copy()
x_train, x_test, y_train, y_test = train_test_split(X.tolist(), y, test_size=0.20, random_state=42, stratify=y)
x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, test_size=0.25, random_state=42, stratify=y_train)

# %%
model = AutoModelForSequenceClassification.from_pretrained('deberta_model', num_labels = 2)
tokenizer = AutoTokenizer.from_pretrained("deberta_tokenizer", model_max_length=512)

# %%
train_encodings = tokenizer(x_train, truncation=True, padding=True, max_length=512)
valid_encodings = tokenizer(x_val, truncation=True, padding=True, max_length=512)
test_encodings = tokenizer(x_test, truncation=True, padding=True, max_length=512)

class MakeTorchData(torch.utils.data.Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        item = {k: torch.tensor(v[idx]) for k, v in self.encodings.items()}
        item["labels"] = torch.tensor(self.labels[idx])
        return item

    def __len__(self):
        return len(self.labels)

# convert our tokenized data into a torch Dataset
train_dataset = MakeTorchData(train_encodings, y_train.ravel())
valid_dataset = MakeTorchData(valid_encodings, y_val.ravel())
test_dataset = MakeTorchData(test_encodings, y_test.ravel())

# %%
from sklearn.metrics import f1_score, accuracy_score, precision_score, recall_score
class F1():
    def info(self):
        return "Nothing Here, lol"
    def compute(self, predictions, references, labels=None, pos_label=1, average="binary", sample_weight=None):
        score = f1_score(
            references, predictions, labels=labels, pos_label=pos_label, average=average, sample_weight=sample_weight
        )
        return {"f1": float(score) if score.size == 1 else score}

# %%
# Create Metrics
def compute_metrics(eval_pred):

  predictions, labels = eval_pred
  
  predictions = np.argmax(predictions, axis=1)

  score_f1 = f1_score(labels, predictions, pos_label=1, average="binary")
  score_acc = accuracy_score(labels, predictions)
  score_pre = precision_score(labels, predictions, pos_label=1 , average="binary", sample_weight=None)
  score_rec = recall_score(labels, predictions, pos_label=1 , average="binary", sample_weight=None)

  return {"accuracy": float(score_acc), "precision": float(score_pre), "recall": float(score_rec), "f1": float(score_f1)}

# %%
training_args = TrainingArguments(
    output_dir='./results_dv1i',          # output directory
    num_train_epochs=10,     # total number of training epochs
    per_device_train_batch_size=16,   # batch size per device during training
    per_device_eval_batch_size=16,   # batch size for evaluation
    warmup_steps=500,                # number of warmup steps for learning rate scheduler
    weight_decay=0.01,               # strength of weight decay
    logging_dir='./logs_dv1i',            # directory for storing logs
    load_best_model_at_end=True,     # load the best model when finished training (default metric is loss)
    metric_for_best_model = "f1",    # select the base metrics
    logging_steps=1000,               # log & save weights each logging_steps
    save_steps=4000,
    evaluation_strategy="steps",     # evaluate each `logging_steps`
) 

# %%
model_name="debertav3"

# %%
trainer = Trainer(
    model=model,                         # the instantiated Transformers model to be trained
    args=training_args,                  # training arguments, defined above
    train_dataset=train_dataset,         # training dataset
    eval_dataset=valid_dataset,          # evaluation dataset
    compute_metrics=compute_metrics,     # the callback that computes metrics of interest
    #optimizers= (torch.optim.AdamW(model.parameters(), lr=1e-3), None)
)

# %%
trainer.train()

# %%
eval_metrics = trainer.evaluate(eval_dataset=test_dataset)
print(eval_metrics)

# %%
y_pred = trainer.predict(test_dataset)

# %%
y_pred2 = list([np.argmax(x) for x in y_pred[0]])

# %%
from sklearn.metrics import classification_report
print(classification_report(y_test, y_pred2, digits=6))


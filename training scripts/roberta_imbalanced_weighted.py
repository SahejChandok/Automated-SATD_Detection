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
df["class"] = df["class"].apply(lambda x: 0 if x == 'WITHOUT_CLASSIFICATION' else 1)
X=df["comment"]
y=df["class"]

counts = np.unique(np.array(y), return_counts=True)[1]
class_weight = [1.0, counts[0] / counts[1]]
class_weight = torch.from_numpy(np.array(class_weight).astype(np.float32)).cuda()

x_train, x_test, y_train, y_test = train_test_split(X.tolist(), y, test_size=0.20, random_state=42, stratify=y)
x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, test_size=0.25, random_state=42, stratify=y_train)

# %%
model = AutoModelForSequenceClassification.from_pretrained('roberta-base_model', num_labels = 2)
tokenizer = AutoTokenizer.from_pretrained("roberta-base_tokenizer", model_max_length=512)

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
# Create Metrics
def compute_metrics(eval_pred):

  predictions, labels = eval_pred

  predictions = np.argmax(predictions, axis=1)

  score_f1 = f1_score(labels, predictions, pos_label=1, average="binary")
  score_acc = accuracy_score(labels, predictions)
  score_pre = precision_score(labels, predictions, pos_label=1 , average="binary")
  score_rec = recall_score(labels, predictions, pos_label=1 , average="binary")

  return {"accuracy": float(score_acc), "precision": float(score_pre), "recall": float(score_rec), "f1": float(score_f1)}

# %%
training_args = TrainingArguments(
    output_dir='./results_rbiw',          # output directory
    num_train_epochs=10,     # total number of training epochs
    per_device_train_batch_size=16,   # batch size per device during training
    per_device_eval_batch_size=16,   # batch size for evaluation
    warmup_steps=500,                # number of warmup steps for learning rate scheduler
    weight_decay=0.01,               # strength of weight decay
    logging_dir='./logs_rbiw',            # directory for storing logs
    load_best_model_at_end=True,     # load the best model when finished training (default metric is loss)
    metric_for_best_model = "f1",    # select the base metrics
    logging_steps=1000,               # log & save weights each logging_steps
    save_steps=4000,
    evaluation_strategy="steps",     # evaluate each `logging_steps`
) 

# %%
model_name="roberta-base"

# %%
class CustomTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False):
        """
        How the loss is computed by Trainer. By default, all models return the loss in the first element.
        Subclass and override for custom behavior.
        """
  
        outputs = model(**inputs)

        # Save past state if it exists
        # TODO: this needs to be fixed and made cleaner later.
        if self.args.past_index >= 0:
            self._past = outputs[self.args.past_index]

        labels = inputs.get("labels")
        logits = outputs.get("logits")

        # compute custom loss (suppose one has 3 labels with different weights)
        #loss_fct = torch.nn.CrossEntropyLoss(weight=class_weight, reduction='mean')
        loss_fct = torch.nn.CrossEntropyLoss(weight = class_weight)
        loss = loss_fct(logits.view(-1, 2), labels.view(-1))
        return (loss, outputs) if return_outputs else loss

# %%
trainer = CustomTrainer(
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



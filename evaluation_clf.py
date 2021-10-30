import math
import numpy as np
import random
from sklearn.metrics import classification_report

import torch
import os
from tqdm import tqdm,trange
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
from keras.preprocessing.sequence import pad_sequences
from transformers import BertTokenizer, BertConfig
from transformers import BertForSequenceClassification, AdamW
from ctextgen.dataset import SST_Dataset
from transformers import get_linear_schedule_with_warmup


def get_special_tokens(tokenizer):

    pad_tok = tokenizer.vocab["[PAD]"]
    sep_tok = tokenizer.vocab["[SEP]"]
    cls_tok = tokenizer.vocab["[CLS]"]

    return pad_tok, sep_tok, cls_tok



max_len = 100
epochs = 8
learning_rate = 5e-7
bert_vocab = 'bert-large-uncased'
batch_num = 4
bert_out_address = './clf/bert_clf_sst_final'


tokenizer=BertTokenizer.from_pretrained("bert-large-uncased", do_lower_case=True)
pad_tok, sep_tok, cls_tok = get_special_tokens(tokenizer)


if not os.path.exists(bert_out_address):
        os.makedirs(bert_out_address)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
n_gpu = torch.cuda.device_count()
print("gpu: ", n_gpu)
print(torch.cuda.is_available())

dataset = SST_Dataset(mbsize=1, emb_dim=300, fixlen=64, init_token='[CLS]', eos_token='[SEP]', pad_token='[PAD]')
train_size = dataset.train_size()
val_size = dataset.validation_size()


tokenized_texts_train = []
labels_train = []
tokenized_texts_val = []
labels_val = []

for i in tqdm(range(train_size)):
    text, label = dataset.next_batch()
    tokenized_texts_train.append(tokenizer.tokenize(dataset.idxs2sentence(text)))
    labels_train.append(label)

for i in tqdm(range(val_size)):
    text, label = dataset.next_validation_batch()
    tokenized_texts_val.append(tokenizer.tokenize(dataset.idxs2sentence(text)))
    labels_val.append(label)





tr_inputs = pad_sequences([tokenizer.convert_tokens_to_ids(txt) for txt in tokenized_texts_train],
                          maxlen=max_len, dtype="long", truncating="post", padding="post",
                          value=pad_tok)
tr_tags = labels_train
tr_masks = [[int(i != pad_tok) for i in ii] for ii in tr_inputs]

print("train examples: {}".format(len(tr_tags)))
val_inputs = pad_sequences([tokenizer.convert_tokens_to_ids(txt) for txt in tokenized_texts_val],
                          maxlen=max_len, dtype="long", truncating="post", padding="post",
                          value=pad_tok)
val_tags = labels_val
val_masks = [[int(i != pad_tok) for i in ii] for ii in val_inputs]
print("val examples: {}".format(len(val_tags)))


tr_inputs = torch.LongTensor(tr_inputs)
val_inputs = torch.LongTensor(val_inputs)
tr_tags = torch.LongTensor(tr_tags)
val_tags = torch.LongTensor(val_tags)
tr_masks = torch.LongTensor(tr_masks)
val_masks = torch.LongTensor(val_masks)


model = BertForSequenceClassification.from_pretrained(bert_vocab,num_labels=2)



train_data = TensorDataset(tr_inputs, tr_masks, tr_tags)
train_sampler = RandomSampler(train_data)
train_dataloader = DataLoader(train_data, sampler=train_sampler, batch_size=batch_num,drop_last=True)

valid_data = TensorDataset(val_inputs, val_masks, val_tags)
valid_sampler = SequentialSampler(valid_data)
valid_dataloader = DataLoader(valid_data, sampler=valid_sampler, batch_size=batch_num)

max_grad_norm = 1.0
num_train_optimization_steps = int(math.ceil(len(tr_inputs) / batch_num) / 1) * epochs


if n_gpu > 1:
    model = torch.nn.DataParallel(model)

model = model.to(device)

FULL_FINETUNING = True
if FULL_FINETUNING:
    param_optimizer = list(model.named_parameters())
    no_decay = ['bias', 'gamma', 'beta']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)],
         'weight_decay_rate': 0.01},
        {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)],
         'weight_decay_rate': 0.0},
    ]

else:
    param_optimizer = list(model.classifier.named_parameters())
    optimizer_grouped_parameters = [{"params": [p for n, p in param_optimizer]}]

total_steps = len(train_dataloader) * epochs
optimizer = AdamW(optimizer_grouped_parameters, lr=learning_rate, eps=1e-9)
scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=0, num_training_steps=total_steps)


seed = 7
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
training_states = []
dev_best_acc = 0.0
dev_best_f1 = 0.1

tokenizer.save_vocabulary(bert_out_address)

print("***** Running training *****")
print("  Num examples = %d" % (len(tr_inputs)))
print("  Batch size = %d" % (batch_num))
print("  Num steps = %d" % (num_train_optimization_steps))
epoch = 0
tolstep = 0
stop_count = 0
min_loss = 1e9
max_accuracy = 0.1
for _ in trange(epochs, desc="Epoch"):
    print()
    epoch += 1
    tr_loss = 0
    tr_accuracy = 0
    nb_tr_examples, nb_tr_steps = 0, 0
    tr_preds, tr_labels = [], []
    model.train()
    for step, batch in zip(tqdm(range(len(train_dataloader)),position=0,desc="Data"), train_dataloader):
        tolstep += 1
        batch = tuple(t.to(device) for t in batch)
        b_input_ids, b_input_mask, b_labels = batch

        optimizer.zero_grad()
        # forward pass
        outputs = model(b_input_ids, token_type_ids=None,
                        attention_mask=b_input_mask, labels=b_labels)
        loss, tr_logits = outputs[:2]
        if n_gpu > 1:
            loss = loss.mean()

        loss.backward()
        optimizer.step()

        tr_loss += loss.item()
        nb_tr_examples += b_input_ids.size(0)
        nb_tr_steps += 1

        tr_batch_labels = b_labels.to("cpu").numpy()
        tr_batch_preds = torch.argmax(tr_logits, axis=1)
        tr_batch_preds = tr_batch_preds.detach().cpu().numpy()


        tr_preds.extend(tr_batch_preds)
        tr_labels.extend(tr_batch_labels)
        torch.nn.utils.clip_grad_norm_(parameters=model.parameters(), max_norm=max_grad_norm)
        scheduler.step()

    tr_loss = tr_loss / nb_tr_steps
    cl_report = classification_report(tr_labels, tr_preds,
                                      target_names=[dataset.idx2label(0), dataset.idx2label(1)],
                                      digits=3)

    print()
    print(f"Train loss: {tr_loss}")
    print(f"Classification Report:\n {cl_report}")
    print()
    print("Running Validation...")
    model.eval()

    eval_loss, eval_accuracy = 0, 0
    nb_eval_steps, nb_eval_examples = 0, 0
    predictions, true_labels = [], []

    for step, batch in enumerate(valid_dataloader):
        batch = tuple(t.to(device) for t in batch)
        b_input_ids, b_input_mask, b_labels = batch
        with torch.no_grad():
            outputs = model(
                b_input_ids,
                token_type_ids=None,
                attention_mask=b_input_mask,
                labels=b_labels,
            )
            tmp_eval_loss, logits = outputs[:2]

        label_ids = b_labels
        val_batch_preds = torch.argmax(logits, axis=1)
        val_batch_preds = val_batch_preds.detach().cpu().numpy()
        val_batch_labels = label_ids.to("cpu").numpy()
        predictions.extend(val_batch_preds)
        true_labels.extend(val_batch_labels)

        eval_loss += tmp_eval_loss.mean().item()
        nb_eval_examples += b_input_ids.size(0)
        nb_eval_steps += 1
    cl_report = classification_report(true_labels, predictions, target_names=[dataset.idx2label(0), dataset.idx2label(1)],
                                      digits=3, output_dict=True)
    accuracy = cl_report['accuracy']
    cl_report = classification_report(true_labels, predictions,
                                      target_names=[dataset.idx2label(0), dataset.idx2label(1)], digits=3)
    eval_loss = eval_loss / nb_eval_steps
    if accuracy > max_accuracy:
        max_accuracy = accuracy
        model_to_save = model.module if hasattr(model, 'module') else model  # Only save the model it-self
        output_model_file = os.path.join(bert_out_address, "best_pytorch_model.bin")
        output_config_file = os.path.join(bert_out_address, "best_config.json")
        torch.save(model_to_save.state_dict(), output_model_file)
        model_to_save.config.to_json_file(output_config_file)

    print(f"Validation loss: {eval_loss}")
    print(f"Classification Report:\n {cl_report}")
    print()
    model_to_save = model.module if hasattr(model, 'module') else model  # Only save the model it-self
    output_model_file = os.path.join(bert_out_address, "pytorch_model.bin")
    output_config_file = os.path.join(bert_out_address, "config.json")
    torch.save(model_to_save.state_dict(), output_model_file)
    model_to_save.config.to_json_file(output_config_file)
print('Training complete!')

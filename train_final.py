import os
import torch

import numpy as np
from torch.autograd import Variable
import random
from ctextgen.dataset import *
from ctextgen.finalmodel import RNN_VAE
from tqdm import trange
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
from transformers import get_linear_schedule_with_warmup, AdamW


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
n_gpu = torch.cuda.device_count()
print("gpu: ", n_gpu)
print(torch.cuda.is_available())

mb_size = 32
sen_len = 32

save_file = 'finalmodels'

seed = 7
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)

dataset = SST_Dataset(emb_dim=300, mbsize=1, fixlen=sen_len)
tr_inputs = []
tr_labels = []
for i in range(dataset.trainlen):
    inputs, labels = dataset.next_batch()
    tr_inputs.append(inputs.view(-1))
    tr_labels.append(labels)


tr_inputs = torch.stack(tr_inputs, dim=0)
tr_labels = torch.stack(tr_labels, dim=0)


train_data = TensorDataset(tr_inputs, tr_labels)
train_sampler = RandomSampler(train_data)
train_dataloader = DataLoader(train_data, sampler=train_sampler, batch_size=mb_size, drop_last=False)


def main():
    model = RNN_VAE(
        dataset.n_vocab, h_dim, z_dim, c_dim, cnew_dim, dropout_prob=0.1, p_word_dropout=0.3, max_sent_len=dataset.fixlen,
        pretrained_embeddings=dataset.get_vocab_vectors(), freeze_embeddings=False,
        gpu=torch.cuda.is_available()
    )


    if n_gpu > 1 and use_parallel:
        model = torch.nn.DataParallel(model)
    model.cuda()
    # Annealing for KL term
    kld_start_inc = int(total_steps / 8)
    kld_end_inc = int(total_steps)
    kld_weight = 0.0
    kld_max = max_weight
    kld_inc = (kld_max - kld_weight) / (kld_end_inc - kld_start_inc)
    optimizer = AdamW(model.parameters(), lr=lr, eps=1e-9)
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=100, num_training_steps=total_steps)

    epoch = 0
    tolstep = 0
    trainning_loss = []
    print("***** Running training *****")
    print("  Num examples = %d" % (len(tr_inputs)))
    print("  Batch size = %d" % (mb_size))
    print("  Num steps = %d" % (total_steps))
    for _ in trange(epochs, desc="Epoch"):
        epoch += 1
        for step, batch in zip(range(len(train_dataloader)), train_dataloader):
            model.train()
            tolstep += 1
            batch = tuple(t.to(device) for t in batch)
            inputs, labels = batch
            inputs = inputs.transpose(0,1)
            labels = labels.transpose(0,1)
            optimizer.zero_grad()
            recon_loss, kl_loss, kl_loss_attn = model(sentence=inputs, labels=labels, use_hmean=use_hmean)

            if use_aneal == False:
                kld_weight = kld_max

            loss = recon_loss + kld_weight * (kl_loss + kl_loss_attn * kl_attn_weight)

            if tolstep > kld_start_inc and kld_weight < kld_max:
                kld_weight += kld_inc

            if use_aneal == False:
                kld_weight = kld_max

            if n_gpu > 1 and use_parallel:
                loss = loss.mean()

            loss.backward()
            optimizer.step()
            grad_norm = torch.nn.utils.clip_grad_norm_(parameters=model.parameters(), max_norm=10.0)
            scheduler.step()

            if tolstep % log_interval == 1:
                model_ = model.module if hasattr(model, 'module') else model
                z = model_.sample_z_prior(1)
                c = Variable(torch.tensor(0))
                c = c.cuda() if model_.gpu else c
                sample_idxs = model_.sample_sentence_attn(z, c, attn_prior=attn_prior)
                sample_sent = dataset.idxs2sentence(sample_idxs)
                print()
                out_loss = 'Iter-{}; Loss: {:.4f}; Recon: {:.4f}; KL: {:.4f}; KL_attn: {:.4f}; Grad_norm: {:.4f}; Kl_weight: {:.4f};'.format(
                    tolstep,
                    loss.item(), recon_loss.item(), kl_loss.item(), kl_loss_attn.item(), grad_norm, kld_weight)
                print(out_loss)
                trainning_loss.append(out_loss)
                trainning_loss.append(sample_sent)
                print('Sample: "{}"'.format(sample_sent))

    print(tolstep)
    if not os.path.exists(save_name):
        os.makedirs(save_name)

    model = model.module if hasattr(model, 'module') else model
    file_name = os.path.join(save_name, 'model.bin')
    torch.save(model.state_dict(), file_name)
    print(save_name)
    print("Trainning Completed !")
    print()
    file_name = os.path.join(save_name, 'loss.txt')
    with open(file_name, 'w') as ott:
        for i in range(len(trainning_loss)):
            ott.write(trainning_loss[i])
            ott.write("\n")



if __name__ == '__main__':

    h_dim = 300
    z_dim = h_dim
    c_dim = 1
    use_hmean = False
    attn_prior = True
    use_parallel = True
    use_aneal = True
    epochs = 200
    total_steps = len(train_dataloader) * epochs
    log_interval = int(total_steps / 20)
    max_weight = 0.3
    cnew_dim = 400
    lr = 1e-3
    kl_attn_weight = 0.5
    save_name = os.path.join(save_file, 'vae_attn_sst_' + str(sen_len) + '_lr_' + str(lr)
                             + '_cnew_' + str(cnew_dim) + '_weight_' + str(kl_attn_weight)
                             + "_" + str(max_weight)
                             + "_2")
    print(save_name)
    main()



import torch
from ctextgen.dataset import SST_Dataset
from ctextgen.finalmodel import RNN_VAE

import pandas as pd
import os
import _locale
_locale._getdefaultlocale = (lambda *args: ['en_US', 'utf8'])
from tqdm import tqdm


textTest = []
text = []
use_gpu = torch.cuda.is_available()
torch.cuda.set_device(1)

mb_size = 32
h_dim = 300
lr_decay_every = 1000000
n_iter = 20000
log_interval = 1000
z_dim = h_dim
c_dim = 1
temper = 1
cnew_dim = 300
save_file = 'finalmodels'
sentence_size = 15000
usesam = True

sen_len = 16
lr = 1e-3
cnew_dim = 400
kl_attn_weight = 0.8
max_weight = 0.6

dataset = SST_Dataset(mbsize=1, emb_dim=300, fixlen=sen_len)

fl = os.path.join(save_file, 'vae_attn_sst_' + str(sen_len) + '_lr_' + str(lr)
                  + '_cnew_' + str(cnew_dim) + '_weight_' + str(kl_attn_weight)
                  + "_" + str(max_weight)
                  + "_2")
save_name = os.path.join(fl, 'model.bin')


save_csv = os.path.join(fl, 'generate.csv')

model = RNN_VAE(
    dataset.n_vocab, h_dim, z_dim, c_dim, cnew_dim, p_word_dropout=0.3, max_sent_len=dataset.fixlen,
    pretrained_embeddings=dataset.get_vocab_vectors(), freeze_embeddings=True,
    gpu=use_gpu
)

out_file_name = save_csv
model.load_state_dict(torch.load(save_name, map_location=lambda storage, loc: storage))

sentences = []
labels = []
print(save_csv)
print('Begin...')

i = 0
for i in tqdm(range(sentence_size)):
    model.eval()
    z = model.sample_z_prior(1)
    c = torch.Tensor([0])
    c = c.view(1, -1)
    c = c.cuda() if model.gpu else c
    sample_idxs = model.sample_sentence_attn(z, c, temp=temper, attn_prior=True, usesam=usesam)
    sentences.append(dataset.idxs2sentence(sample_idxs))
    labels.append(0)

print('complete...')
i = 0

for i in tqdm(range(sentence_size)):
    model.eval()
    z = model.sample_z_prior(1)
    c = torch.Tensor([1])
    c = c.view(1, -1)
    c = c.cuda() if model.gpu else c
    sample_idxs = model.sample_sentence_attn(z, c, temp=temper, attn_prior=True, usesam=usesam)
    sentences.append(dataset.idxs2sentence(sample_idxs))
    labels.append(1)

df = pd.DataFrame({"sentence":sentences, "label":labels})
print(df.head(10))
df.to_csv(out_file_name, index=None)
print(df['sentence'].value_counts())
print(len(df))
print()
print(out_file_name)

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from attention import LuongAttention

class RNN_VAE(nn.Module):

    def __init__(self, n_vocab, h_dim, z_dim, c_dim, cnew_dim, dropout_prob = 0.3, p_word_dropout=0.3, unk_idx=0, pad_idx=1, start_idx=2, eos_idx=3, max_sent_len=100, pretrained_embeddings=None, freeze_embeddings=False, gpu=False):
        super(RNN_VAE, self).__init__()

        self.UNK_IDX = unk_idx
        self.PAD_IDX = pad_idx
        self.START_IDX = start_idx
        self.EOS_IDX = eos_idx
        self.MAX_SENT_LEN = max_sent_len

        self.n_vocab = n_vocab
        self.h_dim = h_dim
        self.z_dim = z_dim
        self.c_dim = c_dim
        self.cnew_dim = cnew_dim
        self.p_word_dropout = p_word_dropout
        self.dropout_prob = dropout_prob
        self.gpu = gpu


        if pretrained_embeddings is None:
            self.emb_dim = h_dim
            self.word_emb = nn.Embedding(n_vocab, h_dim, self.PAD_IDX)
        else:
            self.emb_dim = pretrained_embeddings.size(1)
            self.word_emb = nn.Embedding(n_vocab, self.emb_dim, self.PAD_IDX)

            # Set pretrained embeddings
            self.word_emb.weight.data.copy_(pretrained_embeddings)

            if freeze_embeddings:
                self.word_emb.weight.requires_grad = False

        self.encoder = nn.GRU(self.emb_dim, h_dim)
        self.q_mu = nn.Linear(h_dim, z_dim)
        self.q_logvar = nn.Linear(h_dim, z_dim)
        self.attn = LuongAttention(self.h_dim, self.z_dim, use_cuda=gpu)
        self.attn_q_logvar = nn.Linear(self.h_dim, self.z_dim)
        self.c_new = nn.Linear(self.z_dim + self.c_dim, self.cnew_dim)
        self.z_to_h = nn.Linear(self.z_dim, self.h_dim)

        self.decoder_fc = nn.Linear(z_dim, n_vocab)
        self.decoder_attn = nn.GRU(self.emb_dim + self.h_dim + self.cnew_dim, self.z_dim)
        self.update_c = nn.GRU(self.z_dim, self.cnew_dim)
        self.dropout = nn.Dropout(self.p_word_dropout)

        if self.gpu:
            self.cuda()

    def forward_encoder(self, inputs):
        mask = torch.where(inputs == 1, 0, 1)
        inputs = self.word_emb(inputs)
        return self.forward_encoder_embed(inputs, mask)

    def forward_encoder_embed(self, inputs, mask):
        output, h = self.encoder(inputs, None)
        h = h.view(-1, self.h_dim)
        mu = self.q_mu(h)
        logvar = self.q_logvar(h)

        return mu, logvar, output, mask

    def sample_z(self, mu, logvar):
        eps = Variable(torch.randn(self.z_dim))
        eps = eps.cuda() if self.gpu else eps
        return mu + torch.exp(logvar/2) * eps

    def sample_z_prior(self, mbsize):
        z = Variable(torch.randn(mbsize, self.z_dim))
        z = z.cuda() if self.gpu else z
        return z

    def sample_c_prior(self, mbsize):
        c = Variable(
            torch.from_numpy(np.random.multinomial(1, [0.5, 0.5], mbsize).astype('float32'))
        )
        c = c.cuda() if self.gpu else c
        return c

    def sample_attn(self, attn_mu, attn_logvar):    #att_mu = attention_vector
        eps = Variable(torch.randn(self.h_dim))
        eps = eps.cuda() if self.gpu else eps
        return attn_mu + torch.exp(attn_logvar / 2) * eps

    def sample_attn_prior(self, mbsize):
        attn_vec = Variable(torch.randn(mbsize, self.h_dim))
        attn_vec = attn_vec.cuda() if self.gpu else attn_vec
        return attn_vec



    def forward_decoder_attn(self, inputs, z, enc_h, c, mask=None, use_hmean=False):

        dec_inputs = self.word_dropout(inputs)
        inputs_emb = self.word_emb(dec_inputs)
        inputs_emb = self.dropout(inputs_emb)
        c = c.unsqueeze(0)
        z = z.unsqueeze(0)
        init_h = self.z_to_h(z)

        output_all = []
        for i in range(inputs_emb.size(0)):
            attn_vec, weights = self.attn(enc_h, enc_h.size(1), init_h, init_h.size(1), mask)
            h_source_mu = torch.mean(enc_h, dim=0)
            attn_mu = attn_vec
            attn_logvar = self.attn_q_logvar(attn_vec)
            sample_attn_vec = self.sample_attn(attn_mu, attn_logvar)
            new_attn_vec = sample_attn_vec

            if i == 0:
                c = self.c_new(torch.cat([new_attn_vec, c], dim=2))

            new_inputs_emb = torch.cat([inputs_emb[i].unsqueeze(0), new_attn_vec, c], dim=2)

            outputs, _ = self.decoder_attn(new_inputs_emb, init_h)



            seq_len, mbsize, _ = outputs.size()
            outputs = outputs.view(seq_len * mbsize, -1)

            y = self.decoder_fc(outputs)
            y = y.view(seq_len, mbsize, self.n_vocab)
            output_all.append(y)
            init_h = outputs.unsqueeze(0)
            c, _ = self.update_c(init_h, c)

            if i == 0:
                if use_hmean:
                    attn_kl_loss = torch.mean(0.5 * torch.sum(torch.exp(attn_logvar) + (attn_mu - h_source_mu)**2 -1 - attn_logvar, 1))
                else:
                    attn_kl_loss = torch.mean(
                        0.5 * torch.sum(torch.exp(attn_logvar) + attn_mu ** 2 - 1 - attn_logvar, 1))
            else:
                if use_hmean:
                    attn_kl_loss += torch.mean(0.5 * torch.sum(torch.exp(attn_logvar) + (attn_mu - h_source_mu)**2 -1 - attn_logvar, 1))
                else:
                    attn_kl_loss += torch.mean(
                        0.5 * torch.sum(torch.exp(attn_logvar) + attn_mu ** 2 - 1 - attn_logvar, 1))
        output = torch.cat(output_all, 0)

        return output, attn_kl_loss



    def forward(self, sentence, labels, use_hmean=False):
        self.train()
        mbsize = sentence.size(1)
        pad_words = Variable(torch.LongTensor([self.PAD_IDX])).repeat(1, mbsize)
        pad_words = pad_words.cuda() if self.gpu else pad_words

        enc_inputs = sentence
        dec_inputs = sentence
        dec_targets = torch.cat([sentence[1:], pad_words], dim=0)


        mu, logvar, output_h, mask = self.forward_encoder(enc_inputs)
        z = self.sample_z(mu, logvar)
        c = labels.view(mbsize, -1)

        y, kl_loss_attn = self.forward_decoder_attn(dec_inputs, z, output_h, c, mask, use_hmean)

        recon_loss = F.cross_entropy(
            y.view(-1, self.n_vocab), dec_targets.view(-1), size_average=True, ignore_index=self.PAD_IDX
        )
        kl_loss = torch.mean(0.5 * torch.sum(torch.exp(logvar) + mu**2 - 1 - logvar, 1))


        return recon_loss, kl_loss, kl_loss_attn


    def sample_sentence_attn(self, z, c, attn_mu=None, attn_logvar=None, temp=1, attn_prior=True, usesam=False):

        self.eval()
        word = torch.LongTensor([self.START_IDX])
        word = word.cuda() if self.gpu else word
        word = Variable(word)  # '<start>'
        z = z.view(1, -1)
        c = c.view(1, -1)
        c = c.unsqueeze(0)

        z = z.view(1, 1, -1)
        h = self.z_to_h(z)
        if not isinstance(h, Variable):
            h = Variable(h)

        outputs = []


        for i in range(self.MAX_SENT_LEN):
            if attn_prior:
                sam_attn = self.sample_attn_prior(1)
                sam_attn = sam_attn.unsqueeze(0)
            else:
                sam_attn = self.sample_attn(attn_mu, attn_logvar)
                sam_attn = sam_attn.unsqueeze(0)
            emb = self.word_emb(word).view(1, 1, -1)

            if i == 0:
                c = self.c_new(torch.cat([sam_attn, c], dim=2))

            output, h = self.decoder_attn(torch.cat([emb, sam_attn, c], dim=2), h)
            c, _ = self.update_c(h, c)



            y = self.decoder_fc(output).view(-1)

            y = F.softmax(y / temp, dim=0)
            if usesam:
                idx = torch.multinomial(y, 1)
            else:
                idx = torch.max(y, 0)[1]

            word = Variable(torch.LongTensor([int(idx)]))
            word = word.cuda() if self.gpu else word

            idx = int(idx)

            if idx == self.EOS_IDX:
                break

            outputs.append(idx)

        self.train()

        return outputs


    def word_dropout(self, inputs):
        """
        Do word dropout: with prob `p_word_dropout`, set the word to '<unk>'.
        """
        if isinstance(inputs, Variable):
            data = inputs.data.clone()
        else:
            data = inputs.clone()

        # Sample masks: elems with val 1 will be set to <unk>
        mask = torch.from_numpy(
            np.random.binomial(1, p=self.p_word_dropout, size=tuple(data.size()))
                     .astype('uint8')
        )

        if self.gpu:
            mask = mask.cuda()

        # Set to <unk>
        data[mask] = self.UNK_IDX

        return Variable(data)

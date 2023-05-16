import torch.nn as nn
import torch
import torch.nn.functional as F
import numpy as np
import random
import os
from models.allennlp_beamsearch import BeamSearch
import math
from torch.autograd import Variable


class AttentionShare(nn.Module):
    def __init__(self, input_value_size, input_key_size, output_size, dropout=0.1):
        super(AttentionShare, self).__init__()
        self.input_value_size = input_value_size
        self.input_key_size = input_key_size
        self.attention_size = output_size
        self.dropout = dropout

        self.K = nn.Linear(in_features=input_value_size, out_features=output_size, bias=False)
        self.Q = nn.Linear(in_features=input_key_size, out_features=output_size, bias=False)
        self.V = nn.Linear(in_features=input_value_size, out_features=output_size, bias=False)
        self.output_layer = nn.Sequential(
            nn.Linear(in_features=self.attention_size, out_features=output_size, bias=False),
            nn.Tanh(),
            nn.LayerNorm(output_size),
            nn.Dropout(self.dropout)
        )

    def forward(self, meta_state, hidden_previous):
        K = self.K(meta_state)
        Q = self.Q(hidden_previous).unsqueeze(2)
        V = self.V(meta_state).transpose(-1, -2)

        logits = torch.div(torch.matmul(K, Q), torch.tensor(np.sqrt(self.attention_size)))
        weight = F.softmax(logits, dim=1)
        # weight = F.sigmoid(logits)
        mid_step = torch.matmul(V, weight)
        # mid_step = torch.matmul(V, weight)

        attention = mid_step.squeeze(2)
        attention = self.output_layer(attention)

        return attention, weight

# generate description according to the visual info
class Decoder(nn.Module):
    def __init__(self, args, vocab, multi_modal=False, baseline=False):
        super(Decoder, self).__init__()
        self.word_size = args.word_size
        self.max_words = args.max_words
        self.vocab = vocab
        self.cfg = args
        self.dataset = args.dataset
        self.vocab_size = len(vocab)
        self.beam_size = args.beam_size
        self.batch_size = args.batch_size
        self.query_hidden_size = args.query_hidden_size
        self.decode_hidden_size = args.decode_hidden_size
        self.multi_modal = multi_modal
        self.teacher_forcing_ratio = args.teacher_forcing_ratio
        # print('decoder parameters ------------')
        # print('word_size = ', args.word_size)
        # print('max_words = ', self.max_words)
        # print('vocab = ', self.vocab)
        # print('vocab_size = ', self.vocab_size)
        # print('beam_size = ', self.beam_size)
        # print('use_glove = ', args.use_glove)
        # print('multi-modal = ',self.multi_modal)
        # print('decoder parameters ------------')

        self.fusion_ind = nn.Sequential(
            nn.Linear(2*args.motion_projected_size,1),
            nn.Sigmoid()
        )
        self.fusion_rel = nn.Sequential(
            nn.Linear(2*args.motion_projected_size,1),
            nn.Sigmoid()
        )
        self.embed_ind = nn.Sequential(
            nn.Linear(args.reason_size,args.motion_projected_size),
            nn.LeakyReLU(0.2)
        )
        self.embed_rel = nn.Sequential(
            nn.Linear(args.reason_size,args.motion_projected_size),
            nn.LeakyReLU(0.2)
        )
        # word embedding matrix
        self.word_embed = nn.Embedding(self.vocab_size, self.word_size)
        if args.use_glove:
            self.get_glove_embedding()
        self.word_drop = nn.Dropout(p=args.dropout)

        # attention lstm
        query_input_size = 2*args.motion_projected_size + args.word_size + args.decode_hidden_size

        self.query_lstm = nn.LSTMCell(query_input_size, args.query_hidden_size)
        self.query_lstm_layernorm = nn.LayerNorm(args.query_hidden_size)
        self.query_lstm_drop = nn.Dropout(p=args.dropout)

        # decoder lstm
        lang_decode_hidden_size = args.visual_hidden_size + args.query_hidden_size
        self.lang_lstm = nn.LSTMCell(lang_decode_hidden_size, args.decode_hidden_size)
        self.lang_lstm_layernorm = nn.LayerNorm(args.decode_hidden_size)
        self.lang_lstm_drop = nn.Dropout(p=args.dropout)

        # context from attention
        self.context_att = AttentionShare(input_value_size=args.motion_projected_size*2,
                                          input_key_size=args.query_hidden_size,
                                          output_size=args.visual_hidden_size)
        self.context_layernorm = nn.LayerNorm(args.decode_hidden_size)

        # final output layer
        self.word_restore = nn.Linear(args.decode_hidden_size, self.vocab_size)
        nn.init.xavier_normal_(self.word_restore.weight)
        # testing stage
        # beam search: The BeamSearch class is imported from Allennlp
        # DOCUMENTS: https://docs.allennlp.org/master/api/nn/beam_search/
        self.beam_search = BeamSearch(vocab('<end>'), self.max_words, self.beam_size, per_node_beam_size=self.beam_size)

    def update_beam_size(self, beam_size):
        self.beam_size = beam_size
        self.beam_search = BeamSearch(self.vocab('<end>'), self.max_words, beam_size, per_node_beam_size=beam_size)

    def get_glove_embedding(self):
        #glove_np_path = '/root/data/Reason/src2/dataset/MSR-VTT/msr-vtt_glove.npy'
        glove_np_path = 'dataset/MSR-VTT/msr-vtt_glove.npy'
        if os.path.exists(glove_np_path):
            weight_matrix = np.load(glove_np_path)

        weight_matrix = torch.from_numpy(weight_matrix)
        self.word_embed.load_state_dict({'weight': weight_matrix})
        # self.word_embed.load_state_dict({'weight': weight_matrix})

    def _init_lstm_state(self, d, hidden_size):
        batch_size = d.size(0)
        lstm_state_h = d.data.new(batch_size, hidden_size).zero_()
        lstm_state_c = d.data.new(batch_size, hidden_size).zero_()
        return lstm_state_h, lstm_state_c

    def forward(self, cnn_feats,z_inds,z_rel,appear,motion,captions, max_words ,step_feats=None):
        self.batch_size = cnn_feats.size(0)
        # inference or training
        if captions is None:
            infer = True
        else:
            infer = False
        if max_words is None:
            max_words = self.max_words

        '''
        print("appear",appear.shape)
        print("a",a.shape)
        print("m",m.shape)
        '''

        appear = torch.mean(appear,dim=1)
        motion = torch.mean(motion,dim=1)
        #print("global_feat:",global_feat.shape)
        z_inds = self.embed_ind(z_inds)
        z_rel = self.embed_rel(z_rel)
        a_feats = torch.cat((appear,z_inds),dim=1)
        m_feats = torch.cat((motion,z_rel), dim=1)
        beta_a = self.fusion_ind(a_feats)
        beta_m = self.fusion_rel(m_feats)
        appearance_feats = appear + beta_a*z_inds
        motion_feats = motion + beta_m*z_rel
        global_feat = torch.cat((appearance_feats,motion_feats),dim=1)

        lang_lstm_h, lang_lstm_c = self._init_lstm_state(cnn_feats, self.decode_hidden_size)
        query_lstm_h, query_lstm_c = self._init_lstm_state(cnn_feats, self.query_hidden_size)
        '''
        print("lang_lstm_h:",lang_lstm_h.shape)
        print("lang_lstm_c:",lang_lstm_c.shape)
        print("q_lstm_h:",query_lstm_h.shape)
        print("q_lstm_c:",query_lstm_c.shape)
        '''
        # add a '<start>' sign
        start_id = self.vocab('<start>')
        start_id = cnn_feats.data.new(cnn_feats.size(0)).long().fill_(start_id)
        word = self.word_embed(start_id)
        word = self.word_drop(word)

        outputs = []
        alpha_all = []
        if not infer or self.beam_size == 1:
            for i in range(max_words):
                # lstm input: word + h_(t-1) + context
                word_logits, query_lstm_h, query_lstm_c, lang_lstm_h, lang_lstm_c, alpha =\
                    self.decode(word, query_lstm_h, query_lstm_c, lang_lstm_h, lang_lstm_c, global_feat, cnn_feats)
                # teacher_forcing: a training trick
                use_teacher_forcing = not infer and (random.random() < self.teacher_forcing_ratio)
                # use_teacher_forcing = False
                if use_teacher_forcing:
                    word_id = captions[:, i].cuda()
                else:
                    word_id = word_logits.max(1)[1].cuda()
                word = self.word_embed(word_id)
                word = self.word_drop(word)

                if infer:
                    outputs.append(word_id)
                else:
                    outputs.append(word_logits)
                    alpha_all.append(alpha)

            outputs = torch.stack(outputs, dim=1)
            alpha_all = torch.stack(alpha_all,dim=1)

        else:
            start_state = {'query_lstm_h': query_lstm_h, 'query_lstm_c': query_lstm_c,
                           'lang_lstm_h': lang_lstm_h, 'lang_lstm_c': lang_lstm_c,
                           'cnn_feats': cnn_feats, 'global_feat': global_feat}
            predictions, log_prob = self.beam_search.search(start_id, start_state, self.beam_step)
            max_prob, max_index = torch.topk(log_prob, 1)
            max_index = max_index.squeeze(1)
            for i in range(self.batch_size):
                outputs.append(predictions[i, max_index[i], :])
            outputs = torch.stack(outputs)

        return outputs, alpha_all

    def decode_tokens(self, tokens):
        '''
        convert word index to caption
        :param tokens: input word index
        :return: capiton
        '''
        words = []
        for token in tokens:
            token = token.item()
            if token == self.vocab('<end>'):
                break
            word = self.vocab.idx2word[token]
            words.append(word)
        captions = ' '.join(words)
        return captions

    def caption2wordembedding(self, caption):
        with torch.no_grad():
            word_embed = self.word_embed(caption)
            return word_embed

    def output2wordembedding(self, ouput):
        word_embed_weights = self.word_embed.weight.detach()
        word_embed = torch.matmul(ouput, word_embed_weights)
        return word_embed

    def beam_step(self, last_predictions, current_state):
        '''
        A function that is responsible for computing the next most likely tokens,
            given the current state and the predictions from the last time step.
            The function should accept two arguments. The first being a tensor
            of shape ``(group_size,)``, representing the index of the predicted
            tokens from the last time step, and the second being the current state.
            The ``group_size`` will be ``batch_size * beam_size``, except in the initial
            step, for which it will just be ``batch_size``.
            The function is expected to return a tuple, where the first element
            is a tensor of shape ``(group_size, target_vocab_size)`` containing
            the log probabilities of the tokens for the next step, and the second
            element is the updated state. The tensor in the state should have shape
            ``(group_size, *)``, where ``*`` means any other number of dimensions.
        Returns
        -------
        Tuple[torch.Tensor, torch.Tensor]
            Tuple of ``(predictions, log_probabilities)``, where ``predictions``
            has shape ``(batch_size, beam_size, max_steps)`` and ``log_probabilities``
            has shape ``(batch_size, beam_size)``.
        '''
        group_size = last_predictions.size(0)  # batch_size or batch_size*beam_size
        batch_size = self.batch_size
        log_probs = []
        new_state = {}
        num = int(group_size / batch_size)  # 1 or beam_size
        for k, state in current_state.items():
            if isinstance(state, list):
                state = torch.stack(state, dim=1)
            _, *last_dims = state.size()
            current_state[k] = state.reshape(batch_size, num, *last_dims)
            new_state[k] = []
        for i in range(num):
            # read current state
            query_lstm_h = current_state['query_lstm_h'][:, i, :]
            query_lstm_c = current_state['query_lstm_c'][:, i, :]
            lang_lstm_h = current_state['lang_lstm_h'][:, i, :]
            lang_lstm_c = current_state['lang_lstm_c'][:, i, :]
            cnn_feats = current_state['cnn_feats'][:, i, :]
            global_feat = current_state['global_feat'][:, i, :]
            cnn_feats_2 = None
            # decoding stage
            word_id = last_predictions.reshape(batch_size, -1)[:, i]
            word = self.word_embed(word_id)

            word_logits, query_lstm_h, query_lstm_c, lang_lstm_h, lang_lstm_c, _ = \
                self.decode(word, query_lstm_h, query_lstm_c, lang_lstm_h, lang_lstm_c, global_feat, cnn_feats)

            log_prob = F.log_softmax(word_logits, dim=1)  # b*v
            log_probs.append(log_prob)

            # update new state
            new_state['query_lstm_h'].append(query_lstm_h)
            new_state['query_lstm_c'].append(query_lstm_c)
            new_state['lang_lstm_h'].append(lang_lstm_h)
            new_state['lang_lstm_c'].append(lang_lstm_c)
            new_state['global_feat'].append(global_feat)
            new_state['cnn_feats'].append(cnn_feats)


        # transform log probabilities
        # from list to tensor(batch_size*beam_size, vocab_size)
        log_probs = torch.stack(log_probs, dim=0).permute(1, 0, 2).reshape(group_size, -1)  # group_size*vocab_size

        # transform new state
        # from list to tensor(batch_size*beam_size, *)
        for k, state in new_state.items():
            new_state[k] = torch.stack(state, dim=0)  # (beam_size, batch_size, *)
            _, _, *last_dims = new_state[k].size()
            dim_size = len(new_state[k].size())
            dim_size = range(2, dim_size)
            new_state[k] = new_state[k].permute(1, 0, *dim_size)  # (batch_size, beam_size, *)
            new_state[k] = new_state[k].reshape(group_size, *last_dims)  # (batch_size*beam_size, *)
        return (log_probs, new_state)

    def decode(self, word, query_lstm_h, query_lstm_c, lang_lstm_h, lang_lstm_c, global_feat, cnn_feats):
        # print("[lang_lstm_h,global_feat,word]:",cat1.shape)
        query_h, query_c = self.query_lstm(torch.cat([lang_lstm_h, global_feat, word], dim=1),
                                               (query_lstm_h, query_lstm_c))
        # query_lstm_h = self.att_lstm_drop(query_lstm_h)  B,1024
        query_current = self.query_lstm_drop(self.query_lstm_layernorm(query_h))
        # context from attention
        context, alpha = self.context_att(cnn_feats, query_current)
        # context = self.context_layernorm(context)
        lang_input = torch.cat([context, query_current], dim=1)

        # language decoding
        lang_h, lang_c = self.lang_lstm(lang_input, (lang_lstm_h, lang_lstm_c))
        lang_h = self.lang_lstm_drop(lang_h)
        # final try
        # final_feature = torch.cat([query_current, self.lang_lstm_layernorm(lang_h), context], dim=-1)
        # store log probabilities
        # decoder_output = torch.tanh(final_feature)
        decoder_output = torch.tanh(self.lang_lstm_layernorm(lang_h))
        word_logits = self.word_restore(decoder_output)

        return word_logits, query_h, query_c, lang_h, lang_c, alpha
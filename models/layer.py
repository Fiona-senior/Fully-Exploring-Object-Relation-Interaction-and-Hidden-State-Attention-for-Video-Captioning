from models.sublayer import *
from models.allennlp_beamsearch import BeamSearch
import random
import os


class EncoderVisual(nn.Module):
    def __init__(self, args, input_type='frame+motion', embed=True, baseline=False):
        super(EncoderVisual, self).__init__()
        self.embed = embed
        hidden_size = args.visual_hidden_size
        self.hidden_size = hidden_size
        if embed:
            input_size = args.a_feature_size + args.m_feature_size

            if input_type == 'object':
                input_size = args.a_feature_size
            if input_type == 'motion':
                input_size = args.m_feature_size
            self.input_size = input_size
            print('batch size', args.train_batch_size)
            # batch size = 128
            # project input dimension to hidden_size
            self.linear_embed = nn.Linear(input_size, hidden_size)
            nn.init.xavier_normal_(self.linear_embed.weight)
        # Bi-LSTM
        self.lstm = nn.LSTM(hidden_size, hidden_size, batch_first=True, bidirectional=True)
        self.layernorm_lstm = nn.LayerNorm(hidden_size*2)
        self.drop_lstm = nn.Dropout(args.dropout)
        # Self attention
        self.baseline = baseline
        if not self.baseline:
            self.self_attention = SelfAttention(hidden_size*2, hidden_size*2, hidden_size, args.dropout, True)
            self.layernorm_sa = nn.LayerNorm(hidden_size)
            self.drop_sa = nn.Dropout(args.dropout)
        else:
            self.out_try = nn.Linear(hidden_size*2, hidden_size)
            nn.init.xavier_normal_(self.out_try.weight)


    def _init_lstm_state(self, d):
        batch_size = d.size(0)
        lstm_state_h = d.data.new(2, batch_size, self.hidden_size).zero_()
        lstm_state_c = d.data.new(2, batch_size, self.hidden_size).zero_()
        # 改变矩阵形状并且值置零
        return lstm_state_h, lstm_state_c

    def forward(self, inputs):
        lstm_h, lstm_c = self._init_lstm_state(inputs)
        # lstm
        input_embedding = inputs
        if self.embed:
            input_embedding = self.linear_embed(inputs)
        lstm_out, _ = self.lstm(input_embedding, (lstm_h, lstm_c))
        out = self.drop_lstm(self.layernorm_lstm(lstm_out))
        # self attention
        if not self.baseline:
            out = self.self_attention(out)
            out = self.layernorm_sa(out)
        else:
            out = self.out_try(out)
        # bs * win_len * hidden_size
        return out


# generate description according to the visual info
class Decoder(nn.Module):
    def __init__(self, args, vocab, multi_modal=False, baseline=False, use_fusion=False):
        super(Decoder, self).__init__()

        self.word_size = args.word_size
        self.max_words = args.max_words
        self.vocab = vocab
        self.dataset = args.dataset
        self.vocab_size = len(vocab)
        self.beam_size = args.beam_size
        self.batch_size = args.train_batch_size
        self.query_hidden_size = args.query_hidden_size
        self.decode_hidden_size = args.decode_hidden_size
        self.multi_modal = multi_modal
        self.use_fusion = use_fusion
        # print('decoder parameters ------------')
        # print('word_size = ', args.word_size)
        # print('max_words = ', self.max_words)
        # print('vocab = ', self.vocab)
        # print('vocab_size = ', self.vocab_size)
        # print('beam_size = ', self.beam_size)
        # print('use_glove = ', args.use_glove)
        # print('multi-modal = ',self.multi_modal)
        # print('decoder parameters ------------')

        if multi_modal and use_fusion:
            self.beta_fusion = nn.Sequential(
                nn.Linear(2*args.visual_hidden_size, 1),
                nn.Sigmoid()
            )
        # word embedding matrix
        self.word_embed = nn.Embedding(self.vocab_size, self.word_size)
        if args.use_glove:
            self.get_glove_embedding()
        self.word_drop = nn.Dropout(p=args.dropout)

        # attention lstm
        query_input_size = args.visual_hidden_size + args.word_size + args.decode_hidden_size
        # if self.multi_modal:
        if baseline is False:
            query_input_size += args.visual_hidden_size

        self.query_lstm = nn.LSTMCell(query_input_size, args.query_hidden_size)
        self.query_lstm_layernorm = nn.LayerNorm(args.query_hidden_size)
        self.query_lstm_drop = nn.Dropout(p=args.dropout)

        # decoder lstm
        # lang_decode_hidden_size中的lang 就是作者名字
        lang_decode_hidden_size = args.visual_hidden_size + args.query_hidden_size
        if self.multi_modal and self.use_fusion is False:
            lang_decode_hidden_size += args.visual_hidden_size
        self.lang_lstm = nn.LSTMCell(lang_decode_hidden_size, args.decode_hidden_size)
        self.lang_lstm_layernorm = nn.LayerNorm(args.decode_hidden_size)
        self.lang_lstm_drop = nn.Dropout(p=args.dropout)

        # context from attention
        self.context_att = AttentionShare(input_value_size=args.visual_hidden_size,
                                          input_key_size=args.query_hidden_size,
                                          output_size=args.visual_hidden_size)
        self.context_layernorm = nn.LayerNorm(args.decode_hidden_size)
        if self.multi_modal:
            self.context_att_2 = AttentionShare(input_value_size=args.visual_hidden_size,
                                              input_key_size=args.query_hidden_size,
                                              output_size=args.visual_hidden_size)

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
        glove_np_path = f'./data/{self.dataset}_glove.npy'
        if os.path.exists(glove_np_path):
            weight_matrix = np.load(glove_np_path)
        else:
            glove_dic = {}
            glove_path = '/mnt/CAC593A17C0101D9/DL_projects/Other_projects/video description/RMN-master/data/glove.6B/glove.42B.300d.txt'
            if not os.path.exists(glove_path):
                glove_path = './data/glove.42B.300d.txt'
            with open(f'{glove_path}', 'rb') as f:
                for l in f:
                    line = l.decode().split()
                    word = line[0]
                    vect = np.array(line[1:]).astype(np.float)
                    glove_dic[word] = vect

            weight_matrix = np.zeros((self.word_embed.num_embeddings, self.word_embed.embedding_dim))
            word_found = 0

            for i, word in enumerate(self.vocab.idx2word):
                if word.endswith(','):
                    word = word[:-1]
                try:
                    weight_matrix[i] = glove_dic[word]
                    word_found += 1
                except:
                    # self.word_embed(torch.Tensor(i).type(torch.LongTensor).to(self.device)).unsquezze()
                    weight_matrix[i] = np.random.normal(scale=0.6, size=(self.word_embed.embedding_dim, ))
                    print(word)
            print('found words ', word_found)
            np.save(glove_np_path, weight_matrix)

        weight_matrix = torch.from_numpy(weight_matrix)
        self.word_embed.load_state_dict({'weight': weight_matrix})
        # self.word_embed.load_state_dict({'weight': weight_matrix})

    def _init_lstm_state(self, d, hidden_size):
        batch_size = d.size(0)
        lstm_state_h = d.data.new(batch_size, hidden_size).zero_()
        lstm_state_c = d.data.new(batch_size, hidden_size).zero_()
        return lstm_state_h, lstm_state_c
# outputs, alpha_all = self.decoder(obj_proposals, caption, max_words, teacher_forcing_ratio, motion_proposals)
    def forward(self, cnn_feats, captions, max_words, teacher_forcing_ratio, cnn_feats_2=None, step_feats=None):
        self.batch_size = cnn_feats.size(0)
        # inference or training
        if captions is None:
            infer = True
        else:
            infer = False
        if max_words is None:
            max_words = self.max_words

        if step_feats is not None:
            global_feat = step_feats
        else:
            global_feat = torch.mean(cnn_feats, dim=1)
            if cnn_feats_2 is not None:
                global_feat_2 = torch.mean(cnn_feats_2, dim=1)
                global_feat = torch.cat([global_feat, global_feat_2], dim=-1)

            if cnn_feats_2 is not None and self.multi_modal is False:
                cnn_feats = torch.cat([cnn_feats, cnn_feats_2], dim=1)
        # 小问题：lang和query有什么区别？
        lang_lstm_h, lang_lstm_c = self._init_lstm_state(cnn_feats, self.decode_hidden_size)
        query_lstm_h, query_lstm_c = self._init_lstm_state(cnn_feats, self.query_hidden_size)

        # add a '<start>' sign
        start_id = self.vocab('<start>') # =1
        start_id = cnn_feats.data.new(cnn_feats.size(0)).long().fill_(start_id) # =tensor([1,1,1,...,1])
        word = self.word_embed(start_id)
        word = self.word_drop(word)

        outputs = []
        alpha_all = []
        if not infer or self.beam_size == 1:  # infer = False
            for i in range(max_words):
                # lstm input: word + h_(t-1) + context   # return word_logits, query_h, query_c, lang_h, lang_c, alpha
                word_logits, query_lstm_h, query_lstm_c, lang_lstm_h, lang_lstm_c, alpha =\
                    self.decode(word, query_lstm_h, query_lstm_c, lang_lstm_h, lang_lstm_c, global_feat, cnn_feats, cnn_feats_2, i)
                # teacher_forcing: a training trick

                use_teacher_forcing = not infer and (random.random() < teacher_forcing_ratio)
                # use_teacher_forcing = False
                if use_teacher_forcing:
                    word_id = captions[:, i]
                else:
                    word_id = word_logits.max(1)[1]
                word = self.word_embed(word_id)
                word = self.word_drop(word)

                if infer:
                    outputs.append(word_id)
                else:
                    outputs.append(word_logits)
                    alpha_all.append(alpha)

            outputs = torch.stack(outputs, dim=1)

        else:
            start_state = {'query_lstm_h': query_lstm_h, 'query_lstm_c': query_lstm_c,
                           'lang_lstm_h': lang_lstm_h, 'lang_lstm_c': lang_lstm_c,
                           'cnn_feats': cnn_feats, 'global_feat': global_feat}
            if self.multi_modal:
                start_state.update({'cnn_feats_2': cnn_feats_2})
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
            if self.multi_modal:
                cnn_feats_2 = current_state['cnn_feats_2'][:, i, :]
            else:
                cnn_feats_2 = None
            # decoding stage
            word_id = last_predictions.reshape(batch_size, -1)[:, i]
            word = self.word_embed(word_id)

            word_logits, query_lstm_h, query_lstm_c, lang_lstm_h, lang_lstm_c, _ = \
                self.decode(word, query_lstm_h, query_lstm_c, lang_lstm_h, lang_lstm_c, global_feat, cnn_feats, cnn_feats_2)

            log_prob = F.log_softmax(word_logits, dim=1)  # b*v
            log_probs.append(log_prob)

            # update new state
            new_state['query_lstm_h'].append(query_lstm_h)
            new_state['query_lstm_c'].append(query_lstm_c)
            new_state['lang_lstm_h'].append(lang_lstm_h)
            new_state['lang_lstm_c'].append(lang_lstm_c)
            new_state['global_feat'].append(global_feat)
            new_state['cnn_feats'].append(cnn_feats)
            if self.multi_modal:
                new_state['cnn_feats_2'].append(cnn_feats_2)


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
# word_logits, query_lstm_h, query_lstm_c, lang_lstm_h, lang_lstm_c, alpha =self.decode(word, query_lstm_h, query_lstm_c, lang_lstm_h, lang_lstm_c, global_feat, cnn_feats, cnn_feats_2)
    def decode(self, word, query_lstm_h, query_lstm_c, lang_lstm_h, lang_lstm_c, global_feat, cnn_feats, cnn_feats_2=None):

        query_h, query_c = self.query_lstm(torch.cat([lang_lstm_h, global_feat, word], dim=1),
                                               (query_lstm_h, query_lstm_c))
        # query_lstm_h = self.att_lstm_drop(query_lstm_h)
        query_current = self.query_lstm_drop(self.query_lstm_layernorm(query_h))
        # context from attention
        context, alpha = self.context_att(cnn_feats, query_current)
        # context = self.context_layernorm(context)
        if self.multi_modal:
            context_2, alpha_2 = self.context_att_2(cnn_feats_2, query_current)
            # context_lambda = self.psl_selector(query_current).unsqueeze(2)
            # context_all = torch.stack([context, context_2], dim=2)
            # context_all = torch.matmul(context_all, context_lambda).squeeze()
            if self.use_fusion:
                beta = self.beta_fusion(torch.cat([query_h, lang_lstm_h], dim=-1))
                lang_input = torch.cat([context*beta + context_2 * (1-beta), query_current], dim=1)
            else:
                lang_input = torch.cat([context, context_2, query_current], dim=1)
            alpha = torch.cat([alpha, alpha_2], dim=1)
        else:
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

    def _sample(self, cnn_feats, captions, max_words, teacher_forcing_ratio, cnn_feats_2=None, step_feats=None, opti={}):
        sample_method = opti.get('sample_method', 'greedy')
        beam_size = opti.get('beam_size', 1)
        temperature = opti.get('temperature', 1.0)
        if beam_size > 1 and sample_method in ['greedy', 'beam_search']:
            return self._sample_beam(cnn_feats, captions, max_words, teacher_forcing_ratio, cnn_feats_2=None)

        batch_size = cnn_feats.size(0)

        if captions is None:
            infer = True
        else:
            infer = False
        if max_words is None:
            max_words = self.max_words

        if step_feats is not None:
            global_feat = step_feats
        else:
            global_feat = torch.mean(cnn_feats, dim=1)
            if cnn_feats_2 is not None:
                global_feat_2 = torch.mean(cnn_feats_2, dim=1)
                global_feat = torch.cat([global_feat, global_feat_2], dim=-1)

            if cnn_feats_2 is not None and self.multi_modal is False:
                cnn_feats = torch.cat([cnn_feats, cnn_feats_2], dim=1)

        lang_lstm_h, lang_lstm_c = self._init_lstm_state(cnn_feats, self.decode_hidden_size)
        query_lstm_h, query_lstm_c = self._init_lstm_state(cnn_feats, self.query_hidden_size)

        start_id = self.vocab('<start>') # =1
        start_id = cnn_feats.data.new(cnn_feats.size(0)).long().fill_(start_id) # =tensor([1,1,1,...,1])
        word = self.word_embed(start_id)
        word = self.word_drop(word)

        seq = cnn_feats.new_zeros(batch_size, max_words, dtype=torch.long)
        seqLogprobs = cnn_feats.new_zeros(batch_size, max_words, self.vocab_size + 1)

        outputs = []
        alpha_all = []
        if not infer or self.beam_size == 1:  # infer = False
            for i in range(max_words):
                # lstm input: word + h_(t-1) + context   # return word_logits, query_h, query_c, lang_h, lang_c, alpha
                word_logits, query_lstm_h, query_lstm_c, lang_lstm_h, lang_lstm_c, alpha =\
                    self.decode(word, query_lstm_h, query_lstm_c, lang_lstm_h, lang_lstm_c, global_feat, cnn_feats, cnn_feats_2, i)
                # teacher_forcing: a training trick

                use_teacher_forcing = not infer and (random.random() < teacher_forcing_ratio)
                # use_teacher_forcing = False
                if use_teacher_forcing:
                    word_id = captions[:, i]
                else:
                    word_id = word_logits.max(1)[1]
                word = self.word_embed(word_id)
                word = self.word_drop(word)

                if infer:
                    outputs.append(word_id)
                else:
                    outputs.append(word_logits)
                    alpha_all.append(alpha)

                logprobs = F.log_softmax(word_logits, dim=-1)

                # sample the next_word
                if i == max_words - 1:  # skip if we achieve maximum length
                    break
                if sample_method == 'greedy':
                    sampleLogprobs, it = torch.max(logprobs.data, 1)
                    it = it.view(-1).long()
                else:
                    if temperature == 1.0:
                        prob_prev = torch.exp(logprobs.data)  # fetch prev distribution: shape Nx(M+1)
                    else:
                        # scale logprobs by temperature
                        prob_prev = torch.exp(torch.div(logprobs.data, temperature))
                    it = torch.multinomial(prob_prev, 1).to(logprobs.device)
                    # 从 pro_prev 采样一个，返回位置

                    sampleLogprobs = logprobs.gather(1, it)  # gather the logprobs at sampled positions
                    # 根据索引查找，然后将查找结果以张量矩阵的形式返回
                    # 当dim=1时，index[0][0]的元素是1，那么它想查找的tensor[1][0]中的元素
                    it = it.view(-1).long()  # and flatten indices for downstream processing

                if i >= 1:
                    # stop when all finished
                    if i == 1:
                        unfinished = it > 0
                    else:
                        unfinished = unfinished & (it > 0)
                        # & 就是都要大于0 若有一个为0 那就变成0
                    it = it * unfinished.type_as(it)
                    # .type_as类型转换
                    seq[:, i - 1] = it  # seq[t] the input of t+2 time step
                    seqLogprobs[:, i - 1] = sampleLogprobs.view(-1)
                    if unfinished.sum() == 0:
                        break


            outputs = torch.stack(outputs, dim=1)

        else:
            start_state = {'query_lstm_h': query_lstm_h, 'query_lstm_c': query_lstm_c,
                           'lang_lstm_h': lang_lstm_h, 'lang_lstm_c': lang_lstm_c,
                           'cnn_feats': cnn_feats, 'global_feat': global_feat}
            if self.multi_modal:
                start_state.update({'cnn_feats_2': cnn_feats_2})
            predictions, log_prob = self.beam_search.search(start_id, start_state, self.beam_step)
            max_prob, max_index = torch.topk(log_prob, 1)
            max_index = max_index.squeeze(1)
            for i in range(self.batch_size):
                outputs.append(predictions[i, max_index[i], :])
            outputs = torch.stack(outputs)

        return seq, seqLogprobs



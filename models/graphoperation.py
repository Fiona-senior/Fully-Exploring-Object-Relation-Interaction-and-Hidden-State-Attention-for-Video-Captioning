from models.layer import EncoderVisual, Decoder
from models.sublayer import SelfAttention, AttentionShare
import torch.nn as nn
import torch
from torch.nn.utils.rnn import pack_padded_sequence,pad_packed_sequence
from models.TemporalAttention import MultiHeadAttention, MatrixAttn
import torch.nn.functional as F
import numpy as np
import random
from models.allennlp_beamsearch import BeamSearch
import os


class CapGraphModel(nn.Module):
    def __init__(self, args, vocab, relation_inv):
        super(CapGraphModel, self).__init__()
        self.objectEncoder = objectEncoder(args, vocab)
        self.graphEncoder = graph_encode(args, relation_inv)
        self.VisualEncoder = EncoderVisual(args)
        self.decoder = Decoder(args, vocab, multi_modal=True)

    def forward(self, device, object_one, object_two, object_3, caption_adj, caption_rel, frames, targets, max_words=None, teacher_forcing_ratio=1.0):
        objects = self.objectEncoder(object_one, object_two, object_3)
        gents, glob, grels = self.graphEncoder(caption_adj,caption_rel,(objects,object_3))
        hx = glob
        keys, mask = grels
        mask = mask == 0
        mask = mask.unsqueeze(1).to(device)
        visual_features = self.VisualEncoder(frames)
        outputs = self.decoder(visual_features, hx, keys, targets, max_words, teacher_forcing_ratio, mask, objects, object_3, device)
        return outputs

    def update_beam_size(self, beam_size):
        self.decoder.update_beam_size(beam_size)

    def load_encoder(self, model, model_path):
        model.load_state_dict(torch.load(model_path, map_location='cuda:0'))
        self.encoder = model.encoder
        self.decoder.word_embed = model.decoder.word_embed
        for param in self.decoder.word_embed.parameters():
            param.requires_grad = False

# Bi-LSTM
class objectEncoder(nn.Module):
    def __init__(self,args, vocab):
        super().__init__()
        if args.dataset == "vatex":
            self.seqenc = lseq_encode(args,toks=len(vocab['idx2word']))
        else:
            self.seqenc = lseq_encode(args, toks=len(vocab))

    def pad(self,tensor,length):
        return torch.cat([tensor, tensor.new(length - tensor.size(0), *tensor.size()[1:]).fill_(0)])

    def forward(self,object_one, object_two, object_3, pad=True):
        object_3 = tuple(object_3.tolist())
        _,enc = self.seqenc((object_one,object_two))
        enc = enc[:,2:]
        enc = torch.cat([enc[:,i] for i in range(enc.size(1))],1)
        m = max(object_3)
        encs = [self.pad(x,m) for x in enc.split(object_3)]
        out = torch.stack(encs,0)
        return out

class lseq_encode(nn.Module):

    def __init__(self, args, vocab=None, toks=None):
        super().__init__()
        sz = args.objects_embeddingsize
        self.lemb = nn.Embedding(toks, args.objects_embeddingsize)
        nn.init.xavier_normal_(self.lemb.weight)
        self.input_drop = nn.Dropout(args.dropout)

        self.encoder = nn.LSTM(sz, args.objects_embeddingsize // 2, bidirectional=True, num_layers=args.layers,
                               batch_first=True)

    def _cat_directions(self, h):
        h = torch.cat([h[0:h.size(0):2], h[1:h.size(0):2]], 2)
        return h


    def forward(self, inp):
        l, ilens = inp
        learned_emb = self.lemb(l)
        learned_emb = self.input_drop(learned_emb)
        e = learned_emb
        sent_lens, idxs = ilens.sort(descending=True)
        e = e.index_select(0, idxs)
        e = pack_padded_sequence(e, sent_lens.to('cpu'), batch_first=True)
        e, (h, c) = self.encoder(e)
        e = pad_packed_sequence(e, batch_first=True)[0]
        e = torch.zeros_like(e).scatter(0, idxs.unsqueeze(1).unsqueeze(1).expand(-1, e.size(1), e.size(2)), e)
        h = h.transpose(0, 1)
        h = torch.zeros_like(h).scatter(0, idxs.unsqueeze(1).unsqueeze(1).expand(-1, h.size(1), h.size(2)), h)
        return e, h

class Block(nn.Module):
    def __init__(self,args):
        super().__init__()
        self.attn = MultiHeadAttention(args.hsz,args.hsz,args.hsz,h=4,dropout_p=args.dropout)
        self.l1 = nn.Linear(args.hsz,args.hsz*4)
        self.l2 = nn.Linear(args.hsz*4,args.hsz)
        self.ln_1 = nn.LayerNorm(args.hsz)
        self.ln_2 = nn.LayerNorm(args.hsz)
        self.drop = nn.Dropout(args.dropout)
        self.act = nn.PReLU(args.hsz*4)
        self.gatact = nn.PReLU(args.hsz)

    def forward(self,q,k,m):
        q = self.attn(q,k,mask=m).squeeze(1)
        t = self.ln_1(q)
        q = self.drop(self.l2(self.act(self.l1(t))))
        q = self.ln_1(q+t)
        return q

# deal with the relation and objects with graph transformer together
class graph_encode(nn.Module):
    def __init__(self,args, relation_inv):
        super().__init__()
        self.args = args
        self.renc = nn.Embedding(len(relation_inv),args.objects_embeddingsize)
        nn.init.xavier_normal_(self.renc.weight)

        if args.model == "gat":
            self.gat = nn.ModuleList([MultiHeadAttention(args.objects_embeddingsize,
                                                       args.objects_embeddingsize,args.objects_embeddingsize,
                                                       h=4,dropout_p=args.dropout) for _ in range(args.prop)])
        else:
            self.gat = nn.ModuleList([Block(args) for _ in range(args.prop)])
        self.prop = args.prop
        self.sparse = args.sparse

    def get_device(self):
    # return the device of the tensor, either "cpu"
    # or number specifiing the index of gpu.
        dev = next(self.parameters()).get_device()
        if dev == -1:
            return "cpu"
        return dev

    def pad(self,tensor,length):
        return torch.cat([tensor, tensor.new(length - tensor.size(0), *tensor.size()[1:]).fill_(0)])

    def forward(self,adjs,rels,ents):
        vents,entlens = ents
        # vents are the handled objects; entlens are object_3(the num of objects in every video)
        if self.args.entdetach:
            vents = torch.tensor(vents,requires_grad=False)
        vrels = [self.renc(x) for x in rels]
        glob = []
        graphs = []
        for i,adj in enumerate(adjs):
            vgraph = torch.cat((vents[i][:entlens[i]],vrels[i]),0)
            # concat the entity and relation in the same video
            N = vgraph.size(0)
            if self.sparse:
                lens = [len(x) for x in adj]
                m = max(lens)
                mask = torch.arange(0,m).unsqueeze(0).repeat(len(lens),1).long()
                # mask and vents should be in the same device.
                mask = (mask <= torch.LongTensor(lens).unsqueeze(1)).to(self.get_device())
                mask = (mask == 0).unsqueeze(1)
            else:
                mask = (adj == 0).unsqueeze(1)
            for j in range(self.prop):
                if self.sparse:
                    ngraph = [vgraph[k] for k in adj]
                    ngraph = [self.pad(x,m) for x in ngraph]
                    ngraph = torch.stack(ngraph,0)
                    vgraph = self.gat[j](vgraph.unsqueeze(1),ngraph,mask)
                else:
                    ngraph = vgraph.repeat(N,1).view(N,N,-1).clone().detach().requires_grad_(False)
                    vgraph = self.gat[j](vgraph.unsqueeze(1),ngraph,mask)
                    if self.args.model == 'gat':
                        vgraph = vgraph.squeeze(1)
                        vgraph = self.gat(vgraph)
            graphs.append(vgraph)
            glob.append(vgraph[entlens[i]])
        elens = [x.size(0) for x in graphs]
        gents = [self.pad(x,max(elens)) for x in graphs]
        gents = torch.stack(gents,0)
        elens = torch.LongTensor(elens)
        emask = torch.arange(0,gents.size(1)).unsqueeze(0).repeat(gents.size(0),1).long()
        emask = (emask <= elens.unsqueeze(1))
        glob = torch.stack(glob,0)
        return None,glob,(gents,emask)

# visual encoder : perform lstm+norm+dropout+linear on visual feats
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

            # project input dimension to hidden_size
            if args.dataset == "vatex":
                input_size = 1024
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

class Decoder(nn.Module):
    def __init__(self, args, vocab, multi_modal=False, baseline=False, use_fusion=False):
        super(Decoder, self).__init__()

        self.word_size = args.word_size
        if args.dataset == "vatex":
            self.max_words = 30
        else:
            self.max_words = args.max_words
        self.chose = args.chose
        self.vocab = vocab
        self.dataset = args.dataset
        if args.dataset == "vatex":
            self.vocab_size = len(vocab['idx2word'])
        else:
            self.vocab_size = len(vocab)
        self.beam_size = args.beam_size
        self.batch_size = args.train_batch_size
        self.query_hidden_size = args.query_hidden_size
        self.decode_hidden_size = args.decode_hidden_size
        self.multi_modal = multi_modal
        self.use_fusion = use_fusion

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

        lang_decode_hidden_size = args.visual_hidden_size + args.query_hidden_size
        if self.multi_modal and self.use_fusion is False:
            lang_decode_hidden_size += args.visual_hidden_size
        self.lang_lstm = nn.LSTMCell(lang_decode_hidden_size, args.decode_hidden_size)
        self.lang_lstm_layernorm = nn.LayerNorm(args.hsz * 3)
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
        self.word_restore = nn.Linear(args.hsz * 3, self.vocab_size)
        nn.init.xavier_normal_(self.word_restore.weight)
        # testing stage
        # beam search: The BeamSearch class is imported from Allennlp
        # DOCUMENTS: https://docs.allennlp.org/master/api/nn/beam_search/
        if args.dataset == "vatex":
            self.beam_search = BeamSearch(vocab['word2idx']['<EOS>'], self.max_words, self.beam_size, per_node_beam_size=self.beam_size)
        else:
            self.beam_search = BeamSearch(vocab('<end>'), self.max_words, self.beam_size, per_node_beam_size=self.beam_size)


        self.lstm = nn.LSTMCell(args.hsz * 3, args.hsz)
        self.attn2 = MultiHeadAttention(args.hsz, args.hsz, args.hsz, h=4, dropout_p=args.dropout)
        self.attn = MultiHeadAttention(args.hsz, args.hsz, args.hsz, h=4, dropout_p=args.dropout)
        self.mattn = MatrixAttn(args.hsz * 3, args.hsz)
        self.switch = nn.Linear(args.hsz * 3, 1)
        self.args = args

    def update_beam_size(self, beam_size):
        self.beam_size = beam_size
        self.beam_search = BeamSearch(self.vocab['word2idx']['<EOS>'], self.max_words, beam_size, per_node_beam_size=beam_size)

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

            for i, word in enumerate(self.vocab['idx2word']):
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

    def forward(self, visual_features, hx, keys, captions, max_words, teacher_forcing_ratio, mask, objects, object_3, device):

        self.visual_features = visual_features
        self.keys = keys
        self.mask = mask
        self.device = device
        cx = hx.clone().detach().requires_grad_(True)
        a = torch.zeros_like(hx)
        a2 = self.attn2(hx.unsqueeze(1), visual_features, mask=None).squeeze(1)
        a = torch.cat((a, a2), 1)
        self.objects = objects
        self.object_3 = object_3

        # inference or training
        if captions is None:
            infer = True
        else:
            infer = False
        if max_words is None:
            max_words = self.max_words
        #
        # if step_feats is not None:
        #     global_feat = step_feats
        # else:
        #     global_feat = torch.mean(cnn_feats, dim=1)
        #     if cnn_feats_2 is not None:
        #         global_feat_2 = torch.mean(cnn_feats_2, dim=1)
        #         global_feat = torch.cat([global_feat, global_feat_2], dim=-1)
        #
        #     if cnn_feats_2 is not None and self.multi_modal is False:
        #         cnn_feats = torch.cat([cnn_feats, cnn_feats_2], dim=1)
        # # 小问题：lang和query有什么区别？
        # lang_lstm_h, lang_lstm_c = self._init_lstm_state(cnn_feats, self.decode_hidden_size)
        # query_lstm_h, query_lstm_c = self._init_lstm_state(cnn_feats, self.query_hidden_size)

        # add a '<start>' sign

        if self.args.dataset == "vatex":
            start_id = self.vocab['word2idx']['<BOS>']
        else:
            start_id = self.vocab('<start>')
        start_id = hx.data.new(hx.size(0)).long().fill_(start_id)
        word = self.word_embed(start_id)
        word = self.word_drop(word)

        #             start_state = {'visual_features': visual_features, 'hx': hx,
        #                            'keys': keys, 'mask': mask, 'a': a, 'cx': cx}

# visual_features: torch.Size([32, 26, 1024])
# hx: torch.Size([32, 1024])
# keys: torch.Size([32, 11, 1024])
# mask: torch.Size([32, 1, 11])
# a: torch.Size([32, 2048])
# cx: torch.Size([32, 1024])
        decoder_outputs = []
        outputs = []
        if not infer or self.beam_size == 1:
            for i in range(max_words):

                prev = torch.cat((a, word), 1)
                hx, cx = self.lstm(prev, (hx, cx))
                a = self.attn(hx.unsqueeze(1), keys, mask=mask).squeeze(1)
                a2 = self.attn2(hx.unsqueeze(1), visual_features, mask=None).squeeze(1)
                a = torch.cat((a, a2), 1)
                word_logits = torch.cat((hx, a), 1)
                word_logits = self.lang_lstm_drop(word_logits)
                decoder_output = torch.tanh(self.lang_lstm_layernorm(word_logits))
                word_logits = self.word_restore(decoder_output)
                decoder_outputs.append(decoder_output)

                # teacher_forcing: a training trick
                use_teacher_forcing = not infer and (random.random() < teacher_forcing_ratio)

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

            outputs = torch.stack(outputs, dim=1)

            if self.chose:
                decoder_outputs = torch.stack(decoder_outputs, dim=1)
                s = torch.sigmoid(self.switch(decoder_outputs))

                outputs = s * outputs
                _, z = self.mattn(decoder_outputs, (self.objects, self.object_3))
                embed2vocabsize = nn.Linear(z.shape[-1], self.vocab_size).to(self.device)
                nn.init.xavier_normal_(embed2vocabsize.weight)
                z = embed2vocabsize(z)
                z = (1 - s) * z

                outputs = outputs + z

        else:

            start_state = {'hx': hx, 'a': a, 'cx': cx}

            predictions, log_prob = self.beam_search.search(start_id, start_state, self.beam_step)
            max_prob, max_index = torch.topk(log_prob, 1)
            max_index = max_index.squeeze(1)
            for i in range(self.batch_size):
                outputs.append(predictions[i, max_index[i], :])
            outputs = torch.stack(outputs)

        return outputs

    def decode_tokens(self, tokens):
        '''
        convert word index to caption
        :param tokens: input word index
        :return: capiton
        '''
        words = []
        for token in tokens:
            if token == self.vocab['word2idx']['<EOS>']:
                break
            word = self.vocab['idx2word'][token]
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
            hx = current_state['hx'][:, i, :]
            a = current_state['a'][:, i, :]
            cx = current_state['cx'][:, i, :]

            word_id = last_predictions.reshape(batch_size, -1)[:, i]
            word = self.word_embed(word_id)

            prev = torch.cat((a, word), 1)
            hx, cx = self.lstm(prev, (hx, cx))
            a = self.attn(hx.unsqueeze(1), self.keys, mask=self.mask).squeeze(1)
            a2 = self.attn2(hx.unsqueeze(1), self.visual_features, mask=None).squeeze(1)

            a = torch.cat((a, a2), 1)
            word_logits = torch.cat((hx, a), 1)
            word_logits = self.lang_lstm_drop(word_logits)

            decoder_output = torch.tanh(self.lang_lstm_layernorm(word_logits))
            word_logits = self.word_restore(decoder_output)
            if self.chose:
                s = torch.sigmoid(self.switch(decoder_output))
                word_logits = s * word_logits
                _, z = self.mattn(decoder_output, (self.objects, self.object_3))
                z = z.squeeze()
                embed2vocabsize = nn.Linear(z.shape[-1], self.vocab_size).to(self.device)
                nn.init.xavier_normal_(embed2vocabsize.weight)
                z = embed2vocabsize(z)
                z = (1 - s) * z
                word_logits = word_logits + z


            log_prob = F.log_softmax(word_logits, dim=-1)  # b*v
            log_probs.append(log_prob)

            # update new state
            new_state['hx'].append(hx)
            new_state['a'].append(a)
            new_state['cx'].append(cx)

        # transform log probabilities
        # from list to tensor(batch_size*beam_size, vocab_size)
        log_probs = torch.stack(log_probs, dim=0).permute(1, 0, 2).reshape(-1, self.vocab_size)  # group_size*vocab_size

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

    def decode(self, word, query_lstm_h, query_lstm_c, lang_lstm_h, lang_lstm_c, global_feat, cnn_feats, cnn_feats_2=None):

        query_h, query_c = self.query_lstm(torch.cat([lang_lstm_h, global_feat, word], dim=1),
                                               (query_lstm_h, query_lstm_c))

        query_current = self.query_lstm_drop(self.query_lstm_layernorm(query_h))
        # context from attention
        context, alpha = self.context_att(cnn_feats, query_current)

        if self.multi_modal:
            context_2, alpha_2 = self.context_att_2(cnn_feats_2, query_current)

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
        decoder_output = torch.tanh(self.lang_lstm_layernorm(lang_h))
        word_logits = self.word_restore(decoder_output)

        return word_logits, query_h, query_c, lang_h, lang_c, alpha
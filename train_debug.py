import os
from run_gun import RunGAN
from utils.opt import parse_opt
import torch
import numpy as np
import random
import pickle
import json
from utils.data import get_train_loader, get_eval_loader
from evaluate import convert_data_to_coco_scorer_format
import torch.distributed as dist
from utils.utils import Vocabulary


if __name__ == "__main__":
    args = parse_opt()
    if args.local_rank < 0:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        dist.init_process_group(backend='nccl')
        torch.cuda.set_device(args.local_rank)
        device = torch.device(f'cuda:{args.local_rank}')
    multi_gpu = False if args.local_rank < 0 else True
    print(f'multi-gpu = {multi_gpu}, local_rank = {args.local_rank}')

    if args.dataset == "vatex":
        vocab = {}
        path = "./data/vatex/VATEX/annotation/RET/word2int.json"
        word2idx = open(path, 'r')
        word2idx = json.load(word2idx)
        word2idx['<pad>'] = 10424
        vocab["word2idx"] = word2idx

        path = "./data/vatex/VATEX/annotation/RET/int2word.npy"
        idx2word = np.load(path)
        idx2word = np.append(idx2word, '<pad>')
        vocab["idx2word"] = idx2word

        nwords = len(idx2word)+1
        vocab["nwords"] = nwords

        path8 = "./data/vatex/VATEX/annotation/RET/vatex_relations_inv.json"
        relation_inv = open(path8, 'r')
        relation_inv = json.load(relation_inv)

    else:
        with open(args.vocab_pkl_path, 'rb') as f:
            vocab = pickle.load(f)
        jsonfile2 = open(args.relation_inv_path, 'r')
        relation_inv = json.load(jsonfile2)

    train_loader, train_sampler = get_train_loader(args.train_caption_pkl_path, args.feature_h5_path,
                                         args.region_feature_h5_path, args.Video_objects_triplets_json,
                                                   args.train_batch_size, multi_gpu=multi_gpu)
    test_loader = get_eval_loader(args.test_range, args.feature_h5_path,
                                       args.region_feature_h5_path, args.Video_objects_triplets_json,
                                  args.test_batch_size, multi_gpu=multi_gpu)
    test_reference = convert_data_to_coco_scorer_format(args.test_reference_txt_path)

    seed = 12
    np.random.seed(seed)  # Numpy module.
    random.seed(seed)  # Python random module.
    torch.manual_seed(seed)
    if multi_gpu:
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)  # if you are using multi-GPU.
        # torch.backends.cudnn.benchmark = False
        # torch.backends.cudnn.deterministic = True

    run = RunGAN(args, vocab, relation_inv, device, train_loader=train_loader, test_loader=test_loader,
                 test_reference=test_reference, is_debug=True, multi_gpu=multi_gpu, train_sampler=train_sampler)
    with torch.backends.cudnn.flags(enabled=False):
        run.train()




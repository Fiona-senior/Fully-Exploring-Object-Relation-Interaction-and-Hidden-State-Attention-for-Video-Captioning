import pickle
import h5py
import json
import torch
import torch.utils.data as data
from utils.opt import parse_opt
import os
import glob
import tables as tb
import numpy as np
import getpass
from utils.utils import Vocabulary
opt = parse_opt()

# v2t = V2TDataset(cap_pkl, frame_feature_h5, region_feature_h5)
class V2TDataset(data.Dataset):
    def __init__(self, cap_pkl, frame_feature_h5, region_feature_h5, Video_objects_triplets_json):
        if opt.dataset == "vatex":
            path = "./data/vatex/VATEX/annotation/RET/vatex_captions_vi_len_dict.json"
            jsonfile2 = open(path, 'r')
            vatex_caption_vi_len_dict = json.load(jsonfile2) # {video name: captions}
            self.captions = vatex_caption_vi_len_dict['captions']
            self.lengths = vatex_caption_vi_len_dict['lengths']
            self.video_ids = vatex_caption_vi_len_dict['video_ids']

            # path1 = "./data/vatex/VATEX/public_split/trn_names.npy"
            # self.video_ids = np.load(path1)

            path4 = "./data/vatex/VATEX_ordered_feature/SA/RET.public.i3d/trn_ft.hdf5"
            self.video_feats = h5py.File(path4, 'r')

            path5 = "./data/vatex/VATEX/annotation/RET/vatex_caption_triplets_idx_dict.json"
            jsonfile = open(path5, 'r')
            self.Video_caption_triplets_Dict = json.load(jsonfile)

            path8 = "./data/vatex/VATEX/annotation/RET/vatex_relation_mapping.json"
            jsonfile5 = open(path8, 'r')
            self.relation2id_DIct = json.load(jsonfile5)

            file_path6 = "./data/vatex/VATEX/annotation/RET/vatex_videonames_mapping.json"
            jsonfile3 = open(file_path6, 'r')
            self.vatex_videonames_mapping = json.load(jsonfile3)

            file7 = "./data/vatex/VATEX/annotation/RET/vatex_vnmapping_idx2word.json.npy"
            self.vatex_vnmapping_idx2word = np.load(file7)

            path = "./data/vatex/VATEX/annotation/RET/word2int.json"
            word2idx = open(path, 'r')
            self.word2idx = json.load(word2idx)
            self.word2idx['<pad>'] = 10424

        else:
            if not os.path.exists(cap_pkl):
                cap_pkl = '../data/MSR-VTT/msr-vtt_captions_train.pkl'
            with open(cap_pkl, 'rb') as f:
                # video ids: train ids for videos
                self.captions, self.pos_tags, self.lengths, self.video_ids = pickle.load(f)

            h5 = h5py.File(frame_feature_h5, 'r')
            self.video_feats = h5[opt.feature_h5_feats] # args.feature_h5_feats = 'feats'
            # h5.close()

            jsonfile = open(Video_objects_triplets_json, 'r')
            self.Video_caption_triplets_Dict = json.load(jsonfile)

            jsonfile2 = open(opt.relation2id_path, 'r')
            self.relation2id_DIct = json.load(jsonfile2)

            with open(opt.vocab_pkl_path, 'rb') as f:
                self.vocab = pickle.load(f)

            if not os.path.exists(region_feature_h5):
                file_names = glob.glob('../data/MSR-VTT/msrvtt_region_feature*.h5')
                file_names.sort()
                print(file_names)
                region_feats_all = []
                spatial_feats_all = []
                for file_name in file_names:
                    print(file_name)
                    h5 = h5py.File(file_name, 'r')
                    region_feats = h5[opt.region_visual_feats]
                    spatial_feats = h5[opt.region_spatial_feats]
                    region_feats_all.append(region_feats)
                    spatial_feats_all.append(spatial_feats)
                    print('finished ', file_name)
                print('start concatenate region_feats_all')
                region_feats_all = np.concatenate(region_feats_all, axis=0)
                print('start concatenate spatial_feats_all')
                spatial_feats_all = np.concatenate(spatial_feats_all, axis=0)
                print(region_feats_all.shape)
                h5f = h5py.File('../data/MSR-VTT/msrvtt_region_feature.h5', 'w')
                h5f.create_dataset(opt.region_visual_feats, data=region_feats_all)
                h5f.create_dataset(opt.region_spatial_feats, data=spatial_feats_all)
                h5f.close()

            h5 = h5py.File(region_feature_h5, 'r')
            print("h5.keys:",h5.keys())
            self.region_feats = h5[opt.region_visual_feats]  # args.region_visual_feats = 'vfeats'
            self.spatial_feats = h5[opt.region_spatial_feats]  # 'sfeats'
            # h5.close()
            print('hehe')

    def __getitem__(self, index):
        # print(index)

        if opt.dataset == "vatex":
            caption = self.captions[index]
            caption = torch.Tensor(caption)
            length = self.lengths[index]
            video_id = self.video_ids[index]
            # print(video_id)

            video_feat = self.video_feats[self.vatex_vnmapping_idx2word[video_id]][:]
            video_feat = torch.tensor(video_feat)

            if video_feat.shape[0] != 32:
                repeat_factor = (32 + video_feat.shape[0] - 1) // video_feat.shape[0]
                tensor_repeated = video_feat.repeat((repeat_factor, 1))
                video_feat = tensor_repeated[:32, :]

            video_feat = video_feat.to(torch.float32)

            # print(video_feat.shape)
            # print("----------------")
            objects_triplets = self.Video_caption_triplets_Dict[str(video_id)]
            objects = objects_triplets[1]
            objectshandled = self.vec_ents(objects, self.word2idx)
            object_one = objectshandled[0]
            object_two = objectshandled[1]
            # print("objectshandled:",objectshandled)
            triplets_idx = objects_triplets[2]
            captionhandled = self.mkGraphs(triplets_idx, len(objectshandled[1]))
            caption_adj = captionhandled[0]
            if opt.sparse:
                caption_adj = self.adjToSparse(caption_adj)
            caption_rel = captionhandled[1]

            region_feat = None
            spatial_feat = None
            pos_tag = None

        else:
            caption = self.captions[index]
            pos_tag = self.pos_tags[index]
            length = self.lengths[index]
            video_id = self.video_ids[index]
            video_feat = torch.from_numpy(self.video_feats[video_id])
            region_feat = torch.from_numpy(self.region_feats[video_id])
            spatial_feat = torch.from_numpy(self.spatial_feats[video_id])


            objects_triplets = self.Video_caption_triplets_Dict['vid'+str(video_id+1)]
            objects = objects_triplets[1]
            objectshandled = self.vec_ents(objects, self.vocab.word2idx)
            object_one = objectshandled[0]
            object_two = objectshandled[1]
            # print("objectshandled:",objectshandled)
            triplets_idx = objects_triplets[2]
            captionhandled = self.mkGraphs(triplets_idx, len(objectshandled[1]))
            caption_adj = captionhandled[0]
            if opt.sparse:
                caption_adj = self.adjToSparse(caption_adj)
            caption_rel = captionhandled[1]
        # print("captionhandled:", captionhandled)
        return video_feat, region_feat, spatial_feat, caption, pos_tag, length, video_id, object_one, object_two, caption_adj, caption_rel

    def __len__(self):
        return len(self.captions)

    def vec_ents(self, ex, field):
        # returns tensor and lens
        ex = [[field[x] if x in field else 0 for x in y.strip().split(" ")] for y in ex]
        return self.pad_list(ex, 1)

    def pad_list(self, l, ent=1):
        lens = [len(x) for x in l]
        m = max(lens)
        return torch.stack([self.pad(torch.tensor(x), m, ent) for x in l], 0), torch.LongTensor(lens)

    def pad(self, tensor, length, ent=1):
        return torch.cat([tensor, tensor.new(length - tensor.size(0), *tensor.size()[1:]).fill_(ent)])

    def mkGraphs(self, r, ent):
        # convert triples to entlist with adj and rel matrices
        # x.rel = self.mkGraphs(x.rel,len(x.ent[1]))
        x = r
        rel = [2]
        # global root node
        adjsize = ent + 1 + (2 * len(x))
        adj = torch.zeros(adjsize, adjsize)
        for i in range(ent):
            adj[i, ent] = 1
            adj[ent, i] = 1
        for i in range(adjsize):
            adj[i, i] = 1
        for y in x:
            rel.extend([y[1] + 3, y[1] + 3 + len(self.relation2id_DIct)])
            a = y[0]
            b = y[2]
            c = ent + len(rel) - 2
            d = ent + len(rel) - 1
            adj[a, c] = 1
            adj[c, b] = 1
            adj[b, d] = 1
            adj[d, a] = 1
        rel = torch.LongTensor(rel)
        return (adj, rel)

    def adjToSparse(self, adj):
        sp = []
        for row in adj:
            sp.append(row.nonzero().squeeze(1))
        return sp


class VideoDataset(data.Dataset):

    def __init__(self, eval_range, frame_feature_h5, region_feature_h5, Video_objects_triplets_json):
        if opt.dataset == "vatex":

            self.eval_list = tuple(range(*eval_range))

            # path = "./data/vatex/VATEX/annotation/RET/vatex_captions_vi_len_dict.json"
            # jsonfile2 = open(path, 'r')
            # vatex_caption_vi_len_dict = json.load(jsonfile2)  # {video name: captions}
            # self.captions = vatex_caption_vi_len_dict['captions']
            # self.lengths = vatex_caption_vi_len_dict['lengths']
            # self.video_ids = vatex_caption_vi_len_dict['video_ids']

            # path1 = "./data/vatex/VATEX/public_split/trn_names.npy"
            # self.video_ids = np.load(path1)

            path4 = "./data/vatex/VATEX_ordered_feature/SA/RET.public.i3d/val_ft.hdf5"
            self.video_feats = h5py.File(path4, 'r')

            path5 = "./data/vatex/VATEX/annotation/RET/vatex_caption_triplets_idx_dict.json"
            jsonfile = open(path5, 'r')
            self.Video_caption_triplets_Dict = json.load(jsonfile)

            path8 = "./data/vatex/VATEX/annotation/RET/vatex_relation_mapping.json"
            jsonfile5 = open(path8, 'r')
            self.relation2id_DIct = json.load(jsonfile5)

            file_path6 = "./data/vatex/VATEX/annotation/RET/vatex_videonames_mapping.json"
            jsonfile3 = open(file_path6, 'r')
            self.vatex_videonames_mapping = json.load(jsonfile3)

            file7 = "./data/vatex/VATEX/annotation/RET/vatex_vnmapping_idx2word.json.npy"
            self.vatex_vnmapping_idx2word = np.load(file7)

            path = "./data/vatex/VATEX/annotation/RET/word2int.json"
            word2idx = open(path, 'r')
            self.word2idx = json.load(word2idx)
            self.word2idx['<pad>'] = 10424
        else:

            self.eval_list = tuple(range(*eval_range))
            h5 = h5py.File(frame_feature_h5, 'r')
            self.video_feats = h5[opt.feature_h5_feats]
            h5 = h5py.File(region_feature_h5, 'r')
            self.region_feats = h5[opt.region_visual_feats]
            self.spatial_feats = h5[opt.region_spatial_feats]

            jsonfile = open(Video_objects_triplets_json, 'r')
            self.Video_caption_triplets_Dict = json.load(jsonfile)

            jsonfile2 = open(opt.relation2id_path, 'r')
            self.relation2id_DIct = json.load(jsonfile2)

            with open(opt.vocab_pkl_path, 'rb') as f:
                self.vocab = pickle.load(f)

    def __getitem__(self, index):
        if opt.dataset == "vatex":

            video_id = self.eval_list[index]
            video_feat = self.video_feats[self.vatex_vnmapping_idx2word[video_id]][:]
            video_feat = torch.tensor(video_feat)

            if video_feat.shape[0] != 32:
                repeat_factor = (32 + video_feat.shape[0] - 1) // video_feat.shape[0]
                tensor_repeated = video_feat.repeat((repeat_factor, 1))
                video_feat = tensor_repeated[:32, :]

            video_feat = video_feat.to(torch.float32)

            objects_triplets = self.Video_caption_triplets_Dict[str(video_id)]
            objects = objects_triplets[1]
            objectshandled = self.vec_ents(objects, self.word2idx)
            object_one = objectshandled[0]
            object_two = objectshandled[1]
            # print("objectshandled:",objectshandled)
            triplets_idx = objects_triplets[2]
            captionhandled = self.mkGraphs(triplets_idx, len(objectshandled[1]))
            caption_adj = captionhandled[0]
            if opt.sparse:
                caption_adj = self.adjToSparse(caption_adj)
            caption_rel = captionhandled[1]

            region_feat = None
            spatial_feat = None
            pos_tag = None
        else:

            video_id = self.eval_list[index]
            video_feat = torch.from_numpy(self.video_feats[video_id])
            region_feat = torch.from_numpy(self.region_feats[video_id])
            spatial_feat = torch.from_numpy(self.spatial_feats[video_id])

            objects_triplets = self.Video_caption_triplets_Dict['vid'+str(video_id+1)]
            objects = objects_triplets[1]
            objectshandled = self.vec_ents(objects, self.vocab.word2idx)
            object_one = objectshandled[0]
            object_two = objectshandled[1]
            # print("objectshandled:",objectshandled)
            triplets_idx = objects_triplets[2]
            captionhandled = self.mkGraphs(triplets_idx, len(objectshandled[1]))
            caption_adj = captionhandled[0]
            if opt.sparse:
                caption_adj = self.adjToSparse(caption_adj)
            caption_rel = captionhandled[1]

        return video_feat, region_feat, spatial_feat, video_id,  object_one, object_two, caption_adj, caption_rel

    def __len__(self):
        return len(self.eval_list)

    def vec_ents(self, ex, field):
        # returns tensor and lens
        ex = [[field[x] if x in field else 0 for x in y.strip().split(" ")] for y in ex]

        return self.pad_list(ex, 1)

    def pad_list(self, l, ent=1):
        lens = [len(x) for x in l]
        m = max(lens)
        return torch.stack([self.pad(torch.tensor(x), m, ent) for x in l], 0), torch.LongTensor(lens)

    def pad(self, tensor, length, ent=1):
        return torch.cat([tensor, tensor.new(length - tensor.size(0), *tensor.size()[1:]).fill_(ent)])

    def mkGraphs(self, r, ent):
        # convert triples to entlist with adj and rel matrices
        # x.rel = self.mkGraphs(x.rel,len(x.ent[1]))
        x = r
        rel = [2]
        # global root node
        adjsize = ent + 1 + (2 * len(x))
        adj = torch.zeros(adjsize, adjsize)
        for i in range(ent):
            adj[i, ent] = 1
            adj[ent, i] = 1
        for i in range(adjsize):
            adj[i, i] = 1
        for y in x:
            rel.extend([y[1] + 3, y[1] + 3 + len(self.relation2id_DIct)])
            a = y[0]
            b = y[2]
            c = ent + len(rel) - 2
            d = ent + len(rel) - 1
            adj[a, c] = 1
            adj[c, b] = 1
            adj[b, d] = 1
            adj[d, a] = 1
        rel = torch.LongTensor(rel)
        return (adj, rel)

    def adjToSparse(self, adj):
        sp = []
        for row in adj:
            sp.append(row.nonzero().squeeze(1))
        return sp


def train_collate_fn(data):
    # print("len(data):",len(data))
    # print(data[0])
    # print("len(data[0]):",len(data[0]))
    data.sort(key=lambda x: x[-5], reverse=True)
# video_feat, region_feat, spatial_feat, caption, pos_tag, length, video_id, objects, triplets_idx
    videos, regions, spatials, captions, pos_tags, lengths, video_ids, object_one, object_two, caption_adj, caption_rel = zip(*data)

    # print(videos)

    videos = torch.stack(videos, 0)
    # regions = torch.stack(regions, 0)
    # spatials =torch.stack(spatials, 0)
    captions = torch.stack(captions, 0)
    # pos_tags = torch.stack(pos_tags, 0)

    return videos, regions, spatials, captions, pos_tags, lengths, video_ids, object_one, object_two, caption_adj, caption_rel

def eval_collate_fn(data):
    # collate_fn：如何取样本的，我们可以定义自己的函数来准确地实现想要的功能
    data.sort(key=lambda x: x[-5], reverse=False)

    videos, regions, spatials, video_ids, object_one, object_two, caption_adj, caption_rel = zip(*data)

    videos = torch.stack(videos, 0)
    # regions = torch.stack(regions, 0)
    # spatials = torch.stack(spatials, 0)

    return videos, regions, spatials, video_ids, object_one, object_two, caption_adj, caption_rel

# train_loader = get_train_loader(opt.train_caption_pkl_path, opt.feature_h5_path, opt.region_feature_h5_path)
# 前两个有的，最后一个是14.1g，有点大，正在下载中
# msvd_captions_train.pkl; feature_h5; region_feature_h5
def get_train_loader(cap_pkl, frame_feature_h5, region_feature_h5, Video_objects_triplets_json, batch_size=100, shuffle=True, num_workers=0, pin_memory=True, multi_gpu=False):
    # 是不是我的电脑的意思
    if getpass.getuser() == 'lenovo':
        num_workers = 0
    print('num_workers = ', num_workers)
    v2t = V2TDataset(cap_pkl, frame_feature_h5, region_feature_h5, Video_objects_triplets_json)
    # return video_feat, region_feat, spatial_feat, caption, pos_tag, length, video_id 【就是一堆tensor】
    data_sampler = None
    if multi_gpu:
        data_sampler = torch.utils.data.distributed.DistributedSampler(v2t)
    data_loader = torch.utils.data.DataLoader(dataset=v2t,
                                              batch_size=batch_size,
                                              shuffle=False if multi_gpu else True,
                                              num_workers=num_workers,
                                              collate_fn=train_collate_fn,
                                              pin_memory=pin_memory,
                                              sampler=data_sampler
                                              )
    print("data_loader:",data_loader)
    return data_loader, data_sampler


def get_eval_loader(cap_pkl, frame_feature_h5, region_feature_h5, Video_objects_triplets_json, batch_size=100, shuffle=False, num_workers=0, pin_memory=False, multi_gpu=False):
    vd = VideoDataset(cap_pkl, frame_feature_h5, region_feature_h5, Video_objects_triplets_json)
    data_sampler = None
    if multi_gpu:
        data_sampler = torch.utils.data.distributed.DistributedSampler(vd)
    data_loader = torch.utils.data.DataLoader(dataset=vd,
                                              batch_size=batch_size,
                                              shuffle=shuffle,
                                              num_workers=num_workers,
                                              collate_fn=eval_collate_fn,
                                              pin_memory=pin_memory,
                                              sampler=data_sampler
                                              )
    return data_loader


if __name__ == '__main__':
    train_loader = get_train_loader(opt.train_caption_pkl_path, opt.feature_h5_path, opt.region_feature_h5_path, opt.Video_objects_triplets_json)
    # return data_loader, data_sampler
    print(len(train_loader))
    d = next(iter(train_loader))
    print(d[0].size())
    print(d[1].size())
    print(d[2].size())
    print(d[3].size())
    print(len(d[4]))
    print(d[5])

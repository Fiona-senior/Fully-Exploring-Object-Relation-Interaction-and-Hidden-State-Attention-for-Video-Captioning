import pickle
import h5py
import torch
import torch.utils.data as data
from opt import parse_opt
import os
import glob
import tables as tb
import numpy as np
import getpass
opt = parse_opt()


cap_pkl=opt.train_caption_pkl_path
frame_feature_h5=opt.feature_h5_path
region_feature_h5=opt.region_feature_h5_path
if getpass.getuser() == 'lenovo':
    num_workers = 0
print('num_workers = ', num_workers)

if not os.path.exists(cap_pkl):
    cap_pkl = '../data/MSR-VTT/msr-vtt_captions_train.pkl'
with open(cap_pkl, 'rb') as f:
    # video ids: train ids for videos
    captions, pos_tags, lengths, video_ids = pickle.load(f)


print("cap_pkl is ok.")

h5 = h5py.File(frame_feature_h5, 'r')
print("h5.keys(frame_feature_h5):", h5.keys())
video_feats = h5[opt.feature_h5_feats]
print("video_feats:",video_feats)


print("frame_feature_h5 is ok.")

if not os.path.exists(region_feature_h5):
    print("oh,no!")
else:
    print("yes,you have region_feature_hs.")

h5 = h5py.File(region_feature_h5, 'r')
print("h5.keys(region_feature_h5):", h5.keys())
region_feats = h5[opt.region_visual_feats]
spatial_feats = h5[opt.region_spatial_feats]
# h5.close()
print('hehe')


index=0
caption = captions[index]
print("caption:",caption)
pos_tag = pos_tags[index]
print("pos_tag:",pos_tag)
length = lengths[index]
print("length:",length)
video_id = video_ids[index]
print("video_id:",video_id)
video_feat = torch.from_numpy(video_feats[video_id])
print("video_feat:",video_feat)
region_feat = torch.from_numpy(region_feats[video_id])
print("region_feat:",region_feat)
spatial_feat = torch.from_numpy(spatial_feats[video_id])
print("spatial_feat:",spatial_feat)

data.sort(key=lambda x: x[-1], reverse=False)

videos, regions, spatials, video_ids = zip(*data)

videos = torch.stack(videos, 0)
regions = torch.stack(regions, 0)
spatials = torch.stack(spatials, 0)


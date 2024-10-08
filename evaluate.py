import sys
sys.path.insert(0, './captioneval')
import torch
import pickle
import models
from utils.utils import Vocabulary
from utils.data import get_eval_loader
from cocoeval import COCOScorer, suppress_stdout_stderr
from utils.opt import parse_opt
from tqdm import tqdm
import collections
import time

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def convert_data_to_coco_scorer_format(reference):
    reference_json = {}
    non_ascii_count = 0
    with open(reference, 'r',encoding="utf-8") as f:
        lines = f.readlines()
        for line in lines:
            vid = line.split('\t')[0]
            sent = line.split('\t')[1].strip()
            try:
                sent.encode('ascii', 'ignore').decode('ascii')
            except UnicodeDecodeError:
                non_ascii_count += 1
                continue
            if vid in reference_json:
                reference_json[vid].append({u'video_id': vid, u'cap_id': len(reference_json[vid]),
                                    u'caption': sent.encode('ascii', 'ignore').decode('ascii')})
            else:
                reference_json[vid] = []
                reference_json[vid].append({u'video_id': vid, u'cap_id': len(reference_json[vid]),
                                    u'caption': sent.encode('ascii', 'ignore').decode('ascii')})
    if non_ascii_count:
        print("=" * 20 + "\n" + "non-ascii: " + str(non_ascii_count) + "\n" + "=" * 20)
    return reference_json

# def convert_prediction_old(prediction):
#     prediction_json = {}
#     with open(prediction, 'r') as f:
#         lines = f.readlines()
#         for line in lines:
#             vid = line.split('\t')[0]
#             sent = line.split('\t')[1].strip()
#             prediction_json[vid] = [{u'video_id': vid, u'caption': sent}]
#     return prediction_json

def convert_prediction(prediction):
    prediction_json = {}
    for key, value in prediction.items():
        prediction_json[str(key)] = [{u'video_id': str(key), u'caption': value}]
    return prediction_json

def evaluate(net, opt, eval_loader, reference, multi_modal=False, multi_gpu=False):

    prediction_txt_path = opt.test_prediction_txt_path
    alpha_all = []
    result = collections.OrderedDict()
    start_time = time.time()
    for i, (frames, regions, spatials, video_ids, object_one, object_two, caption_adj, caption_rel) in tqdm(enumerate(eval_loader)):
        print(i)
        print("frames.shape:",frames.shape)
        frames = frames.to(DEVICE)
        # regions = regions.to(DEVICE)
        # spatials = spatials.to(DEVICE)
        if multi_modal:
            # regions = regions[:,:,:opt.num_obj,:].to(DEVICE)
            object_one, object_two, object_3, caption_adj, caption_rel = fixTobatch(object_one, object_two,
                                                                                         caption_adj, caption_rel)
            outputs = net(DEVICE, object_one, object_two, object_3, caption_adj, caption_rel, frames, None)
            #     def forward(self, device, object_one, object_two, object_3, caption_adj, caption_rel, frames, targets, max_words=None, teacher_forcing_ratio=1.0):

        else:
            outputs = net(frames, None)

        for (tokens, vid) in zip(outputs, video_ids):
            if multi_gpu:
                s = net.module.decoder.decode_tokens(tokens.data)
            else:
                s = net.decoder.decode_tokens(tokens.data)
            result[vid] = s

    # with open(prediction_txt_path, 'w') as f:
    #     for vid, s in result.items():
    #         f.write('%d\t%s\n' % (vid, s))
    end_time = time.time()
    print('evaluate inference time:', end_time-start_time)
    prediction_json = convert_prediction(result)

    # compute scores
    scorer = COCOScorer()
    with suppress_stdout_stderr():
        scores, sub_category_score = scorer.score(reference, prediction_json, prediction_json.keys())
    for metric, score in scores.items():
        print('%s: %.6f' % (metric, score * 100))

    if sub_category_score is not None:
        print('Sub Category Score in Spice:')
        for category, score in sub_category_score.items():
            print('%s: %.6f' % (category, score * 100))
    return scores, result, alpha_all, end_time-start_time


def gather_results(net, opt, eval_loader, multi_modal=False, multi_gpu=False):
    result = collections.OrderedDict()
    alpha_all = []
    for i, (frames, regions, spatials, video_ids) in tqdm(enumerate(eval_loader)):
        frames = frames.to(DEVICE)
        regions = regions[:,:,:opt.num_obj,:].to(DEVICE)
        outputs, _, _, alpha = net(frames, regions, None)

        alpha_all.append(alpha)
        for (tokens, vid) in zip(outputs, video_ids):
            if multi_gpu:
                s = net.module.decoder.decode_tokens(tokens.data)
            else:
                s = net.decoder.decode_tokens(tokens.data)
            result[vid] = s
    return result, alpha_all


def evaluate_multi_gpu(result, reference):
    start_time = time.time()
    prediction_json = convert_prediction(result)
    scorer = COCOScorer()
    with suppress_stdout_stderr():
        scores, sub_category_score = scorer.score(reference, prediction_json, prediction_json.keys())
    for metric, score in scores.items():
        print('%s: %.6f' % (metric, score * 100))

    if sub_category_score is not None:
        print('Sub Category Score in Spice:')
        for category, score in sub_category_score.items():
            print('%s: %.6f' % (category, score * 100))

    end_time = time.time()
    return scores, result, end_time - start_time

def fixTobatch(object_one, object_two, caption_adj, caption_rel):
    object_one, object_3 = adjToBatch(object_one)
    object_one = object_one.to(DEVICE)
    object_3 = object_3.to(DEVICE)
    object_two = torch.cat(object_two, 0).to(DEVICE)
    caption_adj = listTo(caption_adj)
    caption_rel = listTo(caption_rel)
    return object_one, object_two, object_3, caption_adj, caption_rel

def adjToBatch( adj):
    lens = [x.size(0) for x in adj]  # 就是将entity的矩阵型转 r×c 其中的r给集中起来
    m = max([x.size(1) for x in adj])  # [x.size(1) for x in adj] 就是将c集中起来，并且选出最大的c
    data = [pad(x.transpose(0, 1), m).transpose(0, 1) for x in adj]
    data = torch.cat(data, 0)
    return data, torch.LongTensor(lens)

def pad(tensor, length, ent=1):
    return torch.cat([tensor, tensor.new(length - tensor.size(0), *tensor.size()[1:]).fill_(ent)])

def listTo( l):
    return [x.to(DEVICE) for x in l]

if __name__ == '__main__':
    opt = parse_opt()

    with open(opt.vocab_pkl_path, 'rb') as f:
        vocab = pickle.load(f)

    net = models.setup(opt, vocab)
    if opt.use_multi_gpu:
        net = torch.nn.DataParallel(net)
    if not opt.eval_metric:
        net.load_state_dict(torch.load(opt.model_pth_path))
    elif opt.eval_metric == 'METEOR':
        net.load_state_dict(torch.load(opt.best_meteor_pth_path))
    elif opt.eval_metric == 'CIDEr':
        net.load_state_dict(torch.load(opt.best_cider_pth_path))
    else:
        raise ValueError('Please choose the metric from METEOR|CIDEr')
    net.to(DEVICE)
    net.eval()

    reference = convert_data_to_coco_scorer_format(opt.test_reference_txt_path)
    metrics = evaluate(opt, net, opt.test_range, opt.test_prediction_txt_path, reference)
    with open(opt.test_score_txt_path, 'a') as f:
        f.write('\nBEST ' + str(opt.eval_metric) + '(beam size = {}):\n'.format(opt.beam_size))
        for k, v in metrics.items():
            f.write('\t%s: %.2f\n' % (k, 100 * v))


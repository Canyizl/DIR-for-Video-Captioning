import pickle
import h5py
import torch
import torch.utils.data as data
from data.opt import parse_opt
opt = parse_opt()

class V2TDataset(data.Dataset):
    def __init__(self, cap_pkl, frame_feature_h5, json_dict):
        with open(cap_pkl, 'rb') as f:
            self.captions, self.pos_tags, self.lengths, self.video_ids = pickle.load(f)
        h5 = h5py.File(frame_feature_h5, 'r')
        self.video_feats = h5[opt.feature_h5_feats]
        self.dataset = json_dict


    def __getitem__(self, index):
        caption = self.captions[index]
        pos_tag = self.pos_tags[index]
        length = self.lengths[index]
        video_id = self.video_ids[index]
        video_feat = torch.from_numpy(self.video_feats[video_id])
        dp = self.dataset[video_id]
        multiclass_inds = torch.tensor(dp['multiclass_inds'])
        multiclass_rels = torch.tensor(dp['relation_label'])

        return video_feat, multiclass_inds,multiclass_rels,caption, pos_tag, length, video_id

    def __len__(self):
        return len(self.captions)


class VideoDataset(data.Dataset):
    def __init__(self, eval_range, frame_feature_h5):
        self.eval_list = tuple(range(*eval_range))
        h5 = h5py.File(frame_feature_h5, 'r')
        self.video_feats = h5[opt.feature_h5_feats]

    def __getitem__(self, index):
        video_id = self.eval_list[index]
        video_feat = torch.from_numpy(self.video_feats[video_id])

        return video_feat,video_id

    def __len__(self):
        return len(self.eval_list)


def train_collate_fn(data):
    data.sort(key=lambda x: x[-1], reverse=True)


    videos,inds,rels,captions, pos_tags, lengths, video_ids = zip(*data)

    videos = torch.stack(videos, 0)
    inds = torch.stack(inds,0)
    rels = torch.stack(rels,0)
    captions = torch.stack(captions, 0)
    pos_tags = torch.stack(pos_tags, 0)
    return videos,inds,rels,captions, pos_tags, lengths, video_ids


def eval_collate_fn(data):
    data.sort(key=lambda x: x[-1], reverse=False)

    videos, video_ids = zip(*data)

    videos = torch.stack(videos, 0)

    return videos,video_ids


def get_train_loader(cfg,json_dict,cap_pkl, frame_feature_h5):
    v2t = V2TDataset(cap_pkl, frame_feature_h5, json_dict)
    data_loader = torch.utils.data.DataLoader(dataset=v2t,
                                              batch_size=cfg.batch_size,
                                              shuffle=True,
                                              num_workers=0,
                                              collate_fn=train_collate_fn,
                                              pin_memory=True)
    return data_loader


def get_eval_loader(cap_pkl, frame_feature_h5):
    vd = VideoDataset(cap_pkl, frame_feature_h5)
    data_loader = torch.utils.data.DataLoader(dataset=vd,
                                              batch_size=10,
                                              shuffle=False,
                                              num_workers=0,
                                              collate_fn=eval_collate_fn,
                                              pin_memory=False)
    return data_loader


if __name__ == '__main__':
    from config import ConfigMARN
    cfg = ConfigMARN()
    cfg.dataset = 'msvd'
    train_loader = get_eval_loader(cfg,None,'../dataset/MSVD/msvd_captions_val.pkl', '../dataset/MSVD/msvd_features.h5')
    #train_loader = get_train_loader(opt.train_caption_pkl_path, opt.feature_h5_path, opt.region_feature_h5_path)
    print(len(train_loader))
    d = next(iter(train_loader))
    print(d[0].size())
    print(d[1].size())
    print(d[2].size())
    print(d[3])
    print(d[4])

import os
import pickle
import torch
import cv2
import random
import warnings
warnings.filterwarnings('ignore')

from utils import Utils,get_w2v_vec,get_datetime_stamp,get_user_yesno_answer,Vocabulary
from data.data import get_train_loader,get_eval_loader
from data.opt import parse_opt
from train import train

import sys
import json
from gensim.models import KeyedVectors
from models.Encoder import EncoderVisual
from models.Decoder import Decoder
from models.model2 import MLP,Reason
from torch.optim.lr_scheduler import StepLR

#set seed for reproducibility
utils = Utils()
utils.set_seed(1)

from config import ConfigMARN


def visual_captioning(path,pred,ids):
    ids = str(ids)
    cap = cv2.VideoCapture(path)
    while (cap.isOpened()):
        ret, frame = cap.read()
        if not ret:
            break
        k = cv2.waitKey(20)
        if (k & 0xff == ord('q')):
            break
        word_x = 10
        word_y = 200
        cv2.putText(frame,  pred, (word_x, word_y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 155), 2)
        cv2.imshow('Video'+ids, frame)

    cap.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    print("Start Init MSRVTT datasets model.....")
    cfg = ConfigMARN()
    cfg.dataset = 'msrvtt'
    vocab_pkl_path = "./dataset/MSR-VTT/msr-vtt_vocab.pkl"
    with open(vocab_pkl_path, 'rb') as f:
        vocab = pickle.load(f)
    cfg.vocab_size = len(vocab)
    cfg.reload = './checkpoints/warm_up.pt'
    if cfg.reload:
        reload_file_path = cfg.reload
        print('Reloading model from {}'.format(reload_file_path))
        saved_model = torch.load(reload_file_path)
        encoder = saved_model['encoder']
        mul_ind = saved_model['mul_ind']
        mul_rel = saved_model['mul_rel']
        encoder.batch_size = cfg.batch_size
        best_loss = saved_model['best_metrics']
        reason_ind = saved_model['reason_ind']
        reason_rel = saved_model['reason_rel']
        decoder = saved_model['decoder']

    print('Loading w2v model...')
    json_path = f"{cfg.dataset}_final.json"
    with open(json_path) as f:
        json_data = json.load(f)
    cfg.dict_path = "./checkpoints/dict2.pt"
    if cfg.dict_path:
        saved_model = torch.load(cfg.dict_path)
        ind_dict = saved_model['ind_dict']
        rel_dict = saved_model['rel_dict']

    cfg.batch_size = 4
    inds, rels, json_data_list = json_data['inds'], json_data['relations'], json_data['dataset']
    cfg.len_inds = len(inds)
    cfg.len_rels = len(rels)
    json_data_dict = {dp['video_id']: dp for dp in json_data_list}
    test_loader = get_eval_loader(cfg.msrvtt_test_range, './dataset/MSR-VTT/msr-vtt_features.h5')
    all_vids = [7393,7422,7423,7432,7433,7440,7441,7443,]
    random.seed(233)

    while(1):

        mode = input("Please choose the way to visualize (random / input): ")
        if mode == "random":
            random_idx = random.randint(1,50) % 8
            input_video = str(all_vids[random_idx])
        elif mode == "input" :
            input_video = input("Input video name is: ")
        else:
            print("Please choose random / input !")
            continue
        samples_video_path = './samples_video/'
        input_video_path = samples_video_path + input_video + '.mp4'
        print("The video path is " + input_video_path )
        print("Start Captioning....")
        encoder.eval()
        mul_ind.eval()
        mul_rel.eval()
        reason_ind.eval()
        reason_rel.eval()
        decoder.eval()

        for data in test_loader:
            video_features, video_ids = data
            for i in range(cfg.batch_size):
                if input_video == str(video_ids[i]):
                    video_features = video_features.to(cfg.device)
                    appear, motion = encoder(video_features)
                    multiclassif_logits = mul_ind(appear)
                    score_base = mul_rel(motion)

                    rels_scores, rels_indices = score_base.sort(descending=True, dim=1)  # inds_size: B,N 100,122
                    _, rels_dis = score_base.sort(descending=False, dim=1)
                    topk = 3
                    topk_rels_indices = rels_indices[:, :topk]
                    disk = 5
                    disk_rels_indices = rels_dis[:, :disk]

                    inds_scores, inds_indices = multiclassif_logits.sort(descending=True, dim=1)  # inds_size: B,N 100,122
                    _, inds_dis = multiclassif_logits.sort(descending=False, dim=1)
                    topk = 3
                    topk_inds_indices = inds_indices[:, :topk]
                    disk = 5
                    disk_inds_indices = inds_dis[:, :disk]

                    loss_ind_ae, loss_ind_moco, z_ind = reason_ind(multiclassif_logits, inds, ind_dict)
                    loss_rel_ae, loss_rel_moco, z_rel = reason_rel(score_base, rels, rel_dict)
                    z_ind = z_ind[1:]
                    z_rel = z_rel[1:]

                    max_len = 26
                    cnn_feats = torch.cat((appear, motion), dim=2)
                    outputs, _ = decoder(cnn_feats, z_ind, z_rel, appear, motion, captions=None, max_words=26)

                    for (tokens, vid) in zip(outputs, video_ids):
                        if(vid == video_ids[i]):
                            s = decoder.decode_tokens(tokens.data)
                            print(str(vid)+":"+s)
                            visual_captioning(input_video_path,s,vid)
                    break
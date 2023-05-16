from pdb import set_trace
import time
import utils
import matplotlib.pyplot as plt
plt.switch_backend('agg')

import torch
import torch.nn as nn
from early_stopper import EarlyStopper
import random
import numpy as np
import collections
import math
from utils import clip_gradient
from tqdm import tqdm
from caption_eval.cocoeval import COCOScorer, suppress_stdout_stderr


def convert_prediction(prediction):
    prediction_json = {}
    for key, value in prediction.items():
        prediction_json[str(key)] = [{u'video_id': str(key), u'caption': value}]
    return prediction_json

def evaluate(cfg,encoder,multiclassifier,cas,reason_ind,reason_rel,decoder,inds,ind_dict,rels,rel_dict,test_loader,test_reference):
    encoder.eval()
    multiclassifier.eval()
    cas.eval()
    reason_ind.eval()
    reason_rel.eval()
    decoder.eval()
    with torch.no_grad():
        result = collections.OrderedDict()
        for i, (video_features, video_ids) in tqdm(enumerate(test_loader)):
            video_features = video_features.to(cfg.device)

            appear, motion = encoder(video_features)
            multiclassif_logits = multiclassifier(appear)

            score_base = cas(motion)
            loss_ind_ae, loss_ind_moco, z_ind = reason_ind(multiclassif_logits, inds, ind_dict)
            loss_rel_ae, loss_rel_moco, z_rel = reason_rel(score_base, rels, rel_dict)
            z_ind = z_ind[1:, :]
            z_rel = z_rel[1:, :]

            cnn_feats = torch.cat((appear, motion), dim=2)
            outputs, _ = decoder(cnn_feats, z_ind, z_rel, appear, motion, captions=None, max_words=26)

            for (tokens, vid) in zip(outputs, video_ids):
                s = decoder.decode_tokens(tokens.data)
                result[vid] = s

        '''
        prediction_txt_path = "D:\kt\project2\KG\src2\dataset\MSVD/msvd_annotations.json"
        with open(prediction_txt_path, 'w') as f:
            for vid, s in result.items():
                f.write('%d\t%s\n' % (vid, s))
        '''

        prediction_json = convert_prediction(result)
        #scores = utils.score(test_reference, prediction_json)

        # compute scores
        scorer = COCOScorer()
        with suppress_stdout_stderr():
            scores, sub_category_score = scorer.score(test_reference, prediction_json, prediction_json.keys())
        for metric, score in scores.items():
            print('%s: %.6f' % (metric, score * 100))

        if sub_category_score is not None:
            print('Sub Category Score in Spice:')
            for category, score in sub_category_score.items():
                print('%s: %.6f' % (category, score * 100))

    return scores

def train(cfg,inds,rels,encoder,mul_ind,mul_rel,reason_ind,reason_rel,decoder,train_dl,val_dl,optimizer,sched,exp_name,train,best_loss,ind_dict,rel_dict,test_ref):
    EarlyStop = EarlyStopper(patience=cfg.patience)

    focus_on = 'CIDEr'

    if best_loss != None :
        EarlyStop.val_loss_min = best_loss
        print("best_loss:",best_loss)

    if cfg.max_epochs < 11:
        for epoch_num in range(cfg.max_epochs):
            epoch_start_time = time.time()
            batch_train_mul_losses = []
            print("Epoch:", epoch_num + 1)
            print("pretrain")
            #print("lr:",sched.get_lr()[0])
            print("opt lr:",optimizer.state_dict()['param_groups'][0]['lr'])
            for iter_,data in enumerate(train_dl):
                new_train_mul_loss = pretrain(cfg,inds,rels,data,encoder,mul_ind,mul_rel,reason_ind,reason_rel,decoder,optimizer,ind_dict,rel_dict,iter_)
                batch_train_mul_losses.append(new_train_mul_loss)
                if iter_ % 1500 == 0:
                    print('Batch:', iter_ , 'cap_loss:', new_train_mul_loss)
            try:
                epoch_train_mul_loss = sum(batch_train_mul_losses) / len(batch_train_mul_losses)
                print("train_mul_loss:",epoch_train_mul_loss)
            except:
                pass
            save_dict = {'encoder': encoder, 'mul_ind': mul_ind, 'mul_rel': mul_rel, 'reason_ind': reason_ind,
                         'reason_rel': reason_rel, 'decoder': decoder, 'ind_dict': ind_dict, 'rel_dict': rel_dict,
                        'best_metrics': 0}
            EarlyStop.save_checkpoint(0, save_dict, "premul")
            print(f'Epoch time: {asMinutes(time.time() - epoch_start_time)}')
    else:
        for epoch_num in range(cfg.max_epochs):
            epoch_start_time = time.time()
            batch_train_cap_losses = []
            batch_train_all_losses = []
            print("Epoch:", epoch_num + 1)
            cfg.beam_size = 5
            #print("lr:",sched.get_lr()[0])
            print("opt lr:",optimizer.state_dict()['param_groups'][0]['lr'])
            for iter_,data in enumerate(train_dl):
                new_train_cap_loss,new_train_all_loss = train_on_batch(cfg,inds,rels,data,encoder,mul_ind,mul_rel,reason_ind,reason_rel,decoder,optimizer,ind_dict,rel_dict,iter_)
                if iter_ % 1500 == 0:
                    print('Batch:', iter_ , 'cap_loss:', new_train_cap_loss,'all_loss:',new_train_all_loss)
                batch_train_cap_losses.append(new_train_cap_loss)
                batch_train_all_losses.append(new_train_all_loss)
            try:
                epoch_train_cap_loss = sum(batch_train_cap_losses) / len(batch_train_cap_losses)
                epoch_train_all_loss = sum(batch_train_all_losses) / len(batch_train_all_losses)
                print('train_cap_loss', epoch_train_cap_loss,'train_all_loss',epoch_train_all_loss)
            except :
                pass
            sched.step()
            print("Test:")
            metrics = evaluate(cfg,encoder,mul_ind,mul_rel,reason_ind,reason_rel,decoder,inds,ind_dict,rels,rel_dict,val_dl,test_ref)
            for k, v in metrics.items():
                if k == focus_on:
                    save_dict = {'encoder': encoder, 'mul_ind': mul_ind, 'mul_rel': mul_rel, 'reason_ind': reason_ind,
                                 'reason_rel': reason_rel, 'decoder': decoder, 'ind_dict': ind_dict,
                                 'rel_dict': rel_dict,
                                 'best_metrics': v}
                    EarlyStop(v, save_dict, exp_name="warm_up")
                    #print('val_metrics:',v)
                    if epoch_num % 5 == 0:
                        EarlyStop.save_long(v,save_dict,"long_train_CIDEr")
            if EarlyStop.early_stop:
                pass
            print(f'Epoch time: {asMinutes(time.time() - epoch_start_time)}')


def train_on_batch(cfg,inds,rels,data,encoder,multiclassifier,cas,reason_ind,reason_rel,decoder,optimizer,ind_dict,rel_dict,iter_):
    encoder.train()
    multiclassifier.train()
    cas.train()
    reason_ind.train()
    reason_rel.train()
    decoder.train()

    ind_criterion = nn.BCEWithLogitsLoss()
    rel_criterion = nn.BCEWithLogitsLoss()
    cap_criterion = nn.CrossEntropyLoss()

    #video_features, inds, rels, _, _, _, _ = data
    video_features,inds_label,rels_label, captions, pos_tags, cap_lens, video_ids = data

    video_features = video_features.to(cfg.device)
    multiclass_inds = inds_label.float().to(cfg.device)
    multiclass_rel = rels_label.float().to(cfg.device)

    optimizer.zero_grad()

    appear,motion = encoder(video_features)
    multiclassif_logits = multiclassifier(appear)
    multiclass_loss = ind_criterion(multiclassif_logits, multiclass_inds)

    score_base = cas(motion)
    rel_loss = rel_criterion(score_base,multiclass_rel)

    # loss1,loss2,z_ind,z_rel = get_reason(multiclassif_logits,score_base,inds,rels,ind_dict,rel_dict,vae,moco)
    loss_ind_ae,loss_ind_moco,z_ind = reason_ind(multiclassif_logits,inds,ind_dict)
    loss_rel_ae,loss_rel_moco,z_rel = reason_rel(score_base,rels,rel_dict)
    z_ind = z_ind[1:,:]
    z_rel = z_rel[1:,:]


    cnn_feats = torch.cat((appear,motion),dim=2)
    max_words = 26
    max_len = 26
    outputs, alphas = decoder(cnn_feats,z_ind,z_rel,appear,motion,captions,max_words)
    captions = captions[:, :max_len]
    targets = captions.to(cfg.device)

    bsz = len(captions)
    # remove pad and flatten outputs
    outputs = torch.cat([outputs[j][:cap_lens[j]] for j in range(bsz)], 0)
    outputs = outputs.view(-1, cfg.vocab_size)

    # remove pad and flatten targets
    targets = torch.cat([targets[j][:cap_lens[j]] for j in range(bsz)], 0)
    targets = targets.view(-1)

    # compute captioning loss
    cap_loss =cap_criterion(outputs, targets)

    loss_ae = loss_ind_ae + loss_rel_ae
    loss_moco = loss_ind_moco + loss_rel_moco
    loss = (multiclass_loss + rel_loss) + (loss_ae + loss_moco) + cap_loss

    loss.backward()
    clip_gradient(optimizer, cfg.grad_clip)
    optimizer.step()

    if iter_ % 200 == 0 :
        with torch.no_grad():
            print("loss_ae:", loss_ae.item() ,"loss_moco:", loss_moco.item())

    return round(cap_loss.item(),5),round(loss.item(),5)

def pretrain(cfg,inds,rels,data,encoder,multiclassifier,cas,reason_ind,reason_rel,decoder,optimizer,ind_dict,rel_dict,iter_):
    encoder.train()
    multiclassifier.train()
    cas.train()


    ind_criterion = nn.BCEWithLogitsLoss()
    rel_criterion = nn.BCEWithLogitsLoss()
    cap_criterion = nn.CrossEntropyLoss()

    #video_features, inds, rels, _, _, _, _ = data
    video_features,inds_label,rels_label, captions, pos_tags, cap_lens, video_ids = data

    video_features = video_features.to(cfg.device)
    multiclass_inds = inds_label.float().to(cfg.device)
    multiclass_rel = rels_label.float().to(cfg.device)

    optimizer.zero_grad()

    appear,motion = encoder(video_features)
    multiclassif_logits = multiclassifier(appear)
    multiclass_loss = ind_criterion(multiclassif_logits, multiclass_inds)

    score_base = cas(motion)
    rel_loss = rel_criterion(score_base,multiclass_rel)

    loss = multiclass_loss + rel_loss

    loss.backward()
    clip_gradient(optimizer, cfg.grad_clip)
    optimizer.step()

    return round(loss.item(),5)


def asMinutes(s):
    m = math.floor(s / 60)
    s -= m * 60
    return '%dm %ds' % (m, s)




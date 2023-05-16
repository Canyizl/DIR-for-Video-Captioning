import torch
import torch.nn as nn
import torch.nn.functional as F

class Filter_Module(nn.Module):
    def __init__(self, len_feature):
        super(Filter_Module, self).__init__()
        self.len_feature = len_feature
        self.conv_1 = nn.Sequential(
            nn.Conv1d(in_channels=self.len_feature, out_channels=512, kernel_size=1,
                    stride=1, padding=0),
            nn.LeakyReLU()
        )

        self.conv_2 = nn.Sequential(
            nn.Conv1d(in_channels=512, out_channels=1, kernel_size=1,
                    stride=1, padding=0),
            nn.Sigmoid()
        )

    def forward(self, x):
        # x: (B, T, F)        
        out = x.permute(0, 2, 1)
        # out: (B, F, T)
        out = self.conv_1(out)
        out = self.conv_2(out)
        out = out.permute(0, 2, 1)
        # out: (B, T, 1)
        return out
        

class CAS_Module(nn.Module):
    def __init__(self, len_feature, num_classes):
        super(CAS_Module, self).__init__()
        self.len_feature = len_feature
        self.conv_1 = nn.Sequential(
            nn.Conv1d(in_channels=self.len_feature, out_channels=2048, kernel_size=3,
                      stride=1, padding=1),
            nn.LeakyReLU()
        )

        self.conv_2 = nn.Sequential(
            nn.Conv1d(in_channels=2048, out_channels=num_classes + 1, kernel_size=1, stride=1, padding=0,bias=False)
        )

        self.drop_out = nn.Dropout(p=0.6)

    def forward(self, x):
        # x: (B, T, F)
        out = x.permute(0, 2, 1)
        # out: (B, F, T)
        out = self.conv_1(out)
        out = self.drop_out(out)
        out = self.conv_2(out)
        out = out.permute(0, 2, 1)
        # out: (B, T, C + 1)
        return out

class CAS(nn.Module):
    def __init__(self, len_feature, num_classes):
        super(CAS, self).__init__()
        self.len_feature = len_feature
        self.conv_1 = nn.Sequential(
            nn.Conv1d(in_channels=self.len_feature, out_channels=512, kernel_size=3,
                      stride=1, padding=1),
            nn.LeakyReLU(0.2)
        )
        self.drop_out = nn.Dropout(p=0.6)
        self.k = 7
        self.conv_2 = nn.Sequential(nn.Conv1d(in_channels=512, out_channels=num_classes, kernel_size=1, stride=1, padding=0,bias=False))

    def forward(self, x):
        # x: (B, T, F)
        out = x.permute(0, 2, 1)
        # out: (B, F, T)
        out = self.conv_1(out)
        out = self.drop_out(out)
        out = self.conv_2(out)
        out = out.permute(0, 2, 1)
        # out: (B, T, C)
        out, _ = out.sort(descending=True, dim=1)
        topk_scores_base = out[:, :self.k, :]
        score = torch.mean(topk_scores_base, dim=1)
        return score

class BaS_Net(nn.Module):
    def __init__(self, cfg,len_feature, num_classes):
        super(BaS_Net, self).__init__()
        self.filter_module = Filter_Module(len_feature).to(cfg.device)
        self.len_feature = len_feature
        self.num_classes = num_classes
        self.cas_module = CAS_Module(len_feature, num_classes).to(cfg.device)
        self.softmax = nn.Softmax(dim=1)

    def forward(self,x):
        fore_weights = self.filter_module(x)
        x_supp = fore_weights * x
        cas_base = self.cas_module(x)
        cas_supp = self.cas_module(x_supp)

        # slicing after sorting is much faster than torch.topk (https://github.com/pytorch/pytorch/issues/22812)
        # score_base = torch.mean(torch.topk(cas_base, self.k, dim=1)[0], dim=1)
        sorted_scores_base, _= cas_base.sort(descending=True, dim=1)
        k = max(sorted_scores_base.size(2)//8,3)
        topk_scores_base = sorted_scores_base[:, :k, :]
        score_base = torch.mean(topk_scores_base, dim=1)

        # score_supp = torch.mean(torch.topk(cas_supp, self.k, dim=1)[0], dim=1)
        sorted_scores_supp, _= cas_supp.sort(descending=True, dim=1)
        topk_scores_supp = sorted_scores_supp[:, :k, :]
        score_supp = torch.mean(topk_scores_supp, dim=1)

        #score_base = self.softmax(score_base)
        #score_supp = self.softmax(score_supp)

        return score_base, cas_base, score_supp, cas_supp, fore_weights


class BaS_Net_loss(nn.Module):
    def __init__(self):
        super(BaS_Net_loss, self).__init__()
        self.alpha = 0.0001
        self.ce_criterion = nn.MultiLabelSoftMarginLoss()

    def forward(self, score_base, score_supp, fore_weights, label):

        label_base = torch.cat((label, torch.ones((label.shape[0], 1)).cuda()), dim=1)
        label_supp = torch.cat((label, torch.zeros((label.shape[0], 1)).cuda()), dim=1)

        '''
        label_base = label_base / torch.sum(label_base, dim=1, keepdim=True)
        label_supp = label_supp / torch.sum(label_supp, dim=1, keepdim=True)
        print("label_base 2:",label_base)
        '''

        loss_base = self.ce_criterion(score_base, label_base.long())
        loss_supp = self.ce_criterion(score_supp, label_supp.long())
        loss_norm = torch.mean(torch.norm(fore_weights, p=1, dim=1))

        loss_total = loss_base + loss_supp + self.alpha * loss_norm

        return loss_total

class AsymmetricLossOptimized(nn.Module):
    ''' Notice - optimized version, minimizes memory allocation and gpu uploading,
    favors inplace operations'''

    def __init__(self, gamma_neg=4, gamma_pos=1, clip=0.05, eps=1e-8, disable_torch_grad_focal_loss=False):
        super(AsymmetricLossOptimized, self).__init__()

        self.gamma_neg = gamma_neg
        self.gamma_pos = gamma_pos
        self.clip = clip
        self.disable_torch_grad_focal_loss = disable_torch_grad_focal_loss
        self.eps = eps

        # prevent memory allocation and gpu uploading every iteration, and encourages inplace operations
        self.targets = self.anti_targets = self.xs_pos = self.xs_neg = self.asymmetric_w = self.loss = None

    def forward(self, x, y):
        """"
        Parameters
        ----------
        x: input logits
        y: targets (multi-label binarized vector)
        """

        self.targets = y
        self.anti_targets = 1 - y

        # Calculating Probabilities
        self.xs_pos = torch.sigmoid(x)
        self.xs_neg = 1.0 - self.xs_pos

        # Asymmetric Clipping
        if self.clip is not None and self.clip > 0:
            self.xs_neg.add_(self.clip).clamp_(max=1)

        # Basic CE calculation
        self.loss = self.targets * torch.log(self.xs_pos.clamp(min=self.eps))
        self.loss.add_(self.anti_targets * torch.log(self.xs_neg.clamp(min=self.eps)))

        # Asymmetric Focusing
        if self.gamma_neg > 0 or self.gamma_pos > 0:
            if self.disable_torch_grad_focal_loss:
                torch._C.set_grad_enabled(False)
            self.xs_pos = self.xs_pos * self.targets
            self.xs_neg = self.xs_neg * self.anti_targets
            self.asymmetric_w = torch.pow(1 - self.xs_pos - self.xs_neg,
                                          self.gamma_pos * self.targets + self.gamma_neg * self.anti_targets)
            if self.disable_torch_grad_focal_loss:
                torch._C.set_grad_enabled(True)
            self.loss *= self.asymmetric_w

        return -self.loss.sum()
import torch
from torch import nn
from torch.nn import functional as F
import torch.utils.checkpoint
from torch import nn
from typing import Union, Tuple, Optional
from dataclasses import dataclass
from transformers.utils import ModelOutput
import math
from tqdm import tqdm

@dataclass
class EncoderOutput(ModelOutput):
    loss: Optional[torch.FloatTensor] = None
    logits: torch.FloatTensor = None


@dataclass
class RankingEncoderOutput(ModelOutput):
    loss: Optional[torch.FloatTensor] = None
    logits: torch.FloatTensor = None
    rank: Optional[torch.Tensor] = None

@dataclass
class InferenceOutputForClassification(ModelOutput):
    logits: torch.FloatTensor = None
    rank: Optional[torch.Tensor] = None

@dataclass
class InferenceOutputForCosineSimilarity(ModelOutput):
    logits: torch.FloatTensor = None

# Pooler class. Copied and adapted from SimCSE code
class Pooler(nn.Module):
    '''
    Parameter-free poolers to get the sentence embedding
    'cls': [CLS] representation with BERT/RoBERTa's MLP pooler.
    'cls_before_pooler': [CLS] representation without the original MLP pooler.
    'avg': average of the last layers' hidden states at each token.
    'avg_top2': average of the last two layers.
    'avg_first_last': average of the first and the last layers.
    '''
    def __init__(self, pooler_type):
        super().__init__()
        self.pooler_type = pooler_type
        assert self.pooler_type in ['cls', 'cls_before_pooler', 'avg', 'avg_top2', 'avg_first_last'], 'unrecognized pooling type %s' % self.pooler_type

    def forward(self, attention_mask, outputs):
        last_hidden = outputs.last_hidden_state
        pooler_output = outputs.pooler_output
        hidden_states = outputs.hidden_states

        if self.pooler_type in ['cls_before_pooler', 'cls']:
            return last_hidden[:, 0]
        elif self.pooler_type == 'avg':
            return ((last_hidden * attention_mask.unsqueeze(-1)).sum(1) / attention_mask.sum(-1).unsqueeze(-1))
        elif self.pooler_type == 'avg_first_last':
            first_hidden = hidden_states[0]
            last_hidden = hidden_states[-1]
            pooled_result = ((first_hidden + last_hidden) / 2.0 * attention_mask.unsqueeze(-1)).sum(1) / attention_mask.sum(-1).unsqueeze(-1)
            return pooled_result
        elif self.pooler_type == 'avg_top2':
            second_last_hidden = hidden_states[-2]
            last_hidden = hidden_states[-1]
            pooled_result = ((last_hidden + second_last_hidden) / 2.0 * attention_mask.unsqueeze(-1)).sum(1) / attention_mask.sum(-1).unsqueeze(-1)
            return pooled_result
        else:
            raise NotImplementedError

class RankingLoss(nn.Module):
    def __init__(self, margin=0.8):
        super(RankingLoss, self).__init__()
        self.margin = margin

    def forward(self, logits, labels, weights=None):
        batch_size = logits.shape[0]
        masks = torch.ones_like(logits, dtype=torch.bool)
        masks[torch.arange(batch_size), labels] = False
        positive_logits = logits[~masks].reshape(batch_size, -1)
        negative_logits = logits[masks].reshape(batch_size, -1)
        
        loss = torch.clamp(self.margin - positive_logits + negative_logits, min=0)

        # 返回平均损失
        return loss.mean()

class AbsoluteLoss(nn.Module):
    def __init__(self, margin_high=0.8, margin_low=0):
        super(AbsoluteLoss, self).__init__()
        self.margin_high = margin_high
        self.margin_low = margin_low

    def forward(self, logits, labels, weights=None):
        batch_size = logits.shape[0]
        masks = torch.ones_like(logits, dtype=torch.bool)
        masks[torch.arange(batch_size), labels] = False
        positive_logits = logits[~masks]
        negative_logits = logits[masks]
        
        loss = torch.clamp(self.margin_high - positive_logits, min=0) \
                + torch.clamp(negative_logits - self.margin_low, min=0)
        return loss.mean()
    
    
# floss = -(aplha_t) * (1-p_t)^gamma * log(p_t)
class FocalLoss(nn.Module):
    def __init__(self, alpha=0.25, gamma=2):
        super(FocalLoss, self).__init__()
        # 当取alpha=1,gamma=0的时候focal_loss等效于cross_entropy_loss
        self.alpha = alpha
        self.gamma = gamma
        self.eps = 1e-7

    def forward(self, logits, labels, weights=None):
        labels = labels.view(-1, 1)

        # 将目标类别的索引转换为one-hot编码形式
        labels_one_hot = torch.zeros_like(logits).scatter_(1, labels, 1)

        # 计算每个类别的概率
        probs = F.softmax(logits, dim=1)
        # 计算Focal Loss
        term1 = self.alpha * (1 - probs) ** self.gamma * torch.log(probs + self.eps)
        term2 = (1 - self.alpha) * probs ** self.gamma * torch.log(1 - probs + self.eps)
        loss = -(labels_one_hot * term1 + (1 - labels_one_hot) * term2)

        return loss.sum(dim=-1).mean()

def law_embeddings_init(model, dataset):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    unique_id = set()
    with torch.no_grad():
        features_tensor = model.law_embeddings.weight.data.to(device)
        for data in tqdm(dataset):
            id = data["article_ids"]
            if id not in unique_id:
                unique_id.add(id)

                outputs = model.backbone(input_ids=torch.Tensor([data["input_ids_2"]]).long().to(device))
                features = model.pooler(None, outputs)
                features_tensor[id] = features
        model.law_embeddings.weight.data.copy_(features_tensor)


class DotProductWithBias(nn.Module):
    def __init__(self, out_dim):
        super(DotProductWithBias, self).__init__()
        self.bias = nn.Parameter(torch.zeros(out_dim))

    def forward(self, features_1, features_2):
        dot_product = (features_1 * features_2).sum(dim=-1, keepdim=False)
        return dot_product + self.bias

class Sigmoid_(nn.Module):
    def __init__(self, hidden_size=None):
        super(Sigmoid_, self).__init__()
        self.factor = math.sqrt(hidden_size)

    def forward(self, features_1, features_2):
        dot_product = (features_1 * features_2).sum(dim=-1, keepdim=False) / self.factor

        sigmoid = 1 / (1 + torch.exp(-dot_product))
        return sigmoid

import numpy as np

def compute_ndcg(logits, k=8):
    # 计算DCG
    def dcg(logits, k=8):
        logits = logits[:, :k]
        return np.sum(logits / np.log2(np.arange(2, k + 2)[np.newaxis ,:]), axis=1)

    # 计算IDCG
    def idcg(logits, k=8):
        sorted_logits = np.sort(logits, axis=-1)[:, ::-1]
        return dcg(sorted_logits, k)

    # 计算nDCG
    k = min(k, logits.shape[-1])
    best = idcg(logits, k)
    ndcg = dcg(logits, k) / best
    return np.mean(ndcg)

def compute_mrr(logits, k=8):
    k = min(k, logits.shape[-1])
    rr = np.sum(logits[:, :k] / np.arange(1, k + 1)[np.newaxis, :], axis=1)
    return np.mean(rr)

def compute_precision(logits):
    return np.mean(logits[:, 0])

def compute_ranking_metrics(logits):
    has_negative = np.any(logits < 0, axis=1)

    # 使用布尔索引剔除包含负数的行
    logits = logits[~has_negative]

    ndgc = compute_ndcg(logits)
    mrr = compute_mrr(logits)
    precision = compute_precision(logits)
    return ndgc, mrr, precision

def compute_ndcg_(logits, k=8):
    return np.mean(np.sum(logits / np.log2(np.arange(2, k + 2)[np.newaxis ,:]), axis=1))

def compute_mrr_(logits, k=8):
    return np.mean(np.sum(logits / np.arange(1, k + 1)[np.newaxis ,:], axis=1))

def compute_percent_(logits, k=8):
    return np.mean(logits, axis=0)

def compute_ranking_metrics_(logits, k=8):
    has_negative = np.any(logits < 0, axis=1)

    # 使用布尔索引剔除包含负数的行
    logits = logits[~has_negative]
    assert np.all(np.count_nonzero(logits, axis=1) <= 1)

    logits = logits[:, :k]
    ndgc = compute_ndcg_(logits, k)
    mrr = compute_mrr_(logits, k)
    percent = compute_percent_(logits, k)
    return ndgc, mrr, percent.tolist()


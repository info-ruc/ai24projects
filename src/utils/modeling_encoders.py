import torch
import torch.utils.checkpoint
from torch.nn.functional import cross_entropy
from torch import nn
from torch.nn.functional import cosine_similarity
from functools import partial
import torch
from .utils import *
from transformers.activations import ACT2FN
from transformers import PreTrainedModel, AutoModel
import logging


logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s: %(message)s')

logger = logging.getLogger(__name__)

class LawsEncoder(PreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.backbone = AutoModel.from_pretrained(
            config.model_name_or_path,
            from_tf=bool('.ckpt' in config.model_name_or_path),
            config=config,
            cache_dir=config.cache_dir,
            revision=config.model_revision,
            use_auth_token=True if config.use_auth_token else None,
            add_pooling_layer=False,
            torch_dtype=config.torch_dtype,
            low_cpu_mem_usage=config.low_cpu_mem_usage,
        ).base_model

        self.law_encoder = self.backbone # = AutoModel.from_pretrained(
        #     config.model_name_or_path,
        #     from_tf=bool('.ckpt' in config.model_name_or_path),
        #     config=config,
        #     cache_dir=config.cache_dir,
        #     add_pooling_layer=False,
        # ).base_model

        classifier_dropout = (
                config.classifier_dropout if config.classifier_dropout is not None else config.hidden_dropout_prob
            )
        if config.transform:
            self.transform = nn.Sequential(
                nn.Dropout(classifier_dropout),
                nn.Linear(config.hidden_size, config.hidden_size),
                ACT2FN[config.hidden_act],
                )
        else:
            self.transform = None

        self.pooler = Pooler(config.pooler_type)
        if config.pooler_type in {'avg_first_last', 'avg_top2'}:
            self.output_hidden_states = True
        else:
            self.output_hidden_states = False
        if config.measure == "cosine":
            self.distance = partial(cosine_similarity, dim=-1)
        elif config.measure == "sigmoid":
            self.distance = sigmoid
        else:
            raise NotImplementedError
        if config.objective == "contrast":
            self.loss_fct = nn.CrossEntropyLoss()
        elif config.objective == "rank":
            self.loss_fct = RankingLoss()
        elif config.objective == "absolute":
            self.loss_fct == AbsoluteLoss()
        elif config.objective == 'mse':
            self.loss_fct = nn.MSELoss()
        else:
            raise ValueError('Only regression and triplet objectives are supported for BiEncoderForClassification')
        self.post_init()

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,

        input_ids_2=None,
        attention_mask_2=None,
        token_type_ids_2=None,
        position_ids_2=None,
        head_mask_2=None,
        inputs_embeds_2=None,
        labels=None,
        **kwargs,
        ):
        bsz = input_ids.shape[0]
        # input_ids_ = input_ids
        # input_ids = self.concat_features(input_ids, input_ids_2)
        # attention_mask = self.concat_features(attention_mask, attention_mask_2)
        # token_type_ids = self.concat_features(token_type_ids, token_type_ids_2)
        # position_ids = self.concat_features(position_ids, position_ids_2)
        # head_mask = self.concat_features(head_mask, head_mask_2)
        # inputs_embeds = self.concat_features(inputs_embeds, inputs_embeds_2)
        
        inputs = self.backbone(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_hidden_states=self.output_hidden_states,
            )
        features_1 = self.pooler(attention_mask, inputs)

        outputs = self.law_encoder(
            input_ids=input_ids_2,
            attention_mask=attention_mask_2,
            token_type_ids=token_type_ids_2,
            position_ids=position_ids_2,
            head_mask=head_mask_2,
            inputs_embeds=inputs_embeds_2,
            output_hidden_states=self.output_hidden_states,
            )
        features_2 = self.pooler(attention_mask, outputs)
        # features_1, features_2 = torch.split(features, bsz, dim=0)  # [sentence1, condtion], [sentence2, condition]
        if self.transform is not None:
            features_1 = self.transform(features_1)

        loss = None
        rank = None

        logits = self.distance(features_1.unsqueeze(1), features_2.unsqueeze(0))

        if labels is not None:
            labels = torch.arange(logits.size(0)).long().to(logits.device)

            loss = self.loss_fct(logits, labels)

            if not self.training:
                logits_mask = unique(input_ids, input_ids_2, logits)
                logits = logits + logits_mask
                sorted_indices = torch.argsort(logits, dim=-1, descending=True)
                rank = torch.gather(torch.eye(bsz, device=logits.device), 1, sorted_indices)

        return RankingEncoderOutput(
            loss=loss,
            logits=logits,
            rank=rank,
        )
    
    def concat_features(self, feature_1=None, feature_2=None):
        if feature_1 is None or feature_2 is None:
            return None
        return torch.cat([feature_1, feature_2], dim=0)
    
    def inference(
        self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,

        input_ids_2=None,
        attention_mask_2=None,
        token_type_ids_2=None,
        position_ids_2=None,
        head_mask_2=None,
        inputs_embeds_2=None,
        **kwargs,
        ):
        self.eval()
        inputs = self.backbone(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_hidden_states=self.output_hidden_states,
            )
        features_1 = self.pooler(attention_mask, inputs)

        outputs = self.law_encoder(
            input_ids=input_ids_2,
            attention_mask=attention_mask_2,
            token_type_ids=token_type_ids_2,
            position_ids=position_ids_2,
            head_mask=head_mask_2,
            inputs_embeds=inputs_embeds_2,
            output_hidden_states=self.output_hidden_states,
            )
        features_2 = self.pooler(attention_mask, outputs)
        if self.transform is not None:
            features_1 = self.transform(features_1)

        logits = self.distance(features_1.unsqueeze(1), features_2.unsqueeze(0))

        return InferenceOutputForCosineSimilarity(
            logits=logits,
        )

  
class LawsEncoderUsingEmbedding(PreTrainedModel):
    '''Encoder model with backbone and classification head.'''
    def __init__(self, config):
        super().__init__(config)
        self.backbone = AutoModel.from_pretrained(
            config.model_name_or_path,
            from_tf=bool('.ckpt' in config.model_name_or_path),
            config=config,
            cache_dir=config.cache_dir,
            revision=config.model_revision,
            use_auth_token=True if config.use_auth_token else None,
            add_pooling_layer=False,
            torch_dtype=config.torch_dtype,
            low_cpu_mem_usage=config.low_cpu_mem_usage,
        ).base_model

        self.law_ids = torch.arange(config.num_article)
        self.law_embeddings = nn.Embedding(config.num_article, config.hidden_size)
        self.topk = True

        classifier_dropout = (
                config.classifier_dropout if config.classifier_dropout is not None else config.hidden_dropout_prob
            )
        if config.transform:
            self.transform = nn.Sequential(
                nn.Dropout(classifier_dropout),
                nn.Linear(config.hidden_size, config.hidden_size),
                ACT2FN[config.hidden_act],
                )
        else:
            self.transform = None

        self.pooler = Pooler(config.pooler_type)
        if config.pooler_type in {'avg_first_last', 'avg_top2'}:
            self.output_hidden_states = True
        else:
            self.output_hidden_states = False
        if config.measure == "cosine":
            self.distance = partial(cosine_similarity, dim=-1)
        elif config.measure == "dot_product":
            assert config.objective in ["classification", "focal"]
            nn.init.xavier_uniform_(self.law_embeddings.weight)
            self.distance = DotProductWithBias(config.num_article)
        elif config.measure == "sigmoid":
            self.distance = Sigmoid_(config.hidden_size) # TODO: have some bugs
        else:
            raise NotImplementedError
        if config.objective in ["contrast", "classification"]:
            self.loss_fct = cross_entropy
        elif config.objective == "rank":
            self.loss_fct = RankingLoss()
        elif config.objective == "absolute":
            self.loss_fct == AbsoluteLoss()
        elif config.objective == 'focal':
            self.loss_fct = FocalLoss(alpha=1.0)
        else:
            raise ValueError('Only regression and triplet objectives are supported for BiEncoderForClassification')
        
        self.post_init()

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,

        article_ids=None,
        labels=None,
        weights=None,
        **kwargs,
        ):
        bsz = input_ids.shape[0]
        device = input_ids.device

        outputs = self.backbone(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_hidden_states=self.output_hidden_states,
            )
        features_1 = self.pooler(attention_mask, outputs)

        if self.transform is not None:
            features_1 = self.transform(features_1)

        loss = None
        rank = None

        if self.training:
            if self.config.objective in ["contrast", "rank", "absolute"]:
                features_2 = self.law_embeddings(article_ids)
                labels = torch.arange(bsz).long().to(device)

            elif self.config.objective in ["classification", "focal"]:
                features_2 = self.law_embeddings(self.law_ids.to(device))
                labels = article_ids.long()

            logits = self.distance(features_1.unsqueeze(1), features_2.unsqueeze(0))
            loss = self.loss_fct(logits, labels, weights)

        else:
            features_2 = self.law_embeddings(self.law_ids.to(device))
            
            logits = self.distance(features_1.unsqueeze(1), features_2.unsqueeze(0))

            if labels is not None:
                labels = article_ids.long()
                loss = self.loss_fct(logits, labels)

            if self.topk:
                _, sorted_indices = logits.topk(bsz, dim=-1)
            else:
                sorted_indices = torch.argsort(logits, dim=-1, descending=True)
            matrix = torch.zeros((bsz, len(self.law_ids)), device=device).long()
            matrix[torch.arange(bsz), labels] = 1
            rank = torch.gather(matrix, 1, sorted_indices)
            logits = torch.gather(logits, 1, sorted_indices)

        return RankingEncoderOutput(
            loss=loss,
            logits=logits,
            rank=rank,
        )
    
    def inference(
        self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        sort=True,
        topk=None,
        **kwargs,
        ):
        self.eval()
        bsz = input_ids.shape[0]

        outputs = self.backbone(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_hidden_states=self.output_hidden_states,
            )
        features_1 = self.pooler(attention_mask, outputs)

        if self.transform is not None:
            features_1 = self.transform(features_1)

        law_ids = self.law_ids.to(input_ids.device)
        features_2 = self.law_embeddings(law_ids)

        logits = self.distance(features_1.unsqueeze(1), features_2.unsqueeze(0))

        if sort:
            if topk is None:
                sorted_indices = torch.argsort(logits, dim=-1, descending=True)
            else:
                _, sorted_indices = logits.topk(topk, dim=-1)
            matrix = self.law_ids.repeat((bsz, 1))
            rank = torch.gather(matrix, 1, sorted_indices)
            logits = torch.gather(logits, 1, sorted_indices)
        else:
            rank = None

        return InferenceOutputForClassification(
            logits=logits,
            rank=rank,
        )
    
  
class LawsEncoderUsingBiEncoder(PreTrainedModel):
    '''Encoder model with backbone and classification head.'''
    def __init__(self, config):
        super().__init__(config)
        self.backbone = AutoModel.from_pretrained(
            config.model_name_or_path,
            from_tf=bool('.ckpt' in config.model_name_or_path),
            config=config,
            cache_dir=config.cache_dir,
            revision=config.model_revision,
            use_auth_token=True if config.use_auth_token else None,
            add_pooling_layer=False,
            torch_dtype=config.torch_dtype,
            low_cpu_mem_usage=config.low_cpu_mem_usage,
        ).base_model

        self.law_encoder = self.backbone

        self.law_ids = torch.arange(config.num_article)
        self.law_embeddings = nn.Embedding(config.num_article, config.hidden_size)
        self.topk = True

        classifier_dropout = (
                config.classifier_dropout if config.classifier_dropout is not None else config.hidden_dropout_prob
            )
        if config.transform:
            self.transform = nn.Sequential(
                nn.Dropout(classifier_dropout),
                nn.Linear(config.hidden_size, config.hidden_size),
                ACT2FN[config.hidden_act],
                )
        else:
            self.transform = None

        self.pooler = Pooler(config.pooler_type)
        if config.pooler_type in {'avg_first_last', 'avg_top2'}:
            self.output_hidden_states = True
        else:
            self.output_hidden_states = False
        if config.measure == "cosine":
            self.distance = partial(cosine_similarity, dim=-1)
        elif config.measure == "dot_product":
            assert config.objective in ["classification", "focal"]
            nn.init.xavier_uniform_(self.law_embeddings.weight)
            self.distance = DotProductWithBias(config.num_article)
        elif config.measure == "sigmoid":
            self.distance = Sigmoid_(config.hidden_size) # TODO: have some bugs
        else:
            raise NotImplementedError
        if config.objective in ["contrast", "classification"]:
            self.loss_fct = cross_entropy
        elif config.objective == "rank":
            self.loss_fct = RankingLoss()
        elif config.objective == "absolute":
            self.loss_fct == AbsoluteLoss()
        elif config.objective == 'focal':
            self.loss_fct = FocalLoss(alpha=1.0)
        else:
            raise ValueError('Only regression and triplet objectives are supported for BiEncoderForClassification')
        
        self.post_init()

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,

        input_ids_2=None,
        attention_mask_2=None,
        token_type_ids_2=None,
        position_ids_2=None,
        head_mask_2=None,
        inputs_embeds_2=None,

        article_ids=None,
        labels=None,
        weights=None,
        **kwargs,
        ):
        bsz = input_ids.shape[0]
        device = input_ids.device

        outputs = self.backbone(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_hidden_states=self.output_hidden_states,
            )
        features_1 = self.pooler(attention_mask, outputs)

        outputs_ = self.law_encoder(
            input_ids=input_ids_2,
            attention_mask=attention_mask_2,
            token_type_ids=token_type_ids_2,
            position_ids=position_ids_2,
            head_mask=head_mask_2,
            inputs_embeds=inputs_embeds_2,
            output_hidden_states=self.output_hidden_states,
            )
        features_2_ = self.pooler(attention_mask, outputs_)

        if self.transform is not None:
            features_1 = self.transform(features_1)

        loss = None
        rank = None

        if self.training:
            if self.config.objective in ["contrast", "rank", "absolute"]:
                features_2 = self.law_embeddings(article_ids)
                labels = torch.arange(bsz).long().to(device)
                loss = torch.mean((features_2_ - features_2.detach()) ** 2)

            elif self.config.objective in ["classification", "focal"]:
                features_2 = self.law_embeddings(self.law_ids.to(device))
                labels = article_ids.long()
                loss = torch.mean((features_2_ - features_2[article_ids].detach()) ** 2)
            
            logits = self.distance(features_1.unsqueeze(1), features_2.unsqueeze(0))
            loss += self.loss_fct(logits, labels, weights)

        else:
            features_2 = self.law_embeddings(self.law_ids.to(device))
            logits = self.distance(features_1.unsqueeze(1), features_2.unsqueeze(0))

            if labels is not None:
                labels = article_ids.long()
                loss = self.loss_fct(logits, labels)

            if self.topk:
                _, sorted_indices = logits.topk(bsz, dim=-1)
            else:
                sorted_indices = torch.argsort(logits, dim=-1, descending=True)
            matrix = torch.zeros((bsz, len(self.law_ids)), device=device).long()
            matrix[torch.arange(bsz), labels] = 1
            rank = torch.gather(matrix, 1, sorted_indices)
            logits = torch.gather(logits, 1, sorted_indices)

        return RankingEncoderOutput(
            loss=loss,
            logits=logits,
            rank=rank,
        )
    
    def inference(
        self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,

        input_ids_2=None,
        attention_mask_2=None,
        token_type_ids_2=None,
        position_ids_2=None,
        head_mask_2=None,
        inputs_embeds_2=None,
        article_ids=None,

        sort=True,
        topk=None,
        **kwargs,
        ):
        self.eval()
        bsz = input_ids.shape[0]

        outputs = self.backbone(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_hidden_states=self.output_hidden_states,
            )
        features_1 = self.pooler(attention_mask, outputs)

        if self.transform is not None:
            features_1 = self.transform(features_1)

        if input_ids_2 is None and article_ids is None:
            law_ids = self.law_ids.to(input_ids.device)
            features_2 = self.law_embeddings(law_ids)

            logits = self.distance(features_1.unsqueeze(1), features_2.unsqueeze(0))

            if sort:
                if topk is None:
                    sorted_indices = torch.argsort(logits, dim=-1, descending=True)
                else:
                    _, sorted_indices = logits.topk(topk, dim=-1)
                matrix = self.law_ids.repeat((bsz, 1))
                rank = torch.gather(matrix, 1, sorted_indices)
                logits = torch.gather(logits, 1, sorted_indices)
            else:
                rank = None

            return InferenceOutputForClassification(
                logits=logits,
                rank=rank,
            )

        else:
            if input_ids_2 is not None:
                outputs = self.law_encoder(
                    input_ids=input_ids_2,
                    attention_mask=attention_mask_2,
                    token_type_ids=token_type_ids_2,
                    position_ids=position_ids_2,
                    head_mask=head_mask_2,
                    inputs_embeds=inputs_embeds_2,
                    output_hidden_states=self.output_hidden_states,
                    )
                features_2 = self.pooler(attention_mask, outputs)
            else:
                features_2 = self.law_embeddings(article_ids)
            
            logits = self.distance(features_1.unsqueeze(1), features_2.unsqueeze(0))

            return InferenceOutputForCosineSimilarity(
                logits=logits,
            )

        


def unique(inputs, outputs, logits):
    bsz = len(inputs)
    mask = torch.zeros((bsz, bsz), device=logits.device, dtype=logits.dtype)
    for i in range(bsz):
        for j in range(i + 1, bsz):
            if torch.equal(inputs[i], inputs[j]) or torch.equal(outputs[i], outputs[j]):
                mask[i][j] = -float('inf')
                mask[j][i] = -float('inf')
    return mask




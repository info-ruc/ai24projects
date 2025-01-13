from .modeling_encoders import LawsEncoder, LawsEncoderUsingEmbedding, LawsEncoderUsingBiEncoder
import torch
import numpy as np
from dataclasses import dataclass
from typing import List, Dict, Any, Optional


@dataclass
class DataCollatorWithPadding:
    pad_token_id: int
    pad_token_type_id: int = 0
    pad_to_multiple_of: Optional[int] = None
    return_tensors: str = 'pt'
    
    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, Any]:
        # get max length of all sequences in features
        max_length = max(max(len(feature[key]) for feature in features) for key in features[0] if key.startswith('input_ids'))
        if self.pad_to_multiple_of is not None:
            max_length = ((max_length + self.pad_to_multiple_of - 1) // self.pad_to_multiple_of) * self.pad_to_multiple_of
        # pad all sequences to max length
        out_features = {}
        for key in features[0].keys():
            if key.startswith('input_ids') or key.startswith('attention_mask') or key.startswith('token_type_ids') or key.startswith("key_ids"):
                if key.startswith('input_ids'):
                    pad_token = self.pad_token_id
                elif key.startswith('attention_mask') or key.startswith('key_ids'):
                    pad_token = 0
                else:
                    pad_token = self.pad_token_type_id
                out_features[key] = [feature[key] + [pad_token] * (max_length - len(feature[key])) for feature in features]
            else:
                out_features[key] = [feature[key] for feature in features]
        if self.return_tensors == 'pt':
            out_features = {key: torch.tensor(value) for key, value in out_features.items()}
        elif self.return_tensors == 'np':
            out_features = {key: np.array(value) for key, value in out_features.items()}
        return out_features


from .article_tokenzier import ArticleTokenizer
from transformers import AutoTokenizer
from tqdm import tqdm
def law_encoder_using_embedding_init(base_model: LawsEncoder, tokenizer, article_tokenzier) -> LawsEncoderUsingEmbedding:   
    config = base_model.config
    # article_tokenzier = ArticleTokenizer(file, articles)
    config.num_article = len(article_tokenzier)
    model = LawsEncoderUsingEmbedding(config)
    # tokenizer = AutoTokenizer.from_pretrained(
    #     config.model_name_or_path,
    #     use_fast=True,
    #     revision="main",
    #     use_auth_token=None,
    #     )
    model.load_state_dict(base_model.state_dict(), strict=False)
    device_ = model.device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    base_model.to(device)
    model.to(device)

    with torch.no_grad():
        features_tensor = model.law_embeddings.weight.data.to(device)
        for idx, source in tqdm(enumerate(article_tokenzier.idx_to_source)):
            if source is None:
                continue
            inputs = tokenizer([source], return_tensors="pt", padding=True, max_length=512, truncation=True).to(device)
            outputs = base_model.law_encoder(**inputs)
            features = base_model.pooler(inputs["attention_mask"], outputs)
            features_tensor[idx] = features[0]
        model.law_embeddings.weight.data.copy_(features_tensor)
    return model.to(device_)


def get_model_cls(model_type):
    if model_type == "v0":
        return LawsEncoder
    elif model_type == "v1":
        return LawsEncoderUsingEmbedding
    elif model_type == "v2":
        return LawsEncoderUsingBiEncoder
    else:
        raise NotImplementedError



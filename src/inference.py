from transformers import AutoTokenizer, AutoConfig
from utils.article_tokenzier import ArticleTokenizer
from utils.modeling_encoders import LawsEncoder, LawsEncoderUsingEmbedding
from utils.modeling_utils import law_encoder_using_embedding_init, get_model_cls
import torch

model_name = "hfl/chinese-roberta-wwm-ext"
#model_name = "/mnt/d/AI/model/chinese/chinese-roberta-wwm-ext"
model_type= "v1"
measure = "dot_product"
objective =  'classification'  #"focal" #"contrast" # rank #  #另外两个是为v2准备的，但是现阶段没有完全调试好
checkpoint = "checkpoint-5280/" # 
article_vocab_file="article_source.json"

#model_path = "output/" + model_name.replace("/", "__") + f"/encoder_{model_type}__trans_True__obj_{objective}__bsz_8__lr_1e-5__wd_0.1__s_42__weight_False/{checkpoint}pytorch_model.bin"
model_path = "output/" + model_name.replace("/", "__") + f"/encoder_{model_type}__meas_{measure}__obj_{objective}__bsz_8__lr_1e-5__wd_0.1__s_42__weight_False/{checkpoint}pytorch_model.bin"
def load_tokenizer_and_model():
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    article_tokenizer = ArticleTokenizer(file=article_vocab_file)
    config = AutoConfig.from_pretrained(model_name, num_labels=1,)

    config.update(
            {
                "use_auth_token": False,
                "model_revision": "main",
                "cache_dir": None,
                "model_name_or_path": model_name,
                "objective": objective,
                "measure": measure,
                "pooler_type": "cls",
                "transform": True,
                "torch_dtype": None,
                "low_cpu_mem_usage": False,
                "num_article": len(article_tokenizer),
            }
        )

    model_cls = get_model_cls(model_type)
    model = model_cls(config=config)

    model.load_state_dict(torch.load(model_path))
    return tokenizer, model, article_tokenizer


def inference_v0(model, tokenizer, inputs, articles):
    inputs_ = tokenizer(inputs, return_tensors="pt", max_length=512, truncation=True, padding=True)
    inputs_2 = tokenizer(articles, return_tensors="pt", max_length=512, truncation=True, padding=True)

    inputs_["input_ids_2"] = inputs_2["input_ids"]
    inputs_["token_type_ids_2"] = inputs_2["token_type_ids"]
    inputs_["attention_mask_2"] = inputs_2["attention_mask"]

    # 使用模型进行推理
    outputs = model.inference(**inputs_)
    print(inputs)
    print(articles)
    print(f"案例法条的置信矩阵为{outputs.logits}")

def inference_v1(model, tokenizer, inputs, article_tokenizer):
    inputs_ = tokenizer(inputs, return_tensors="pt", max_length=512, truncation=True, padding=True)
    inputs_["topk"] = 8
    # 使用模型进行推理
    outputs = model.inference(**inputs_)
    print(inputs[0])
    for i in range(8):
        print(article_tokenizer.convert_idx_to_article(outputs.rank[0][i:i+1])[0], ": ", float(outputs.logits[0][i]))
    print("------------------------------------------------")
    for i in range(3):
        print("有", float(outputs.logits[0][i]), "置信处以以下法条")
        print(article_tokenizer.convert_idx_to_source(outputs.rank[0][i:i+1]))

if __name__ == "__main__":
    tokenizer, model, article_tokenizer = load_tokenizer_and_model()
    inputs = ["基于下列案件，推测可能的判决结果。\n经审理查明，2015年6月21日15时许，被告人白某某在大东区小河沿公交车站乘坐被害人张某某驾驶的133路公交车，当车辆行驶至沈阳市大东区东陵西路26号附近时，被告人白某某因未能下车而与司机张某某发生争执，并在该公交车行驶中用手拉拽档杆，被证人韩某某拉开后，被告人白某某又用手拉拽司机张某某的右胳膊，导致该车失控撞向右侧马路边停放的轿车和一个路灯杆，路灯杆折断后将福锅记炖品店的牌匾砸坏。后经被害人张某某报警，公安人员赶至现场将被告人白某某传唤到案。经鉴定，公交车受损价值人民币5,189.9元，轿车受损价值人民币1,449.57元，路灯杆受损价值人民币2,927.15元，福锅记饭店牌匾受损价值人民币9,776元，本案损失价值共计人民币19,342.6元。上述事实，被告人白某某在庭审中亦无异议，被害人张某某、朱某某、詹某某陈述，证人韩某某的证言，现场勘察笔录，视听资料，鉴定结论书，被告人白某某的供述与辩解等证据证实，足以认定。"]
    articles = [
        "《民法典》第一千零七十九条: 夫妻一方要求离婚的，可以由有关组织进行调解或者直接向人民法院提起离婚诉讼。人民法院审理离婚案件，应当进行调解；如果感情确已破裂，调解无效的，应当准予离婚。有下列情形之一，调解无效的，应当准予离婚：（一）重婚或者与他人同居；（二）实施家庭暴力或者虐待、遗弃家庭成员；（三）有赌博、吸毒等恶习屡教不改；（四）因感情不和分居满二年；（五）其他导致夫妻感情破裂的情形。一方被宣告失踪，另一方提起离婚诉讼的，应当准予离婚。经人民法院判决不准离婚后，双方又分居满一年，一方再次提起离婚诉讼的，应当准予离婚。",
        "《矿产资源法》第二十条: 非经国务院授权的有关主管部门同意，不得在下列地区开采矿产资源：（一）港口、机场、国防工程设施圈定地区以内；（二）重要工业区、大型水利工程设施、城镇市政工程设施附近一定距离以内；（三）铁路、重要公路两侧一定距离以内；（四）重要河流、堤坝两侧一定距离以内；（五）国家划定的自然保护区、重要风景区，国家重点保护的不能移动的历史文物和名胜古迹所在地；（六）国家规定不得开采矿产资源的其他地区。",
        '《刑法》第一百一十四条: 放火、决水、爆炸以及投放毒害性、放射性、传染病病原体等物质或者以其他危险方法危害公共安全，尚未造成严重后果的，处三年以上十年以下有期徒刑。'
    ]
    #inference_v0(model, tokenizer, inputs, articles)
    
    #model = law_encoder_using_embedding_init(model, tokenizer, article_tokenizer)
    inference_v1(model, tokenizer, inputs, article_tokenizer)
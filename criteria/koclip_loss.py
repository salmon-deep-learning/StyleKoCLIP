
import torch
import clip
import torch 
from collections import OrderedDict
import math
import torch
import clip
from transformers import AutoModel, AutoTokenizer, RobertaForSequenceClassification, pipeline

class TextEncoder(torch.nn.Module):
    def __init__(self):
        super(TextEncoder, self).__init__()
        self.model = RobertaForSequenceClassification.from_pretrained("klue/roberta-small", num_labels=512)
        self.model = self.model.eval()

        self.projection = torch.nn.Linear(512, 512)
        self.projection = self.projection.train()

    def forward(self, input_ids, attention_mask):
        
        x = self.model(input_ids, attention_mask=attention_mask)[0]
        x = self.projection(x)
        return x

def copyStateDict(state_dict):
    if list(state_dict.keys())[0].startswith("module"):
        start_idx = 1
    else:
        start_idx = 0
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        name = ".".join(k.split(".")[start_idx:])
        new_state_dict[name] = v
    return new_state_dict


class KoCLIPLoss(torch.nn.Module):

    def __init__(self, opts):
        super(KoCLIPLoss, self).__init__()
        self.model, self.preprocess = clip.load("ViT-B/32", device="cuda")
        self.upsample = torch.nn.Upsample(scale_factor=7)
        self.avg_pool = torch.nn.AvgPool2d(kernel_size=opts.stylegan_size // 32)
        self.text_encoder = TextEncoder()
        self.tokenizer = AutoTokenizer.from_pretrained("klue/roberta-small", use_fast=True)

    def forward(self, image, text):
        # (256 x 7) x (256 x 7) 
        # (32 x 7) x (32 x 7)
        # 224 x 224
        image = self.avg_pool(self.upsample(image))
        # 1 x 512
        image_embedding = self.model.encode_image(image)
        text_tensor = self.tokenizer(
                text,
                return_tensors='pt',
                truncation=True,
                max_length=self.tokenizer.model_max_length,
                padding="max_length",
                add_special_tokens=True,
                return_token_type_ids=False
            )
        input_ids = text_tensor['input_ids'][0]
        attention_mask = text_tensor['attention_mask'][0]
        # 1 x 512
        text_embedding = self.text_encoder(input_ids, attention_mask)

        image_embedding = image_embedding / image_embedding.norm(dim=-1, keepdim=True)    
        text_embedding = text_embedding / text_embedding.norm(dim=-1, keepdim=True)

        # scalar
        similarity = (image_embedding @ text_embedding.T)[0]
        # cosine distance
        loss = 1 - similarity
        return loss
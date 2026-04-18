import torch
import torch.nn as nn
from transformers.modeling_utils import PreTrainedModel
from diffusers.configuration_utils import ConfigMixin, register_to_config
from diffusers.models.modeling_utils import ModelMixin

class Image2Token(ModelMixin, ConfigMixin):
    @register_to_config
    def __init__(self, visual_hidden_size=1280, text_hidden_size=768, max_length=77, num_layers=3):
        super(Image2Token, self).__init__()
        
        self.visual_proj = nn.Linear(visual_hidden_size, text_hidden_size)
        self.text_hidden_size = text_hidden_size
        
        if num_layers>0:
            self.query = nn.Parameter(torch.randn((1, max_length, text_hidden_size)))
            decoder_layer = nn.TransformerDecoderLayer(d_model=text_hidden_size, nhead=text_hidden_size//64, batch_first=True)
            self.i2t = nn.TransformerDecoder(decoder_layer, num_layers=num_layers)
        else:
            self.i2t = None

    def forward(self, x):
        b=x.shape[0]
        out = self.visual_proj(x).view(b,-1,self.text_hidden_size)
        if self.i2t is not None:
            out = self.i2t(self.query.repeat(b,1,1), out)

        return out
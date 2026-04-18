import torch
import torch.nn as nn
from transformers.modeling_utils import PreTrainedModel
from diffusers.configuration_utils import ConfigMixin, register_to_config
from diffusers.models.modeling_utils import ModelMixin


class IDEncoder(nn.Module):
    def __init__(self, width=1280, context_dim=2048, num_token=5):
        super().__init__()
        self.num_token = num_token
        self.context_dim = context_dim
        h1 = min((context_dim * num_token) // 4, 1024)
        h2 = min((context_dim * num_token) // 2, 1024)
        self.body = nn.Sequential(
            nn.Linear(width, h1),
            nn.LayerNorm(h1),
            nn.LeakyReLU(),
            nn.Linear(h1, h2),
            nn.LayerNorm(h2),
            nn.LeakyReLU(),
            nn.Linear(h2, context_dim * num_token),
        )

        for i in range(5):
            setattr(
                self,
                f'mapping_{i}',
                nn.Sequential(
                    nn.Linear(1024, 1024),
                    nn.LayerNorm(1024),
                    nn.LeakyReLU(),
                    nn.Linear(1024, 1024),
                    nn.LayerNorm(1024),
                    nn.LeakyReLU(),
                    nn.Linear(1024, context_dim),
                ),
            )

            setattr(
                self,
                f'mapping_patch_{i}',
                nn.Sequential(
                    nn.Linear(1024, 1024),
                    nn.LayerNorm(1024),
                    nn.LeakyReLU(),
                    nn.Linear(1024, 1024),
                    nn.LayerNorm(1024),
                    nn.LeakyReLU(),
                    nn.Linear(1024, context_dim),
                ),
            )

    def forward(self, x, y):
        # x shape [N, C]
        x = self.body(x)
        x = x.reshape(-1, self.num_token, self.context_dim)

        hidden_states = ()
        for i, emb in enumerate(y):
            hidden_state = getattr(self, f'mapping_{i}')(emb[:, :1]) + getattr(self, f'mapping_patch_{i}')(
                emb[:, 1:]
            ).mean(dim=1, keepdim=True)
            hidden_states += (hidden_state,)
        hidden_states = torch.cat(hidden_states, dim=1)

        return torch.cat([x, hidden_states], dim=1)

class ID2Token(ModelMixin, ConfigMixin):
    @register_to_config
    def __init__(self, id_dim=512, text_hidden_size=768, max_length=77, num_layers=3):
        super(ID2Token, self).__init__()
        
        self.id_proj = nn.Linear(id_dim, text_hidden_size)
        self.text_hidden_size = text_hidden_size
        
        if num_layers>0:
            self.query = nn.Parameter(torch.randn((1, max_length, text_hidden_size)))
            decoder_layer = nn.TransformerDecoderLayer(d_model=text_hidden_size, nhead=text_hidden_size//64, batch_first=True)
            self.id2t = nn.TransformerDecoder(decoder_layer, num_layers=num_layers)
        else:
            self.id2t = None

    def forward(self, x):
        b=x.shape[0]
        out = self.id_proj(x).view(b,-1,self.text_hidden_size)
        if self.id2t is not None:
            out = self.id2t(self.query.repeat(b,1,1), out)

        return out

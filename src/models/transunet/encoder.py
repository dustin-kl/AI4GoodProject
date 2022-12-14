import math
import copy
import torch
import torch.nn as nn
from .resnet_v2 import ResNetV2
from .cnn import CNN
from torch.nn.modules.utils import _pair

def np2th(weights, conv=False):
    """Possibly convert HWIO to OIHW."""
    if conv:
        weights = weights.transpose([3, 2, 0, 1])
    return torch.from_numpy(weights)


def swish(x):
    return x * torch.sigmoid(x)

ACT2FN = {"gelu": torch.nn.functional.gelu, "relu": torch.nn.functional.relu, "swish": swish}

class Attention(nn.Module):
    def __init__(self, params):
        super(Attention, self).__init__()
        self.num_attention_heads = params["transformer"]["num_heads"]
        self.attention_head_size = int(params["hidden_size"] / self.num_attention_heads)
        self.all_head_size = self.num_attention_heads * self.attention_head_size

        self.query = nn.Linear(params["hidden_size"], self.all_head_size)
        self.key = nn.Linear(params["hidden_size"], self.all_head_size)
        self.value = nn.Linear(params["hidden_size"], self.all_head_size)

        self.out = nn.Linear(params["hidden_size"], params["hidden_size"])
        self.attn_dropout = nn.Dropout(params["transformer"]["attention_dropout_rate"])
        self.proj_dropout = nn.Dropout(params["transformer"]["attention_dropout_rate"])

        self.softmax = nn.Softmax(dim=-1)

    def transpose_for_scores(self, x):
        new_x_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)
        x = x.view(*new_x_shape)
        return x.permute(0, 2, 1, 3)

    def forward(self, hidden_states):
        mixed_query_layer = self.query(hidden_states)
        mixed_key_layer = self.key(hidden_states)
        mixed_value_layer = self.value(hidden_states)

        query_layer = self.transpose_for_scores(mixed_query_layer)
        key_layer = self.transpose_for_scores(mixed_key_layer)
        value_layer = self.transpose_for_scores(mixed_value_layer)

        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
        attention_scores = attention_scores / math.sqrt(self.attention_head_size)
        attention_probs = self.softmax(attention_scores)
        #weights = attention_probs if self.vis else None
        weights = None
        attention_probs = self.attn_dropout(attention_probs)

        context_layer = torch.matmul(attention_probs, value_layer)
        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
        context_layer = context_layer.view(*new_context_layer_shape)
        attention_output = self.out(context_layer)
        attention_output = self.proj_dropout(attention_output)
        return attention_output, weights


class Mlp(nn.Module):
    def __init__(self, params):
        super(Mlp, self).__init__()
        self.fc1 = nn.Linear(params["hidden_size"], params["transformer"]["mlp_dim"])
        self.fc2 = nn.Linear(params["transformer"]["mlp_dim"], params["hidden_size"])
        self.act_fn = ACT2FN["gelu"]
        self.dropout = nn.Dropout(params["transformer"]["dropout_rate"])

        self._init_weights()

    def _init_weights(self):
        nn.init.xavier_uniform_(self.fc1.weight)
        nn.init.xavier_uniform_(self.fc2.weight)
        nn.init.normal_(self.fc1.bias, std=1e-6)
        nn.init.normal_(self.fc2.bias, std=1e-6)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act_fn(x)
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.dropout(x)
        return x


class Embeddings(nn.Module):
    """Construct the embeddings from patch, position embeddings.
    """
    def __init__(self, params, img_size, in_channels=3):
        super(Embeddings, self).__init__()
        self.hybrid = None
        self.params = params
        #img_size = _pair(img_size)
        img_size = (768,1152)
        if params["patches"]["grid"] is not None:   # ResNet
            grid_size = params["patches"]["grid"]
            patch_size = (img_size[0] // 16 // grid_size[0], img_size[1] // 16 // grid_size[1])
            patch_size_real = (patch_size[0] * 16, patch_size[1] * 16)
            #patch_size_real = patch_size
            n_patches = (img_size[0] // patch_size_real[0]) * (img_size[1] // patch_size_real[1])  
            #patch_size = (grid_size[0], grid_size[1])
            #n_patches = (img_size[0] // patch_size[0]) * (img_size[1] // patch_size[1])
            self.hybrid = True
        else:
            patch_size = _pair(params["patches"]["size"])
            n_patches = (img_size[0] // patch_size[0]) * (img_size[1] // patch_size[1])
            self.hybrid = False
        #print("patch_size", patch_size)
        #print("NPATCH", n_patches)
        if self.hybrid:
            #self.hybrid_model = ResNetV2(block_units=params["resnet"]["num_layers"], width_factor=params["resnet"]["width_factor"])
            self.hybrid_model = CNN(in_channels=4)
            in_channels = self.hybrid_model.width * 16
        #self.patch_embeddings = nn.Conv2d(in_channels=in_channels,
        #                               out_channels=params["hidden_size"],
        #                               kernel_size=patch_size,
        #                               stride=patch_size)
        self.patch_embeddings = nn.Conv2d(in_channels=in_channels,
                                       out_channels=params["hidden_size"],
                                       kernel_size=1,
                                       stride=1)
        #self.position_embeddings = nn.Parameter(torch.zeros(1, n_patches, params["hidden_size"]))
        self.position_embeddings= nn.Parameter(torch.zeros(1, 3456, params["hidden_size"]))

        self.dropout = nn.Dropout(params["transformer"]["dropout_rate"])

    def forward(self, x):
        if self.hybrid:
            x, features = self.hybrid_model(x)
        else:
            features = None
        #print("HYBRID", x.shape)
        x = self.patch_embeddings(x)  # (B, hidden. n_patches^(1/2), n_patches^(1/2))
        #print("PATCH EMBEDDINGS", x.shape)
        x = x.flatten(2)
        x = x.transpose(-1, -2)  # (B, n_patches, hidden)

        embeddings = x + self.position_embeddings
        embeddings = self.dropout(embeddings)
        return embeddings, features

class TransformerBlock(nn.Module):
    def __init__(self, params):
        super(TransformerBlock, self).__init__()
        self.hidden_size = params["hidden_size"]
        self.attention_norm = nn.LayerNorm(params["hidden_size"], eps=1e-6)
        self.ffn_norm = nn.LayerNorm(params["hidden_size"], eps=1e-6)
        self.ffn = Mlp(params)
        self.attn = Attention(params)

    def forward(self, x):
        h = x
        x = self.attention_norm(x)
        x, weights = self.attn(x)
        x = x + h

        h = x
        x = self.ffn_norm(x)
        x = self.ffn(x)
        x = x + h
        return x, weights

class Transformer(nn.Module):
    def __init__(self, params):
        super(Transformer, self).__init__()
        self.layer = nn.ModuleList()
        self.encoder_norm = nn.LayerNorm(params["hidden_size"], eps=1e-6)
        for _ in range(params["transformer"]["num_layers"]):
            layer = TransformerBlock(params)
            self.layer.append(copy.deepcopy(layer))

    def forward(self, hidden_states):
        attn_weights = []
        for layer_block in self.layer:
            hidden_states, weights = layer_block(hidden_states)
        encoded = self.encoder_norm(hidden_states)
        return encoded, attn_weights

class Encoder(nn.Module):
    def __init__(self, params, img_size, in_channels=3):
        super(Encoder, self).__init__()
        self.embeddings = Embeddings(params, img_size=img_size, in_channels=in_channels)
        self.transformer = Transformer(params)

    def forward(self, input_ids):
        #print("ENC IN", input_ids.shape)
        embedding_output, features = self.embeddings(input_ids)
        #print("EMB OUT", embedding_output.shape)
        encoded, attn_weights = self.transformer(embedding_output)  # (B, n_patch, hidden)
        #encoded, attn_weights = embedding_output, None
        #print("ENC OUT", encoded.shape)
        return encoded, attn_weights, features

import torch
import torch.nn as nn
import torch.nn.functional as F
from .layers import *

def build_net(model_name):
    class ModelError(Exception):
        def __init__(self, msg):
            self.msg = msg
        def __str__(self):
            return self.msg
    return DeHambaNet()

class DeHambaNet(nn.Module):
    def __init__(self,
                 img_size=64,
                 patch_size=1,
                 in_chans=3,
                 embed_dim=48,
                 depths=(2, 2, 2, 2),
                 d_state = 16,
                 mlp_ratio=2.,
                 use_checkpoint=True,
                 upscale=2,
                 img_range=1.):
        super(DeHambaNet, self).__init__()
        factory_kwargs = {"device": 'cuda', "dtype": None}
        num_in_ch = in_chans
        num_out_ch = in_chans
        num_feat = 64
        self.img_range = img_range
        if in_chans == 3:
            rgb_mean = (0.4488, 0.4371, 0.4040)
            self.mean = torch.Tensor(rgb_mean).view(1, 3, 1, 1)
        else:
            self.mean = torch.zeros(1, 1, 1, 1, **factory_kwargs)
        self.upscale = upscale
        self.mlp_ratio=mlp_ratio

        self.conv_first = nn.Conv2d(num_in_ch, embed_dim, 3, 1, 1, **factory_kwargs) # 3 -> 96
        self.conv_last = nn.Conv2d(embed_dim, num_out_ch, 3, 1, 1, device='cuda') # 96 -> 3

        # ------------------------- 2, deep feature extraction ------------------------- #
        self.num_layers = 1
        self.embed_dim = embed_dim
        self.num_features = embed_dim

        # transfer 2D feature map into 1D token sequence, pay attention to whether using normalization
        self.patch_embed = PatchEmbed(
            img_size=img_size,
            patch_size=patch_size,
            in_chans=embed_dim,
            embed_dim=embed_dim)
        
        self.patches_resolution = self.patch_embed.patches_resolution

        # return 2D feature map from 1D token sequence
        self.patch_unembed = PatchUnEmbed(
            img_size=img_size,
            patch_size=patch_size,
            in_chans=embed_dim,
            embed_dim=embed_dim)

        # build Residual State Space Group (RSSG)
        self.layers = nn.ModuleList()
        for i_layer in range(self.num_layers): 
            layer = ResidualGroup(
                embed_dim=embed_dim,
                input_resolution=(self.patches_resolution),
                depth=depths[i_layer], #6
                d_state = d_state,
                mlp_ratio=self.mlp_ratio,
                use_checkpoint=use_checkpoint,
                img_size=img_size,
                patch_size=patch_size,
                in_chans=in_chans
            )
            self.layers.append(layer)

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    @torch.jit.ignore
    def no_weight_decay(self):
        return {'absolute_pos_embed'}

    @torch.jit.ignore
    def no_weight_decay_keywords(self):
        return {'relative_position_bias_table'}

    def forward_features(self, x):
        x_size = (x.shape[2], x.shape[3])
        x.to('cuda')
        x = self.patch_embed(x) # N,L,C  [4, 96, 64, 64] -> [4, 4096, 96]

        for layer in self.layers:
            x = layer(x, x_size)

        x = self.patch_unembed(x, x_size) # [4, 4096, 96] -> [4, 96, 64, 64]

        return x

    def forward(self, x):
        self.mean = self.mean.type_as(x)
        x = (x - self.mean) * self.img_range
        x.to("cuda")

        x_first = self.conv_first(x)
        res = self.forward_features(x_first) + x_first
        x = x + self.conv_last(res)

        x = x / self.img_range + self.mean

        return x

    def flops(self):
        flops = 0
        h, w = self.patches_resolution
        flops += h * w * 3 * self.embed_dim * 9
        flops += self.patch_embed.flops()
        for layer in self.layers:
            flops += layer.flops()
        flops += h * w * 3 * self.embed_dim * self.embed_dim
        return flops

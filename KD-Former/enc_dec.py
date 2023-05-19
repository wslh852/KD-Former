import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np
import logging
from functools import partial
from collections import OrderedDict
from einops import rearrange, repeat

from timm.data import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
from timm.models.helpers import load_pretrained
from timm.models.layers import DropPath, to_2tuple, trunc_normal_
from timm.models.registry import register_model
class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)
        self.layer_norm = nn.LayerNorm(out_features, eps=1e-6)
    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
       # x=self.layer_norm(x)
        return x

def get_pad_mask(seq, pad_idx):
    return (seq != pad_idx).unsqueeze(-2)


def get_subsequent_mask(seq):
    ''' For masking out the subsequent info. '''
    sz_b, len_s = seq.size()
    subsequent_mask = (1 - torch.triu(
        torch.ones((1, len_s, len_s), device=seq.device), diagonal=1)).bool()
    return subsequent_mask

class Attention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        # NOTE scale factor was wrong in my original version, can set manually to be compat with prev weights
        self.scale = qk_scale or head_dim ** -0.5

        self.q = nn.Linear(dim, dim, bias=qkv_bias)
        self.k=nn.Linear(dim, dim, bias=qkv_bias)
        self.v=nn.Linear(dim, dim, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)
        self.layer_norm = nn.LayerNorm(dim, eps=1e-6)
        
    def forward(self, x,k=None,v=None,mask=None):
        B, N, C = x.shape
        q = self.q(x).reshape(B, x.shape[1], self.num_heads, C // self.num_heads).permute( 0, 2, 1, 3)
        k = self.k(k).reshape(B, k.shape[1], self.num_heads, C // self.num_heads).permute( 0, 2, 1, 3)
        v = self.v(v).reshape(B, v.shape[1], self.num_heads, C // self.num_heads).permute( 0, 2, 1, 3)
        attn = (q @ k.transpose(-2, -1)) * self.scale
        if mask is not None:
            mask = mask.unsqueeze(1)
            attn = attn.masked_fill(mask == 0, -1e9)
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)
        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        #x =self.layer_norm(x)
        return x

class Block(nn.Module):

    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm,mask=None):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = Attention(
            dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop)
        # NOTE: drop path for stochastic depth, we shall see if this is better than dropout here
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

    def forward(self, x,k=None,v=None,mask=None):

        x = x + self.drop_path(self.attn(x,k,v))
        x = x + self.drop_path(self.mlp(x))
        
        return x



class Encoder_Decoder(nn.Module):
    def __init__(self, num_frame=49, decoder_frame=26, num_joints=32,  embed_dim_ratio=8, in_chans=3, depth=1,
                 num_heads=8, mlp_ratio=2., qkv_bias=True, qk_scale=None,
                 drop_rate=0.2, attn_drop_rate=0.2, drop_path_rate=0.2,  norm_layer=None):
        """    ##########hybrid_backbone=None, representation_size=None,
        Args:
            num_frame (int, tuple): input frame number
            num_joints (int, tuple): joints number
            in_chans (int): number of input channels, 2D joints have 2 channels: (x,y)
            embed_dim_ratio (int): embedding dimension ratio
            depth (int): depth of transformer
            num_heads (int): number of attention heads
            mlp_ratio (int): ratio of mlp hidden dim to embedding dim
            qkv_bias (bool): enable bias for qkv if True
            qk_scale (float): override default qk scale of head_dim ** -0.5 if set
            drop_rate (float): dropout rate
            attn_drop_rate (float): attention dropout rate
            drop_path_rate (float): stochastic depth rate
            norm_layer: (nn.Module): normalization layer
        """
        super().__init__()
        self.input_dim=num_joints*in_chans
        self.num_joints=num_joints
        self.num_frame=num_frame
        self.decoder_frame=decoder_frame
        norm_layer = norm_layer or partial(nn.LayerNorm, eps=1e-6)
        embed_dim = embed_dim_ratio * num_joints   #### temporal embed_dim is num_joints * spatial embedding dim ratio
       

        ### spatial patch embedding
        self.Spatial_patch_to_embedding = nn.Linear(in_chans, embed_dim_ratio)
        self.Spatial_patch_to_embedding_f = nn.Linear(in_chans, embed_dim_ratio)
        self.Spatial_pos_embed = nn.Parameter(torch.zeros(1, num_joints, embed_dim_ratio))
       
        self.Temporal_pos_embed = nn.Parameter(torch.zeros(1, num_frame, embed_dim))
            
        self.decoder_patch_to_embedding = nn.Linear(96, embed_dim)
        self.decoder_pos_embed = nn.Parameter(torch.zeros(1, decoder_frame, embed_dim))
        self.encoder_pos_embed = nn.Parameter(torch.zeros(1, decoder_frame, embed_dim))
        self.pos_drop = nn.Dropout(p=drop_rate)


        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]  # stochastic depth decay rule

        self.Spatial_blocks = nn.ModuleList([
            Block(
                dim=embed_dim_ratio, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
                drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[i], norm_layer=norm_layer)
            for i in range(depth)])
        
        self.Spatial_blocks_f = nn.ModuleList([
            Block(
                dim=embed_dim_ratio, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
                drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[i], norm_layer=norm_layer)
            for i in range(depth)])

        self.blocks = nn.ModuleList([
            Block(
                dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
                drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[i], norm_layer=norm_layer)
            for i in range(depth)])
        
        self.blocks_f = nn.ModuleList([
            Block(
                dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
                drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[i], norm_layer=norm_layer)
            for i in range(depth)])
        
        self.decoder_blocks = nn.ModuleList([
            Block(
                dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
                drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[i], norm_layer=norm_layer,mask=None)
            for i in range(depth)])
        
        self.decoder_blocks_f = nn.ModuleList([
            Block(
                dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
                drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[i], norm_layer=norm_layer,mask=None)
            for i in range(depth)])
        
        self.Spatial_norm = norm_layer(embed_dim_ratio)
        self.Temporal_norm = norm_layer(embed_dim)

        ####### A easy way to implement weighted mean

        self.out_put=nn.Linear( embed_dim, self.input_dim)



    def encoder_Spatial_forward_features(self, x):
        b, _, f, p = x.shape  ##### b is batch size, f is number of frames, p is number of joints
        x = rearrange(x, 'b c f p  -> (b f) p  c', )   
        x = self.Spatial_patch_to_embedding(x)
        x += self.Spatial_pos_embed
        x = self.pos_drop(x)
        for blk in self.Spatial_blocks:
            x= blk(x,x,x)
        x = rearrange(x, '(b f) w c -> b f (w c)', f=f)
        return x
    
    def encoder_Spatial_forward_features_f(self, x):
        b, _, f, p = x.shape  ##### b is batch size, f is number of frames, p is number of joints
        x = rearrange(x, 'b c f p  -> (b f) p  c', )
        x = self.Spatial_patch_to_embedding(x)
        x += self.Spatial_pos_embed
        x = self.pos_drop(x)
        for blk in self.Spatial_blocks_f:
            x = blk(x,x,x)
        #x = self.Spatial_norm(x)
        x = rearrange(x, '(b f) w c -> b f (w c)', f=f)
        return x

    def encoder_forward_features(self, x):
        x += self.Temporal_pos_embed
        x = self.pos_drop(x)
        for blk in self.blocks:
            x = blk(x,x,x)
        return x
    
    def encoder_forward_features_f(self, x):
        x += self.Temporal_pos_embed
        x = self.pos_drop(x)
        for blk in self.blocks_f:
            x = blk(x,x,x)
        return x
    
    def decoder_forward_features(self, encoder_out,encoder_out_f,tar,i=None):
        tar=tar.reshape(tar.shape[0],tar.shape[1],32,-1)
        tar = tar.permute(0, 3, 1, 2)
        tar=self.encoder_Spatial_forward_features(tar)
        tar += self.decoder_pos_embed
        tar = self.pos_drop(tar)
        encoder_out = self.pos_drop(encoder_out)
        encoder_out_f = self.pos_drop(encoder_out_f)
        #encoder_out_f =encoder_out_f + encoder_out
        for blk in self.decoder_blocks:
            tar = blk(tar, encoder_out_f, encoder_out_f)
        for blk in self.decoder_blocks_f:
            tar = blk(tar, encoder_out, encoder_out)
        return tar
    

    def forward(self, inp,tar):
        #encoder
        inp_x=inp[:,:,:self.input_dim].reshape(inp.shape[0],inp.shape[1],self.num_joints,-1)
        inp_f=inp[:,:,self.input_dim:].reshape(inp.shape[0],inp.shape[1],self.num_joints,-1)
        inp_x = inp_x.permute(0, 3, 1, 2)
        inp_f = inp_f.permute(0, 3, 1, 2)
        sp_onp = self.encoder_Spatial_forward_features(inp_x)
        encoder_out_x = self.encoder_forward_features(sp_onp)
        sp_onp_f = self.encoder_Spatial_forward_features_f(inp_f)
        encoder_out_f = self.encoder_forward_features_f(sp_onp_f)
        #decoder
        tar=tar[:,0:1,:].repeat(1,self.decoder_frame,1)
        de_self_att = self.decoder_forward_features(encoder_out_x, encoder_out_f, tar[:,:,:self.input_dim])
        out=self.out_put(de_self_att)
        out=out+tar[:, :, :self.input_dim]
        return out
    


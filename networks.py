from dataclasses import dataclass
from typing import Dict, Optional, Tuple, Union, Any, Callable

import torch
import torch.nn as nn
import torch.nn.functional as F

from diffusers.configuration_utils import ConfigMixin, register_to_config
from diffusers.loaders import UNet2DConditionLoadersMixin
from diffusers.utils import BaseOutput, logging
from diffusers.utils.torch_utils import is_torch_version
from diffusers.models.attention_processor import CROSS_ATTENTION_PROCESSORS, AttentionProcessor, AttnProcessor
from diffusers.models.embeddings import TimestepEmbedding, Timesteps
from diffusers.models.modeling_utils import ModelMixin
from diffusers.models.unets.unet_3d_blocks import (
    UNetMidBlockSpatioTemporal,
    get_down_block as gdb, 
    get_up_block as gub,
)
from diffusers.models.resnet import (
    Downsample2D,
    SpatioTemporalResBlock,
    Upsample2D,
)
from diffusers.models.transformers.transformer_temporal import TransformerSpatioTemporalModel
from diffusers.models.attention_processor import Attention
from diffusers.utils import deprecate
from diffusers.utils.import_utils import is_xformers_available

from network_utils import DragEmbedding, get_2d_sincos_pos_embed

logger = logging.get_logger(__name__)  # pylint: disable=invalid-name


if is_xformers_available():
    import xformers
    import xformers.ops


class AllToFirstXFormersAttnProcessor:
    r"""
    Processor for implementing memory efficient attention using xFormers.

    Args:
        attention_op (`Callable`, *optional*, defaults to `None`):
            The base
            [operator](https://facebookresearch.github.io/xformers/components/ops.html#xformers.ops.AttentionOpBase) to
            use as the attention operator. It is recommended to set to `None`, and allow xFormers to choose the best
            operator.
    """

    def __init__(self, attention_op: Optional[Callable] = None):
        self.attention_op = attention_op

    def __call__(
        self,
        attn: Attention,
        hidden_states: torch.FloatTensor,
        encoder_hidden_states: Optional[torch.FloatTensor] = None,
        attention_mask: Optional[torch.FloatTensor] = None,
        temb: Optional[torch.FloatTensor] = None,
        *args,
        **kwargs,
    ) -> torch.FloatTensor:
        if len(args) > 0 or kwargs.get("scale", None) is not None:
            deprecation_message = "The `scale` argument is deprecated and will be ignored. Please remove it, as passing it will raise an error in the future. `scale` should directly be passed while calling the underlying pipeline component i.e., via `cross_attention_kwargs`."
            deprecate("scale", "1.0.0", deprecation_message)

        residual = hidden_states

        if attn.spatial_norm is not None:
            hidden_states = attn.spatial_norm(hidden_states, temb)

        input_ndim = hidden_states.ndim

        if input_ndim == 4:
            batch_size, channel, height, width = hidden_states.shape
            hidden_states = hidden_states.view(batch_size, channel, height * width).transpose(1, 2)

        batch_size, key_tokens, _ = (
            hidden_states.shape if encoder_hidden_states is None else encoder_hidden_states.shape
        )

        assert encoder_hidden_states is None
        attention_mask = attn.prepare_attention_mask(attention_mask, key_tokens, batch_size)
        if attention_mask is not None:
            # expand our mask's singleton query_tokens dimension:
            #   [batch*heads,            1, key_tokens] ->
            #   [batch*heads, query_tokens, key_tokens]
            # so that it can be added as a bias onto the attention scores that xformers computes:
            #   [batch*heads, query_tokens, key_tokens]
            # we do this explicitly because xformers doesn't broadcast the singleton dimension for us.
            _, query_tokens, _ = hidden_states.shape
            attention_mask = attention_mask.expand(-1, query_tokens, -1)

        if attn.group_norm is not None:
            hidden_states = attn.group_norm(hidden_states.transpose(1, 2)).transpose(1, 2)

        query = attn.to_q(hidden_states)
        key = attn.to_k(hidden_states.view(-1, 14, *hidden_states.shape[1:])[:, 0])[:, None].expand(-1, 14, -1, -1).flatten(0, 1)
        value = attn.to_v(hidden_states.view(-1, 14, *hidden_states.shape[1:])[:, 0])[:, None].expand(-1, 14, -1, -1).flatten(0, 1)

        query = attn.head_to_batch_dim(query).contiguous()
        key = attn.head_to_batch_dim(key).contiguous()
        value = attn.head_to_batch_dim(value).contiguous()

        hidden_states = xformers.ops.memory_efficient_attention(
            query, key, value, attn_bias=attention_mask, op=self.attention_op, scale=attn.scale
        )
        hidden_states = hidden_states.to(query.dtype)
        hidden_states = attn.batch_to_head_dim(hidden_states)

        # linear proj
        hidden_states = attn.to_out[0](hidden_states)
        # dropout
        hidden_states = attn.to_out[1](hidden_states)

        if input_ndim == 4:
            hidden_states = hidden_states.transpose(-1, -2).reshape(batch_size, channel, height, width)

        if attn.residual_connection:
            hidden_states = hidden_states + residual

        hidden_states = hidden_states / attn.rescale_output_factor

        return hidden_states


class CrossAttnDownBlockSpatioTemporalWithFlow(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        temb_channels: int,
        flow_channels: int,
        num_layers: int = 1,
        transformer_layers_per_block: Union[int, Tuple[int]] = 1,
        num_attention_heads: int = 1,
        cross_attention_dim: int = 1280,
        add_downsample: bool = True,
        num_frames: int = 14,
        pos_embed_dim: int = 64,
        drag_token_cross_attn: bool = True,
        use_modulate: bool = True,
        drag_embedder_out_channels = (256, 320, 320),
        num_max_drags: int = 5,
    ):
        super().__init__()
        resnets = []
        attentions = []
        flow_convs = []
        if drag_token_cross_attn:
            drag_token_mlps = []
        self.num_max_drags = num_max_drags
        self.num_frames = num_frames
        self.pos_embed_dim = pos_embed_dim
        self.drag_token_cross_attn = drag_token_cross_attn

        self.has_cross_attention = True
        self.num_attention_heads = num_attention_heads
        self.use_modulate = use_modulate
        if isinstance(transformer_layers_per_block, int):
            transformer_layers_per_block = [transformer_layers_per_block] * num_layers

        for i in range(num_layers):
            in_channels = in_channels if i == 0 else out_channels
            resnets.append(
                SpatioTemporalResBlock(
                    in_channels=in_channels,
                    out_channels=out_channels,
                    temb_channels=temb_channels,
                    eps=1e-6,
                )
            )
            attentions.append(
                TransformerSpatioTemporalModel(
                    num_attention_heads,
                    out_channels // num_attention_heads,
                    in_channels=out_channels,
                    num_layers=transformer_layers_per_block[i],
                    cross_attention_dim=cross_attention_dim,
                )
            )
            flow_convs.append(
                DragEmbedding(
                    conditioning_channels=flow_channels, 
                    conditioning_embedding_channels=out_channels * 2 if use_modulate else out_channels,
                    block_out_channels = drag_embedder_out_channels,
                )
            )
            if drag_token_cross_attn:
                drag_token_mlps.append(
                    nn.Sequential(
                        nn.Linear(pos_embed_dim * 2 + out_channels * 2, cross_attention_dim),
                        nn.SiLU(),
                        nn.Linear(cross_attention_dim, cross_attention_dim),
                    )
                )
        self.attentions = nn.ModuleList(attentions)
        self.resnets = nn.ModuleList(resnets)
        self.flow_convs = nn.ModuleList(flow_convs)
        if drag_token_cross_attn:
            self.drag_token_mlps = nn.ModuleList(drag_token_mlps)
        if add_downsample:
            self.downsamplers = nn.ModuleList(
                [
                    Downsample2D(
                        out_channels,
                        use_conv=True,
                        out_channels=out_channels,
                        padding=1,
                        name="op",
                    )
                ]
            )
        else:
            self.downsamplers = None

        self.pos_embedding = {res: torch.tensor(get_2d_sincos_pos_embed(self.pos_embed_dim, res)) for res in [32, 16, 8, 4, 2]}
        self.pos_embedding_prepared = False

        self.gradient_checkpointing = False

    def forward(
        self,
        hidden_states: torch.FloatTensor,
        temb: Optional[torch.FloatTensor] = None,
        encoder_hidden_states: Optional[torch.FloatTensor] = None,
        image_only_indicator: Optional[torch.Tensor] = None,
        flow: Optional[torch.Tensor] = None,
        drag_original: Optional[torch.Tensor] = None,  # (batch_frame, num_points, 4)
    ) -> Tuple[torch.FloatTensor, Tuple[torch.FloatTensor, ...]]:
        output_states = ()

        batch_frame = hidden_states.shape[0]

        if self.drag_token_cross_attn:
            encoder_hidden_states_ori = encoder_hidden_states

        if not self.pos_embedding_prepared:
            for res in self.pos_embedding:
                self.pos_embedding[res] = self.pos_embedding[res].to(hidden_states)
            self.pos_embedding_prepared = True

        blocks = list(zip(self.resnets, self.attentions, self.flow_convs))
        for bid, (resnet, attn, flow_conv) in enumerate(blocks):
            if self.training and self.gradient_checkpointing:  # TODO

                def create_custom_forward(module, return_dict=None):
                    def custom_forward(*inputs):
                        if return_dict is not None:
                            return module(*inputs, return_dict=return_dict)
                        else:
                            return module(*inputs)

                    return custom_forward

                ckpt_kwargs: Dict[str, Any] = {"use_reentrant": False} if is_torch_version(">=", "1.11.0") else {}
                hidden_states = torch.utils.checkpoint.checkpoint(
                    create_custom_forward(resnet),
                    hidden_states,
                    temb,
                    image_only_indicator,
                    **ckpt_kwargs,
                )

                if flow is not None:
                    # flow shape is (batch_frame, 40, h, w)
                    drags = flow.view(-1, self.num_frames, *flow.shape[1:])
                    drags = drags.chunk(self.num_max_drags, dim=2)  # (batch, frame, 4, h, w) x 10
                    drags = torch.stack(drags, dim=0)  # 10, batch, frame, 4, h, w
                    invalid_flag = torch.all(drags == -1, dim=(2, 3, 4, 5))
                    if self.use_modulate:
                        scale, shift = flow_conv(flow).chunk(2, dim=1)
                    else:
                        scale = 0
                        shift = flow_conv(flow)
                    hidden_states = hidden_states * (1 + scale) + shift
                    # print(self.drag_token_cross_attn)
                    if self.drag_token_cross_attn:
                        drag_token_mlp = self.drag_token_mlps[bid]
                        pos_embed = self.pos_embedding[scale.shape[-1]]
                        pos_embed = pos_embed.reshape(1, scale.shape[-1], scale.shape[-1], -1).permute(0, 3, 1, 2)
                        grid = (drag_original[..., :2] * 2 - 1)[:, None]
                        grid_end = (drag_original[..., 2:] * 2 - 1)[:, None]
                        drags_pos_start = F.grid_sample(pos_embed.repeat(batch_frame, 1, 1, 1), grid, padding_mode="border", mode="bilinear", align_corners=False).squeeze(dim=2)
                        drags_pos_end = F.grid_sample(pos_embed.repeat(batch_frame, 1, 1, 1), grid_end, padding_mode="border", mode="bilinear", align_corners=False).squeeze(dim=2)
                        features = F.grid_sample(hidden_states.detach(), grid, padding_mode="border", mode="bilinear", align_corners=False).squeeze(dim=2)
                        features_end = F.grid_sample(hidden_states.detach(), grid_end, padding_mode="border", mode="bilinear", align_corners=False).squeeze(dim=2)

                        drag_token_in = torch.cat([features, features_end, drags_pos_start, drags_pos_end], dim=1).permute(0, 2, 1)
                        drag_token_out = drag_token_mlp(drag_token_in)
                        # Mask the invalid drags
                        drag_token_out = drag_token_out.view(batch_frame // self.num_frames, self.num_frames, self.num_max_drags, -1)
                        drag_token_out = drag_token_out.permute(2, 0, 1, 3)
                        drag_token_out = drag_token_out.masked_fill(invalid_flag[..., None, None].expand_as(drag_token_out), 0)
                        drag_token_out = drag_token_out.permute(1, 2, 0, 3).flatten(0, 1)
                        encoder_hidden_states = torch.cat([encoder_hidden_states_ori, drag_token_out], dim=1)

                hidden_states = attn(
                    hidden_states,
                    encoder_hidden_states=encoder_hidden_states,
                    image_only_indicator=image_only_indicator,
                    return_dict=False,
                )[0]
            else:
                hidden_states = resnet(
                    hidden_states,
                    temb,
                    image_only_indicator=image_only_indicator,
                )
                if flow is not None:
                    # flow shape is (batch_frame, 40, h, w)
                    drags = flow.view(-1, self.num_frames, *flow.shape[1:])
                    drags = drags.chunk(self.num_max_drags, dim=2)  # (batch, frame, 4, h, w) x 10
                    drags = torch.stack(drags, dim=0)  # 10, batch, frame, 4, h, w
                    invalid_flag = torch.all(drags == -1, dim=(2, 3, 4, 5))
                    if self.use_modulate:
                        scale, shift = flow_conv(flow).chunk(2, dim=1)
                    else:
                        scale = 0
                        shift = flow_conv(flow)
                    hidden_states = hidden_states * (1 + scale) + shift
                    if self.drag_token_cross_attn:
                        drag_token_mlp = self.drag_token_mlps[bid]
                        pos_embed = self.pos_embedding[scale.shape[-1]]
                        pos_embed = pos_embed.reshape(1, scale.shape[-1], scale.shape[-1], -1).permute(0, 3, 1, 2)
                        grid = (drag_original[..., :2] * 2 - 1)[:, None]
                        grid_end = (drag_original[..., 2:] * 2 - 1)[:, None]
                        drags_pos_start = F.grid_sample(pos_embed.repeat(batch_frame, 1, 1, 1), grid, padding_mode="border", mode="bilinear", align_corners=False).squeeze(dim=2)
                        drags_pos_end = F.grid_sample(pos_embed.repeat(batch_frame, 1, 1, 1), grid_end, padding_mode="border", mode="bilinear", align_corners=False).squeeze(dim=2)
                        features = F.grid_sample(hidden_states.detach(), grid, padding_mode="border", mode="bilinear", align_corners=False).squeeze(dim=2)
                        features_end = F.grid_sample(hidden_states.detach(), grid_end, padding_mode="border", mode="bilinear", align_corners=False).squeeze(dim=2)

                        drag_token_in = torch.cat([features, features_end, drags_pos_start, drags_pos_end], dim=1).permute(0, 2, 1)
                        drag_token_out = drag_token_mlp(drag_token_in)
                        # Mask the invalid drags
                        drag_token_out = drag_token_out.view(batch_frame // self.num_frames, self.num_frames, self.num_max_drags, -1)
                        drag_token_out = drag_token_out.permute(2, 0, 1, 3)
                        drag_token_out = drag_token_out.masked_fill(invalid_flag[..., None, None].expand_as(drag_token_out), 0)
                        drag_token_out = drag_token_out.permute(1, 2, 0, 3).flatten(0, 1)
                        encoder_hidden_states = torch.cat([encoder_hidden_states_ori, drag_token_out], dim=1)
                hidden_states = attn(
                    hidden_states,
                    encoder_hidden_states=encoder_hidden_states,
                    image_only_indicator=image_only_indicator,
                    return_dict=False,
                )[0]

            output_states = output_states + (hidden_states,)

        if self.downsamplers is not None:
            for downsampler in self.downsamplers:
                hidden_states = downsampler(hidden_states)

            output_states = output_states + (hidden_states,)

        return hidden_states, output_states


class CrossAttnUpBlockSpatioTemporalWithFlow(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        prev_output_channel: int,
        temb_channels: int,
        flow_channels: int,
        resolution_idx: Optional[int] = None,
        num_layers: int = 1,
        transformer_layers_per_block: Union[int, Tuple[int]] = 1,
        resnet_eps: float = 1e-6,
        num_attention_heads: int = 1,
        cross_attention_dim: int = 1280,
        add_upsample: bool = True,
        num_frames: int = 14,
        pos_embed_dim: int = 64,
        drag_token_cross_attn: bool = True,
        use_modulate: bool = True,
        drag_embedder_out_channels = (256, 320, 320),
        num_max_drags: int = 5,
    ):
        super().__init__()
        resnets = []
        attentions = []
        flow_convs = []
        if drag_token_cross_attn:
            drag_token_mlps = []
        self.num_max_drags = num_max_drags

        self.drag_token_cross_attn = drag_token_cross_attn

        self.num_frames = num_frames
        self.pos_embed_dim = pos_embed_dim

        self.has_cross_attention = True
        self.num_attention_heads = num_attention_heads
        self.use_modulate = use_modulate

        if isinstance(transformer_layers_per_block, int):
            transformer_layers_per_block = [transformer_layers_per_block] * num_layers

        for i in range(num_layers):
            res_skip_channels = in_channels if (i == num_layers - 1) else out_channels
            resnet_in_channels = prev_output_channel if i == 0 else out_channels

            resnets.append(
                SpatioTemporalResBlock(
                    in_channels=resnet_in_channels + res_skip_channels,
                    out_channels=out_channels,
                    temb_channels=temb_channels,
                    eps=resnet_eps,
                )
            )
            attentions.append(
                TransformerSpatioTemporalModel(
                    num_attention_heads,
                    out_channels // num_attention_heads,
                    in_channels=out_channels,
                    num_layers=transformer_layers_per_block[i],
                    cross_attention_dim=cross_attention_dim,
                )
            )
            flow_convs.append(
                DragEmbedding(
                    conditioning_channels=flow_channels, 
                    conditioning_embedding_channels=out_channels * 2 if use_modulate else out_channels,
                    block_out_channels = drag_embedder_out_channels,
                )
            )
            if drag_token_cross_attn:
                drag_token_mlps.append(
                    nn.Sequential(
                        nn.Linear(pos_embed_dim * 2 + out_channels * 2, cross_attention_dim),
                        nn.SiLU(),
                        nn.Linear(cross_attention_dim, cross_attention_dim),
                    )
                )
        self.attentions = nn.ModuleList(attentions)
        self.resnets = nn.ModuleList(resnets)
        self.flow_convs = nn.ModuleList(flow_convs)
        
        if drag_token_cross_attn:
            self.drag_token_mlps = nn.ModuleList(drag_token_mlps)
        if add_upsample:
            self.upsamplers = nn.ModuleList([Upsample2D(out_channels, use_conv=True, out_channels=out_channels)])
        else:
            self.upsamplers = None

        self.pos_embedding = {res: torch.tensor(get_2d_sincos_pos_embed(pos_embed_dim, res)) for res in [32, 16, 8, 4, 2]}
        self.pos_embedding_prepared = False

        self.gradient_checkpointing = False
        self.resolution_idx = resolution_idx

    def forward(
        self,
        hidden_states: torch.FloatTensor,
        res_hidden_states_tuple: Tuple[torch.FloatTensor, ...],
        temb: Optional[torch.FloatTensor] = None,
        encoder_hidden_states: Optional[torch.FloatTensor] = None,
        image_only_indicator: Optional[torch.Tensor] = None,
        flow: Optional[torch.Tensor] = None,
        drag_original: Optional[torch.Tensor] = None,  # (batch_frame, num_points, 4)
    ) -> torch.FloatTensor:
        batch_frame = hidden_states.shape[0]

        if self.drag_token_cross_attn:
            encoder_hidden_states_ori = encoder_hidden_states
        
        if not self.pos_embedding_prepared:
            for res in self.pos_embedding:
                self.pos_embedding[res] = self.pos_embedding[res].to(hidden_states)
            self.pos_embedding_prepared = True

        for bid, (resnet, attn, flow_conv) in enumerate(zip(self.resnets, self.attentions, self.flow_convs)):
            # pop res hidden states
            res_hidden_states = res_hidden_states_tuple[-1]
            res_hidden_states_tuple = res_hidden_states_tuple[:-1]

            hidden_states = torch.cat([hidden_states, res_hidden_states], dim=1)

            if self.training and self.gradient_checkpointing:  # TODO
                def create_custom_forward(module, return_dict=None):
                    def custom_forward(*inputs):
                        if return_dict is not None:
                            return module(*inputs, return_dict=return_dict)
                        else:
                            return module(*inputs)

                    return custom_forward

                ckpt_kwargs: Dict[str, Any] = {"use_reentrant": False} if is_torch_version(">=", "1.11.0") else {}
                hidden_states = torch.utils.checkpoint.checkpoint(
                    create_custom_forward(resnet),
                    hidden_states,
                    temb,
                    image_only_indicator,
                    **ckpt_kwargs,
                )
                if flow is not None:
                    # flow shape is (batch_frame, 40, h, w)
                    drags = flow.view(-1, self.num_frames, *flow.shape[1:])
                    drags = drags.chunk(self.num_max_drags, dim=2)  # (batch, frame, 4, h, w) x 10
                    drags = torch.stack(drags, dim=0)  # 10, batch, frame, 4, h, w
                    invalid_flag = torch.all(drags == -1, dim=(2, 3, 4, 5))
                    if self.use_modulate:
                        scale, shift = flow_conv(flow).chunk(2, dim=1)
                    else:
                        scale = 0
                        shift = flow_conv(flow)
                    hidden_states = hidden_states * (1 + scale) + shift
                    if self.drag_token_cross_attn:
                        drag_token_mlp = self.drag_token_mlps[bid]
                        pos_embed = self.pos_embedding[scale.shape[-1]]
                        pos_embed = pos_embed.reshape(1, scale.shape[-1], scale.shape[-1], -1).permute(0, 3, 1, 2)
                        grid = (drag_original[..., :2] * 2 - 1)[:, None]
                        grid_end = (drag_original[..., 2:] * 2 - 1)[:, None]
                        drags_pos_start = F.grid_sample(pos_embed.repeat(batch_frame, 1, 1, 1), grid, padding_mode="border", mode="bilinear", align_corners=False).squeeze(dim=2)
                        drags_pos_end = F.grid_sample(pos_embed.repeat(batch_frame, 1, 1, 1), grid_end, padding_mode="border", mode="bilinear", align_corners=False).squeeze(dim=2)
                        features = F.grid_sample(hidden_states.detach(), grid, padding_mode="border", mode="bilinear", align_corners=False).squeeze(dim=2)
                        features_end = F.grid_sample(hidden_states.detach(), grid_end, padding_mode="border", mode="bilinear", align_corners=False).squeeze(dim=2)

                        drag_token_in = torch.cat([features, features_end, drags_pos_start, drags_pos_end], dim=1).permute(0, 2, 1)
                        drag_token_out = drag_token_mlp(drag_token_in)
                        # Mask the invalid drags
                        drag_token_out = drag_token_out.view(batch_frame // self.num_frames, self.num_frames, self.num_max_drags, -1)
                        drag_token_out = drag_token_out.permute(2, 0, 1, 3)
                        drag_token_out = drag_token_out.masked_fill(invalid_flag[..., None, None].expand_as(drag_token_out), 0)
                        drag_token_out = drag_token_out.permute(1, 2, 0, 3).flatten(0, 1)
                        encoder_hidden_states = torch.cat([encoder_hidden_states_ori, drag_token_out], dim=1)

                hidden_states = attn(
                    hidden_states,
                    encoder_hidden_states=encoder_hidden_states,
                    image_only_indicator=image_only_indicator,
                    return_dict=False,
                )[0]
            else:
                hidden_states = resnet(
                    hidden_states,
                    temb,
                    image_only_indicator=image_only_indicator,
                )
                if flow is not None:
                    # flow shape is (batch_frame, 40, h, w)
                    drags = flow.view(-1, self.num_frames, *flow.shape[1:])
                    drags = drags.chunk(self.num_max_drags, dim=2)  # (batch, frame, 4, h, w) x 10
                    drags = torch.stack(drags, dim=0)  # 10, batch, frame, 4, h, w
                    invalid_flag = torch.all(drags == -1, dim=(2, 3, 4, 5))
                    if self.use_modulate:
                        scale, shift = flow_conv(flow).chunk(2, dim=1)
                    else:
                        scale = 0
                        shift = flow_conv(flow)
                    hidden_states = hidden_states * (1 + scale) + shift
                    if self.drag_token_cross_attn:
                        drag_token_mlp = self.drag_token_mlps[bid]
                        pos_embed = self.pos_embedding[scale.shape[-1]]
                        pos_embed = pos_embed.reshape(1, scale.shape[-1], scale.shape[-1], -1).permute(0, 3, 1, 2)
                        grid = (drag_original[..., :2] * 2 - 1)[:, None]
                        grid_end = (drag_original[..., 2:] * 2 - 1)[:, None]
                        drags_pos_start = F.grid_sample(pos_embed.repeat(batch_frame, 1, 1, 1), grid, padding_mode="border", mode="bilinear", align_corners=False).squeeze(dim=2)
                        drags_pos_end = F.grid_sample(pos_embed.repeat(batch_frame, 1, 1, 1), grid_end, padding_mode="border", mode="bilinear", align_corners=False).squeeze(dim=2)
                        features = F.grid_sample(hidden_states.detach(), grid, padding_mode="border", mode="bilinear", align_corners=False).squeeze(dim=2)
                        features_end = F.grid_sample(hidden_states.detach(), grid_end, padding_mode="border", mode="bilinear", align_corners=False).squeeze(dim=2)

                        drag_token_in = torch.cat([features, features_end, drags_pos_start, drags_pos_end], dim=1).permute(0, 2, 1)
                        drag_token_out = drag_token_mlp(drag_token_in)
                        # Mask the invalid drags
                        drag_token_out = drag_token_out.view(batch_frame // self.num_frames, self.num_frames, self.num_max_drags, -1)
                        drag_token_out = drag_token_out.permute(2, 0, 1, 3)
                        drag_token_out = drag_token_out.masked_fill(invalid_flag[..., None, None].expand_as(drag_token_out), 0)
                        drag_token_out = drag_token_out.permute(1, 2, 0, 3).flatten(0, 1)
                        encoder_hidden_states = torch.cat([encoder_hidden_states_ori, drag_token_out], dim=1)

                hidden_states = attn(
                    hidden_states,
                    encoder_hidden_states=encoder_hidden_states,
                    image_only_indicator=image_only_indicator,
                    return_dict=False,
                )[0]

        if self.upsamplers is not None:
            for upsampler in self.upsamplers:
                hidden_states = upsampler(hidden_states)

        return hidden_states


def get_down_block(
    with_concatenated_flow: bool = False,
    *args,
    **kwargs,
):
    NEEDED_KEYS = [
        "in_channels",
        "out_channels",
        "temb_channels",
        "flow_channels",
        "num_layers",
        "transformer_layers_per_block",
        "num_attention_heads",
        "cross_attention_dim",
        "add_downsample",
        "pos_embed_dim",
        'use_modulate',
        "drag_token_cross_attn",
        "drag_embedder_out_channels",
        "num_max_drags",
    ]
    if not with_concatenated_flow or args[0] == "DownBlockSpatioTemporal":
        kwargs.pop("flow_channels", None)
        kwargs.pop("pos_embed_dim", None)
        kwargs.pop("use_modulate", None)
        kwargs.pop("drag_token_cross_attn", None)
        kwargs.pop("drag_embedder_out_channels", None)
        kwargs.pop("num_max_drags", None)
        return gdb(*args, **kwargs)
    elif args[0] == "CrossAttnDownBlockSpatioTemporal":
        for key in list(kwargs.keys()):
            if key not in NEEDED_KEYS:
                kwargs.pop(key, None)
        return CrossAttnDownBlockSpatioTemporalWithFlow(*args[1:], **kwargs)
    else:
        raise ValueError(f"Unknown block type {args[0]}")
    

def get_up_block(
    with_concatenated_flow: bool = False,
    *args,
    **kwargs,
):
    NEEDED_KEYS = [
        "in_channels",
        "out_channels",
        "prev_output_channel",
        "temb_channels",
        "flow_channels",
        "resolution_idx",
        "num_layers",
        "transformer_layers_per_block",
        "resnet_eps",
        "num_attention_heads",
        "cross_attention_dim",
        "add_upsample",
        "pos_embed_dim",
        "use_modulate",
        "drag_token_cross_attn",
        "drag_embedder_out_channels",
        "num_max_drags",
    ]
    if not with_concatenated_flow or args[0] == "UpBlockSpatioTemporal":
        kwargs.pop("flow_channels", None)
        kwargs.pop("pos_embed_dim", None)
        kwargs.pop("use_modulate", None)
        kwargs.pop("drag_token_cross_attn", None)
        kwargs.pop("drag_embedder_out_channels", None)
        kwargs.pop("num_max_drags", None)
        return gub(*args, **kwargs)
    elif args[0] == "CrossAttnUpBlockSpatioTemporal":
        for key in list(kwargs.keys()):
            if key not in NEEDED_KEYS:
                kwargs.pop(key, None)
        return CrossAttnUpBlockSpatioTemporalWithFlow(*args[1:], **kwargs)
    else:
        raise ValueError(f"Unknown block type {args[0]}")


@dataclass
class UNetSpatioTemporalConditionOutput(BaseOutput):
    """
    The output of [`UNetSpatioTemporalConditionModel`].

    Args:
        sample (`torch.FloatTensor` of shape `(batch_size, num_frames, num_channels, height, width)`):
            The hidden states output conditioned on `encoder_hidden_states` input. Output of last layer of model.
    """

    sample: torch.FloatTensor = None


class UNetDragSpatioTemporalConditionModel(ModelMixin, ConfigMixin, UNet2DConditionLoadersMixin):
    r"""
    A conditional Spatio-Temporal UNet model that takes a noisy video frames, conditional state, and a timestep and
    returns a sample shaped output.

    This model inherits from [`ModelMixin`]. Check the superclass documentation for it's generic methods implemented
    for all models (such as downloading or saving).

    Parameters:
        sample_size (`int` or `Tuple[int, int]`, *optional*, defaults to `None`):
            Height and width of input/output sample.
        in_channels (`int`, *optional*, defaults to 8): Number of channels in the input sample.
        out_channels (`int`, *optional*, defaults to 4): Number of channels in the output.
        down_block_types (`Tuple[str]`, *optional*, defaults to `("CrossAttnDownBlockSpatioTemporal", "CrossAttnDownBlockSpatioTemporal", "CrossAttnDownBlockSpatioTemporal", "DownBlockSpatioTemporal")`):
            The tuple of downsample blocks to use.
        up_block_types (`Tuple[str]`, *optional*, defaults to `("UpBlockSpatioTemporal", "CrossAttnUpBlockSpatioTemporal", "CrossAttnUpBlockSpatioTemporal", "CrossAttnUpBlockSpatioTemporal")`):
            The tuple of upsample blocks to use.
        block_out_channels (`Tuple[int]`, *optional*, defaults to `(320, 640, 1280, 1280)`):
            The tuple of output channels for each block.
        addition_time_embed_dim: (`int`, defaults to 256):
            Dimension to to encode the additional time ids.
        projection_class_embeddings_input_dim (`int`, defaults to 768):
            The dimension of the projection of encoded `added_time_ids`.
        layers_per_block (`int`, *optional*, defaults to 2): The number of layers per block.
        cross_attention_dim (`int` or `Tuple[int]`, *optional*, defaults to 1280):
            The dimension of the cross attention features.
        transformer_layers_per_block (`int`, `Tuple[int]`, or `Tuple[Tuple]` , *optional*, defaults to 1):
            The number of transformer blocks of type [`~models.attention.BasicTransformerBlock`]. Only relevant for
            [`~models.unet_3d_blocks.CrossAttnDownBlockSpatioTemporal`],
            [`~models.unet_3d_blocks.CrossAttnUpBlockSpatioTemporal`],
            [`~models.unet_3d_blocks.UNetMidBlockSpatioTemporal`].
        num_attention_heads (`int`, `Tuple[int]`, defaults to `(5, 10, 10, 20)`):
            The number of attention heads.
        dropout (`float`, *optional*, defaults to 0.0): The dropout probability to use.
    """

    _supports_gradient_checkpointing = True

    @register_to_config
    def __init__(
        self,
        sample_size: Optional[int] = None,
        in_channels: int = 8,
        out_channels: int = 4,
        down_block_types: Tuple[str] = (
            "CrossAttnDownBlockSpatioTemporal",
            "CrossAttnDownBlockSpatioTemporal",
            "CrossAttnDownBlockSpatioTemporal",
            "DownBlockSpatioTemporal",
        ),
        up_block_types: Tuple[str] = (
            "UpBlockSpatioTemporal",
            "CrossAttnUpBlockSpatioTemporal",
            "CrossAttnUpBlockSpatioTemporal",
            "CrossAttnUpBlockSpatioTemporal",
        ),
        block_out_channels: Tuple[int] = (320, 640, 1280, 1280),
        addition_time_embed_dim: int = 256,
        projection_class_embeddings_input_dim: int = 768,
        layers_per_block: Union[int, Tuple[int]] = 2,
        cross_attention_dim: Union[int, Tuple[int]] = 1024,
        transformer_layers_per_block: Union[int, Tuple[int], Tuple[Tuple]] = 1,
        num_attention_heads: Union[int, Tuple[int]] = (5, 10, 20, 20),
        num_frames: int = 25,
        num_drags: int = 10,
        cond_dropout_prob: float = 0.1,
        pos_embed_dim: int = 64,
        drag_token_cross_attn: bool = True,
        use_modulate: bool = True,
        drag_embedder_out_channels = (256, 320, 320),
    ):
        super().__init__()

        self.sample_size = sample_size
        self.cond_dropout_prob = cond_dropout_prob
        self.drag_token_cross_attn = drag_token_cross_attn

        self.pos_embed_dim = pos_embed_dim
        self.use_modulate = use_modulate

        flow_channels = 6 * num_drags

        # Check inputs
        if len(down_block_types) != len(up_block_types):
            raise ValueError(
                f"Must provide the same number of `down_block_types` as `up_block_types`. `down_block_types`: {down_block_types}. `up_block_types`: {up_block_types}."
            )

        if len(block_out_channels) != len(down_block_types):
            raise ValueError(
                f"Must provide the same number of `block_out_channels` as `down_block_types`. `block_out_channels`: {block_out_channels}. `down_block_types`: {down_block_types}."
            )

        if not isinstance(num_attention_heads, int) and len(num_attention_heads) != len(down_block_types):
            raise ValueError(
                f"Must provide the same number of `num_attention_heads` as `down_block_types`. `num_attention_heads`: {num_attention_heads}. `down_block_types`: {down_block_types}."
            )

        if isinstance(cross_attention_dim, list) and len(cross_attention_dim) != len(down_block_types):
            raise ValueError(
                f"Must provide the same number of `cross_attention_dim` as `down_block_types`. `cross_attention_dim`: {cross_attention_dim}. `down_block_types`: {down_block_types}."
            )

        if not isinstance(layers_per_block, int) and len(layers_per_block) != len(down_block_types):
            raise ValueError(
                f"Must provide the same number of `layers_per_block` as `down_block_types`. `layers_per_block`: {layers_per_block}. `down_block_types`: {down_block_types}."
            )

        # input
        self.conv_in = nn.Conv2d(
            in_channels,
            block_out_channels[0],
            kernel_size=3,
            padding=1,
        )

        # time
        time_embed_dim = block_out_channels[0] * 4

        self.time_proj = Timesteps(block_out_channels[0], True, downscale_freq_shift=0)
        timestep_input_dim = block_out_channels[0]

        self.time_embedding = TimestepEmbedding(timestep_input_dim, time_embed_dim)

        self.down_blocks = nn.ModuleList([])
        self.up_blocks = nn.ModuleList([])

        if isinstance(num_attention_heads, int):
            num_attention_heads = (num_attention_heads,) * len(down_block_types)

        if isinstance(cross_attention_dim, int):
            cross_attention_dim = (cross_attention_dim,) * len(down_block_types)

        if isinstance(layers_per_block, int):
            layers_per_block = [layers_per_block] * len(down_block_types)

        if isinstance(transformer_layers_per_block, int):
            transformer_layers_per_block = [transformer_layers_per_block] * len(down_block_types)

        blocks_time_embed_dim = time_embed_dim

        # down
        output_channel = block_out_channels[0]
        for i, down_block_type in enumerate(down_block_types):
            input_channel = output_channel
            output_channel = block_out_channels[i]
            is_final_block = i == len(block_out_channels) - 1

            down_block = get_down_block(
                True,
                down_block_type,
                num_layers=layers_per_block[i],
                transformer_layers_per_block=transformer_layers_per_block[i],
                in_channels=input_channel,
                out_channels=output_channel,
                temb_channels=blocks_time_embed_dim,
                add_downsample=not is_final_block,
                resnet_eps=1e-5,
                cross_attention_dim=cross_attention_dim[i],
                num_attention_heads=num_attention_heads[i],
                resnet_act_fn="silu",
                flow_channels=flow_channels,
                pos_embed_dim=pos_embed_dim,
                use_modulate=use_modulate,
                drag_token_cross_attn=drag_token_cross_attn,
                drag_embedder_out_channels=drag_embedder_out_channels,
                num_max_drags=num_drags,
            )
            self.down_blocks.append(down_block)

        # mid
        self.mid_block = UNetMidBlockSpatioTemporal(
            block_out_channels[-1],
            temb_channels=blocks_time_embed_dim,
            transformer_layers_per_block=transformer_layers_per_block[-1],
            cross_attention_dim=cross_attention_dim[-1],
            num_attention_heads=num_attention_heads[-1],
        )

        # count how many layers upsample the images
        self.num_upsamplers = 0

        # up
        reversed_block_out_channels = list(reversed(block_out_channels))
        reversed_num_attention_heads = list(reversed(num_attention_heads))
        reversed_layers_per_block = list(reversed(layers_per_block))
        reversed_cross_attention_dim = list(reversed(cross_attention_dim))
        reversed_transformer_layers_per_block = list(reversed(transformer_layers_per_block))

        output_channel = reversed_block_out_channels[0]
        for i, up_block_type in enumerate(up_block_types):
            is_final_block = i == len(block_out_channels) - 1

            prev_output_channel = output_channel
            output_channel = reversed_block_out_channels[i]
            input_channel = reversed_block_out_channels[min(i + 1, len(block_out_channels) - 1)]

            # add upsample block for all BUT final layer
            if not is_final_block:
                add_upsample = True
                self.num_upsamplers += 1
            else:
                add_upsample = False

            up_block = get_up_block(
                True,
                up_block_type,
                num_layers=reversed_layers_per_block[i] + 1,
                transformer_layers_per_block=reversed_transformer_layers_per_block[i],
                in_channels=input_channel,
                out_channels=output_channel,
                prev_output_channel=prev_output_channel,
                temb_channels=blocks_time_embed_dim,
                add_upsample=add_upsample,
                resnet_eps=1e-5,
                resolution_idx=i,
                cross_attention_dim=reversed_cross_attention_dim[i],
                num_attention_heads=reversed_num_attention_heads[i],
                resnet_act_fn="silu",
                flow_channels=flow_channels,
                pos_embed_dim=pos_embed_dim,
                use_modulate=use_modulate,
                drag_token_cross_attn=drag_token_cross_attn,
                drag_embedder_out_channels=drag_embedder_out_channels,
                num_max_drags=num_drags,
            )
            self.up_blocks.append(up_block)
            prev_output_channel = output_channel

        # out
        self.conv_norm_out = nn.GroupNorm(num_channels=block_out_channels[0], num_groups=32, eps=1e-5)
        self.conv_act = nn.SiLU()

        self.conv_out = nn.Conv2d(
            block_out_channels[0],
            out_channels,
            kernel_size=3,
            padding=1,
        )

        self.num_drags = num_drags

        self.pos_embedding = {res: torch.tensor(get_2d_sincos_pos_embed(self.pos_embed_dim, res)) for res in [32, 16, 8, 4, 2]}
        self.pos_embedding_prepared = False

    @property
    def attn_processors(self) -> Dict[str, AttentionProcessor]:
        r"""
        Returns:
            `dict` of attention processors: A dictionary containing all attention processors used in the model with
            indexed by its weight name.
        """
        # set recursively
        processors = {}

        def fn_recursive_add_processors(
            name: str,
            module: torch.nn.Module,
            processors: Dict[str, AttentionProcessor],
        ):
            if hasattr(module, "get_processor"):
                processors[f"{name}.processor"] = module.get_processor(return_deprecated_lora=True)

            for sub_name, child in module.named_children():
                fn_recursive_add_processors(f"{name}.{sub_name}", child, processors)

            return processors

        for name, module in self.named_children():
            fn_recursive_add_processors(name, module, processors)

        return processors

    def set_attn_processor(self, processor: Union[AttentionProcessor, Dict[str, AttentionProcessor]]):
        r"""
        Sets the attention processor to use to compute attention.

        Parameters:
            processor (`dict` of `AttentionProcessor` or only `AttentionProcessor`):
                The instantiated processor class or a dictionary of processor classes that will be set as the processor
                for **all** `Attention` layers.

                If `processor` is a dict, the key needs to define the path to the corresponding cross attention
                processor. This is strongly recommended when setting trainable attention processors.

        """
        count = len(self.attn_processors.keys())

        if isinstance(processor, dict) and len(processor) != count:
            raise ValueError(
                f"A dict of processors was passed, but the number of processors {len(processor)} does not match the"
                f" number of attention layers: {count}. Please make sure to pass {count} processor classes."
            )

        def fn_recursive_attn_processor(name: str, module: torch.nn.Module, processor):
            if hasattr(module, "set_processor"):
                if not isinstance(processor, dict):
                    module.set_processor(processor)
                else:
                    module.set_processor(processor.pop(f"{name}.processor"))

            for sub_name, child in module.named_children():
                fn_recursive_attn_processor(f"{name}.{sub_name}", child, processor)

        for name, module in self.named_children():
            fn_recursive_attn_processor(name, module, processor)

    def set_default_attn_processor(self):
        """
        Disables custom attention processors and sets the default attention implementation.
        """
        if all(proc.__class__ in CROSS_ATTENTION_PROCESSORS for proc in self.attn_processors.values()):
            processor = AttnProcessor()
        else:
            raise ValueError(
                f"Cannot call `set_default_attn_processor` when attention processors are of type {next(iter(self.attn_processors.values()))}"
            )

        self.set_attn_processor(processor)

    def _set_gradient_checkpointing(self, module, value=False):
        if hasattr(module, "gradient_checkpointing"):
            module.gradient_checkpointing = value

    # Copied from diffusers.models.unets.unet_3d_condition.UNet3DConditionModel.enable_forward_chunking
    def enable_forward_chunking(self, chunk_size: Optional[int] = None, dim: int = 0) -> None:
        """
        Sets the attention processor to use [feed forward
        chunking](https://huggingface.co/blog/reformer#2-chunked-feed-forward-layers).

        Parameters:
            chunk_size (`int`, *optional*):
                The chunk size of the feed-forward layers. If not specified, will run feed-forward layer individually
                over each tensor of dim=`dim`.
            dim (`int`, *optional*, defaults to `0`):
                The dimension over which the feed-forward computation should be chunked. Choose between dim=0 (batch)
                or dim=1 (sequence length).
        """
        if dim not in [0, 1]:
            raise ValueError(f"Make sure to set `dim` to either 0 or 1, not {dim}")

        # By default chunk size is 1
        chunk_size = chunk_size or 1

        def fn_recursive_feed_forward(module: torch.nn.Module, chunk_size: int, dim: int):
            if hasattr(module, "set_chunk_feed_forward"):
                module.set_chunk_feed_forward(chunk_size=chunk_size, dim=dim)

            for child in module.children():
                fn_recursive_feed_forward(child, chunk_size, dim)

        for module in self.children():
            fn_recursive_feed_forward(module, chunk_size, dim)

    def _convert_drag_to_concatting_image(self, drags: torch.Tensor, current_resolution: int) -> torch.Tensor:
        batch_size, num_frames, num_points, _ = drags.shape
        num_channels = 6
        concatting_image = -torch.ones(
            batch_size, num_frames, num_channels * num_points, current_resolution, current_resolution
        ).to(drags)
        
        not_all_zeros = drags.any(dim=-1).repeat_interleave(num_channels, dim=-1)[..., None, None]
        y_grid, x_grid = torch.meshgrid(torch.arange(current_resolution), torch.arange(current_resolution), indexing='ij')
        y_grid = y_grid.to(drags)[None, None, None]  # (1, 1, 1, res, res)
        x_grid = x_grid.to(drags)[None, None, None]  # (1, 1, 1, res, res)
        x0 = (drags[..., 0] * current_resolution - 0.5).round().clip(0, current_resolution - 1)
        x_src = (drags[..., 0] * current_resolution - x0)[..., None, None]  # (batch, num_frames, num_points, 1, 1)
        x0 = x0[..., None, None]  # (batch, num_frames, num_points, 1, 1)
        x0 = torch.stack([
            x0, x0, 
            torch.zeros_like(x0) - 1, torch.zeros_like(x0) - 1, 
            torch.zeros_like(x0) - 1, torch.zeros_like(x0) - 1,
        ], dim=3).view(batch_size, num_frames, num_channels * num_points, 1, 1)

        y0 = (drags[..., 1] * current_resolution - 0.5).round().clip(0, current_resolution - 1)
        y_src = (drags[..., 1] * current_resolution - y0)[..., None, None]  # (batch, num_frames, num_points, 1, 1)
        y0 = y0[..., None, None]  # (batch, num_frames, num_points, 1, 1)
        y0 = torch.stack([
            y0, y0, 
            torch.zeros_like(y0) - 1, torch.zeros_like(y0) - 1, 
            torch.zeros_like(y0) - 1, torch.zeros_like(y0) - 1,
        ], dim=3).view(batch_size, num_frames, num_channels * num_points, 1, 1)

        x1 = (drags[..., 2] * current_resolution - 0.5).round().clip(0, current_resolution - 1)
        x_tgt = (drags[..., 2] * current_resolution - x1)[..., None, None]  # (batch, num_frames, num_points, 1, 1)
        x1 = x1[..., None, None]  # (batch, num_frames, num_points, 1, 1)
        x1 = torch.stack([
            torch.zeros_like(x1) - 1, torch.zeros_like(x1) - 1, 
            x1, x1, 
            torch.zeros_like(x1) - 1, torch.zeros_like(x1) - 1
        ], dim=3).view(batch_size, num_frames, num_channels * num_points, 1, 1)

        y1 = (drags[..., 3] * current_resolution - 0.5).round().clip(0, current_resolution - 1)
        y_tgt = (drags[..., 3] * current_resolution - y1)[..., None, None]  # (batch, num_frames, num_points, 1, 1)
        y1 = y1[..., None, None]  # (batch, num_frames, num_points, 1, 1)
        y1 = torch.stack([
            torch.zeros_like(y1) - 1, torch.zeros_like(y1) - 1, 
            y1, y1, 
            torch.zeros_like(y1) - 1, torch.zeros_like(y1) - 1
        ], dim=3).view(batch_size, num_frames, num_channels * num_points, 1, 1)

        drags_final = drags[:, -1:, :, :].expand_as(drags)
        x_final = (drags_final[..., 2] * current_resolution - 0.5).round().clip(0, current_resolution - 1)
        x_final_tgt = (drags_final[..., 2] * current_resolution - x_final)[..., None, None]  # (batch, num_frames, num_points, 1, 1)
        x_final = x_final[..., None, None]  # (batch, num_frames, num_points, 1, 1)
        x_final = torch.stack([
            torch.zeros_like(x_final) - 1, torch.zeros_like(x_final) - 1, 
            torch.zeros_like(x_final) - 1, torch.zeros_like(x_final) - 1, 
            x_final, x_final
        ], dim=3).view(batch_size, num_frames, num_channels * num_points, 1, 1)

        y_final = (drags_final[..., 3] * current_resolution - 0.5).round().clip(0, current_resolution - 1)
        y_final_tgt = (drags_final[..., 3] * current_resolution - y_final)[..., None, None]  # (batch, num_frames, num_points, 1, 1)
        y_final = y_final[..., None, None]  # (batch, num_frames, num_points, 1, 1)
        y_final = torch.stack([
            torch.zeros_like(y_final) - 1, torch.zeros_like(y_final) - 1, 
            torch.zeros_like(y_final) - 1, torch.zeros_like(y_final) - 1, 
            y_final, y_final
        ], dim=3).view(batch_size, num_frames, num_channels * num_points, 1, 1)

        value_image = torch.stack([
            x_src, y_src, 
            x_tgt, y_tgt, 
            x_final_tgt, y_final_tgt
        ], dim=3).view(batch_size, num_frames, num_channels * num_points, 1, 1)
        value_image = value_image.expand_as(concatting_image)
        start_mask = (x_grid == x0) & (y_grid == y0) & not_all_zeros
        end_mask = (x_grid == x1) & (y_grid == y1) & not_all_zeros
        final_mask = (x_grid == x_final) & (y_grid == y_final) & not_all_zeros
        concatting_image[start_mask] = value_image[start_mask]
        concatting_image[end_mask] = value_image[end_mask]
        concatting_image[final_mask] = value_image[final_mask]
        return concatting_image
    
    def zero_init(self):
        for block in self.down_blocks:
            if hasattr(block, "flow_convs"):
                for flow_conv in block.flow_convs:
                    try:
                        nn.init.constant_(flow_conv.conv_out.weight, 0)
                        nn.init.constant_(flow_conv.conv_out.bias, 0)
                    except:
                        nn.init.constant_(flow_conv.weight, 0)

        for block in self.up_blocks:
            if hasattr(block, "flow_convs"):
                for flow_conv in block.flow_convs:
                    try:
                        nn.init.constant_(flow_conv.conv_out.weight, 0)
                        nn.init.constant_(flow_conv.conv_out.bias, 0)
                    except:
                        nn.init.constant_(flow_conv.weight, 0)

    def forward(
        self,
        sample: torch.FloatTensor,
        timestep: Union[torch.Tensor, float, int],
        image_latents: torch.FloatTensor,
        encoder_hidden_states: torch.Tensor,
        added_time_ids: torch.Tensor,
        drags: torch.Tensor,

        force_drop_ids: Optional[torch.Tensor] = None,
    ) -> Union[UNetSpatioTemporalConditionOutput, Tuple]:
        r"""
        The [`UNetSpatioTemporalConditionModel`] forward method.

        Args:
            sample (`torch.FloatTensor`):
                The noisy input tensor with the following shape `(batch, num_frames, channel, height, width)`.
            image_latents (`torch.FloatTensor`):
                The clean conditioning tensor of the first frame of the image with shape `(batch, num_channels, height, width)`.
            timestep (`torch.FloatTensor` or `float` or `int`): The number of timesteps to denoise an input.
            encoder_hidden_states (`torch.FloatTensor`):
                The encoder hidden states with shape `(batch, sequence_length, cross_attention_dim)`.
            added_time_ids: (`torch.FloatTensor`):
                The additional time ids with shape `(batch, num_additional_ids)`. These are encoded with sinusoidal
                embeddings and added to the time embeddings.
            drags (`torch.Tensor`):
                The drags tensor with shape `(batch, num_frames, num_points, 4)`.
            return_dict (`bool`, *optional*, defaults to `True`):
                Whether or not to return a [`~models.unet_slatio_temporal.UNetSpatioTemporalConditionOutput`] instead
                of a plain tuple.
        Returns:
            [`~models.unet_slatio_temporal.UNetSpatioTemporalConditionOutput`] or `tuple`:
                If `return_dict` is True, an [`~models.unet_slatio_temporal.UNetSpatioTemporalConditionOutput`] is
                returned, otherwise a `tuple` is returned where the first element is the sample tensor.
        """
        batch_size, num_frames = sample.shape[:2]

        if not self.pos_embedding_prepared:
            for res in self.pos_embedding:
                self.pos_embedding[res] = self.pos_embedding[res].to(drags)
            self.pos_embedding_prepared = True

        # 0. prepare for cfg
        drag_drop_ids = None
        if (self.training and self.cond_dropout_prob > 0) or force_drop_ids is not None:
            if force_drop_ids is None:
                drag_drop_ids = torch.rand(batch_size, device=sample.device) < self.cond_dropout_prob
            else:
                drag_drop_ids = (force_drop_ids == 1)
            drags = drags * ~drag_drop_ids[:, None, None, None]

        sample = torch.cat([sample, image_latents[:, None].repeat(1, num_frames, 1, 1, 1)], dim=2)
        # 1. time
        timesteps = timestep
        if not torch.is_tensor(timesteps):
            # TODO: this requires sync between CPU and GPU. So try to pass timesteps as tensors if you can
            # This would be a good case for the `match` statement (Python 3.10+)
            is_mps = sample.device.type == "mps"
            if isinstance(timestep, float):
                dtype = torch.float32 if is_mps else torch.float64
            else:
                dtype = torch.int32 if is_mps else torch.int64
            timesteps = torch.tensor([timesteps], dtype=dtype, device=sample.device)
        elif len(timesteps.shape) == 0:
            timesteps = timesteps[None].to(sample.device)

        # broadcast to batch dimension in a way that's compatible with ONNX/Core ML
        timesteps = timesteps.expand(batch_size)
        
        drag_encodings = {res: self._convert_drag_to_concatting_image(drags, res) for res in [32, 16, 8]}

        t_emb = self.time_proj(timesteps)

        # `Timesteps` does not contain any weights and will always return f32 tensors
        # but time_embedding might actually be running in fp16. so we need to cast here.
        # there might be better ways to encapsulate this.
        t_emb = t_emb.to(dtype=sample.dtype)
        emb = self.time_embedding(t_emb)

        # Flatten the batch and frames dimensions
        # sample: [batch, frames, channels, height, width] -> [batch * frames, channels, height, width]
        sample = sample.flatten(0, 1)
        # Repeat the embeddings num_video_frames times
        # emb: [batch, channels] -> [batch * frames, channels]
        emb = emb.repeat_interleave(num_frames, dim=0)
        # encoder_hidden_states: [batch, 1, channels] -> [batch * frames, 1, channels]
        encoder_hidden_states = encoder_hidden_states.repeat_interleave(num_frames, dim=0)

        # 2. pre-process
        sample = self.conv_in(sample)

        image_only_indicator = torch.zeros(batch_size, num_frames, dtype=sample.dtype, device=sample.device)

        down_block_res_samples = (sample,)
        for downsample_block in self.down_blocks:
            if hasattr(downsample_block, "has_cross_attention") and downsample_block.has_cross_attention:
                flow = drag_encodings[sample.shape[-1]]

                sample, res_samples = downsample_block(
                    hidden_states=sample,
                    temb=emb,
                    encoder_hidden_states=encoder_hidden_states,
                    image_only_indicator=image_only_indicator,
                    flow=flow.flatten(0, 1),
                    drag_original=drags.flatten(0, 1),
                )
            else:
                sample, res_samples = downsample_block(
                    hidden_states=sample,
                    temb=emb,
                    image_only_indicator=image_only_indicator,
                )

            down_block_res_samples += res_samples

        # 4. mid
        sample = self.mid_block(
            hidden_states=sample,
            temb=emb,
            encoder_hidden_states=encoder_hidden_states,
            image_only_indicator=image_only_indicator,
        )
        # 5. up
        for i, upsample_block in enumerate(self.up_blocks):
            res_samples = down_block_res_samples[-len(upsample_block.resnets) :]
            down_block_res_samples = down_block_res_samples[: -len(upsample_block.resnets)]
            if hasattr(upsample_block, "has_cross_attention") and upsample_block.has_cross_attention:
                flow = drag_encodings[sample.shape[-1]]
                sample = upsample_block(
                    hidden_states=sample,
                    temb=emb,
                    res_hidden_states_tuple=res_samples,
                    encoder_hidden_states=encoder_hidden_states,
                    image_only_indicator=image_only_indicator,
                    flow=flow.flatten(0, 1),
                    drag_original=drags.flatten(0, 1),
                )
            else:
                sample = upsample_block(
                    hidden_states=sample,
                    temb=emb,
                    res_hidden_states_tuple=res_samples,
                    image_only_indicator=image_only_indicator,
                )

        # 6. post-process
        sample = self.conv_norm_out(sample)
        sample = self.conv_act(sample)
        sample = self.conv_out(sample)

        # 7. Reshape back to original shape
        sample = sample.reshape(batch_size, num_frames, *sample.shape[1:])
        return sample


if __name__ == "__main__":
    puppet_master = UNetDragSpatioTemporalConditionModel(num_drags=5)

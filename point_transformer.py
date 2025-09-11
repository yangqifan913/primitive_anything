from functools import partial
from addict import Dict
import torch
import torch.nn as nn
from torch.nn.init import trunc_normal_
import ocnn
from utils.utils import offset2bincount, offset2batch, batch2offset
from collections import OrderedDict
from utils.serialization.default import encode
import sys


class Point(Dict):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # If one of "offset" or "batch" do not exist, generate by the existing one
        if "batch" not in self.keys() and "offset" in self.keys():
            self["batch"] = offset2batch(self.offset)
        elif "offset" not in self.keys() and "batch" in self.keys():
            self["offset"] = batch2offset(self.batch)

    def serialization(self, order="z", depth=None, shuffle_orders=False):
        """
        Point Cloud Serialization

        relay on ["grid_coord" or "coord" + "grid_size", "batch", "feat"]
        """
        self["order"] = order
        assert "batch" in self.keys()
        if "grid_coord" not in self.keys():
            # if you don't want to operate GridSampling in data augmentation,
            # please add the following augmentation into your pipline:
            # dict(type="Copy", keys_dict={"grid_size": 0.01}),
            # (adjust `grid_size` to what your want)
            assert {"grid_size", "coord"}.issubset(self.keys())

            self["grid_coord"] = torch.div(
                self.coord - self.coord.min(0)[0], self.grid_size, rounding_mode="trunc"
            ).int()

        if depth is None:
            # Adaptive measure the depth of serialization cube (length = 2 ^ depth)
            depth = int(self.grid_coord.max() + 1).bit_length()
        self["serialized_depth"] = depth
        # Maximum bit length for serialization code is 63 (int64)
        assert depth * 3 + len(self.offset).bit_length() <= 63
        # Here we follow OCNN and set the depth limitation to 16 (48bit) for the point position.
        # Although depth is limited to less than 16, we can encode a 655.36^3 (2^16 * 0.01) meter^3
        # cube with a grid size of 0.01 meter. We consider it is enough for the current stage.
        # We can unlock the limitation by optimizing the z-order encoding function if necessary.
        assert depth <= 16

        # The serialization codes are arranged as following structures:
        # [Order1 ([n]),
        #  Order2 ([n]),
        #   ...
        #  OrderN ([n])] (k, n)
        # print(f"grid_coord: {self.grid_coord.shape}, batch: {self.batch.shape}")
        code = [
            encode(self.grid_coord, self.batch, depth, order=order_) for order_ in order
        ]
        code = torch.stack(code)
        order = torch.argsort(code)
        inverse = torch.zeros_like(order).scatter_(
            dim=1,
            index=order,
            src=torch.arange(0, code.shape[1], device=order.device).repeat(
                code.shape[0], 1
            ),
        )

        if shuffle_orders:
            perm = torch.randperm(code.shape[0])
            code = code[perm]
            order = order[perm]
            inverse = inverse[perm]

        self["serialized_code"] = code
        self["serialized_order"] = order
        self["serialized_inverse"] = inverse

    def octreelization(self, depth=None, full_depth=None):
        """
        Point Cloud Octreelization

        Generate octree with OCNN
        relay on ["grid_coord", "batch", "feat"]
        """
        assert (
            ocnn is not None
        ), "Please follow https://github.com/octree-nn/ocnn-pytorch install ocnn."
        assert {"feat", "batch"}.issubset(self.keys())
        # add 1 to make grid space support shift order
        if "grid_coord" not in self.keys():
            # if you don't want to operate GridSampling in data augmentation,
            # please add the following augmentation into your pipline:
            # dict(type="Copy", keys_dict={"grid_size": 0.01}),
            # (adjust `grid_size` to what your want)
            assert {"grid_size", "coord"}.issubset(self.keys())
            self["grid_coord"] = torch.div(
                self.coord - self.coord.min(0)[0], self.grid_size, rounding_mode="trunc"
            ).int()
        if depth is None:
            if "depth" in self.keys():
                depth = self.depth
            else:
                depth = int(self.grid_coord.max() + 1).bit_length()
        if full_depth is None:
            full_depth = 1
        self["depth"] = depth
        assert depth <= 16  # maximum in ocnn

        # [0, 2**depth] -> [0, 2] -> [-1, 1]
        coord = self.grid_coord / 2 ** (self.depth - 1) - 1.0
        point = ocnn.octree.Points(
            points=coord,
            features=self.feat,
            batch_id=self.batch.unsqueeze(-1),
            batch_size=self.batch[-1] + 1,
        )
        octree = ocnn.octree.Octree(
            depth=depth,
            full_depth=full_depth,
            batch_size=self.batch[-1] + 1,
            device=coord.device,
        )
        octree.build_octree(point)
        octree.construct_all_neigh()

        query_pts = torch.cat([self.grid_coord, point.batch_id], dim=1).contiguous()
        inverse = octree.search_xyzb(query_pts, depth, True)
        assert torch.sum(inverse < 0) == 0  # all mapping should be valid
        inverse_ = torch.unique(inverse)
        order = torch.zeros_like(inverse_).scatter_(
            dim=0,
            index=inverse,
            src=torch.arange(0, inverse.shape[0], device=inverse.device),
        )
        self["octree"] = octree
        self["octree_order"] = order
        self["octree_inverse"] = inverse

class PointModule(nn.Module):
    r"""PointModule
    placeholder, all module subclass from this will take Point in PointSequential.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)


class PointSequential(PointModule):
    r"""A sequential container.
    Modules will be added to it in the order they are passed in the constructor.
    Alternatively, an ordered dict of modules can also be passed in.
    """

    def __init__(self, *args, **kwargs):
        super().__init__()
        if len(args) == 1 and isinstance(args[0], OrderedDict):
            for key, module in args[0].items():
                self.add_module(key, module)
        else:
            for idx, module in enumerate(args):
                self.add_module(str(idx), module)
        for name, module in kwargs.items():
            if sys.version_info < (3, 6):
                raise ValueError("kwargs only supported in py36+")
            if name in self._modules:
                raise ValueError("name exists.")
            self.add_module(name, module)

    def __getitem__(self, idx):
        if not (-len(self) <= idx < len(self)):
            raise IndexError("index {} is out of range".format(idx))
        if idx < 0:
            idx += len(self)
        it = iter(self._modules.values())
        for i in range(idx):
            next(it)
        return next(it)

    def __len__(self):
        return len(self._modules)

    def add(self, module, name=None):
        if name is None:
            name = str(len(self._modules))
            if name in self._modules:
                raise KeyError("name exists")
        self.add_module(name, module)

    def forward(self, input):
        for k, module in self._modules.items():
            # Point module
            if isinstance(module, PointModule):
                input = module(input)

            elif isinstance(module, ocnn.nn.OctreeConv):
                input.octree.features[-1] = module(
                    input.feat[input.octree_order],
                    input.octree,
                    input.depth)
                input.feat = input.octree.features[-1][input.octree_inverse]
            # PyTorch module
            else:
                if isinstance(input, Point):
                    input.feat = module(input.feat)
                    if "sparse_conv_feat" in input.keys():
                        input.sparse_conv_feat = input.sparse_conv_feat.replace_feature(
                            input.feat
                        )
                else:
                    input = module(input)
        return input
    
class PDNorm(PointModule):
    def __init__(
        self,
        num_features,
        norm_layer,
        context_channels=256,
        conditions=("ScanNet", "S3DIS", "Structured3D"),
        decouple=True,
        adaptive=False,
    ):
        super().__init__()
        self.conditions = conditions
        self.decouple = decouple
        self.adaptive = adaptive
        if self.decouple:
            self.norm = nn.ModuleList([norm_layer(num_features) for _ in conditions])
        else:
            self.norm = norm_layer
        if self.adaptive:
            self.modulation = nn.Sequential(
                nn.SiLU(), nn.Linear(context_channels, 2 * num_features, bias=True)
            )

    def forward(self, point):
        assert {"feat", "condition"}.issubset(point.keys())
        if isinstance(point.condition, str):
            condition = point.condition
        else:
            condition = point.condition[0]
        if self.decouple:
            assert condition in self.conditions
            norm = self.norm[self.conditions.index(condition)]
        else:
            norm = self.norm
        point.feat = norm(point.feat)
        if self.adaptive:
            assert "context" in point.keys()
            shift, scale = self.modulation(point.context).chunk(2, dim=1)
            point.feat = point.feat * (1.0 + scale) + shift
        return point
    
class RPE(torch.nn.Module):
    def __init__(self, patch_size, num_heads):
        super().__init__()
        self.patch_size = patch_size
        self.num_heads = num_heads
        self.pos_bnd = int((4 * patch_size) ** (1 / 3) * 2)
        self.rpe_num = 2 * self.pos_bnd + 1
        self.rpe_table = torch.nn.Parameter(torch.zeros(3 * self.rpe_num, num_heads))
        torch.nn.init.trunc_normal_(self.rpe_table, std=0.02)

    def forward(self, coord):
        idx = (
            coord.clamp(-self.pos_bnd, self.pos_bnd)  # clamp into bnd
            + self.pos_bnd  # relative position to positive index
            + torch.arange(3, device=coord.device) * self.rpe_num  # x, y, z stride
        )
        out = self.rpe_table.index_select(0, idx.reshape(-1))
        out = out.view(idx.shape + (-1,)).sum(3)
        out = out.permute(0, 3, 1, 2)  # (N, K, K, H) -> (N, H, K, K)
        return out


class SerializedAttention(PointModule):
    def __init__(
        self,
        channels,
        num_heads,
        patch_size,
        qkv_bias=True,
        qk_scale=None,
        attn_drop=0.0,
        proj_drop=0.0,
        order_index=0,
        enable_rpe=True,
    ):
        super().__init__()
        assert channels % num_heads == 0
        self.channels = channels
        self.num_heads = num_heads
        self.scale = qk_scale or (channels // num_heads) ** -0.5
        self.order_index = order_index
        self.enable_rpe = enable_rpe

        # when disable flash attention, we still don't want to use mask
        # consequently, patch size will auto set to the
        # min number of patch_size_max and number of points
        self.patch_size_max = patch_size
        self.patch_size = 0
        self.attn_drop = torch.nn.Dropout(attn_drop)

        self.qkv = torch.nn.Linear(channels, channels * 3, bias=qkv_bias)
        self.proj = torch.nn.Linear(channels, channels)
        self.proj_drop = torch.nn.Dropout(proj_drop)
        self.softmax = torch.nn.Softmax(dim=-1)
        self.rpe = RPE(patch_size, num_heads) if self.enable_rpe else None

    @torch.no_grad()
    def get_rel_pos(self, point, order):
        K = self.patch_size
        rel_pos_key = f"rel_pos_{self.order_index}"
        if rel_pos_key not in point.keys():
            grid_coord = point.grid_coord[order]
            grid_coord = grid_coord.reshape(-1, K, 3)
            point[rel_pos_key] = grid_coord.unsqueeze(2) - grid_coord.unsqueeze(1)
        return point[rel_pos_key]

    @torch.no_grad()
    def get_padding_and_inverse(self, point):
        pad_key = "pad"
        unpad_key = "unpad"
        cu_seqlens_key = "cu_seqlens_key"
        if (
            pad_key not in point.keys()
            or unpad_key not in point.keys()
            or cu_seqlens_key not in point.keys()
        ):
            offset = point.offset
            bincount = offset2bincount(offset)
            bincount_pad = (
                torch.div(
                    bincount + self.patch_size - 1,
                    self.patch_size,
                    rounding_mode="trunc",
                )
                * self.patch_size
            )
            # only pad point when num of points larger than patch_size
            mask_pad = bincount > self.patch_size
            bincount_pad = ~mask_pad * bincount + mask_pad * bincount_pad
            _offset = nn.functional.pad(offset, (1, 0))
            _offset_pad = nn.functional.pad(torch.cumsum(bincount_pad, dim=0), (1, 0))
            pad = torch.arange(_offset_pad[-1], device=offset.device)
            unpad = torch.arange(_offset[-1], device=offset.device)
            cu_seqlens = []
            for i in range(len(offset)):
                unpad[_offset[i] : _offset[i + 1]] += _offset_pad[i] - _offset[i]
                if bincount[i] != bincount_pad[i]:
                    pad[
                        _offset_pad[i + 1]
                        - self.patch_size
                        + (bincount[i] % self.patch_size) : _offset_pad[i + 1]
                    ] = pad[
                        _offset_pad[i + 1]
                        - 2 * self.patch_size
                        + (bincount[i] % self.patch_size) : _offset_pad[i + 1]
                        - self.patch_size
                    ]
                pad[_offset_pad[i] : _offset_pad[i + 1]] -= _offset_pad[i] - _offset[i]
                cu_seqlens.append(
                    torch.arange(
                        _offset_pad[i],
                        _offset_pad[i + 1],
                        step=self.patch_size,
                        dtype=torch.int32,
                        device=offset.device,
                    )
                )
            point[pad_key] = pad
            point[unpad_key] = unpad
            point[cu_seqlens_key] = nn.functional.pad(
                torch.concat(cu_seqlens), (0, 1), value=_offset_pad[-1]
            )
        return point[pad_key], point[unpad_key], point[cu_seqlens_key]

    def forward(self, point):
        self.patch_size = min(
            offset2bincount(point.offset).min().tolist(), self.patch_size_max
        )

        H = self.num_heads
        K = self.patch_size
        C = self.channels

        pad, unpad, cu_seqlens = self.get_padding_and_inverse(point)

        order = point.serialized_order[self.order_index][pad]
        inverse = unpad[point.serialized_inverse[self.order_index]]

        # padding and reshape feat and batch for serialized point patch
        qkv = self.qkv(point.feat)[order]

        # encode and reshape qkv: (N', K, 3, H, C') => (3, N', H, K, C')
        q, k, v = (
            qkv.reshape(-1, K, 3, H, C // H).permute(2, 0, 3, 1, 4).unbind(dim=0)
        )
        # attn
        attn = (q * self.scale) @ k.transpose(-2, -1)  # (N', H, K, K)
        if self.enable_rpe:
            attn = attn + self.rpe(self.get_rel_pos(point, order))
        attn = self.softmax(attn)
        attn = self.attn_drop(attn).to(qkv.dtype)
        feat = (attn @ v).transpose(1, 2).reshape(-1, C)
        feat = feat[inverse]

        # ffn
        feat = self.proj(feat)
        feat = self.proj_drop(feat)
        point.feat = feat
        return point


class MLP(nn.Module):
    def __init__(
        self,
        in_channels,
        hidden_channels=None,
        out_channels=None,
        act_layer=nn.GELU,
        drop=0.0,
    ):
        super().__init__()
        out_channels = out_channels or in_channels
        hidden_channels = hidden_channels or in_channels
        self.fc1 = nn.Linear(in_channels, hidden_channels)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_channels, out_channels)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x

class DropPath(nn.Module):
    def __init__(self, drop_prob=0.0):
        super().__init__()
        self.drop_prob = drop_prob
        self.keep_prob = 1 - drop_prob

    def forward(self, x):
        if not self.training or self.drop_prob == 0.0:
            return x
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        random_tensor = self.keep_prob + torch.rand(
            shape, dtype=x.dtype, device=x.device
        )
        random_tensor.floor_()  
        output = x.div(self.keep_prob) * random_tensor
        return output

class Block(PointModule):
    def __init__(
        self,
        channels,
        num_heads,
        patch_size=48,
        mlp_ratio=4.0,
        qkv_bias=True,
        qk_scale=None,
        attn_drop=0.0,
        proj_drop=0.0,
        drop_path=0.0,
        norm_layer=nn.LayerNorm,
        act_layer=nn.GELU,
        pre_norm=True,
        order_index=0,
        enable_rpe=True,
    ):
        super().__init__()
        self.channels = channels
        self.pre_norm = pre_norm

        self.cpe = PointSequential(
            ocnn.nn.OctreeConv(
                channels,
                channels,
                kernel_size=[3],
                stride=1,
                nempty=True,
                use_bias=True,
            ),
            nn.Linear(channels, channels),
            norm_layer(channels),
        )

        self.norm1 = PointSequential(norm_layer(channels))
        self.attn = SerializedAttention(
            channels=channels,
            patch_size=patch_size,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            qk_scale=qk_scale,
            attn_drop=attn_drop,
            proj_drop=proj_drop,
            order_index=order_index,
            enable_rpe=enable_rpe,
        )
        self.norm2 = PointSequential(norm_layer(channels))
        self.mlp = PointSequential(
            MLP(
                in_channels=channels,
                hidden_channels=int(channels * mlp_ratio),
                out_channels=channels,
                act_layer=act_layer,
                drop=proj_drop,
            )
        )
        self.drop_path = PointSequential(
            DropPath(drop_path) if drop_path > 0.0 else nn.Identity()
        )

    def forward(self, point: Point):
        shortcut = point.feat
        point = self.cpe(point)
        point.feat = shortcut + point.feat
        shortcut = point.feat
        if self.pre_norm:
            point = self.norm1(point)
        point = self.drop_path(self.attn(point))
        point.feat = shortcut + point.feat
        if not self.pre_norm:
            point = self.norm1(point)

        shortcut = point.feat
        if self.pre_norm:
            point = self.norm2(point)
        point = self.drop_path(self.mlp(point))
        point.feat = shortcut + point.feat
        if not self.pre_norm:
            point = self.norm2(point)
        point.octree.features[-1] = point.feat[point.octree_order]
        return point


class GridPooling(PointModule):
    def __init__(
        self,
        in_channels,
        out_channels,
        stride=2,
        norm_layer=None,
        act_layer=None,
        shuffle_orders=True,
        traceable=True,  # record parent and cluster
    ):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels

        self.stride = stride
        self.shuffle_orders = shuffle_orders
        self.traceable = traceable

        self.proj = nn.Linear(in_channels, out_channels)
        if norm_layer is not None:
            self.norm = PointSequential(norm_layer(out_channels))
        if act_layer is not None:
            self.act = PointSequential(act_layer())

    def segment_csr_pytorch(self, feat, idx_ptr, reduce):
        num_segments = len(idx_ptr) - 1 
        segment_id = torch.bucketize(torch.arange(len(feat), device=feat.device), idx_ptr, right=True) - 1
        if reduce == "mean":
            out = torch.zeros((num_segments, feat.shape[1]), dtype=feat.dtype, device=feat.device)
            out.scatter_reduce_(0, segment_id.unsqueeze(1).expand(-1, feat.shape[1]), feat, reduce='mean', include_self=False)
        elif reduce == "max":
            out = torch.full((num_segments, feat.shape[1]), float('-inf'), dtype=feat.dtype, device=feat.device)
            out.scatter_reduce_(0, segment_id.unsqueeze(1).expand(-1, feat.shape[1]), feat, reduce='amax')
        return out
    
    def forward(self, point: Point):
        if "grid_coord" in point.keys():
            grid_coord = point.grid_coord
        elif {"coord", "grid_size"}.issubset(point.keys()):
            grid_coord = torch.div(
                point.coord - point.coord.min(0)[0],
                point.grid_size,
                rounding_mode="trunc",
            ).int()
        else:
            raise AssertionError(
                "[gird_coord] or [coord, grid_size] should be include in the Point"
            )
        grid_coord = torch.div(grid_coord, self.stride, rounding_mode="trunc")
        grid_coord = grid_coord | point.batch.view(-1, 1) << 48
        grid_coord, cluster, counts = torch.unique(
            grid_coord,
            sorted=True,
            return_inverse=True,
            return_counts=True,
            dim=0,
        )
        grid_coord = grid_coord & ((1 << 48) - 1)
        # indices of point sorted by cluster, for torch_scatter.segment_csr
        _, indices = torch.sort(cluster)
        # index pointer for sorted point, for torch_scatter.segment_csr
        idx_ptr = torch.cat([counts.new_zeros(1), torch.cumsum(counts, dim=0)])
        # head_indices of each cluster, for reduce attr e.g. code, batch
        head_indices = indices[idx_ptr[:-1]]
        point_dict = Dict(
            feat = self.segment_csr_pytorch(self.proj(point.feat)[indices], idx_ptr, "max"),
            coord = self.segment_csr_pytorch(point.coord[indices], idx_ptr, "mean"),
            grid_coord=grid_coord,
            batch=point.batch[head_indices],
        )

        if self.traceable:
            point_dict["pooling_inverse"] = cluster
            point_dict["pooling_parent"] = point
        order = point.order
        point = Point(point_dict)
        if self.norm is not None:
            point = self.norm(point)
        if self.act is not None:
            point = self.act(point)
        point.serialization(order=order, shuffle_orders=self.shuffle_orders)
        point.octreelization()
        return point


class GridUnpooling(PointModule):
    def __init__(
        self,
        in_channels,
        skip_channels,
        out_channels,
        norm_layer=None,
        act_layer=None,
        traceable=False,  # record parent and cluster
    ):
        super().__init__()
        self.proj = PointSequential(nn.Linear(in_channels, out_channels))
        self.proj_skip = PointSequential(nn.Linear(skip_channels, out_channels))

        if norm_layer is not None:
            self.proj.add(norm_layer(out_channels))
            self.proj_skip.add(norm_layer(out_channels))

        if act_layer is not None:
            self.proj.add(act_layer())
            self.proj_skip.add(act_layer())

        self.traceable = traceable

    def forward(self, point):
        assert "pooling_parent" in point.keys()
        assert "pooling_inverse" in point.keys()
        parent = point.pop("pooling_parent")
        inverse = point.pop("pooling_inverse")
        feat = point.feat

        skip = Point(
            coord=parent.coord.clone(),
            feat=parent.feat.clone(),
            offset=parent.offset.clone(),
        )

        parent = self.proj_skip(parent)
        parent.feat = parent.feat + self.proj(point).feat[inverse]
        parent.octree.features[-1] = parent.feat[parent.octree_order]

        if self.traceable:
            point.feat = feat
            parent["unpooling_parent"] = point
            parent["unpooling_skip"] = skip
        return parent


class Embedding(PointModule):
    def __init__(
        self,
        in_channels,
        embed_channels,
        norm_layer=None,
        act_layer=None,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.embed_channels = embed_channels

        # self.stem = PointSequential(
        #     ocnn.nn.OctreeConv(
        #         in_channels,
        #         embed_channels,
        #         kernel_size=[3],
        #         stride=1,
        #         nempty=True,
        #         use_bias=True,)
        # )
        self.stem = PointSequential(linear=nn.Linear(in_channels, embed_channels))
        if norm_layer is not None:
            self.stem.add(norm_layer(embed_channels), name="norm")
        if act_layer is not None:
            self.stem.add(act_layer(), name="act")

    def forward(self, point: Point):
        point = self.stem(point)
        return point

class SegmentHead(nn.Module):
    def __init__(
        self,
        num_classes,
        backbone_out_channels,
    ):
        super().__init__()
        self.seg_head = (
            nn.Linear(backbone_out_channels, num_classes)
        )

    def forward(self, point):
        if isinstance(point, Point):
            feat = point.feat
        else:
            feat = point
        seg_logits = self.seg_head(feat)
        return seg_logits
        
class PointTransformerV3(PointModule):
    def __init__(
        self,
        in_channels=3,
        order=("z", "z-trans"),
        stride=(2, 2, 2, 2),
        enc_depths=(2, 2, 2, 6, 2),
        enc_channels=(32, 64, 128, 256, 512),
        enc_num_head=(2, 4, 8, 16, 32),
        enc_patch_size=(48, 48, 48, 48, 48),
        mlp_ratio=4,
        qkv_bias=True,
        qk_scale=None,
        attn_drop=0.0,
        proj_drop=0.0,
        drop_path=0.3,
        pre_norm=True,
        shuffle_orders=True,
        enable_rpe=True,
        traceable=True,
    ):
        super().__init__()
        self.num_stages = len(enc_depths)
        self.order = [order] if isinstance(order, str) else order
        self.shuffle_orders = shuffle_orders

        assert ocnn is not None, "Please install ocnn-pytorch with `pip install ocnn`"

        assert self.num_stages == len(stride) + 1
        assert self.num_stages == len(enc_depths)
        assert self.num_stages == len(enc_channels)
        assert self.num_stages == len(enc_num_head)
        assert self.num_stages == len(enc_patch_size)
        assert self.num_stages == len(dec_depths) + 1
        assert self.num_stages == len(dec_channels) + 1
        assert self.num_stages == len(dec_num_head) + 1
        assert self.num_stages == len(dec_patch_size) + 1


        bn_layer = partial(nn.BatchNorm1d, eps=1e-3, momentum=0.01)
        ln_layer = nn.LayerNorm
        # activation layers
        act_layer = nn.GELU

        self.embedding = Embedding(
            in_channels=in_channels,
            embed_channels=enc_channels[0],
            norm_layer=bn_layer,
            act_layer=act_layer,
        )

        # encoder
        enc_drop_path = [
            x.item() for x in torch.linspace(0, drop_path, sum(enc_depths))
        ]
        self.enc = PointSequential()
        for s in range(self.num_stages):
            enc_drop_path_ = enc_drop_path[
                sum(enc_depths[:s]) : sum(enc_depths[: s + 1])
            ]
            enc = PointSequential()
            if s > 0:
                enc.add(
                    GridPooling(
                        in_channels=enc_channels[s - 1],
                        out_channels=enc_channels[s],
                        stride=stride[s - 1],
                        norm_layer=bn_layer,
                        act_layer=act_layer,
                    ),
                    name="down",
                )
            for i in range(enc_depths[s]):
                enc.add(
                    Block(
                        channels=enc_channels[s],
                        num_heads=enc_num_head[s],
                        patch_size=enc_patch_size[s],
                        mlp_ratio=mlp_ratio,
                        qkv_bias=qkv_bias,
                        qk_scale=qk_scale,
                        attn_drop=attn_drop,
                        proj_drop=proj_drop,
                        drop_path=enc_drop_path_[i],
                        norm_layer=ln_layer,
                        act_layer=act_layer,
                        pre_norm=pre_norm,
                        order_index=i % len(self.order),
                        enable_rpe=enable_rpe,
                    ),
                    name=f"block{i}",
                )
            if len(enc) != 0:
                self.enc.add(module=enc, name=f"enc{s}")
        self.apply(self._init_weights)

    @staticmethod
    def _init_weights(module):
        if isinstance(module, nn.Linear):
            trunc_normal_(module.weight, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, ocnn.nn.OctreeConv):
            trunc_normal_(module.weights, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)

    def forward(self, data_dict):
        point = Point(data_dict)
        # print(f"point: {point.grid_coord.shape}, batch: {point.batch.shape}")
        point.serialization(order=self.order, shuffle_orders=self.shuffle_orders)
        point.octreelization()

        point = self.embedding(point)
        
        # 构建完整的映射链：原始点 -> Serialization -> Octreelization -> Pooling链
        # 注意：不同Block使用不同order，但只影响特征重排，不影响点的空间关系
        
        # 1. 首先应用serialization的逆映射（使用第一个order作为基准）
        # serialized_inverse.shape = (k, n)，k为order数量，n为点数
        serialized_inverse = point.serialized_inverse[0]  # [N_orig] -> [serialized_idx]
        # 2. 然后应用octreelization的逆映射  
        # octree_inverse.shape = (n,) 或 (1, n)
        octree_inverse = point.octree_inverse[0] if point.octree_inverse.dim() > 1 else point.octree_inverse
        # 3. 组合前两步：原始点 -> 八叉树化后的点
        orig2cur = octree_inverse[serialized_inverse]

        for stage in self.enc:            # enc0, enc1, ..., enc4
            point = stage(point)
            # 若该 stage 含有下采样（GridPooling），会带上 pooling_inverse
            if "pooling_inverse" in point:
                inv = point.pooling_inverse            # 形状：[前一层点数] -> [本层点索引]
                orig2cur = inv[orig2cur]

        # 此时 point 为 encoder 最终输出：
        # point.feat: [N_enc, C_enc]，C_enc=enc_channels[-1]
        # orig2cur: [N_orig]，把每个原始点映射到 encoder 点索引（多对一）
        return point, orig2cur
import torch
import torch.nn as nn
from models.mmformer.Transformer import mmTransformerModel, TransformerModel
from models.mmformer.PositionalEncoding import FixedPositionalEncoding,LearnedPositionalEncoding
from models.mmformer.Unet_skipconnection import Unet
import torch.nn.functional as F


class mmFormer(nn.Module):
    def __init__(
        self,
        img_dim,
        patch_dim,
        num_channels,
        num_classes,
        embedding_dim,
        num_heads,
        num_layers,
        hidden_dim,
        dropout_rate=0.0,
        attn_dropout_rate=0.0,
        conv_patch_representation=True,
        positional_encoding_type="learned",
    ):
        super(mmFormer, self).__init__()

        assert embedding_dim % num_heads == 0
        assert img_dim % patch_dim == 0

        self.img_dim = img_dim
        self.embedding_dim = embedding_dim
        self.num_heads = num_heads
        self.patch_dim = patch_dim
        self.num_channels = num_channels
        self.dropout_rate = dropout_rate
        self.attn_dropout_rate = attn_dropout_rate
        self.conv_patch_representation = conv_patch_representation

        self.num_patches = int((img_dim // patch_dim) ** 3)
        self.seq_length = self.num_patches
        self.flatten_dim = 128 * num_channels

        self.linear_encoding = nn.Linear(self.flatten_dim, self.embedding_dim)

        self.shadow_tokens = []
        self.position_encoding = []
        self.pe_dropout = []
        self.intra_transformer = []
        for i in range(self.num_channels):
            self.shadow_tokens.append(torch.zeros(1, self.seq_length, self.embedding_dim).cuda())
            # self.shadow_tokens.append(nn.Parameter(torch.zeros(1, 512, 512)).cuda())
            if positional_encoding_type == "learned":
                self.position_encoding.append(LearnedPositionalEncoding(
                    self.seq_length, self.embedding_dim, self.seq_length
                ))
            elif positional_encoding_type == "fixed":
                self.position_encoding.append(FixedPositionalEncoding(
                    self.embedding_dim,
                ))
            self.pe_dropout.append(nn.Dropout(p=self.dropout_rate))
            self.intra_transformer.append(mmTransformerModel(
                num_channels,
                embedding_dim,
                num_layers,
                num_heads,
                hidden_dim,
                self.dropout_rate,
                self.attn_dropout_rate,
                ))

        # self.shadow_tokens = nn.ParameterList(self.shadow_tokens)
        self.position_encoding = nn.ModuleList(self.position_encoding)
        self.pe_dropout = nn.ModuleList(self.pe_dropout)
        self.intra_transformer = nn.ModuleList(self.intra_transformer)

        self.inter_position_encoding = LearnedPositionalEncoding(self.seq_length*self.num_channels, self.embedding_dim, self.seq_length*self.num_channels)
        self.inter_pe_dropout = nn.Dropout(p=self.dropout_rate)
        self.fusion = nn.Sequential(nn.LayerNorm(self.seq_length*self.num_channels), nn.LeakyReLU(), nn.Linear(self.seq_length*self.num_channels, self.seq_length))
        
        self.inter_transformer = mmTransformerModel(
            num_channels,
            embedding_dim,
            num_layers,
            num_heads,
            hidden_dim,
            self.dropout_rate,
            self.attn_dropout_rate,
            )
        self.pre_head_ln = nn.InstanceNorm3d(embedding_dim)

        if self.conv_patch_representation:

            self.conv_x_list = []
            for i in range(self.num_channels):
                self.conv_x_list.append(nn.Conv3d(
                    256,
                    self.embedding_dim,
                    kernel_size=3,
                    stride=1,
                    padding=1
                ))
            self.conv_x_list = nn.ModuleList(self.conv_x_list)

        self.Unet_list = []
        self.bn_list = []
        self.relu_list = []
        for i in range(self.num_channels):
            self.Unet_list.append(Unet(in_channels=1, base_channels=16, num_classes=4))
            # self.bn_list.append(nn.BatchNorm3d(256))
            self.bn_list.append(nn.InstanceNorm3d(256))
            self.relu_list.append(nn.LeakyReLU(inplace=True))
        self.Unet_list = nn.ModuleList(self.Unet_list)
        self.bn_list = nn.ModuleList(self.bn_list)
        self.relu_list = nn.ModuleList(self.relu_list)

        self.decoder = Decoder(self.embedding_dim, num_classes)
        self.shared_decoder = Decoder(self.embedding_dim, num_classes)

    def encode(self, x, missing_modal):
        if self.conv_patch_representation:
            # combine embedding with conv patch distribution
            # x1_1, x2_1, x3_1, x4_1, x = self.Unet(x)
            # x = self.bn(x)
            # x = self.relu(x)
            # x = self.conv_x(x)
            # x = x.permute(0, 2, 3, 4, 1).contiguous()
            # x = x.view(x.size(0), -1, self.embedding_dim)

            x = list(torch.chunk(x, self.num_channels, dim=1))
            # combine embedding with conv patch distribution

            x1_1_list = []
            x2_1_list = []
            x3_1_list = []
            x4_1_list = []
            existing_modal = [x for x in [0, 1, 2, 3] if x not in missing_modal]
            if len(existing_modal) > 0:
                x1_1, x2_1, x3_1, x4_1, x[existing_modal[0]] = self.Unet_list[existing_modal[0]](x[existing_modal[0]])
                x1_1_list.append(x1_1)
                x2_1_list.append(x2_1)
                x3_1_list.append(x3_1)
                x4_1_list.append(x4_1)
            for i in range(1, len(existing_modal)):
                x1_1_tmp, x2_1_tmp, x3_1_tmp, x4_1_tmp, x[existing_modal[i]] = self.Unet_list[existing_modal[i]](x[existing_modal[i]])
                x1_1_list.append(x1_1_tmp)
                x2_1_list.append(x2_1_tmp)
                x3_1_list.append(x3_1_tmp)
                x4_1_list.append(x4_1_tmp)
                x1_1 = x1_1 + x1_1_tmp
                x2_1 = x2_1 + x2_1_tmp
                x3_1 = x3_1 + x3_1_tmp
                x4_1 = x4_1 + x4_1_tmp

            x_temp = []
            for i in existing_modal:
                x[i] = self.bn_list[i](x[i])
                x[i] = self.relu_list[i](x[i])
                x[i] = self.conv_x_list[i](x[i])
                x_temp.append(x[i])
                x[i] = x[i].permute(0, 2, 3, 4, 1).contiguous()
                x[i] = x[i].view(x[i].size(0), -1, self.embedding_dim)

        # else:
        #     x = self.Unet(x)
        #     x = self.bn(x)
        #     x = self.relu(x)
        #     x = (
        #         x.unfold(2, 2, 2)
        #         .unfold(3, 2, 2)
        #         .unfold(4, 2, 2)
        #         .contiguous()
        #     )
        #     x = x.view(x.size(0), x.size(1), -1, 8)
        #     x = x.permute(0, 2, 3, 1).contiguous()
        #     x = x.view(x.size(0), -1, self.flatten_dim)
        #     x = self.linear_encoding(x)

        # x = self.shadow_tokens
        for i in missing_modal:
            x[i] = self.shadow_tokens[i]
        # if missing_modal.int() < 4:
        #     x[missing_modal.int()] = self.shadow_tokens[missing_modal.int()].repeat(x[i].size(0), 1, 1)

        for i in existing_modal:
            x[i] = self.position_encoding[i](x[i])
            x[i] = self.pe_dropout[i](x[i])
            x[i] = self.intra_transformer[i](x[i])

        # apply transformer
        x = torch.cat(x, dim=1)
        x = self.inter_position_encoding(x)
        x = self.inter_pe_dropout(x)
        x = self.inter_transformer(x)

        x = x.transpose(1, 2)
        x = self.fusion(x)
        x = x.transpose(1, 2)
        # x = self.pre_head_ln(x)
        x = self._reshape_output(x)

        return x1_1, x2_1, x3_1, x4_1, x, x1_1_list, x2_1_list, x3_1_list, x4_1_list, x_temp

    def forward(self, x, missing_modal, auxillary_output_layers=[1, 2, 3, 4]):

        x1_1, x2_1, x3_1, x4_1, encoder_output, x1_1_list, x2_1_list, x3_1_list, x4_1_list, x_temp = self.encode(x, missing_modal)

        decoder_output = self.decoder(x1_1, x2_1, x3_1, x4_1, encoder_output)

        auxillary_outputs = []
        for i in range(len(x_temp)):
            auxillary_outputs.append(self.shared_decoder(x1_1_list[i], x2_1_list[i], x3_1_list[i], x4_1_list[i], x_temp[i])[0])
        for i in decoder_output[1:]:
            auxillary_outputs.append(i)

        return decoder_output[0], auxillary_outputs

    def _get_padding(self, padding_type, kernel_size):
        assert padding_type in ['SAME', 'VALID']
        if padding_type == 'SAME':
            _list = [(k - 1) // 2 for k in kernel_size]
            return tuple(_list)
        return tuple(0 for _ in kernel_size)

    def _reshape_output(self, x):
        x = x.view(
            x.size(0),
            int(self.img_dim / self.patch_dim),
            int(self.img_dim / self.patch_dim),
            int(self.img_dim / self.patch_dim),
            self.embedding_dim,
        )
        x = x.permute(0, 4, 1, 2, 3).contiguous()

        return x


class Decoder(nn.Module):
    def __init__(self, embedding_dim, num_classes):
        super(Decoder, self).__init__()

        self.embedding_dim = embedding_dim
        self.num_classes = num_classes

        self.Softmax = nn.Softmax(dim=1)

        self.Enblock8_1 = EnBlock1(in_channels=self.embedding_dim)
        self.Enblock8_2 = EnBlock2(in_channels=self.embedding_dim // 2)

        self.DeUp5 = DeUp_Cat(in_channels=self.embedding_dim // 2, out_channels=self.embedding_dim // 4)
        self.DeBlock5 = DeBlock(in_channels=self.embedding_dim // 4)
        self.endconv5 = nn.Conv3d(self.embedding_dim // 4, self.num_classes, kernel_size=3, padding=1)

        self.DeUp4 = DeUp_Cat(in_channels=self.embedding_dim // 4, out_channels=self.embedding_dim // 8)
        self.DeBlock4 = DeBlock(in_channels=self.embedding_dim // 8)
        self.endconv4 = nn.Conv3d(self.embedding_dim // 8, self.num_classes, kernel_size=3, padding=1)

        self.DeUp3 = DeUp_Cat(in_channels=self.embedding_dim // 8, out_channels=self.embedding_dim // 16)
        self.DeBlock3 = DeBlock(in_channels=self.embedding_dim // 16)
        self.endconv3 = nn.Conv3d(self.embedding_dim // 16, self.num_classes, kernel_size=3, padding=1)

        self.DeUp2 = DeUp_Cat(in_channels=self.embedding_dim // 16, out_channels=self.embedding_dim // 32)
        self.DeBlock2 = DeBlock(in_channels=self.embedding_dim // 32)

        self.endconv = nn.Conv3d(self.embedding_dim // 32, self.num_classes, kernel_size=1)

    def forward(self, x1_1, x2_1, x3_1, x4_1, x):
        # x8 = encoder_outputs[all_keys[0]]
        x8 = x
        x8 = self.Enblock8_1(x8)
        x8 = self.Enblock8_2(x8)

        y5 = self.DeUp5(x8, x4_1)  # (1, 128, 16, 16, 16)
        y5 = self.DeBlock5(y5)
        y5_tmp = self.endconv5(y5)
        y5_tmp = F.interpolate(y5_tmp, scale_factor=8, mode='trilinear', align_corners=False)
        y5_tmp = self.Softmax(y5_tmp)

        y4 = self.DeUp4(y5, x3_1)  # (1, 64, 32, 32, 32)
        y4 = self.DeBlock4(y4)
        y4_tmp = self.endconv4(y4)
        y4_tmp = F.interpolate(y4_tmp, scale_factor=4, mode='trilinear', align_corners=False)
        y4_tmp = self.Softmax(y4_tmp)

        y3 = self.DeUp3(y4, x2_1)  # (1, 32, 64, 64, 64)
        y3 = self.DeBlock3(y3)
        y3_tmp = self.endconv3(y3)
        y3_tmp = F.interpolate(y3_tmp, scale_factor=2, mode='trilinear', align_corners=False)
        y3_tmp = self.Softmax(y3_tmp)

        y2 = self.DeUp2(y3, x1_1)  # (1, 16, 128, 128, 128)
        y2 = self.DeBlock2(y2)

        y = self.endconv(y2)  # (1, 4, 128, 128, 128)
        y = self.Softmax(y)
        return y, y3_tmp, y4_tmp, y5_tmp




# class BTS(TransformerBTS):
#     def __init__(
#         self,
#         img_dim,
#         patch_dim,
#         num_channels,
#         num_classes,
#         embedding_dim,
#         num_heads,
#         num_layers,
#         hidden_dim,
#         dropout_rate=0.0,
#         attn_dropout_rate=0.0,
#         conv_patch_representation=True,
#         positional_encoding_type="learned",
#     ):
#         super(BTS, self).__init__(
#             img_dim=img_dim,
#             patch_dim=patch_dim,
#             num_channels=num_channels,
#             embedding_dim=embedding_dim,
#             num_heads=num_heads,
#             num_layers=num_layers,
#             hidden_dim=hidden_dim,
#             dropout_rate=dropout_rate,
#             attn_dropout_rate=attn_dropout_rate,
#             conv_patch_representation=conv_patch_representation,
#             positional_encoding_type=positional_encoding_type,
#         )
#
#         self.num_classes = num_classes
#
#         self.Softmax = nn.Softmax(dim=1)
#
#         self.Enblock8_1 = EnBlock1(in_channels=self.embedding_dim)
#         self.Enblock8_2 = EnBlock2(in_channels=self.embedding_dim // 2)
#
#         self.DeUp5 = DeUp_Cat(in_channels=self.embedding_dim//2, out_channels=self.embedding_dim//4)
#         self.DeBlock5 = DeBlock(in_channels=self.embedding_dim//4)
#
#         self.DeUp4 = DeUp_Cat(in_channels=self.embedding_dim//4, out_channels=self.embedding_dim//8)
#         self.DeBlock4 = DeBlock(in_channels=self.embedding_dim//8)
#
#         self.DeUp3 = DeUp_Cat(in_channels=self.embedding_dim//8, out_channels=self.embedding_dim//16)
#         self.DeBlock3 = DeBlock(in_channels=self.embedding_dim//16)
#
#         self.DeUp2 = DeUp_Cat(in_channels=self.embedding_dim//16, out_channels=self.embedding_dim//32)
#         self.DeBlock2 = DeBlock(in_channels=self.embedding_dim//32)
#
#         self.endconv = nn.Conv3d(self.embedding_dim // 32, 4, kernel_size=1)
#
#
#     def decode(self, x1_1, x2_1, x3_1, x4_1, x, intmd_layers=[1, 2, 3, 4]):
#
#         # assert intmd_layers is not None, "pass the intermediate layers for MLA"
#         # encoder_outputs = {}
#         # all_keys = []
#         # for i in intmd_layers:
#         #     val = str(2 * i - 1)
#         #     _key = 'Z' + str(i)
#         #     all_keys.append(_key)
#         #     encoder_outputs[_key] = intmd_x[val]
#         # all_keys.reverse()
#
#         # x8 = encoder_outputs[all_keys[0]]
#         x8 = x
#         x8 = self._reshape_output(x8)
#         x8 = self.Enblock8_1(x8)
#         x8 = self.Enblock8_2(x8)
#
#         y5 = self.DeUp5(x8, x4_1)  # (1, 128, 16, 16, 16)
#         y5 = self.DeBlock5(y5)
#
#         y4 = self.DeUp4(y5, x3_1)  # (1, 64, 32, 32, 32)
#         y4 = self.DeBlock4(y4)
#
#         y3 = self.DeUp3(y4, x2_1)  # (1, 32, 64, 64, 64)
#         y3 = self.DeBlock3(y3)
#
#         y2 = self.DeUp2(y3, x1_1)  # (1, 16, 128, 128, 128)
#         y2 = self.DeBlock2(y2)
#
#         y = self.endconv(y2)      # (1, 4, 128, 128, 128)
#         y = self.Softmax(y)
#         return y

class EnBlock1(nn.Module):
    def __init__(self, in_channels):
        super(EnBlock1, self).__init__()

        # self.bn1 = nn.BatchNorm3d(in_channels // 2)
        self.bn1 = nn.InstanceNorm3d(in_channels // 2)
        self.relu1 = nn.LeakyReLU(inplace=True)
        # self.bn2 = nn.BatchNorm3d(in_channels // 2)
        self.bn2 = nn.InstanceNorm3d(in_channels // 2)
        self.relu2 = nn.LeakyReLU(inplace=True)
        self.conv1 = nn.Conv3d(in_channels, in_channels // 2, kernel_size=3, padding=1)
        self.conv2 = nn.Conv3d(in_channels // 2, in_channels // 2, kernel_size=3, padding=1)

    def forward(self, x):
        x1 = self.conv1(x)
        x1 = self.bn1(x1)
        x1 = self.relu1(x1)
        x1 = self.conv2(x1)
        x1 = self.bn2(x1)
        x1 = self.relu2(x1)

        return x1


class EnBlock2(nn.Module):
    def __init__(self, in_channels):
        super(EnBlock2, self).__init__()

        self.conv1 = nn.Conv3d(in_channels, in_channels, kernel_size=3, padding=1)
        # self.bn1 = nn.BatchNorm3d(in_channels)
        self.bn1 = nn.InstanceNorm3d(in_channels)
        self.relu1 = nn.LeakyReLU(inplace=True)
        # self.bn2 = nn.BatchNorm3d(in_channels)
        self.bn2 = nn.InstanceNorm3d(in_channels)
        self.relu2 = nn.LeakyReLU(inplace=True)
        self.conv2 = nn.Conv3d(in_channels, in_channels, kernel_size=3, padding=1)

    def forward(self, x):
        x1 = self.conv1(x)
        x1 = self.bn1(x1)
        x1 = self.relu1(x1)
        x1 = self.conv2(x1)
        x1 = self.bn2(x1)
        x1 = self.relu2(x1)
        x1 = x1 + x

        return x1


class DeUp_Cat(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(DeUp_Cat, self).__init__()
        self.conv1 = nn.Conv3d(in_channels, out_channels, kernel_size=1)
        self.conv2 = nn.ConvTranspose3d(out_channels, out_channels, kernel_size=2, stride=2)
        self.conv3 = nn.Conv3d(out_channels*2, out_channels, kernel_size=1)

    def forward(self, x, prev):
        x1 = self.conv1(x)
        y = self.conv2(x1)
        # y = y + prev
        y = torch.cat((prev, y), dim=1)
        y = self.conv3(y)
        return y

class DeBlock(nn.Module):
    def __init__(self, in_channels):
        super(DeBlock, self).__init__()

        # self.bn1 = nn.BatchNorm3d(in_channels)
        self.bn1 = nn.InstanceNorm3d(in_channels)
        self.relu1 = nn.LeakyReLU(inplace=True)
        self.conv1 = nn.Conv3d(in_channels, in_channels, kernel_size=3, padding=1)
        self.conv2 = nn.Conv3d(in_channels, in_channels, kernel_size=3, padding=1)
        # self.bn2 = nn.BatchNorm3d(in_channels)
        self.bn2 = nn.InstanceNorm3d(in_channels)
        self.relu2 = nn.LeakyReLU(inplace=True)

    def forward(self, x):
        x1 = self.conv1(x)
        x1 = self.bn1(x1)
        x1 = self.relu1(x1)
        x1 = self.conv2(x1)
        x1 = self.bn2(x1)
        x1 = self.relu2(x1)
        x1 = x1 + x

        return x1


def get_mmFormer(dataset='brats', _conv_repr=True, _pe_type="learned"):

    if dataset.lower() == 'brats':
        img_dim = 128
        num_classes = 4

    num_channels = 4
    patch_dim = 16
    aux_layers = [1, 2, 3, 4]
    model = mmFormer(
        img_dim,
        patch_dim,
        num_channels,
        num_classes,
        embedding_dim=512,
        num_heads=8,
        num_layers=1,
        hidden_dim=4096,
        dropout_rate=0.1,
        attn_dropout_rate=0.1,
        conv_patch_representation=_conv_repr,
        positional_encoding_type=_pe_type,
    )

    return aux_layers, model

import torch
import torch.nn as nn

def normalization(planes, norm='bn'):
    if norm == 'bn':
        m = nn.BatchNorm3d(planes)
    elif norm == 'gn':
        m = nn.GroupNorm(4, planes)
    elif norm == 'in':
        m = nn.InstanceNorm3d(planes)
    elif norm == 'sync_bn':
        m = SynchronizedBatchNorm3d(planes)
    else:
        raise ValueError('normalization type {} is not supported'.format(norm))
    return m

class general_conv1d(nn.Module):
    def __init__(self, in_ch, out_ch, k_size=3, stride=1, padding=1, pad_type='zeros', norm='in', is_training=True, act_type='lrelu', relufactor=0.2):
        super(general_conv1d, self).__init__()
        self.conv = nn.Conv1d(in_channels=in_ch, out_channels=out_ch, kernel_size=k_size, stride=stride, padding=padding, padding_mode=pad_type, bias=True)

    def forward(self, x):
        x = self.conv(x)
        return x

class general_conv3d_prenorm(nn.Module):
    def __init__(self, in_ch, out_ch, k_size=3, stride=1, padding=1, pad_type='zeros', norm='in', is_training=True, act_type='lrelu', relufactor=0.2):
        super(general_conv3d_prenorm, self).__init__()
        self.conv = nn.Conv3d(in_channels=in_ch, out_channels=out_ch, kernel_size=k_size, stride=stride, padding=padding, padding_mode=pad_type, bias=True)

        self.norm = normalization(out_ch, norm=norm)
        if act_type == 'relu':
            self.activation = nn.ReLU(inplace=True)
        elif act_type == 'lrelu':
            self.activation = nn.LeakyReLU(negative_slope=relufactor, inplace=True)


    def forward(self, x):
        x = self.norm(x)
        x = self.activation(x)
        x = self.conv(x)
        return x

class general_conv3d(nn.Module):
    def __init__(self, in_ch, out_ch, k_size=3, stride=1, padding=1, pad_type='zeros', norm='in', is_training=True, act_type='lrelu', relufactor=0.2):
        super(general_conv3d, self).__init__()
        self.conv = nn.Conv3d(in_channels=in_ch, out_channels=out_ch, kernel_size=k_size, stride=stride, padding=padding, padding_mode=pad_type, bias=True)

        self.norm = normalization(out_ch, norm=norm)
        if act_type == 'relu':
            self.activation = nn.ReLU(inplace=True)
        elif act_type == 'lrelu':
            self.activation = nn.LeakyReLU(negative_slope=relufactor, inplace=True)


    def forward(self, x):
        x = self.conv(x)
        x = self.norm(x)
        x = self.activation(x)
        return x

class prm_generator_laststage(nn.Module):
    def __init__(self, in_channel=64, norm='in', num_cls=4):
        super(prm_generator_laststage, self).__init__()

        self.embedding_layer = nn.Sequential(
                            general_conv3d(in_channel*4, int(in_channel//4), k_size=1, padding=0, stride=1),
                            general_conv3d(int(in_channel//4), int(in_channel//4), k_size=3, padding=1, stride=1),
                            general_conv3d(int(in_channel//4), in_channel, k_size=1, padding=0, stride=1))

        self.prm_layer = nn.Sequential(
                            general_conv3d(in_channel, 16, k_size=1, stride=1, padding=0),
                            nn.Conv3d(16, num_cls, kernel_size=1, padding=0, stride=1, bias=True),
                            nn.Softmax(dim=1))

    def forward(self, x):
        seg = self.prm_layer(self.embedding_layer(x))
        return seg

class prm_generator(nn.Module):
    def __init__(self, in_channel=64, norm='in', num_cls=4):
        super(prm_generator, self).__init__()

        self.embedding_layer = nn.Sequential(
                            general_conv3d(in_channel*4, int(in_channel//4), k_size=1, padding=0, stride=1),
                            general_conv3d(int(in_channel//4), int(in_channel//4), k_size=3, padding=1, stride=1),
                            general_conv3d(int(in_channel//4), in_channel, k_size=1, padding=0, stride=1))


        self.prm_layer = nn.Sequential(
                            general_conv3d(in_channel*2, 16, k_size=1, stride=1, padding=0),
                            nn.Conv3d(16, num_cls, kernel_size=1, padding=0, stride=1, bias=True),
                            nn.Softmax(dim=1))

    def forward(self, x1, x2):
        seg = self.prm_layer(torch.cat((x1, self.embedding_layer(x2)), dim=1))
        return seg

####modal fusion in each region
class modal_fusion(nn.Module):
    def __init__(self, in_channel=64):
        super(modal_fusion, self).__init__()
        self.weight_layer = nn.Sequential(
                            nn.Conv3d(4*in_channel+1, 128, 1, padding=0, bias=True),
                            nn.LeakyReLU(negative_slope=0.2, inplace=True),
                            nn.Conv3d(128, 4, 1, padding=0, bias=True))
        self.sigmoid = nn.Sigmoid()

    def forward(self, x, prm, region_name):
        B, K, C, H, W, Z = x.size()

        prm_avg = torch.mean(prm, dim=(3,4,5), keepdim=False) + 1e-7
        feat_avg = torch.mean(x, dim=(3,4,5), keepdim=False) / prm_avg

        feat_avg = feat_avg.view(B, K*C, 1, 1, 1)
        feat_avg = torch.cat((feat_avg, prm_avg[:, 0, 0, ...].view(B, 1, 1, 1, 1)), dim=1)
        weight = torch.reshape(self.weight_layer(feat_avg), (B, K, 1))
        weight = self.sigmoid(weight).view(B, K, 1, 1, 1, 1)

        ###we find directly using weighted sum still achieve competing performance
        region_feat = torch.sum(x * weight, dim=1)
        return region_feat

###fuse region feature
class region_fusion_laststage(nn.Module):
    def __init__(self, in_channel=64, num_cls=4):
        super(region_fusion_laststage, self).__init__()
        self.fusion_layer = nn.Sequential(
                        general_conv3d(in_channel*num_cls, in_channel, k_size=1, padding=0, stride=1),
                        general_conv3d(in_channel, in_channel, k_size=3, padding=1, stride=1),
                        general_conv3d(in_channel, in_channel, k_size=1, padding=0, stride=1))

    def forward(self, x):
        B, _, _, H, W, Z = x.size()
        x = torch.reshape(x, (B, -1, H, W, Z))
        return self.fusion_layer(x)

class region_fusion(nn.Module):
    def __init__(self, in_channel=64, num_cls=4):
        super(region_fusion, self).__init__()
        self.fusion_layer = nn.Sequential(
                        general_conv3d(in_channel*num_cls, in_channel, k_size=1, padding=0, stride=1),
                        general_conv3d(in_channel, in_channel, k_size=3, padding=1, stride=1),
                        # general_conv3d(in_channel, in_channel, k_size=1, padding=0, stride=1)
                        )

    def forward(self, x):
        return self.fusion_layer(x)

class fusion_prenorm(nn.Module):
    def __init__(self, in_channel=64, num_cls=4):
        super(fusion_prenorm, self).__init__()
        self.fusion_layer = nn.Sequential(
                        general_conv3d_prenorm(in_channel*num_cls, in_channel, k_size=1, padding=0, stride=1),
                        general_conv3d_prenorm(in_channel, in_channel, k_size=3, padding=1, stride=1),
                        general_conv3d_prenorm(in_channel, in_channel, k_size=1, padding=0, stride=1))

    def forward(self, x):
        return self.fusion_layer(x)

class region_aware_modal_fusion(nn.Module):
    def __init__(self, in_channel=64, norm='in', num_cls=4):
        super(region_aware_modal_fusion, self).__init__()
        self.num_cls = num_cls

        self.modal_fusion = nn.ModuleList([modal_fusion(in_channel=in_channel) for i in range(num_cls)])
        self.region_fusion = region_fusion(in_channel=in_channel, num_cls=num_cls)
        self.short_cut = nn.Sequential(
                        general_conv3d(in_channel*4, in_channel, k_size=1, padding=0, stride=1),
                        general_conv3d(in_channel, in_channel, k_size=3, padding=1, stride=1),
                        general_conv3d(in_channel, in_channel//2, k_size=1, padding=0, stride=1))

        self.clsname_list = ['BG', 'NCR/NET', 'ED', 'ET'] ##BRATS2020 and BRATS2018
        self.clsname_list = ['BG', 'NCR', 'ED', 'NET', 'ET'] ##BRATS2015

    def forward(self, x, prm):
        B, _, H, W, Z = x.size()
        y = x.view(B, 4, -1, H, W, Z)
        B, K, C, H, W, Z = y.size()

        prm = torch.unsqueeze(prm, 2).repeat(1, 1, C, 1, 1, 1)
        ###divide modal features into different regions
        flair = y[:, 0:1, ...] * prm
        t1ce = y[:, 1:2, ...] * prm
        t1 = y[:, 2:3, ...] * prm
        t2 = y[:, 3:4, ...] * prm

        modal_feat = torch.stack((flair, t1ce, t1, t2), dim=1)
        region_feat = [modal_feat[:, :, i, :, :] for i in range(self.num_cls)]

        ###modal fusion in each region
        region_fused_feat = []
        for i in range(self.num_cls):
            region_fused_feat.append(self.modal_fusion[i](region_feat[i], prm[:, i:i+1, ...], self.clsname_list[i]))
        region_fused_feat = torch.stack(region_fused_feat, dim=1)
        '''
        region_fused_feat = torch.stack((self.modal_fusion[0](region_feat[0], prm[:, 0:1, ...], 'BG'),
                                         self.modal_fusion[1](region_feat[1], prm[:, 1:2, ...], 'NCR/NET'),
                                         self.modal_fusion[2](region_feat[2], prm[:, 2:3, ...], 'ED'),
                                         self.modal_fusion[3](region_feat[3], prm[:, 3:4, ...], 'ET')), dim=1)
        '''

        ###gain final feat with a short cut
        final_feat = torch.cat((self.region_fusion(region_fused_feat), self.short_cut(y.view(B, -1, H, W, Z))), dim=1)
        return final_feat

import torch
import torch.nn as nn
import torchvision.models as models
import torch.nn.functional as F
from model.networks.resample2d_package.resample2d import Resample2d
from model.networks.block_extractor.block_extractor   import BlockExtractor
from model.networks.local_attn_reshape.local_attn_reshape   import LocalAttnReshape
from model.networks.base_network import BaseNetwork
from model.networks.base_function import get_nonlinearity_layer, get_norm_layer, coord_conv, ResBlocks

from util import util
import numpy as np


class MultiAffineRegularizationLoss(nn.Module):
    def __init__(self, kz_dic):
        super(MultiAffineRegularizationLoss, self).__init__()
        self.kz_dic=kz_dic
        self.method_dic={}
        for key in kz_dic:
            instance = AffineRegularizationLoss(kz_dic[key])
            self.method_dic[key] = instance
        self.layers = sorted(kz_dic, reverse=True) 
 
    def __call__(self, flow_fields):
        loss=0
        for i in range(len(flow_fields)):
            method = self.method_dic[self.layers[i]]
            loss += method(flow_fields[i])
        return loss



class AffineRegularizationLoss(nn.Module):
    """docstring for AffineRegularizationLoss"""
    # kernel_size: kz
    def __init__(self, kz):
        super(AffineRegularizationLoss, self).__init__()
        self.kz = kz
        self.criterion = torch.nn.L1Loss()
        self.extractor = BlockExtractor(kernel_size=kz)
        self.reshape = LocalAttnReshape()

        temp = np.arange(kz)
        A = np.ones([kz*kz, 3])
        A[:, 0] = temp.repeat(kz)
        A[:, 1] = temp.repeat(kz).reshape((kz,kz)).transpose().reshape(kz**2)
        AH = A.transpose()
        k = np.dot(A, np.dot(np.linalg.inv(np.dot(AH, A)), AH)) - np.identity(kz**2) #K = (A((AH A)^-1)AH - I)
        self.kernel = np.dot(k.transpose(), k)
        self.kernel = torch.from_numpy(self.kernel).unsqueeze(1).view(kz**2, kz, kz).unsqueeze(1)

    def __call__(self, flow_fields):
        grid = self.flow2grid(flow_fields)

        grid_x = grid[:,0,:,:].unsqueeze(1)
        grid_y = grid[:,1,:,:].unsqueeze(1)
        weights = self.kernel.type_as(flow_fields)
        loss_x = self.calculate_loss(grid_x, weights)
        loss_y = self.calculate_loss(grid_y, weights)
        return loss_x+loss_y


    def calculate_loss(self, grid, weights):
        results = nn.functional.conv2d(grid, weights)   # KH K B [b, kz*kz, w, h]
        b, c, h, w = results.size()
        kernels_new = self.reshape(results, self.kz)
        f = torch.zeros(b, 2, h, w).type_as(kernels_new) + float(int(self.kz/2))
        grid_H = self.extractor(grid, f)
        result = torch.nn.functional.avg_pool2d(grid_H*kernels_new, self.kz, self.kz)
        loss = torch.mean(result)*self.kz**2
        return loss

    def flow2grid(self, flow_field):
        b,c,h,w = flow_field.size()
        x = torch.arange(w).view(1, -1).expand(h, -1).type_as(flow_field).float() 
        y = torch.arange(h).view(-1, 1).expand(-1, w).type_as(flow_field).float()
        grid = torch.stack([x,y], dim=0)
        grid = grid.unsqueeze(0).expand(b, -1, -1, -1)
        return flow_field+grid


        

class AdversarialLoss(nn.Module):
    r"""
    Adversarial loss
    https://arxiv.org/abs/1711.10337
    """

    def __init__(self, type='nsgan', target_real_label=1.0, target_fake_label=0.0):
        r"""
        type = nsgan | lsgan | hinge
        """
        super(AdversarialLoss, self).__init__()

        self.type = type
        self.register_buffer('real_label', torch.tensor(target_real_label))
        self.register_buffer('fake_label', torch.tensor(target_fake_label))

        if type == 'nsgan':
            self.criterion = nn.BCELoss()

        elif type == 'lsgan':
            self.criterion = nn.MSELoss()

        elif type == 'hinge':
            self.criterion = nn.ReLU()

    def __call__(self, outputs, is_real, for_dis=None):
        if self.type == 'hinge':
            if for_dis:
                if is_real:
                    outputs = -outputs
                return self.criterion(1 + outputs).mean()
            else:
                return (-outputs).mean()

        else:
            labels = (self.real_label if is_real else self.fake_label).expand_as(outputs)
            loss = self.criterion(outputs, labels)
            return loss


class FeatureMappingsLoss(nn.Module):
    def __init__(self, weights=[1.0, 1.0, 1.0, 1.0, 1.0, 1.0]):
        r"""
        type = nsgan | lsgan | hinge
        """
        super(FeatureMappingsLoss, self).__init__()
        self.criterion = torch.nn.L1Loss()
        self.weights = weights
    
    def forward(self, layers1, layers2):
        loss = 0
        for i in range(len(layers1)):
            b,n,h,w = layers1[i].shape
            loss +=  self.weights[i] * self.criterion(layers1[i], layers2[i])
        return loss


class PatchDiscriminatorWithLayers(BaseNetwork):
    """
    Patch Discriminator Network for Local 70*70 fake/real
    :param input_nc: number of channels in input
    :param ndf: base filter channel
    :param img_f: the largest channel for the model
    :param layers: down sample layers
    :param norm: normalization function 'instance, batch, group'
    :param activation: activation function 'ReLU, SELU, LeakyReLU, PReLU'
    :param use_spect: use spectral normalization or not
    :param use_coord: use CoordConv or nor
    """
    def __init__(self, input_nc=3, ndf=64, img_f=512, layers=3, n_res=1, multi_dim_out = False,
                norm='batch', activation='LeakyReLU', use_spect=True,
                use_coord=False):
        super(PatchDiscriminatorWithLayers, self).__init__()
        self.layers = layers
        self.n_res =n_res
        self.multi_dim_out = multi_dim_out
        norm_layer = get_norm_layer(norm_type=norm)
        nonlinearity = get_nonlinearity_layer(activation_type=activation)

        kwargs = {'kernel_size': 3, 'stride': 2, 'padding': 1, 'bias': False}

        sequence = []
        if norm_layer is not None:
            sequence +=[norm_layer(input_nc)]
        sequence += [
            coord_conv(input_nc, ndf, use_spect, use_coord, **kwargs),
            nonlinearity,
        ]
        setattr(self,'enc0',nn.Sequential(*sequence))
        mult = 1
        for i in range(1, layers):
            sequence = []
            mult_prev = mult
            mult = min(2 ** i, img_f // ndf)
            if norm_layer is not None:
                sequence +=[norm_layer(ndf * mult_prev)]
            sequence +=[
                coord_conv(ndf * mult_prev, ndf * mult, use_spect, use_coord, **kwargs),
                nonlinearity,
            ]
            setattr(self, 'enc'+str(i),nn.Sequential(*sequence))
        
        if self.n_res>0:
            setattr(self, 'res',ResBlocks(self.n_res,ndf * mult,norm_layer=norm_layer,
                nonlinearity=nonlinearity,use_spect=use_spect,use_coord=use_coord))

        sequence=[]
        mult_prev = mult
        mult = min(2 ** i, img_f // ndf)
        kwargs = {'kernel_size': 3, 'stride': 1, 'padding': 1, 'bias': False}
        if norm_layer is not None:
            sequence +=[norm_layer(ndf * mult_prev)]
        sequence += [
            coord_conv(ndf * mult_prev, ndf * mult, use_spect, use_coord, **kwargs),
            nonlinearity,
            coord_conv(ndf * mult, 1, use_spect, use_coord, **kwargs),
        ]
        self.final_model = nn.Sequential(*sequence)

    def forward(self, x):
        layers = []
        for i in range(self.layers):
            x = getattr(self, 'enc'+str(i))(x)
            layers.append(x)
        if self.n_res > 0:
            x = getattr(self, 'res')(x)
        if not self.multi_dim_out:
            x = self.model(x)
        return x, layers


class VGGLoss(nn.Module):
    r"""
    Perceptual loss, VGG-based
    https://arxiv.org/abs/1603.08155
    https://github.com/dxyang/StyleTransfer/blob/master/utils.py
    """

    def __init__(self, weights=[1.0, 1.0, 1.0, 1.0, 1.0]):
        super(VGGLoss, self).__init__()
        self.criterion = torch.nn.L1Loss()
        self.weights = weights
        # self.vgg = vgg

    def compute_gram(self, x):
        b, ch, h, w = x.size()
        f = x.view(b, ch, w * h)
        f_T = f.transpose(1, 2)
        G = f.bmm(f_T) / (h * w * ch)
        return G
        
    def __call__(self, x_vgg, y_vgg):
        # normalize img
        #TODO
        # Compute features
        # x_vgg, y_vgg = self.vgg(x), self.vgg(y)

        content_loss = 0.0
        content_loss += self.weights[0] * self.criterion(x_vgg['relu1_1'], y_vgg['relu1_1'])
        content_loss += self.weights[1] * self.criterion(x_vgg['relu2_1'], y_vgg['relu2_1'])
        content_loss += self.weights[2] * self.criterion(x_vgg['relu3_1'], y_vgg['relu3_1'])
        content_loss += self.weights[3] * self.criterion(x_vgg['relu4_1'], y_vgg['relu4_1'])
        content_loss += self.weights[3] * self.criterion(x_vgg['relu5_1'], y_vgg['relu5_1'])

        # Compute loss
        style_loss = 0.0
        style_loss += self.criterion(self.compute_gram(x_vgg['relu1_2']), self.compute_gram(y_vgg['relu1_2']))
        style_loss += self.criterion(self.compute_gram(x_vgg['relu2_2']), self.compute_gram(y_vgg['relu2_2']))
        style_loss += self.criterion(self.compute_gram(x_vgg['relu3_4']), self.compute_gram(y_vgg['relu3_4']))
        style_loss += self.criterion(self.compute_gram(x_vgg['relu4_4']), self.compute_gram(y_vgg['relu4_4']))
        style_loss += self.criterion(self.compute_gram(x_vgg['relu5_2']), self.compute_gram(y_vgg['relu5_2']))


        return content_loss, style_loss

class StyleLoss(nn.Module):
    r"""
    Perceptual loss, VGG-based
    https://arxiv.org/abs/1603.08155
    https://github.com/dxyang/StyleTransfer/blob/master/utils.py
    """

    def __init__(self):
        super(StyleLoss, self).__init__()
        self.criterion = torch.nn.L1Loss()

    def compute_gram(self, x):
        b, ch, h, w = x.size()
        f = x.view(b, ch, w * h)
        f_T = f.transpose(1, 2)
        G = f.bmm(f_T) / (h * w * ch)

        return G

    def __call__(self, x_vgg, y_vgg):
        # Compute features
        # x_vgg, y_vgg = self.vgg(x), self.vgg(y)

        # Compute loss
        style_loss = 0.0
        style_loss += self.criterion(self.compute_gram(x_vgg['relu1_2']), self.compute_gram(y_vgg['relu1_2']))
        style_loss += self.criterion(self.compute_gram(x_vgg['relu2_2']), self.compute_gram(y_vgg['relu2_2']))
        style_loss += self.criterion(self.compute_gram(x_vgg['relu3_4']), self.compute_gram(y_vgg['relu3_4']))
        style_loss += self.criterion(self.compute_gram(x_vgg['relu4_4']), self.compute_gram(y_vgg['relu4_4']))

        return style_loss



class PerceptualLoss(nn.Module):
    r"""
    Perceptual loss, VGG-based
    https://arxiv.org/abs/1603.08155
    https://github.com/dxyang/StyleTransfer/blob/master/utils.py
    """

    def __init__(self, weights=[1.0, 1.0, 1.0, 1.0, 1.0]):
        super(PerceptualLoss, self).__init__()
        self.criterion = torch.nn.L1Loss()
        self.weights = weights

    def __call__(self,  x_vgg, y_vgg):
        # Compute features
        # x_vgg, y_vgg = self.vgg(x), self.vgg(y)
        content_loss = 0.0
        content_loss += self.weights[0] * self.criterion(x_vgg['relu1_1'], y_vgg['relu1_1'])
        content_loss += self.weights[1] * self.criterion(x_vgg['relu2_1'], y_vgg['relu2_1'])
        content_loss += self.weights[2] * self.criterion(x_vgg['relu3_1'], y_vgg['relu3_1'])
        content_loss += self.weights[3] * self.criterion(x_vgg['relu4_1'], y_vgg['relu4_1'])

        return content_loss


class ContextSimilarityLoss(nn.Module):
    '''
    https://github.com/roimehrez/contextualLoss/blob/master/CX/CX_distance.py
    cosine similarity implementation
    '''
    def __init__(self, sigma=0.5, b=1.0):
        super(ContextSimilarityLoss, self).__init__()
        self.sigma = sigma
        self.b = b

    def center_by_T(self, featureI, featureT):
        # Calculate mean channel vector for feature map.
        meanT = featureT.mean(0, keepdim=True).mean(2, keepdim=True).mean(3, keepdim=True)
        return featureI - meanT, featureT - meanT

    def l2_normalize_channelwise(self, features):
        # Normalize on channel dimension (axis=1)
        norms = features.norm(p=2, dim=1, keepdim=True)
        features = features.div(norms)
        return features

    def patch_decomposition(self, features):
        N, C, H, W = features.shape
        assert N == 1
        P = H * W
        # NCHW --> 1x1xCxHW --> HWxCx1x1
        patches = features.view(1, 1, C, P).permute((3, 2, 0, 1))
        return patches

    def calc_relative_distances(self, raw_dist, axis=1):
        epsilon = 1e-5
        div = torch.min(raw_dist, dim=axis, keepdim=True)[0]
        relative_dist = raw_dist / (div + epsilon)
        return relative_dist

    def calc_CX(self, dist, axis=1):
        W = torch.exp((self.b - dist) / self.sigma)
        W_sum = W.sum(dim=axis, keepdim=True)
        return W.div(W_sum)

    def feature_loss(self, featureT, featureI):
        '''
        :param featureT: target
        :param featureI: inference
        :return:
        '''
        # NCHW
        # print(featureI.shape)

        featureI, featureT = self.center_by_T(featureI, featureT)

        featureI = self.l2_normalize_channelwise(featureI)
        featureT = self.l2_normalize_channelwise(featureT)

        dist = []
        N = featureT.size()[0]
        for i in range(N):
            # NCHW
            featureT_i = featureT[i, :, :, :].unsqueeze(0)
            # NCHW
            featureI_i = featureI[i, :, :, :].unsqueeze(0)
            featureT_patch = self.patch_decomposition(featureT_i)
            # Calculate cosine similarity
            # See the torch document for functional.conv2d
            dist_i = F.conv2d(featureI_i, featureT_patch)
            dist.append(dist_i)

        # NCHW
        dist = torch.cat(dist, dim=0)

        raw_dist = (1. - dist) / 2.

        relative_dist = self.calc_relative_distances(raw_dist)

        CX = self.calc_CX(relative_dist)

        CX = CX.max(dim=3)[0].max(dim=2)[0]
        CX = CX.mean(1)
        CX = -torch.log(CX)
        CX = torch.mean(CX)
        return CX
    
    def __call__(self, x_vgg, y_vgg, layers = ['relu3_2','relu4_2','relu5_2']):
        # Compute features
        # x_vgg, y_vgg = self.vgg(x), self.vgg(y)
        
        # layers = ['relu4_2']
        cx_style_loss = 0
        for layer in layers:
            featureT, featureI = x_vgg[layer], y_vgg[layer]
            if 'relu2_2'==layer:
                # featureI = featureI[:,:,::2,::2]
                # featureT = featureT[:,:,::2,::2]
                featureI = nn.AvgPool2d(2)(featureI)
                featureT = nn.AvgPool2d(2)(featureT)
            cx_style_loss += self.feature_loss(featureT, featureI)
        return cx_style_loss


class PerceptualCorrectness(nn.Module):
    r"""

    """

    def __init__(self, layer=['relu1_1','relu2_1','relu3_1','relu4_1','relu5_1']):
        super(PerceptualCorrectness, self).__init__()
        # self.vgg= vgg
        self.layer = layer  
        self.eps=1e-8 
        self.resample = Resample2d(4, 1, sigma=2)

    def __call__(self, target_vgg, source_vgg, flow_list, used_layers, mask=None, use_bilinear_sampling=False):
        used_layers=sorted(used_layers, reverse=True)
        # self.target=target
        # self.source=source
        # self.target_vgg, self.source_vgg = self.vgg(target), self.vgg(source)
        self.target_vgg, self.source_vgg = target_vgg, source_vgg
        loss = 0
        for i in range(len(flow_list)):
            loss += self.calculate_loss(flow_list[i], self.layer[used_layers[i]], mask, use_bilinear_sampling)



        return loss

    def calculate_loss(self, flow, layer, mask=None, use_bilinear_sampling=False):
        target_vgg = self.target_vgg[layer]
        source_vgg = self.source_vgg[layer]
        [b, c, h, w] = target_vgg.shape

        # maps = F.interpolate(maps, [h,w]).view(b,-1)
        if flow.shape[2]!=h or flow.shape[3]!=w:
            flow = F.interpolate(flow, [h,w])

        target_all = target_vgg.view(b, c, -1)                      #[b C N2]
        source_all = source_vgg.view(b, c, -1).transpose(1,2)       #[b N2 C]


        source_norm = source_all/(source_all.norm(dim=2, keepdim=True)+self.eps)
        target_norm = target_all/(target_all.norm(dim=1, keepdim=True)+self.eps)
        try:
            correction = torch.bmm(source_norm, target_norm)                       #[b N2 N2]
        except:
            print("An exception occurred")
            print(source_norm.shape)
            print(target_norm.shape)
        (correction_max,max_indices) = torch.max(correction, dim=1)

        # interple with bilinear sampling
        if use_bilinear_sampling:
            input_sample = self.bilinear_warp(source_vgg, flow).view(b, c, -1)
        else:
            input_sample = self.resample(source_vgg, flow).view(b, c, -1)

        correction_sample = F.cosine_similarity(input_sample, target_all)    #[b 1 N2]
        loss_map = torch.exp(-correction_sample/(correction_max+self.eps))
        if mask is None:
            loss = torch.mean(loss_map) - torch.exp(torch.tensor(-1).type_as(loss_map))
        else:
            mask=F.interpolate(mask, size=(target_vgg.size(2), target_vgg.size(3)))
            mask=mask.view(-1, target_vgg.size(2)*target_vgg.size(3))
            loss_map = loss_map - torch.exp(torch.tensor(-1).type_as(loss_map))
            loss = torch.sum(mask * loss_map)/(torch.sum(mask)+self.eps)

        # print(correction_sample[0,2076:2082])
        # print(correction_max[0,2076:2082])
        # coor_x = [32,32]
        # coor = max_indices[0,32+32*64]
        # coor_y = [int(coor%64), int(coor/64)]
        # source = F.interpolate(self.source, [64,64])
        # target = F.interpolate(self.target, [64,64])
        # source_i = source[0]
        # target_i = target[0]

        # source_i = source_i.view(3, -1)
        # source_i[:,coor]=-1
        # source_i[0,coor]=1
        # source_i = source_i.view(3,64,64)
        # target_i[:,32,32]=-1
        # target_i[0,32,32]=1
        # lists = str(int(torch.rand(1)*100))
        # img_numpy = util.tensor2im(source_i.data)
        # util.save_image(img_numpy, 'source'+lists+'.png')
        # img_numpy = util.tensor2im(target_i.data)
        # util.save_image(img_numpy, 'target'+lists+'.png')
        return loss

    def bilinear_warp(self, source, flow):
        [b, c, h, w] = source.shape
        x = torch.arange(w).view(1, -1).expand(h, -1).type_as(source).float() / (w-1)
        y = torch.arange(h).view(-1, 1).expand(-1, w).type_as(source).float() / (h-1)
        grid = torch.stack([x,y], dim=0)
        grid = grid.unsqueeze(0).expand(b, -1, -1, -1)
        grid = 2*grid - 1
        flow = 2*flow/torch.tensor([w, h]).view(1, 2, 1, 1).expand(b, -1, h, w).type_as(flow)
        grid = (grid+flow).permute(0, 2, 3, 1)
        input_sample = F.grid_sample(source, grid).view(b, c, -1)
        return input_sample


class VGG19Limited(torch.nn.Module):
    def __init__(self, pretrained_path = None):
        super(VGG19Limited, self).__init__()
        self.register_buffer("mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer("std", torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))
        if pretrained_path is not None:
            vgg19 = models.vgg19(pretrained=False)
            vgg19.load_state_dict(torch.load(pretrained_path)) 
        else:
            vgg19 = models.vgg19(pretrained=True)       
        features = vgg19.features
        self.relu1_1 = torch.nn.Sequential()
        self.relu1_2 = torch.nn.Sequential()

        self.relu2_1 = torch.nn.Sequential()
        self.relu2_2 = torch.nn.Sequential()

        self.relu3_1 = torch.nn.Sequential()
        self.relu3_2 = torch.nn.Sequential()
        self.relu3_3 = torch.nn.Sequential()
        self.relu3_4 = torch.nn.Sequential()

        self.relu4_1 = torch.nn.Sequential()
        self.relu4_2 = torch.nn.Sequential()
        self.relu4_3 = torch.nn.Sequential()
        self.relu4_4 = torch.nn.Sequential()

        self.relu5_1 = torch.nn.Sequential()
        self.relu5_2 = torch.nn.Sequential()
        # self.relu5_3 = torch.nn.Sequential()
        # self.relu5_4 = torch.nn.Sequential()

        for x in range(2):
            self.relu1_1.add_module(str(x), features[x])

        for x in range(2, 4):
            self.relu1_2.add_module(str(x), features[x])

        for x in range(4, 7):
            self.relu2_1.add_module(str(x), features[x])

        for x in range(7, 9):
            self.relu2_2.add_module(str(x), features[x])

        for x in range(9, 12):
            self.relu3_1.add_module(str(x), features[x])

        for x in range(12, 14):
            self.relu3_2.add_module(str(x), features[x])

        for x in range(14, 16):
            self.relu3_2.add_module(str(x), features[x])

        for x in range(16, 18):
            self.relu3_4.add_module(str(x), features[x])

        for x in range(18, 21):
            self.relu4_1.add_module(str(x), features[x])

        for x in range(21, 23):
            self.relu4_2.add_module(str(x), features[x])

        for x in range(23, 25):
            self.relu4_3.add_module(str(x), features[x])

        for x in range(25, 27):
            self.relu4_4.add_module(str(x), features[x])

        for x in range(27, 30):
            self.relu5_1.add_module(str(x), features[x])

        for x in range(30, 32):
            self.relu5_2.add_module(str(x), features[x])

        # for x in range(32, 34):
        #     self.relu5_3.add_module(str(x), features[x])

        # for x in range(34, 36):
        #     self.relu5_4.add_module(str(x), features[x])

        # don't need the gradients, just want the features
        for param in self.parameters():
            param.requires_grad = False
        self.eval()
        
    def optimized_forward(self, x, max_layer = 'relu5_2'):
        result = {}
        relu1_1 = self.relu1_1(x)
        result['relu1_1']=relu1_1
        if max_layer in result.keys(): return result

        relu1_2 = self.relu1_2(relu1_1)
        result['relu1_2']=relu1_2
        if max_layer in result.keys(): return result

        relu2_1 = self.relu2_1(relu1_2)
        result['relu2_1']=relu2_1
        if max_layer in result.keys(): return result

        relu2_2 = self.relu2_2(relu2_1)
        result['relu2_2']=relu2_2
        if max_layer in result.keys(): return result

        relu3_1 = self.relu3_1(relu2_2)
        result['relu3_1']=relu3_1
        if max_layer in result.keys(): return result

        relu3_2 = self.relu3_2(relu3_1)
        result['relu3_2']=relu3_2
        if max_layer in result.keys(): return result

        relu3_3 = self.relu3_3(relu3_2)
        result['relu3_3']=relu3_3
        if max_layer in result.keys(): return result

        relu3_4 = self.relu3_4(relu3_3)
        result['relu3_4']=relu3_4
        if max_layer in result.keys(): return result

        relu4_1 = self.relu4_1(relu3_4)
        result['relu4_1']=relu4_1
        if max_layer in result.keys(): return result

        relu4_2 = self.relu4_2(relu4_1)
        result['relu4_2']=relu4_2
        if max_layer in result.keys(): return result

        relu4_3 = self.relu4_3(relu4_2)
        result['relu4_3']=relu4_3
        if max_layer in result.keys(): return result

        relu4_4 = self.relu4_4(relu4_3)
        result['relu4_4']=relu4_4
        if max_layer in result.keys(): return result

        relu5_1 = self.relu5_1(relu4_4)
        result['relu5_1']=relu5_1
        if max_layer in result.keys(): return result

        relu5_2 = self.relu5_2(relu5_1)
        result['relu5_2']=relu5_2
        return result

    def forward(self, x, max_layer = 'relu5_2'):
        #change stats to vgg
        # mean = torch.FloatTensor([0.485, 0.456, 0.406]).to(x.device)[None,:,None,None]
        # std = torch.FloatTensor([0.229, 0.224, 0.225]).to(x.device)[None,:,None,None]

        x = (x + 1)/2 # [-1, 1] => [0, 1]
        x = (x - self.mean)/self.std

        res = self.optimized_forward(x,max_layer)

        out = {
            'relu1_1': res.get('relu1_1'),
            'relu1_2': res.get('relu1_2'),

            'relu2_1': res.get('relu2_1'),
            'relu2_2': res.get('relu2_2'),

            'relu3_1': res.get('relu3_1'),
            'relu3_2': res.get('relu3_2'),
            'relu3_3': res.get('relu3_3'),
            'relu3_4': res.get('relu3_4'),

            'relu4_1': res.get('relu4_1'),
            'relu4_2': res.get('relu4_2'),
            'relu4_3': res.get('relu4_3'),
            'relu4_4': res.get('relu4_4'),

            'relu5_1': res.get('relu5_1'),
            'relu5_2': res.get('relu5_2'),
            # 'relu5_3': relu5_3,
            # 'relu5_4': relu5_4,
        }
        return out
import torch
import torch.nn as nn
import torch.nn.functional as F
from CLIPSeg.models.clipseg import CLIPDensePredT
import spacy
from torchvision import transforms
from utils import text_token


class Pre_trained(nn.Module):
    def __init__(self, CLIPSEG_MODEL_PATH, rank):
        super(Pre_trained, self).__init__()
        self.rank = rank
        
        '''load Clipseg'''
        self.ClipsegModel = CLIPDensePredT(version='ViT-B/16', reduce_dim=64, complex_trans_conv=True).to(rank)
        self.ClipsegModel.load_state_dict(
            torch.load(CLIPSEG_MODEL_PATH, weights_only=True, map_location=torch.device(f'cuda:{rank}')),
            strict=False
        )
        self.ClipsegModel.eval()
        for param in self.ClipsegModel.parameters():
            param.requires_grad = False
        '''load spacy model'''
        self.nlp = spacy.load("en_core_web_sm")

        '''fine_tuning'''
        self.upsampling = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.LeakyReLU(inplace=True),
            nn.ConvTranspose2d(64, 64, kernel_size=4, stride=4),
            nn.LeakyReLU(inplace=True),
            nn.ConvTranspose2d(64, 64, kernel_size=4, stride=4),
            nn.LeakyReLU(inplace=True)
        )
        self.mask_feature = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(64, 32, kernel_size=3, padding=1),
            nn.LeakyReLU(inplace=True)
        )

    def clipseg_preprocess(self, image_ir, image_vi, prompt):
        height, width = image_ir.size(1), image_ir.size(2)
        image_ir = image_ir.repeat(3, 1, 1)
        assert image_ir.shape == image_vi.shape, 'assert shape error'
        transform = transforms.Compose([
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            transforms.Resize((352, 352)),
        ])
        input_img = [transform(image_ir), transform(image_vi)]
        input_img = torch.stack(input_img, dim=0)
        assert input_img.shape == torch.Size([2, 3, 352, 352])
        prompt = [prompt] * 2
        with torch.no_grad():
            seg_result, feature = self.ClipsegModel(input_img, prompt, h=height, w=width)
            if prompt != [0, 0] and prompt !=['image', 'image']:
                seg_result = torch.sigmoid(seg_result)
                seg_result = F.interpolate(seg_result, size=(height, width), mode='bilinear',align_corners=False)
        assert feature.shape == torch.Size([2, 64, 22, 22])
        return seg_result, feature

    def get_mask_feature(self, image_ir, image_vi, targets):
        batch, height, width = image_ir.size(0), image_ir.size(2), image_ir.size(3)
        mask_fea = []
        for i in range(batch):
            seg_result, clipseg_fea = self.clipseg_preprocess(image_ir[i], image_vi[i], targets[i])
            ir_seg = seg_result[0].unsqueeze(dim=0)
            vi_seg = seg_result[1].unsqueeze(dim=0)
            seg_result = torch.cat((ir_seg, vi_seg), dim=1)
            # print(seg_result.shape)
            assert seg_result.shape == torch.Size([1, 2, height, width]), 'seg error'
            clipseg_fea1 = self.upsampling(clipseg_fea)
            assert clipseg_fea1.shape == torch.Size([2, 64, 352, 352]), 'upsampling error'
            clipseg_fea2 = F.interpolate(clipseg_fea1, size=(height, width), mode='bilinear', align_corners=False)
            assert clipseg_fea2.shape == torch.Size([2, 64, height, width]), 'interpolate error'
            maskfeat = self.mask_feature(clipseg_fea2)  # [2 32 height width]
            assert maskfeat.shape == torch.Size([2, 32, height, width])
            ir_fea = maskfeat[0].unsqueeze(dim=0)
            vi_fea = maskfeat[1].unsqueeze(dim=0)
            mask_fea.append(torch.cat((ir_fea, vi_fea, seg_result), dim=1))
        mask_fea = torch.cat(mask_fea, dim=0)
        assert mask_fea.shape == torch.Size([batch, 66, height, width])
        return mask_fea


class Net(Pre_trained):
    def __init__(self, CLIPSEG_MODEL_PATH, rank):
        super(Net, self).__init__(CLIPSEG_MODEL_PATH, rank)
        self.irconv = nn.Conv2d(1, 64, kernel_size=3, padding=1)
        self.viconv = nn.Conv2d(3, 64, kernel_size=3, padding=1)
        self.maskconv = nn.Conv2d(66, 64, kernel_size=3, padding=1, bias=False)
        self.dconv1 = nn.Conv2d(128, 128, kernel_size=3, padding=1)
        self.dconv2 = nn.Conv2d(256, 128, kernel_size=3, padding=1)
        self.dconv3 = nn.Conv2d(256, 128, kernel_size=3, padding=1)
        self.dconv4 = nn.Conv2d(256, 128, kernel_size=3, padding=1)
        self.dconv5 = nn.Conv2d(256, 128, kernel_size=3, padding=1)
        self.dconv6 = nn.Conv2d(128, 64, kernel_size=3, padding=1)
        self.dconv7 = nn.Conv2d(64, 3, kernel_size=3, padding=1)
        self.relu = nn.ReLU(inplace=True)
        self.leaky = nn.LeakyReLU(inplace=True)
        self.processmodule = PModule()
        self.ResBlock1 = ResBlockM()
        self.ResBlock2 = ResBlockM()
        self.ResBlock3 = ResBlockM()
        self.imagemoudle = IMoudle()
        self.CBA1 = CBAMoudle()
        self.CBA2 = CBAMoudle()
        self.CBA3 = CBAMoudle()
        self.CBA4 = CBAMoudle()

    def forward(self, image_ir, image_vi, prompts, filename=0):
        batch, height, width = image_ir.size(0), image_ir.size(2), image_ir.size(3)
        '''target、process '''
        targets, process = text_token(self.nlp, prompts, filename)
        assert len(targets) == batch, 'target and image should be paired'
        process_embedding = self.ClipsegModel.get_embedding(process)
        w, b = self.processmodule(process_embedding)  # w->[3 b 64]  b->[3 b 64]
        assert w.shape == torch.Size([3, batch, 64])
        mask = self.get_mask_feature(image_ir, image_vi, targets)
        m_fea1 = self.maskconv(mask)
        m_fea1 = torch.tanh(m_fea1)
        m_fea2 = self.ResBlock1(m_fea1, mask, w[0], b[0])
        m_fea3 = self.ResBlock2(m_fea2, mask, w[1], b[1])
        m_fea4 = self.ResBlock3(m_fea3, mask, w[2], b[2])
        w_moudle, b_moudle = self.imagemoudle(m_fea4)
        assert w_moudle.shape == torch.Size([4, batch, 128, height, width]), 'w_moudle shape error'
        '''encode'''
        vi_fea = self.viconv(image_vi)
        vi_fea = self.relu(vi_fea)
        ir_fea = self.irconv(image_ir)
        ir_fea = self.relu(ir_fea)
        image_fea1 = torch.cat([ir_fea, vi_fea], dim=1)
        image_fea2 = self.CBA1(image_fea1, w_moudle[0], b_moudle[0])
        image_fea3 = self.CBA2(image_fea2, w_moudle[1], b_moudle[1])
        image_fea4 = self.CBA3(image_fea3, w_moudle[2], b_moudle[2])
        fusion_fea = self.CBA4(image_fea4, w_moudle[3], b_moudle[3])
        assert fusion_fea.shape == torch.Size([batch, 128, height, width]), 'encoder error'
        '''decode'''
        fusion_fea1 = self.dconv1(fusion_fea)
        fusion_fea1 = self.relu(fusion_fea1)
        fusion_fea2 = torch.cat([image_fea4, fusion_fea1], dim=1)
        fusion_fea2 = self.dconv2(fusion_fea2)
        fusion_fea2 = self.relu(fusion_fea2)
        fusion_fea3 = torch.cat([image_fea3, fusion_fea2], dim=1)
        fusion_fea3 = self.dconv3(fusion_fea3)
        fusion_fea3 = self.relu(fusion_fea3)
        fusion_fea4 = torch.cat([image_fea2, fusion_fea3], dim=1)
        fusion_fea4 = self.dconv4(fusion_fea4)
        fusion_fea4 = self.relu(fusion_fea4)
        fusion_fea5 = torch.cat([image_fea1, fusion_fea4], dim=1)
        fusion_fea5 = self.dconv5(fusion_fea5)
        fusion_fea5 = self.relu(fusion_fea5)
        fusion_feature = self.dconv6(fusion_fea5)
        fusion_feature = self.relu(fusion_feature)
        fusion_result = self.dconv7(fusion_feature)
        fusion_result = self.relu(fusion_result)
        assert fusion_result.shape == torch.Size([batch, 3, height, width]), 'decoder error'
        '''Loss_D'''
        delta_i = image_fea1-fusion_fea # [b 128 h w]
        delta_i = delta_i.mean(dim=1, keepdim=True) # [b 1 h w]
        delta_i = delta_i.view(batch, -1)  # flat [b h*w]
        m_fea4 = m_fea4.mean(dim=1, keepdim=True) # [b 1 h w]
        beta = m_fea4.view(batch, -1)  # flat [b h*w]
        loss_d = 1 - F.cosine_similarity(delta_i, beta, dim=1)
        assert loss_d.shape == torch.Size([batch]), 'loss_d error'
        loss_d = loss_d.mean()  
        
        return fusion_result, 5*loss_d.item()


class PModule(nn.Module):
    def __init__(self):
        super(PModule, self).__init__()
        self.wconv1 = nn.Conv1d(1, 64, 512, bias=False)
        self.wconv2 = nn.Conv1d(1, 64, 512, bias=False)
        self.wconv3 = nn.Conv1d(1, 64, 512, bias=False)
        self.bconv1 = nn.Conv1d(1, 64, 512, bias=False)
        self.bconv2 = nn.Conv1d(1, 64, 512, bias=False)
        self.bconv3 = nn.Conv1d(1, 64, 512, bias=False)

    def forward(self, embedding):
        embedding = embedding.unsqueeze(dim=1).to(torch.float32)
        w = [self.wconv1(embedding).squeeze(dim=-1),
             self.wconv2(embedding).squeeze(dim=-1),
             self.wconv3(embedding).squeeze(dim=-1)]
        b = [self.bconv1(embedding).squeeze(dim=-1),
             self.bconv2(embedding).squeeze(dim=-1),
             self.bconv3(embedding).squeeze(dim=-1)]
        w = torch.stack(w, dim=0)
        b = torch.stack(b, dim=0)
        return w, b


class ResBlockM(nn.Module):
    def __init__(self):
        super(ResBlockM, self).__init__()
        self.conv1 = nn.Conv2d(64, 64, kernel_size=3, padding=1, bias=False)
        self.conv2 = nn.Conv2d(130, 64, kernel_size=3, padding=1, bias=False)
        self.leaky = nn.LeakyReLU(inplace=True)

    def forward(self, feature, mask, w, b):
        feature1 = self.conv1(feature)
        feature1 = self.leaky(feature1)
        w = w.unsqueeze(dim=2).unsqueeze(dim=3)
        b = b.unsqueeze(dim=2).unsqueeze(dim=3)
        feature1 = feature1 * w + b
        featureca = torch.cat([feature1, mask], dim=1)
        feature2 = self.conv2(featureca)
        feature2 = self.leaky(feature2)
        output = feature2 + feature
        return output


class IMoudle(nn.Module):
    def __init__(self):
        super(IMoudle, self).__init__()
        self.wconv1 = nn.Conv2d(64, 128, kernel_size=3, padding=1, bias=False)
        self.wconv2 = nn.Conv2d(64, 128, kernel_size=3, padding=1, bias=False)
        self.wconv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1, bias=False)
        self.wconv4 = nn.Conv2d(64, 128, kernel_size=3, padding=1, bias=False)
        self.bconv1 = nn.Conv2d(64, 128, kernel_size=3, padding=1, bias=False)
        self.bconv2 = nn.Conv2d(64, 128, kernel_size=3, padding=1, bias=False)
        self.bconv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1, bias=False)
        self.bconv4 = nn.Conv2d(64, 128, kernel_size=3, padding=1, bias=False)

    def forward(self, feature):
        w = [self.wconv1(feature),
             self.wconv2(feature),
             self.wconv3(feature),
             self.wconv4(feature)]
        b = [self.bconv1(feature),
             self.bconv2(feature),
             self.bconv3(feature),
             self.bconv4(feature)]
        w = torch.stack(w, dim=0)
        b = torch.stack(b, dim=0)
        return w, b


class CBAMoudle(nn.Module):
    def __init__(self):
        super(CBAMoudle, self).__init__()
        '''channel att'''
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.mlp = nn.Sequential(
            nn.Conv2d(128, 16, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(16, 128, kernel_size=1)
        )
        self.sigmoid = nn.Sigmoid()
        '''spatial att'''
        self.convs = nn.Conv2d(2, 1, kernel_size=7, padding=3)

    def forward(self, input_fea, weight, bias):
        # channel att
        cha_att = self.sigmoid(self.mlp(self.max_pool(input_fea)) + self.mlp(self.avg_pool(input_fea)))
        feature = input_fea * cha_att
        # spatial
        smax_pool, _ = torch.max(feature, dim=1, keepdim=True)
        savg_pool = torch.mean(feature, dim=1, keepdim=True)
        spa_att = self.sigmoid(self.convs(torch.cat((smax_pool, savg_pool), dim=1)))
        feature1 = spa_att * feature
        # module
        feature2 = feature1 * weight + bias
        output = feature2 + feature1 + input_fea
        return output

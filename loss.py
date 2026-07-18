import torch
import torch.nn as nn
import torch.nn.functional as F

class Fusionloss(nn.Module):
    def __init__(self, rank):
        super(Fusionloss, self).__init__()
        self.sobelconv = Sobelxy(rank)

    def forward(self, image_ir, image_vi, fusion_image, mask):
        batch, height, width = image_ir.shape[0], image_ir.shape[2], image_ir.shape[3]
        totalp = height*width
        lamda = []
        alpha = []
        for item in range(batch):
          mask_temp = mask[item]  
          reg_p = torch.sum(mask_temp)  # 1
          if reg_p!=0 and reg_p!=totalp:
            lamda_temp = totalp/reg_p*0.5
            alpha_temp = totalp/(totalp-reg_p)
          elif reg_p == 0:
            alpha_temp = 1.0
            lamda_temp = 0.0
          elif reg_p == totalp:
            lamda_temp = 1.0
            alpha_temp = 0.0
          lamda.append(lamda_temp)
          alpha.append(alpha_temp)
        lamda = torch.tensor(lamda).view(batch, 1, 1, 1).to(image_ir.device)
        alpha = torch.tensor(alpha).view(batch, 1, 1, 1).to(image_ir.device)
        
        # YCrCb
        imagevi_y = image_vi[:, :1]  # [b 1 h w]
        imagevi_cr = image_vi[:, 1:2]
        imagevi_cb = image_vi[:, 2:3]
        fusion_y = fusion_image[:, :1]
        fusion_cr = fusion_image[:, 1:2]
        fusion_cb = fusion_image[:, 2:3]
        assert imagevi_y.shape == image_ir.shape == imagevi_cr.shape
        
        # loss_in
        in_max = torch.max(imagevi_y, image_ir) # [b 1 h w]
        region = mask * fusion_y + (1 - mask) * in_max
        no_region = (1-mask) * fusion_y + mask * in_max
        region_loss_in = 5 * F.l1_loss(lamda * in_max, lamda * region)
        no_region_loss_in = 5 * F.l1_loss(alpha * in_max, alpha * no_region)
        # loss_grad
        viy_grad = self.sobelconv(imagevi_y)
        ir_grad = self.sobelconv(image_ir)
        fusion_grad = self.sobelconv(fusion_y)
        grad_max = torch.max(viy_grad, ir_grad)
        region = mask * fusion_grad + (1 - mask) * grad_max
        no_region = (1-mask) * fusion_grad + mask * grad_max
        region_loss_grad = 10 * F.l1_loss(lamda * grad_max, lamda * region)
        no_region_loss_grad = 10 * F.l1_loss(alpha * grad_max, alpha * no_region)
        # loss_cr
        region = mask * fusion_cr + (1 - mask) * imagevi_cr
        no_region = (1-mask) * fusion_cr + mask * imagevi_cr
        region_loss_cr = 5 * F.l1_loss(lamda * imagevi_cr, lamda * region)
        no_region_loss_cr = F.l1_loss(alpha * imagevi_cr, alpha * no_region)
        # loss_cb
        region = mask * fusion_cb + (1 - mask) * imagevi_cb
        no_region = (1-mask) * fusion_cb + mask * imagevi_cb
        region_loss_cb = 5 * F.l1_loss(lamda * imagevi_cb, lamda * region)
        no_region_loss_cb = F.l1_loss(alpha * imagevi_cb, alpha * no_region)
        
        # loss_total
        region_loss = region_loss_in + region_loss_grad + region_loss_cr + region_loss_cb
        no_region_loss = no_region_loss_in + no_region_loss_grad + no_region_loss_cr + no_region_loss_cb
        loss_total = region_loss + no_region_loss
        return loss_total, region_loss, no_region_loss
    

class Sobelxy(nn.Module):
    def __init__(self, rank):
        super(Sobelxy, self).__init__()
        kernelx = [[-1, 0, 1],
                  [-2, 0, 2],
                  [-1, 0, 1]]
        kernely = [[1, 2, 1],
                  [0, 0, 0],
                  [-1, -2, -1]]
        kernelx = torch.FloatTensor(kernelx).unsqueeze(0).unsqueeze(0)
        kernely = torch.FloatTensor(kernely).unsqueeze(0).unsqueeze(0)
        self.weightx = nn.Parameter(data=kernelx, requires_grad=False).to(rank)
        self.weighty = nn.Parameter(data=kernely, requires_grad=False).to(rank)

    def forward(self, x):
        sobelx = F.conv2d(x, self.weightx, padding=1)
        sobely = F.conv2d(x, self.weighty, padding=1)
        grad = torch.abs(sobelx)+torch.abs(sobely)
        return grad
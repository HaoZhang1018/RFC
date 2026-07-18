import torch
from torchvision import transforms
import torch.nn.functional as F


def seg_image(model, image_ir, image_vi, prompt, rank):
    b, h, w = image_ir.shape[0], image_ir.shape[2], image_ir.shape[3]
    region_mask = [i for i, l in enumerate(prompt) if l != 0 and l != 'image']
    none_mask = [i for i, l in enumerate(prompt) if l == 0]
    image_mask = [i for i, l in enumerate(prompt) if l == 'image']
    # region mask
    if len(region_mask):
        image_ir = image_ir[region_mask]
        image_vi = image_vi[region_mask]
        target = []
        for i in region_mask:
            target.append(prompt[i])
        seg_result = clipseg_image(model, image_ir, image_vi, target)  # [b 1 h w]
    else:
        seg_result = []
    # result
    no_mask = torch.zeros(1, 1, h, w).to(rank)
    im_mask = torch.ones(1, 1, h, w).to(rank)
    result = []
    k = 0
    for i in range(len(prompt)):
        if i in none_mask:
            result.append(no_mask)
        elif i in image_mask:
            result.append(im_mask)
        else:
            result.append(seg_result[k].unsqueeze(0))  # [1 1 h w]
            k = k + 1
    result = torch.cat(result, dim=0)  # b 1 h w
    assert result.shape == torch.Size([b, 1, h, w]), 'seg error'
    return result


def clipseg_image(model, image_ir, image_vi, prompt):
    image_ir = image_ir.repeat(1, 3, 1, 1)
    transform = transforms.Compose([
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        transforms.Resize((352, 352)),
    ])
    b, h, w = image_ir.shape[0], image_ir.shape[2], image_ir.shape[3]
    ir_image = []
    vi_image = []
    for i in range(b):
        img_ir = transform(image_ir[i])  # [3 352 352]
        img_vi = transform(image_vi[i])  # [3 352 352]
        ir_image.append(img_ir)
        vi_image.append(img_vi)
    img_ir = torch.stack(ir_image)  # [b 3 352 352]
    img_vi = torch.stack(vi_image)  # [b 3 352 352]
    assert img_ir.shape == img_vi.shape, 'ir and vi should be paired'
    input_image = torch.cat((img_ir, img_vi), dim=0)  # [2b 3 352 352]
    prompt = prompt * 2  # [2b]
    with torch.no_grad():
        preds = model(input_image, prompt)[0]
    resize_image = F.interpolate(preds, size=[h, w], mode='bilinear', align_corners=False)  # [2b 1 h w]
    seg_result = []
    for i in range(b):
        resize_image[i][0] = torch.sigmoid(resize_image[i][0])
        resize_image[i + b][0] = torch.sigmoid(resize_image[i + b][0])
        temp_result = torch.tanh(resize_image[i][0] + resize_image[i + b][0])  # [h w]
        seg_result.append(temp_result)
    seg_result = torch.stack(seg_result)  # [b h w]
    seg_result = seg_result.unsqueeze(dim=1)  # [b 1 h w]
    assert seg_result.shape == torch.Size([b, 1, h, w]), 'seg_result error'
    return seg_result


def RGB2YCrCb(input_im):
    im_flat = input_im.transpose(1, 3).transpose(1, 2).reshape(-1, 3)  # (nhw,c)
    R = im_flat[:, 0]
    G = im_flat[:, 1]
    B = im_flat[:, 2]
    Y = 0.299 * R + 0.587 * G + 0.114 * B
    Cr = (R - Y) * 0.713 + 0.5
    Cb = (B - Y) * 0.564 + 0.5
    Y = torch.unsqueeze(Y, 1)
    Cr = torch.unsqueeze(Cr, 1)
    Cb = torch.unsqueeze(Cb, 1)
    temp = torch.cat((Y, Cr, Cb), dim=1)
    size = list(input_im.size())
    out = (
        temp.reshape(size[0], size[2], size[3], 3)
        .transpose(1, 3)
        .transpose(2, 3)
    )
    return out


def YCrCb2RGB(input_im):
    im_flat = input_im.transpose(1, 3).transpose(1, 2).reshape(-1, 3)
    mat = torch.tensor(
        [[1.0, 1.0, 1.0], [1.403, -0.714, 0.0], [0.0, -0.344, 1.773]]
    )
    bias = torch.tensor([0.0 / 255, -0.5, -0.5])
    temp = (im_flat + bias).mm(mat)
    size = list(input_im.size())
    out = (
        temp.reshape(size[0], size[2], size[3], 3)
        .transpose(1, 3)
        .transpose(2, 3)
    )
    return out


def text_token(nlp, prompts, filename):
    target=[]
    process=[]
    for prompt in prompts:
        region=[]
        proc=[]
        prompt = prompt.lower()
        if prompt == 'none' or prompt == '':
            target.append(0)
            process.append('the photo')
        else:
            doc=nlp(prompt)
            if doc[0].dep_=='ROOT' and doc[0].pos_=='VERB':
                for i,token in enumerate(doc):
                    if token.dep_ in ['pobj','poss','compound','amod']:
                        region.append(token.text)
                        proc.append('photo')
                    elif token.dep_=='nsubj': 
                        if (i>0 and doc[i-1].dep_=='case') or (i<len(doc)-1 and doc[i+1].dep_=='prep'):
                            proc.append(token.text)
                        else:
                            region.append(token.text)
                            proc.append('photo')
                    else:
                        proc.append(token.text)
            else:
                if 'of' in prompt:
                    for token in doc:
                        if token.dep_=='pobj':
                            region.append(token.text)
                            proc.append('photo')
                        else:
                            proc.append(token.text)
                else:
                    for i,token in enumerate(doc):
                        if token.dep_=='nsubj' and i>0:
                            region.append(token.text)
                            proc.append('photo')
                        else:
                            proc.append(token.text)
            region=' '.join(region)
            proc=' '.join(proc)
            if region != '':
                target.append(region)
            else:
                print(prompt)
            if proc != '':
                process.append(proc)
    assert len(target)==len(process)==len(prompts),'token error'
    #print(f'target: {target}\nprocess: {process}')
    #print(filename)
    return target, process

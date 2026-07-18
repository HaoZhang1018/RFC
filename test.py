import os
from net import Net
from PIL import Image
import numpy as np
from torchvision import transforms
import torch
import time
from tqdm import tqdm

CLIPSEG_MODEL_PATH = 'CLIPSeg/weights/rd64-uni-refined.pth'

def main():

    print('Loading model...')
    device = 'cuda'
    model = Net(CLIPSEG_MODEL_PATH, rank=0)
    checkpoint = torch.load('./model/model.pth', weights_only=True)
    model.load_state_dict(checkpoint['model_state_dict'], strict=False)
    model.eval()
    model.to(device)

    print('Start testing...')
    root_path = 'dataset/test'
    for dataset in ['FMB', 'LLVIP', 'M3FD', 'MFNet']:
        ir_path = os.path.join(root_path, dataset, 'ir')
        vi_path = os.path.join(root_path, dataset, 'vi')
        save_path = os.path.join('result', dataset)
        os.makedirs(save_path, exist_ok=True)
        filenames = os.listdir(ir_path)
        print(f'dataset: {dataset}')
        for name in tqdm(filenames, total=len(filenames)):
            ir_image = Image.open(os.path.join(ir_path, name)).convert('L')
            vi_image = Image.open(os.path.join(vi_path, name)).convert('RGB')
            ir_image = transforms.ToTensor()(ir_image).unsqueeze(0).to(device)
            vi_image = transforms.ToTensor()(vi_image).unsqueeze(0).to(device)
            text = ['enhance the visibility of the image']
            with torch.no_grad():
                result = model(ir_image, vi_image, text, name)
            fusion_image = result.squeeze(0).permute(1, 2, 0)
            fusion_image = fusion_image.cpu().numpy()
            fusion_image = np.clip(fusion_image * 255, 0, 255).astype(np.uint8)
            fusion_image = Image.fromarray(fusion_image, mode='RGB')
            fusion_image.save(os.path.join(save_path, name))
        print('Finished testing')

if __name__ == '__main__':
    main()
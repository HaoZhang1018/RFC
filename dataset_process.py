from torchvision import transforms
import os
from torch.utils.data.dataset import Dataset
from PIL import Image
import torch


class dataset_gen(Dataset):
    def __init__(self, dataset_path, shared_scale):
        super(dataset_gen, self).__init__()
        self.datapath = dataset_path
        self.filenames_vi = os.listdir(os.path.join(self.datapath, 'vi'))
        self.maskname = os.listdir(os.path.join(self.datapath, 'groundtruth', 'mask'))
        self.length = len(self.filenames_vi)
        # self.scale_range = [320, 736]
        self.shared_scale = shared_scale
        self.current_scale = 576

    def __len__(self):
        return self.length

    def __getitem__(self, index):
        self.current_scale = self.shared_scale.value
        name = self.filenames_vi[index]
        ir_path = os.path.join(self.datapath, 'ir', name)
        vi_path = os.path.join(self.datapath, 'vi', name)
        text_path = os.path.join(self.datapath, 'text', name.split('.')[0] + '.txt')
        gtir_path = os.path.join(self.datapath, 'groundtruth', 'ir', name)
        gtvi_path = os.path.join(self.datapath, 'groundtruth', 'vi', name)

        raw_ir = Image.open(ir_path).convert('L')
        raw_vi = Image.open(vi_path).convert('RGB')
        gt_ir = Image.open(gtir_path).convert('L')
        gt_vi = Image.open(gtvi_path).convert('RGB')

        new_width = int(self.current_scale * 1.4)
        new_height = int(self.current_scale)

        raw_ir = raw_ir.resize((new_width, new_height), Image.LANCZOS)
        raw_vi = raw_vi.resize((new_width, new_height), Image.LANCZOS)
        gt_ir = gt_ir.resize((new_width, new_height), Image.LANCZOS)
        gt_vi = gt_vi.resize((new_width, new_height), Image.LANCZOS)
        image_ir = transforms.ToTensor()((raw_ir))
        image_vi = transforms.ToTensor()((raw_vi))
        gt_ir = transforms.ToTensor()((gt_ir))
        gt_vi = transforms.ToTensor()((gt_vi))

        assert image_vi.shape == gt_vi.shape, 'shape error'

        with open(text_path, 'r') as f:
            prompt = f.readline().strip()
        '''read mask'''
        if name in self.maskname:
            mask = Image.open(os.path.join(self.datapath, 'groundtruth', 'mask', name)).convert('L')
            mask = mask.resize((new_width, new_height), Image.LANCZOS)
            mask = transforms.ToTensor()(mask)
        else:
            mask = torch.zeros(1, new_height, new_width)
        assert image_ir.shape == gt_ir.shape == mask.shape, 'shape error'
        return image_ir, image_vi, gt_ir, gt_vi, prompt, mask, name.split('.')[0]


import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
from dataset_process import dataset_gen
from torch.utils.data import DataLoader, DistributedSampler
from net import Net
from loss import Fusionloss
import time
import logging
import torch
from torch.utils.tensorboard import SummaryWriter
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
from utils import RGB2YCrCb
from multiprocessing import Value, Lock
import random
import torch.distributed as dist
import datetime

train_path = '/data2/ZYP/dataset/RFC/train' #'./dataset/train/'
log_path = './log/'
checkpoint_path = './checkpoint/'
CLIPSEG_MODEL_PATH = 'CLIPSeg/weights/rd64-uni-refined.pth'

lr = 0.0002
epochs = 60

def train(rank, world_size, scale_range, shared_scale):
    setup(rank, world_size)
    lock = Lock()
    if dist.get_rank() == 0:
        writer = SummaryWriter(log_dir=f'runs')
    
    '''load my model'''
    model = Net(CLIPSEG_MODEL_PATH, rank).to(rank)
    model = DDP(model, device_ids=[rank])
    optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=lr, weight_decay=1e-4)
    start_epoch = 0
    '''
    checkpoint = torch.load(os.path.join(checkpoint_path, 'epoch40.pth'), weights_only=True, map_location='cuda:{}'.format(rank))
    model.module.load_state_dict(checkpoint['model_state_dict'], strict=False)
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    start_epoch = checkpoint['epoch']
    '''
    train_dataset = dataset_gen(train_path, shared_scale)
    print(f"train_dataset length:{train_dataset.length}")
    sampler = DistributedSampler(train_dataset, shuffle=True)
    
    train_loader = DataLoader(
        dataset=train_dataset,
        batch_size=1,
        shuffle=False,  # avoid shuffle
        num_workers=4,
        pin_memory=True,
        sampler=sampler
    )
    train_loader.n_iter = len(train_loader)

    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.65, last_epoch=start_epoch-1)

    Loss = Fusionloss(rank)
    
    accumulation_steps = 8  # grad accumulate
    optimizer.zero_grad()

    print('Start training...')
    start_time = time.time()
    for epoch in range(start_epoch, epochs):
        sampler.set_epoch(epoch)  # shuffle data
        for it, (image_ir, image_vi, gt_ir, gt_vi, prompts, mask, filename) in enumerate(train_loader):
            image_ir = image_ir.to(rank)
            image_vi = image_vi.to(rank)
            gt_ir = gt_ir.to(rank)
            gt_vi = gt_vi.to(rank)
            mask = mask.to(rank)
            '''model'''
            result, loss_d = model(image_ir, image_vi, prompts, filename)
            '''loss'''
            result_ycrcb = RGB2YCrCb(result)
            gtvi_ycrcb = RGB2YCrCb(gt_vi)
            loss_fusion, region_loss, no_region_loss = Loss(gt_ir, gtvi_ycrcb, result_ycrcb, mask)
            loss_total = loss_fusion+loss_d
            '''weight update'''
            loss_total.backward()
            if (it + 1) % accumulation_steps == 0 or (it + 1) == train_loader.n_iter:
              optimizer.step()
              optimizer.zero_grad()
            '''log'''
            batchs_passed = train_loader.n_iter * epoch + it + 1
            total_batchs = train_loader.n_iter * epochs
            end_time = time.time()
            batch_time = (end_time - start_time) / batchs_passed
            left_time = int(batch_time * (total_batchs - batchs_passed))
            left_time = str(datetime.timedelta(seconds=left_time))
            if dist.get_rank() == 0 and batchs_passed % 10 == 0:
                message = ', '.join([
                    f'E{epoch+1}',
                    f'S{batchs_passed}/{total_batchs}',
                    f'L: {loss_total:.4f}',
                    f'nr_loss: {no_region_loss:.4f}',
                    f'r_loss: {region_loss:.4f}', 
                    f'loss_d: {loss_d:.4f}', 
                    f'lr:{optimizer.param_groups[0]['lr']}',
                    f'left_time: {left_time}',
                ])
                print(message)
                logging.info(message)
                writer.add_scalar('Total Loss', loss_total, batchs_passed)
                writer.add_scalar('no_region_loss', no_region_loss, batchs_passed)
                writer.add_scalar('loss_d', loss_d, batchs_passed)
                writer.add_scalar('region_loss', region_loss, batchs_passed)
            
            if dist.get_rank() == 0 and batchs_passed % 200 == 0:
                new_scale = random.choice(range(scale_range[0], scale_range[1] + 1, 32))
                with lock:
                    shared_scale.value = new_scale
            
        scheduler.step()  # update lr
        if (epoch + 1) % 10 == 0 and dist.get_rank() == 0:
            temp_path = os.path.join(checkpoint_path, f'epoch{epoch + 1}.pth')
            state_dict = {k: v for k, v in model.module.state_dict().items() if model.module.get_parameter(k).requires_grad}
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': state_dict,  # state_dict
                'optimizer_state_dict': optimizer.state_dict(),
            }, temp_path)
            print(f'model saved at epoch {epoch + 1}')
            logging.info(f"Fusion Model Save to: {temp_path}")
    print('Finished Training')
    writer.close()
    torch.distributed.destroy_process_group() 


def setup(rank, world_size):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12356' 
    torch.distributed.init_process_group(
        backend="nccl",
        rank=rank,
        world_size=world_size
    )
    torch.cuda.set_device(rank)

if __name__ == "__main__":
    os.makedirs(log_path, exist_ok=True)
    os.makedirs(checkpoint_path, exist_ok=True)
    logging.basicConfig(
        filename=log_path + 'training.log',  
        filemode='a',  
        level=logging.INFO,  
        format='%(asctime)s - %(levelname)s - %(message)s'
    )

    scale_range = [480, 576]
    shared_scale = Value('i', scale_range[1], lock=False) 

    world_size = torch.cuda.device_count()
    for i in range(world_size):
        print(f"GPU {i}: {torch.cuda.get_device_name(i)}")
    mp.spawn(train, args=(world_size, scale_range, shared_scale), nprocs=world_size, join=True)



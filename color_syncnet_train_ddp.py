from os.path import dirname, join, basename, isfile
from tqdm import tqdm

from models import SyncNet_color as SyncNet
import audio

import torch
from torch import nn
from torch import optim
import torch.backends.cudnn as cudnn
from torch.utils import data as data_utils
import numpy as np

from glob import glob
from torch.utils.data import Dataset
import os, random, cv2, argparse
from hparams import hparams, HParams, get_image_list
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed import init_process_group, destroy_process_group
from torch.utils.data.distributed import DistributedSampler
import torch.multiprocessing as mp
from dotenv import load_dotenv
load_dotenv()
import wandb


# syncnet_T = 5
# syncnet_mel_step_size = 16


class Dataset(Dataset):
    def __init__(self, split, data_root, img_size=96, syncnet_T=5, syncnet_mel_step_size=16, sample_rate=16000, fps=25):
        self.all_videos = get_image_list(data_root, split)
        self.syncnet_T = syncnet_T
        self.syncnet_mel_step_size = syncnet_mel_step_size
        self.img_size = img_size
        self.sample_rate = sample_rate
        self.fps = fps 

    def get_frame_id(self, frame):
        return int(basename(frame).split('.')[0])

    def get_window(self, start_frame):
        start_id = self.get_frame_id(start_frame)
        vidname = dirname(start_frame)

        window_fnames = []
        for frame_id in range(start_id, start_id + self.syncnet_T):
            frame = join(vidname, '{}.jpg'.format(frame_id))
            if not isfile(frame):
                return None
            window_fnames.append(frame)
        return window_fnames

    def crop_audio_window(self, spec, start_frame):
        # num_frames = (T x hop_size * fps) / sample_rate
        start_frame_num = self.get_frame_id(start_frame)
        start_idx = int(80. * (start_frame_num / float(self.fps)))

        end_idx = start_idx + self.syncnet_mel_step_size

        return spec[start_idx : end_idx, :]


    def __len__(self):
        return len(self.all_videos)

    def __getitem__(self, idx):
        while 1:
            idx = random.randint(0, len(self.all_videos) - 1)
            vidname = self.all_videos[idx]

            img_names = list(glob(join(vidname, '*.jpg')))
            if len(img_names) <= 3 * self.syncnet_T:
                continue
            img_name = random.choice(img_names)
            wrong_img_name = random.choice(img_names)
            while wrong_img_name == img_name:
                wrong_img_name = random.choice(img_names)

            if random.choice([True, False]):
                y = torch.ones(1).float()
                chosen = img_name
            else:
                y = torch.zeros(1).float()
                chosen = wrong_img_name

            window_fnames = self.get_window(chosen)
            if window_fnames is None:
                continue

            window = []
            all_read = True
            for fname in window_fnames:
                img = cv2.imread(fname)
                if img is None:
                    all_read = False
                    break
                try:
                    img = cv2.resize(img, (self.img_size, self.img_size))
                except Exception as e:
                    all_read = False
                    break

                window.append(img)

            if not all_read: continue

            try:
                wavpath = join(vidname, "audio.wav")
                wav = audio.load_wav(wavpath, self.sample_rate)

                orig_mel = audio.melspectrogram(wav).T
            except Exception as e:
                continue

            mel = self.crop_audio_window(orig_mel.copy(), img_name)

            if (mel.shape[0] != self.syncnet_mel_step_size):
                continue

            # H x W x 3 * T
            x = np.concatenate(window, axis=2) / 255.
            x = x.transpose(2, 0, 1)
            x = x[:, x.shape[1]//2:]

            x = torch.FloatTensor(x)
            mel = torch.FloatTensor(mel.T).unsqueeze(0)

            return x, mel, y


def cosine_loss(a, v, y):
    logloss = nn.BCELoss()
    d = nn.functional.cosine_similarity(a, v)
    loss = logloss(d.unsqueeze(1), y)

    return loss

def train(device, model, train_data_loader, test_data_loader, optimizer,
          checkpoint_dir=None, checkpoint_interval=None, syncnet_eval_interval=None, 
          nepochs=None, start_epoch=0, global_step=0, best_eval_loss=1e3):

    
    resumed_step = start_epoch
    total_epochs = nepochs - start_epoch
    eval_loss = best_eval_loss 
    if device == 0:
        pbar = tqdm(total=total_epochs, desc='Training Progress')
    for global_epoch in range(start_epoch + 1, nepochs + 1):
        
        running_loss = 0.
        
        for step, (x, mel, y) in enumerate(train_data_loader):
            model.train()
            optimizer.zero_grad(set_to_none=True)

            # Transform data to CUDA device
            x = x.to(device)

            mel = mel.to(device)

            a, v = model(mel, x)
            y = y.to(device)

            loss = cosine_loss(a, v, y)
            loss.backward()
            optimizer.step()

            global_step += 1
            cur_session_steps = global_step - resumed_step
            running_loss += loss.item()

            # if global_step % checkpoint_interval and device == 0:
            #     save_checkpoint(
            #         model, optimizer, global_step, checkpoint_dir, global_epoch)

            if global_step % syncnet_eval_interval == 0 and device == 0:
                with torch.no_grad():
                    eval_loss = eval_model(test_data_loader, global_step, device, model, checkpoint_dir)
                    if eval_loss < best_eval_loss:
                        best_eval_loss = eval_loss
                        save_checkpoint(
                            model, optimizer, global_step, checkpoint_dir, global_epoch, prefix='best_', best_eval_loss=best_eval_loss)
                wandb.log({'eval/best_eval_loss': best_eval_loss, 'eval/loss': eval_loss}, step=global_step)
                wandb.log({'train/loss': running_loss / (step + 1), 'epoch': global_epoch}, step=global_step)
        if device == 0:
            pbar.set_description('epoch: {} Loss: {} eval loss: {} best_eval_loss: {}'.format(global_epoch, running_loss / (step + 1), eval_loss, best_eval_loss))
            pbar.update(1)
    if device == 0:
        pbar.close()

def eval_model(test_data_loader, global_step, device, model, checkpoint_dir):
    eval_steps = 1400
    print('Evaluating for {} steps'.format(eval_steps))
    losses = []
    while 1:
        for step, (x, mel, y) in enumerate(test_data_loader):

            model.eval()

            # Transform data to CUDA device
            x = x.to(device)

            mel = mel.to(device)

            a, v = model(mel, x)
            y = y.to(device)

            loss = cosine_loss(a, v, y)
            losses.append(loss.item())

            if step > eval_steps: break

        averaged_loss = sum(losses) / len(losses)
        #print(averaged_loss)
        #wandb.log({'eval/loss': averaged_loss})
        return averaged_loss

def save_checkpoint(model, optimizer, step, checkpoint_dir, epoch, prefix='', best_eval_loss=1e3):

    checkpoint_path = join(
        checkpoint_dir, "{}checkpoint_step{:09d}.pth".format(prefix, step))
    optimizer_state = optimizer.state_dict() 
    torch.save({
        "state_dict": model.module.state_dict(), # for ddp model 
        "optimizer": optimizer_state,
        "global_step": step,
        "global_epoch": epoch,
        "best_eval_loss": best_eval_loss
    }, checkpoint_path)
    print("Saved checkpoint:", checkpoint_path)

def _load(checkpoint_path, device):
    if device != 'cpu':
        checkpoint = torch.load(checkpoint_path, map_location='cuda:{}'.format(device))
    else:
        checkpoint = torch.load(checkpoint_path,
                                map_location=lambda storage, loc: storage)
    return checkpoint

def load_checkpoint(path, model, optimizer, device, reset_optimizer=False):
   

    print("Load checkpoint from: {}".format(path))
    checkpoint = _load(path, device)
    model.load_state_dict(checkpoint["state_dict"])
    if not reset_optimizer:
        optimizer_state = checkpoint["optimizer"]
        if optimizer_state is not None:
            print("Load optimizer state from {}".format(path))
            optimizer.load_state_dict(checkpoint["optimizer"])
    global_step = checkpoint["global_step"]
    global_epoch = checkpoint["global_epoch"]
    best_eval_loss = checkpoint.get("best_eval_loss", 1e3)

    return model, global_step, global_epoch, best_eval_loss


def ddp_setup(rank, world_size):
    """
    Args:
        rank: Unique identifier of each process
        world_size: Total number of processes
    """
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "12355"
    torch.cuda.set_device(rank)
    init_process_group(backend="nccl", rank=rank, world_size=world_size)

def main(rank: int, world_size: int, hparams:HParams):
    ddp_setup(rank, world_size)
   
    if rank == 0: 
         # 加载 wandb 配置并登录
        wandb_api_key = os.getenv('WANDB_API_KEY')
        wandb.login(key=wandb_api_key, host='http://10.10.185.1:8080')
        if hparams.checkpoint_path is not None:
            wandb.init(entity='lingz0124', project=hparams.project_name, id=hparams.wandb_id, resume="must")
        else:
            wandb.init(project=hparams.project_name, config=hparams, name=hparams.model_name + '_' + hparams.experiment_id)
         # 手动同步到 hparams 对象
        for key in wandb.config.keys():
            if hasattr(hparams, key):
                setattr(hparams, key, wandb.config[key])
        hparams.set_hparam('wandb_id', wandb.run.id)
        if not os.path.exists(hparams.checkpoint_dir): os.mkdir(hparams.checkpoint_dir)
        # save yaml configuration file 
        config_file = os.path.join(hparams.checkpoint_dir, 'hparams.yaml')
        hparams.save_to_yaml(config_file)
    
        hparams_list = [hparams]
    else:
        hparams_list = [None]
    torch.distributed.broadcast_object_list(hparams_list, src=0)
    hparams = hparams_list[0]

    # Dataset and Dataloader setup
    dataset_params = {
        'data_root': hparams.data_root,
        'img_size': hparams.img_size,
        'syncnet_T': hparams.syncnet_T,
        'syncnet_mel_step_size': hparams.syncnet_mel_step_size,
        'sample_rate': hparams.sample_rate,
        'fps': hparams.fps
    }
    train_dataset = Dataset('train',  **dataset_params)
    test_dataset = Dataset('val',  **dataset_params)

    train_data_loader = data_utils.DataLoader(
        train_dataset, batch_size=hparams.syncnet_batch_size, shuffle=False,
        sampler=DistributedSampler(train_dataset, shuffle=True),
        num_workers=hparams.num_workers, 
        pin_memory=True,  # 加速数据加载
        drop_last=True     # 避免最后一个不完整 batch
        )

    test_data_loader = data_utils.DataLoader(
        test_dataset, batch_size=hparams.syncnet_batch_size,
        shuffle=False, sampler=DistributedSampler(test_dataset, shuffle=False),
        num_workers=hparams.num_workers, pin_memory=True)

    device = rank 
    # Model
    model = SyncNet().to(device)
   
    print('total trainable params {}'.format(sum(p.numel() for p in model.parameters() if p.requires_grad)))

    optimizer = optim.Adam([p for p in model.parameters() if p.requires_grad],
                           lr=hparams.syncnet_lr)

    if hparams.checkpoint_path is not None:
        model, global_step, global_epoch, best_eval_loss=load_checkpoint(hparams.checkpoint_path, model, optimizer, device, reset_optimizer=False)
    else:
        global_step = 0
        global_epoch = 0
        best_eval_loss = 1e3
    model = DDP(model, device_ids=[device])
    train(device, model, train_data_loader, test_data_loader, optimizer,
          checkpoint_dir=hparams.checkpoint_dir,
          checkpoint_interval=hparams.syncnet_checkpoint_interval,
          syncnet_eval_interval=hparams.syncnet_eval_interval, 
          nepochs=hparams.nepochs, start_epoch=global_epoch, global_step=global_step, best_eval_loss=best_eval_loss)
    if rank == 0:
        wandb.finish()
    destroy_process_group()



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Code to train the expert lip-sync discriminator')

    parser.add_argument("--data_root", help="Root folder of the preprocessed LRS2 dataset", required=True)

    parser.add_argument('--checkpoint_dir', help='Save checkpoints to this directory', required=True, type=str)
    parser.add_argument('--checkpoint_path', help='Resumed from this checkpoint', default=None, type=str)
    parser.add_argument('--config_file', help='config yaml file', default=None, type=str)


    args = parser.parse_args()
    if args.config_file is not None:
        hparams.load_from_yaml(args.config_file) # override hyperparameters with config file
    # combine with args
    hparams.update_params(args)
    print(f'hparams: {hparams.data}')
    world_size = torch.cuda.device_count()  
    print(f'world_size: {world_size}')
    mp.spawn(main, args=(world_size, hparams), nprocs=world_size)
    

   
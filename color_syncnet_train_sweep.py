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

import os, random, cv2, argparse
from hparams import hparams, get_image_list
from tools.utils import timing_decorator
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed import init_process_group, destroy_process_group
from torch.utils.data.distributed import DistributedSampler
from contextlib import nullcontext

from dotenv import load_dotenv
load_dotenv()
wandb_api_key = os.getenv('WANDB_API_KEY')
import wandb
wandb.login(key=wandb_api_key, host='http://10.10.185.1:8080')
parser = argparse.ArgumentParser(description='Code to train the expert lip-sync discriminator')

parser.add_argument("--data_root", help="Root folder of the preprocessed LRS2 dataset", required=True)

parser.add_argument('--checkpoint_dir', help='Save checkpoints to this directory', required=True, type=str)
parser.add_argument('--checkpoint_path', help='Resumed from this checkpoint', default=None, type=str)
parser.add_argument('--config_file', help='config yaml file',default=None, type=str)


args = parser.parse_args()





use_cuda = torch.cuda.is_available()
print('use_cuda: {}'.format(use_cuda))

syncnet_T = 5
syncnet_mel_step_size = 16

 # various inits, derived attributes, I/O setup
ddp = int(os.environ.get('RANK', -1)) != -1 # is this a ddp run?
if ddp:
    init_process_group(backend=hparams.backend)
    ddp_rank = int(os.environ['RANK'])
    ddp_local_rank = int(os.environ['LOCAL_RANK'])
    ddp_world_size = int(os.environ['WORLD_SIZE'])
    device = f'cuda:{ddp_local_rank}'
    torch.cuda.set_device(device)
    master_process = ddp_rank == 0 # this process will do logging, checkpointing etc.
    seed_offset = ddp_rank # each process gets a different seed
    # world_size number of processes will be training simultaneously, so we can scale
    # down the desired gradient accumulation iterations per process proportionally
    if hparams.gradient_accumulation_steps > 1:
        assert hparams.gradient_accumulation_steps % ddp_world_size == 0
        hparams.gradient_accumulation_steps //= ddp_world_size
else:
    # if not ddp, we are running on a single gpu, and one process
    master_process = True
    seed_offset = 0
    ddp_world_size = 1
    device = 'cuda:0' if use_cuda else 'cpu'

torch.manual_seed(1337 + seed_offset)
random.seed(1337 + seed_offset)
np.random.seed(1337 + seed_offset)
torch.backends.cuda.matmul.allow_tf32 = True # allow tf32 on matmul
torch.backends.cudnn.allow_tf32 = True # allow tf32 on cudnn
device_type = 'cuda' if 'cuda' in device else 'cpu' # for later use in torch.autocast
# note: float16 data type will automatically use a GradScaler
ptdtype = torch.float32 #{'float32': torch.float32, 'bfloat16': torch.bfloat16, 'float16': torch.float16}[hparams.dtype]

ctx = nullcontext() if device_type == 'cpu' else torch.amp.autocast(device_type=device_type, dtype=ptdtype) 




class Dataset(object):
    def __init__(self, split):
        self.all_videos = get_image_list(args.data_root, split)

    def get_frame_id(self, frame):
        return int(basename(frame).split('.')[0])

    def get_window(self, start_frame):
        start_id = self.get_frame_id(start_frame)
        vidname = dirname(start_frame)

        window_fnames = []
        for frame_id in range(start_id, start_id + syncnet_T):
            frame = join(vidname, '{}.jpg'.format(frame_id))
            if not isfile(frame):
                return None
            window_fnames.append(frame)
        return window_fnames

    def crop_audio_window(self, spec, start_frame):
        # num_frames = (T x hop_size * fps) / sample_rate
        start_frame_num = self.get_frame_id(start_frame)
        start_idx = int(80. * (start_frame_num / float(hparams.fps)))

        end_idx = start_idx + syncnet_mel_step_size

        return spec[start_idx : end_idx, :]


    def __len__(self):
        return len(self.all_videos)

    def __getitem__(self, idx):
        while 1:
            idx = random.randint(0, len(self.all_videos) - 1)
            vidname = self.all_videos[idx]

            img_names = list(glob(join(vidname, '*.jpg')))
            if len(img_names) <= 3 * syncnet_T:
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
                    img = cv2.resize(img, (hparams.img_size, hparams.img_size))
                except Exception as e:
                    all_read = False
                    break

                window.append(img)

            if not all_read: continue

            try:
                wavpath = join(vidname, "audio.wav")
                wav = audio.load_wav(wavpath, hparams.sample_rate)

                orig_mel = audio.melspectrogram(wav).T
            except Exception as e:
                continue

            mel = self.crop_audio_window(orig_mel.copy(), img_name)

            if (mel.shape[0] != syncnet_mel_step_size):
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
          checkpoint_dir=None, checkpoint_interval=None, nepochs=None):

    global_step = 0
    global_epoch = 0
    best_eval_loss = 1e3 
    resumed_step = global_step
    raw_model = model.module if ddp else model
    # 创建一个总的进度条，表示整个训练过程
    total_steps = nepochs * len(train_data_loader)
    with tqdm(total=nepochs, desc='Training Progress') as pbar:
        while global_epoch < nepochs:
            #print('Starting Epoch: {}'.format(global_epoch))
            running_loss = 0.
            #prog_bar = tqdm(enumerate(train_data_loader))
            #train_data_loader.sampler.set_epoch(global_epoch) 
            for step, (x, mel, y) in enumerate(train_data_loader):
                model.train()
                optimizer.zero_grad()

                # Transform data to CUDA device
                x = x.to(device)

                mel = mel.to(device)
                y = y.to(device)
                with ctx:
                    a, v = model(mel, x)
                   

                loss = cosine_loss(a, v, y)
                loss.backward()
                optimizer.step()

                global_step += 1
                cur_session_steps = global_step - resumed_step
                running_loss += loss.item()
                # only save best eval model, uncomment to save all
                # if global_step == 1 or global_step % checkpoint_interval == 0:
                #     save_checkpoint(
                #         model, optimizer, global_step, checkpoint_dir, global_epoch)

                if global_step % hparams.syncnet_eval_interval == 0 and master_process:
                    with torch.no_grad():
                        eval_loss = eval_model(test_data_loader, global_step, device, model, checkpoint_dir)
                        if eval_loss < best_eval_loss:
                            best_eval_loss = eval_loss
                            save_checkpoint(
                                raw_model, optimizer, global_step, checkpoint_dir, global_epoch, prefix='best_')
                    wandb.log({'eval/best_eval_loss': best_eval_loss, 'eval/loss': eval_loss})
                    wandb.log({'train/loss': running_loss / (step + 1), "epoch": global_epoch + 1})
           
            pbar.set_description(f'Epoch {global_epoch + 1}/{nepochs} Loss: {running_loss / (step + 1):.4f}')
            pbar.update(1)
            global_epoch += 1
            
@torch.no_grad() 
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
            y = y.to(device)
            with ctx:
                a, v = model(mel, x)

            loss = cosine_loss(a, v, y)
            losses.append(loss.item())

            if step > eval_steps: break

        averaged_loss = sum(losses) / len(losses)
        #print(averaged_loss)
        #wandb.log({'eval/loss': averaged_loss})
        return averaged_loss

def save_checkpoint(model, optimizer, step, checkpoint_dir, epoch, prefix=''):

    checkpoint_path = join(
        checkpoint_dir, "{}checkpoint_step{:09d}.pth".format(prefix, step))
    optimizer_state = optimizer.state_dict() if hparams.save_optimizer_state else None
    torch.save({
        "state_dict": model.state_dict(),
        "optimizer": optimizer_state,
        "global_step": step,
        "global_epoch": epoch,
    }, checkpoint_path)
    print("Saved checkpoint:", checkpoint_path)

def _load(checkpoint_path, device):
    if use_cuda:
        checkpoint = torch.load(checkpoint_path, map_location=device)
    else:
        checkpoint = torch.load(checkpoint_path,
                                map_location=lambda storage, loc: storage)
    return checkpoint

def load_checkpoint(path, device, model, optimizer,  reset_optimizer=False):
    global global_step
    global global_epoch

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

    return model

def update_hparams(hparams, args):
    for key, value in args.items():
        if hasattr(hparams, key):
            setattr(hparams, key, value)
        else:
            hparams.set_hparam(key, value)

@timing_decorator 
def run():
   
   
    # 初始化 wandb
    if master_process:
        # 将args 参数更新到 hparams 对象
        if args.config_file is not None:
            hparams.load_from_yaml(args.config_file) # override hyperparameters with config file
        update_hparams(hparams, vars(args))
        wandb.init(project=hparams.project_name, config=hparams.data)
    
        # 将sweep搜索的超参数更新到hparams
        config = wandb.config
        # 手动同步到 hparams 对象
        for key in wandb.config.keys():
            if hasattr(hparams, key):
                setattr(hparams, key, wandb.config[key])

        print("Updated hparams:", hparams)
        run_id = wandb.run.id 
        checkpoint_dir = os.path.join(args.checkpoint_dir, run_id)
        checkpoint_path = args.checkpoint_path
        if not os.path.exists(checkpoint_dir): 
            os.mkdir(checkpoint_dir)
        # save yaml configuration file 
        config_file = os.path.join(checkpoint_dir, 'hparams.yaml')
        hparams.save_to_yaml(config_file)
    # 分布式同步：确保所有进程加载相同 hparams
    if ddp:
        torch.distributed.barrier()  # 等待主进程完成文件写入
        hparams.load_from_yaml(config_file)  # 所有进程读取同一配置
    # Dataset and Dataloader setup
    train_dataset = Dataset('train')
    test_dataset = Dataset('val')

    if ddp:
        train_sampler = DistributedSampler(train_dataset, shuffle=True)
        test_sampler = DistributedSampler(test_dataset, shuffle=False)
    else:
        train_sampler = None
        test_sampler = None
    num_workers = hparams.num_workers // ddp_world_size if ddp else hparams.num_workers
    if ddp:
        batch_size_per_gpu = hparams.syncnet_batch_size // ddp_world_size
    else:
        batch_size_per_gpu = hparams.syncnet_batch_size
    train_data_loader = data_utils.DataLoader(
        train_dataset, 
        batch_size=batch_size_per_gpu, 
        sampler=train_sampler,
        shuffle=(not ddp),  # DDP 时用 sampler 控制 shuffle
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True
    )

    test_data_loader = data_utils.DataLoader(
        test_dataset, 
        batch_size=batch_size_per_gpu,
        sampler=test_sampler,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=False
    )
   
    # train_data_loader = data_utils.DataLoader(
    #     train_dataset, batch_size=hparams.syncnet_batch_size, shuffle=True,
    #     num_workers=hparams.num_workers)

    # test_data_loader = data_utils.DataLoader(
    #     test_dataset, batch_size=hparams.syncnet_batch_size,
    #     num_workers=8)

    #device = torch.device(device if use_cuda else "cpu")

    # Model
    model = SyncNet().to(device)
    print('total trainable params {}'.format(sum(p.numel() for p in model.parameters() if p.requires_grad)))

    optimizer = optim.Adam([p for p in model.parameters() if p.requires_grad],
                           lr=hparams.syncnet_lr)

    if checkpoint_path is not None:
        load_checkpoint(checkpoint_path, device, model, optimizer, reset_optimizer=False)
    
    # # compile the model
    # if hparams.compile:
    #     print("compiling the model... (takes a ~minute)")
    #     unoptimized_model = model
    #     model = torch.compile(model) # requires PyTorch 2.0

    # wrap model into DDP container
    if ddp:
        model = DDP(model, device_ids=[ddp_local_rank])

    train(device, model, train_data_loader, test_data_loader, optimizer,
          checkpoint_dir=checkpoint_dir,
          checkpoint_interval=hparams.syncnet_checkpoint_interval,
          nepochs=hparams.nepochs)

    if ddp:
        destroy_process_group()
sweep_config = {    
    'method': 'bayes',
    'name': 'syncnet_sweep',
    'metric': {'name': 'eval/loss', 'goal': 'minimize'},
    'parameters': {
        'syncnet_lr': {
            'min': 0.00001,
            'max': 0.001,
            'distribution': 'log_uniform_values'
        },
        'syncnet_batch_size': {
            'values': [32, 64, 128]
        }
    },
    'early_terminate': {
        'type': 'hyperband',
        'min_iter': 3,
        'eta': 2
    }, 
    # 添加全局终止条件
    'stop': {
        'max_runs': 50,  # 最多运行50次试验
    }
}
sweep_id = wandb.sweep(sweep_config, project=hparams.project_name, entity='lingz0124')

wandb.agent(sweep_id, function=run, count=5)

# if __name__ == "__main__":

#     run(args)
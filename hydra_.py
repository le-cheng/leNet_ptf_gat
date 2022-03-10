
import hydra
import torch
from torch import distributed
import torch.multiprocessing as mp
from omegaconf import DictConfig, OmegaConf

from utils import find_free_port, get_logger


@hydra.main(config_path='config', config_name='cls')
def mmmm(cfg: DictConfig):
    port = find_free_port()
    cfg.dist_url = f"tcp://localhost:{port}"
    cfg.ngpus_per_node = torch.cuda.device_count()
    cfg.world_size = cfg.ngpus_per_node * cfg.nodes   
    print('\nWorld_size: {}\n'.format(cfg.world_size))  
    mp.spawn(main, nprocs=cfg.ngpus_per_node, args=(cfg,)) 

def main(gpu, cfg):
    
            # 起始打印
    distributed.init_process_group(
                backend='nccl', init_method=cfg.dist_url, world_size=cfg.world_size, rank=gpu)   
    if gpu == 0:
        logger =  get_logger('multitrain.log')
        a = cfg.manual_seed    
    if gpu == 0:   
        logger.info("=> start code ...")
    print(1)

if __name__ == '__main__':
    mmmm()

from omegaconf import OmegaConf
from ..uda_models import uda_models


def load_config_runtime(
    config_path
):
    cfg = OmegaConf.load(config_path)

    # process generator args from default and config
    gen_args = OmegaConf.create({
        'generator_args': {
            cfg.training.generator: OmegaConf.merge(
                OmegaConf.structured(uda_models.make_dataclass_from_args())[cfg.training.generator], 
                OmegaConf.structured(cfg.generator_args)
            )
        }
    })
    cfg.generator_args.clear()
    cfg = OmegaConf.merge(cfg, gen_args)
    
    return cfg

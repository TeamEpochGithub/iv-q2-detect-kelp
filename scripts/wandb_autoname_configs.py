from omegaconf import OmegaConf

from src.utils.replace_list_with_dict import replace_list_with_dict
import wandb
from copy import deepcopy

import wandb
from datetime import datetime, timedelta

time_window = timedelta(days=7)

if __name__ == "__main__":
    # Initialize the Wandb API
    api = wandb.Api()

    # Get the current date and time
    now = datetime.now()

    # iterate over all runs
    total = 0
    updated = 0
    for run in wandb.Api().runs("team-epoch-iv/detect-kelp"):
        if datetime.strptime(run.created_at, "%Y-%m-%dT%H:%M:%S") < (now - time_window):
            # wandb returns run sorted by date, so we can break the loop once we reach runs older than 2 days
            break
        cfg = run.config
        cfg_dict = replace_list_with_dict(deepcopy(cfg))

        # convert the dictionary to an OmegaConf object
        conf = OmegaConf.create(cfg_dict)

        if 'model' not in conf:
            # skip ensembles etc
            continue
        try:
            # define zero to be either a string or an integer
            zero = list(conf.model.model_loop_pipeline.model_blocks_pipeline.model_blocks.keys())[0]
            one = list(conf.model.model_loop_pipeline.pretrain_pipeline.pretrain_steps.keys())[1]

            # get the name of the model
            name = \
            conf.model.model_loop_pipeline.model_blocks_pipeline.model_blocks[zero].model.model._target_.split(".")[-1]
            if name == "Unet":
                name = f"{conf.model.model_loop_pipeline.model_blocks_pipeline.model_blocks[zero].model.model.encoder_name}-{name}"
            name += f"-f{len(conf.model.model_loop_pipeline.pretrain_pipeline.pretrain_steps[one].columns)}"
            name += f"-bs{conf.model.model_loop_pipeline.model_blocks_pipeline.model_blocks[zero].batch_size}"
            name += f"-e{conf.model.model_loop_pipeline.model_blocks_pipeline.model_blocks[zero].epochs}"
            name += f"-{conf.model.model_loop_pipeline.model_blocks_pipeline.model_blocks[zero].criterion._target_.split('.')[-1]}"

            if 'autoname' in run.config and run.config['autoname'] == name:
                continue

            # add a field 'autoname'
            print(name)
            updated += 1
            run.config['autoname'] = name
            run.update()
        except:
            print('Skipping', run.name)

        total += 1
        print(total, 'runs processed')
    print('Updated', updated, 'runs')

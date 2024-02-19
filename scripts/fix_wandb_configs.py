from src.utils.replace_list_with_dict import replace_list_with_dict
import wandb
from copy import deepcopy

if __name__ == "__main__":
    # log in to wandb
    wandb.login()

    # iterate over all runs
    total = 0
    updated = 0
    for run in wandb.Api().runs("team-epoch-iv/detect-kelp"):
        cfg = run.config
        cfg_dict = replace_list_with_dict(deepcopy(cfg))
        if cfg != cfg_dict:
            print('Updating', run.name)
            run.config.update(cfg_dict)
            run.update()
            updated += 1
        total += 1
        print(total, 'runs processed')
    print('Updated', updated, 'runs')

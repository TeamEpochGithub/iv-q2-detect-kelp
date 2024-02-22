from src.utils.replace_list_with_dict import replace_list_with_dict
import wandb
from copy import deepcopy

latest_update = "2024-02-19T19:00:00"

if __name__ == "__main__":
    # log in to wandb
    wandb.login()

    # iterate over all runs
    total = 0
    updated = 0
    for run in wandb.Api().runs("team-epoch-iv/detect-kelp"):
        if run.created_at < latest_update:
            # wandb returns run sorted by date, so we can break the loop once we reach runs older than 2 days
            break
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

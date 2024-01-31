import os
import shutil

# Set working directory to root of the project.

if __name__ == "__main__":
    to_move = [
        "WU193724",
        "VC433864",
        "KA317221",
        "UE455968",
        "PJ346120",
        "OB141843",
        "ED316890",
        "IP560237",
        "CH665185",
        "FL603399",
        "YI406351",
        "TF297833",
        "GW781453",
        "OD359870",
        "LP202365",
    ]
    to_move_satellite = [f"{i}_satellite.tif" for i in to_move]
    to_move_kelp = [f"{i}_kelp.tif" for i in to_move]
    train_dir = "data/raw/train_satellite"
    labels_dir = "data/raw/train_kelp"

    move_dir_satellite = "data/moved/train_satellite"
    move_dir_kelp = "data/moved/train_kelp"

    # Create data/moved if it does not exist
    if not os.path.exists("data/moved"):
        os.makedirs("data/moved")

    if not os.path.exists(move_dir_satellite):
        os.makedirs(move_dir_satellite)

    if not os.path.exists(move_dir_kelp):
        os.makedirs(move_dir_kelp)

    try:
        for i in to_move_satellite:
            shutil.move(os.path.join(train_dir, i), os.path.join(move_dir_satellite, i))

        for i in to_move_kelp:
            shutil.move(os.path.join(labels_dir, i), os.path.join(move_dir_kelp, i))
    except:
        print("Error moving files, they may already be moved.")
    else:
        print(f"All files moved to {move_dir_satellite} and {move_dir_kelp}.")

    # Assert that the files are not in the original directory
    for i in to_move_satellite:
        assert not os.path.exists(os.path.join(train_dir, i))

    for i in to_move_kelp:
        assert not os.path.exists(os.path.join(labels_dir, i))

    # Get number of files in data/raw/train_satellite
    num_files = len(os.listdir(train_dir))

    # Get number of files in data/raw/train_kelp
    num_files = len(os.listdir(labels_dir))
    print(f"Number of files in data/raw/train_satellite: {num_files}")
    print(f"Number of files in data/raw/train_kelp: {num_files}")

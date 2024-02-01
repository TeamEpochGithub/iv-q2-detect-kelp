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
    train_dir = "data/moved/train_satellite"
    labels_dir = "data/moved/train_kelp"

    move_dir_satellite = "data/raw/train_satellite"
    move_dir_kelp = "data/raw/train_kelp"

    try:
        for i in to_move_satellite:
            shutil.move(os.path.join(train_dir, i), os.path.join(move_dir_satellite, i))

        for i in to_move_kelp:
            shutil.move(os.path.join(labels_dir, i), os.path.join(move_dir_kelp, i))
    except:
        print("Error moving files, they may already be moved.")
    else:
        print(f"All files moved successfully to {move_dir_satellite} and {move_dir_kelp}")

    # Assert that the files are not in the original directory
    for i in to_move_satellite:
        assert not os.path.exists(os.path.join(train_dir, i))

    for i in to_move_kelp:
        assert not os.path.exists(os.path.join(labels_dir, i))

    # Get number of files in data/raw/train_satellite
    num_files = len(os.listdir(move_dir_satellite))

    # Get number of files in data/raw/train_kelp
    num_files_kelp = len(os.listdir(move_dir_kelp))

    print(f"Number of files in data/raw/train_satellite: {num_files}")
    print(f"Number of files in data/raw/train_kelp: {num_files_kelp}")

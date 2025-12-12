Create conda environment:
conda env create -f env.yml

Activate yml environment:
conda activate fovenv

Generate pooled fine tuned training parquets:
python -m scripts.build_avtrack360_pooled

To see the users in pooled training parquet:
python -c "import pandas as pd; df = pd.read_parquet('data/avtrack360_train.parquet'); print(sorted(df['user_id'].unique()))"

And in the finetune parquet:
python -c "import pandas as pd; df = pd.read_parquet('data/avtrack360_val.parquet'); print(sorted(df['user_id'].unique()))"

The lists are here in case you would rather skip those steps:
Pooled training user ids:
[3, 4, 5, 6, 7, 8, 10, 11, 12, 13, 14, 15, 16, 18, 19, 20, 21, 22, 24, 25, 26, 28, 31, 32, 33, 34, 35, 37, 38, 39, 40, 41, 42, 43, 44, 46, 47, 48]

Fine tune training user ids:
[2, 9, 17, 23, 27, 29, 30, 36, 45, 49]

Pick any user id in the fine tune training list and fine tune the chosen user:
python -m scripts.finetune_user --user_id <val_user_id>

Then run evaluate system for Final Exam task of week 4:
python -m scripts.evaluate_system --user_id <val_user_id>

Outside of the print results as 3 graphs for each horizon will also be in:
<repo_path>/eft-conformal-fov-360/results/eval_user_<id>_prefetch.png
<repo_path>/eft-conformal-fov-360/results/eval_user_<id>_deadline.png
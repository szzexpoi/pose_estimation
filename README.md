# pose_estimation

This code implements a simple framework for articulated pose estimation. In the current stage, the experiments are restricted to LSP dataset.

### Generating clustering assignment
```
python cluster_relation.py --data ./joints_scaled_cropped.mat --human_exp_dir xxx --num_cluster 11
```

### Training ResNet baseline on individual joints
```
python main.py --checkpoint_dir ./log_baseline --batch_size 32 --lr 1e-3 --joint_size 60
```

### Training Side-by-side model
After training the baseline on individual joints, call

```
python main_pair.py --checkpoint_dir ./log_side_by_side --joint_size 60 --batch_size 32 --lr 1e-2 --pretrained ./log_baseline/model_best.pth --freeze_backbone 1 --hidden_size 100

```

### Training Relational model
First training the (partial) model on individual joints:
```
python main.py --checkpoint_dir ./log_rel_individual --batch_size 32 --lr 1e-3 --joint_size 60 --use_rel 1
```

```
python main_pair.py --checkpoint_dir ./log_side_by_side --joint_size 60 --batch_size 32 --lr 1e-2 --pretrained ./log_rel_individual/model_best.pth --freeze_backbone 1 --hidden_size 100 --use_rel 1 --use_pos 1

```

### Evaluating Models on individual joints
```
python main.py --mode eval --weights ./log_xxx/model_best.pth --batch_size 32 --lr 1e-3 --joint_size 60
```
Note that `use_rel` should be set accordingly.

### Evaluating Models on target-cue pairs
```
python main_pair.py --weights ./log_xxx/model_best.pth --joint_size 60 --batch_size 32 --lr 1e-2 --pretrained ./log_rel_individual/model_best.pth --freeze_backbone 1 --hidden_size 100

```
Note that `use_rel`, `use_pos` should be set accordingly. There are also three arguments for ablation study, namely:
1. `--ablation`: for testing with 65 images selected for human experiments.
2. `--rotate`: for testing with different degrees of rotation on cue.
3. `--cue_size`: for testing with a cue size different from the target one. 

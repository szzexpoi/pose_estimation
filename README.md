# pose_estimation

This code implements a simple framework for articulated pose estimation. In the current stage, the experiments are restricted to LSP dataset.

### Generating clustering assignment
```
python cluster_relation.py --data ./joints_scaled_cropped.mat --human_exp_dir xxx --num_cluster 11
```

### Training
```
python main.py --checkpoint_dir ./log_baseline --alpha 1 --batch_size 32 --alpha 0.5 --lr 1e-5 --joint_size 60
```

#################### MODEL w Speed
#Scene10 Input:Shared Lidar Output: Control
#python -m training.train_lidar_wSpeed --data ~/Documents/AutoCast_Shared_Dataset --ego-only --batch-size 64 --num-dataloader-workers 8 --shared True -T 5 --init-lr 0.0001 --num-steps-per-log 100  --num-epochs 201 --frame-stack 2 
#Scene 6 Input: Shared Lidar Output: Control
#python -m training.train_lidar_wSpeed --data ~/Documents/AutoCast_Scene6_Dataset --ego-only --batch-size 64 --num-dataloader-workers 8 --shared True -T 5 --init-lr 0.0001 --num-steps-per-log 100  --num-epochs 201 --frame-stack 2 
#!/bin/bash
DATA_FOLDER=CoDrive_Training
#Scene6 Input:Shared Lidar Output:Brake and path
#python -m training.train_lidar --data ~/Documents/AutoCast_Scene6_Dataset --ego-only --batch-size 64 --num-dataloader-workers 8 --shared True -T 5 --init-lr 0.001 --num-steps-per-log 100  --num-epochs 101

#Scene6 Input:Non-Shared Lidar Output:Brake and path
#python -m training.train_lidar --data ~/Documents/AutoCast_Scene6_Dataset --ego-only --batch-size 64 --num-dataloader-workers 8 --shared False -T 5 --init-lr 0.001 --num-steps-per-log 100  --num-epochs 101

#Scene10 Input:Shared Lidar Output:Brake and path
#python -m training.train_lidar --data ~/Documents/AutoCast_Shared_Dataset --ego-only --batch-size 64 --num-dataloader-workers 8 --shared True -T 5 --init-lr 0.001 --num-steps-per-log 100  --num-epochs 101

#Scene10 Input:Non-Shared Lidar Output:Brake and path
#python -m training.train_lidar --data ~/Documents/AutoCast_Shared_Dataset --ego-only --batch-size 64 --num-dataloader-workers 8 --shared False -T 5 --init-lr 0.001 --num-steps-per-log 100  --num-epochs 101

#################### MODEL w Speed
#Scene10 Input:Shared Lidar Output: Control
#python3 -m training.train_lidar_wSpeed \
#  --data ${DATA_FOLDER} \
#  --ego-only \
#  --batch-size 64 \
#  --num-dataloader-workers 8 \
#  --shared True \
#  -T 5 \
#  --init-lr 0.0001 \
#  --num-steps-per-log 100 \
#  --num-epochs 201 \
#  --frame-stack 2
#Scene 6 Input: Shared Lidar Output: Control
#python3 -m training.train_lidar_wSpeed \
#  --data ${DATA_FOLDER} \
#  --ego-only \
#  --batch-size 64 \
#  --num-dataloader-workers 8 \
#  --shared True \
#  -T 5 \
#  --init-lr 0.0001 \
#  --num-steps-per-log 100  \
#  --num-epochs 201 \
#  --frame-stack 2
#--use-speed



#Scene10 Input:Non-Shared Lidar Output:Brake and path
#python -m training.train_lidar_wSpeed --data ~/Documents/AutoCast_Shared_Dataset --ego-only --batch-size 64 --num-dataloader-workers 8 --shared False -T 5 --init-lr 0.001 --num-steps-per-log 100  --num-epochs 101


#################### Voxel Model
#Test Input: Shared Lidar Voxel Output: Control
#python -m training.train_lidar_voxel --data ~/Documents/AutoCast_Scene6_Dataset --ego-only --batch-size 32 --num-dataloader-workers 4 --shared True -T 5 --init-lr 0.001 --num-steps-per-log 100  --num-epochs 351 --frame-stack 3 --device cuda --use-speed 
#Test Input: Shared Lidar Voxel Output: Control
#python -m training.train_lidar_voxel --data ~/Documents/AutoCast_Scene6_Dataset --ego-only --batch-size 32 --num-dataloader-workers 4 --shared True -T 5 --init-lr 0.001 --num-steps-per-log 100  --num-epochs 351 --frame-stack 3 --device cuda

#################### DAgger Voxel
#python -m training.train_dagger_lidar_voxel --data ~/Documents/AutoCast_Scene6_Dataset --ego-only --batch-size 32 --num-dataloader-workers 4 --shared True -T 5 --init-lr 0.001 --num-steps-per-log 100  --num-epochs 351 --frame-stack 2 --device cuda --use-speed --finetune wandb/run-20210222_072513-xb6xewzg/files/model-200.th --beta 0.5 --sampling-frequency 25 --checkpoint-frequency 25

#################### Representation Learning
python -m training.train_lidar_voxel_concat --data ~/Documents/AutoCast_6/Train --batch-size 1 --num-dataloader-workers 4  -T 5 --init-lr 0.001 --num-steps-per-log 100  --num-epochs 500 --frame-stack 2 --device cuda --eval-data ~/Documents/AutoCast_Test 

#python -m training.train_dagger_transformer_voxel --data ~/Documents/AutoCast_6 --ego-only --batch-size 32 --num-dataloader-workers 4 --shared True -T 5 --init-lr 0.001 --num-steps-per-log 100  --num-epochs 2001 --frame-stack 2 --device cuda --finetune  wandb/run-20210331_185709-1iwm460b/files/model-75.th --beta 0.5 --sampling-frequency 25 --checkpoint-frequency 25 --max_num_neighbors 2 --eval-data ~/Documents/AutoCast_Test

#################### Graph Model
#python -m training.train_v2v --data ~/Documents/AutoCast_6 --batch-size 1 --num-dataloader-workers 1  -T 5 --init-lr 0.001 --num-steps-per-log 100  --num-epochs 500 --frame-stack 1 --device cuda 
#python -m training.train_dagger_v2v --data ~/Documents/AutoCast_6 --ego-only --batch-size 1 --num-dataloader-workers 1 --shared True -T 5 --init-lr 0.001 --num-steps-per-log 100  --num-epochs 1000 --frame-stack 1 --device cuda --finetune wandb/run-20210311_131815-16jlndo7/files/model-25.th --beta 0.95 --sampling-frequency 25 --checkpoint-frequency 25 --max_num_neighbors 2


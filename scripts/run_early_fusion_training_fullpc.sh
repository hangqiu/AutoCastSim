#!/bin/bash
echo -n "Train Data Collection, Validation Data Collection, Behavior Cloning or DAgger, or Evaluation? [data-train, data-val, bc,dagger,eval] "
read MODE
echo -n "Enter Scenario No. [6,8,10]: "
read SCEN
#echo -n "Sharing?[True, False]: "
#read SHARING

# Things to pay attention to
# for both BC and Dagger
# 1. Change --data to specify where the training data is stored

# For Dagger
# Make sure you start CARLA instances with correct port in a tmux pane
# ./scripts/launch_carla.sh [GPU_ID, 0] [NUM_WORKERS, 1] [PORT,2001] must be 2001 for now, hardcoded....
# Make sure mosquitto process are killed before running, although the script cleans by default after its own running

# 1. Change --finetune as the model that you want to continue training from BC
# 2. Change --benchmark_config to specify the dagger sampling configurations
# 3. I sample both data with collider and without collider, remove as you wish

# Must collect train and validation data before training
BATCHSIZE=32
WORKERS=4
CARLA_WORKERS=3
FRAMESTACK=2
TRAIN_SIZE=12
VAL_SIZE=12
DAGGER_SIZE=3
BGTRAFFIC=120
if [[ $SCEN == 6 ]]
then
  BGTRAFFIC=30
fi

# One train/val folder for all cases,
# a different data folder for each model and its corresponding dagger data and eval data
#TrainValFolder=/hdd1/AutoCast/Scen${SCEN}_bg${BGTRAFFIC}_fullpc/
#TrainValFolder=CoDrive_Dataset/Scen${SCEN}_bg${BGTRAFFIC}_fullpc/
TrainValFolder=~/Documents/AutoCast_${SCEN}_fullpc/

DATAFOLDER=${TrainValFolder}
#DATAFOLDER=/hdd2/AutoCast/Scen${SCEN}_bg${BGTRAFFIC}_nosharing/
#DATAFOLDER=CoDrive_Dataset/Scen${SCEN}_bg${BGTRAFFIC}_nosharing/
#DATAFOLDER=~/Documents/AutoCast_6_nosharing




if [[ $MODE == data-train ]]
then

AGENT=AVR/autocast_agents/simple_agent.py
CONFIG=benchmark/scene${SCEN}.json
OUTPUTDIR=${TrainValFolder}/Train/
kill $(pgrep ray)  
kill $(pgrep mosquitto)

python3 parallel_scenario_runner.py  \
  --agent $AGENT \
  --reloadWorld  \
  --port 2001 \
  --trafficManagerPort 3123 \
  --mqttport 4884 \
  --bgtraffic 0 \
  --num-worker 3 \
  --file --sharing \
  --benchmark_config $CONFIG \
  --commlog  \
  --lean \
  --fullpc \
  --outputdir $OUTPUTDIR \
  --resample-config 'random_uniform' \
  --num-config 3
fi

if [[ $MODE == data-val ]]
then

AGENT=AVR/autocast_agents/simple_agent.py
CONFIG=benchmark/scene${SCEN}.json
OUTPUTDIR=${TrainValFolder}/Val/
kill $(pgrep ray)  
kill $(pgrep mosquitto)

python3 parallel_scenario_runner.py  \
  --agent $AGENT \
  --reloadWorld  \
  --port 2001 \
  --trafficManagerPort 3123 \
  --mqttport 4884 \
  --bgtraffic $BGTRAFFIC \
  --num-workers $CARLA_WORKERS \
  --file --sharing \
  --benchmark_config $CONFIG \
  --commlog  \
  --lean \
  --fullpc \
  --outputdir $OUTPUTDIR \
  --resample-config 'random_uniform' \
  --num-config $VAL_SIZE
fi


if [[ $MODE == bc ]]
then
#################### BC
#Test Input: Shared Lidar Voxel Output: Control
python3 -m training.train_lidar_voxel \
  --num-epochs 501 \
  --data $TrainValFolder/Train/ \
  --daggerdata $DATAFOLDER/Dagger/ \
  --ego-only \
  --batch-size $BATCHSIZE \
  --num-dataloader-workers $WORKERS \
  --shared -T 5 --init-lr 0.001 \
  --num-steps-per-log 100  \
  --frame-stack $FRAMESTACK \
  --device cuda --use-speed --eval-data $TrainValFolder/Val/
fi

if [[ $MODE == dagger ]]
then
# Make sure you kill all carla processes
#################### DAgger
RUN=latest-run
RUN=run-20210425_104932-nqjx9cmu
CHECKPOINT=wandb/${RUN}/files/model-250.th
#CHECKPOINT=wandb/run-20210410_000152-j9lqqqzv/files/model-100.th
BETA=0.5
CONFIG=benchmark/scene${SCEN}.json

#kill $(pgrep CarlaUE4)
kill $(pgrep ray)  
kill $(pgrep mosquitto)

python3 -m training.train_dagger_lidar_voxel \
  --num-epochs 351 \
  --data $TrainValFolder/Train/ \
  --daggerdata $DATAFOLDER/Dagger/ \
  --num-workers $CARLA_WORKERS \
  --ego-only \
  --batch-size $BATCHSIZE \
  --num-dataloader-workers $WORKERS \
  --shared -T 5 --init-lr 0.001 \
  --num-steps-per-log 100  \
  --frame-stack $FRAMESTACK \
  --device cuda --use-speed \
  --finetune $CHECKPOINT \
  --lean \
  --beta $BETA --sampling-frequency 25 --checkpoint-frequency 25  \
  --benchmark_config $CONFIG \
  --bgtraffic $BGTRAFFIC \
  --eval-data $TrainValFolder/Val/ \
  --resample-config 'random_uniform' \
  --num-config $DAGGER_SIZE 
fi


if [[ $MODE == eval ]]
then
#################### Evaluation
RUN=6-fullpc-dagger
CHECKPOINTITER=125
AGENTCONFIG=wandb/${RUN}/files/config.yaml
AGENT=NeuralAgents/dagger_agent.py
CONFIG=benchmark/scene${SCEN}.json
OUTPUTDIR=${DATAFOLDER}/eval_fullpc/
CARLA_WORKERS=6
kill $(pgrep ray)  
kill $(pgrep mosquitto)

python3 parallel_scenario_runner.py  \
  --agent $AGENT \
  --agentConfig $AGENTCONFIG \
  --reloadWorld  \
  --port 2001 \
  --trafficManagerPort 3123 \
  --mqttport 4884 \
  --bgtraffic $BGTRAFFIC \
  --num-workers $CARLA_WORKERS \
  --file \
  --sharing \
  --benchmark_config $CONFIG \
  --num_checkpoint $CHECKPOINTITER \
  --beta 0.0 \
  --commlog  \
  --lean \
  --fullpc \
  --outputdir $OUTPUTDIR\
  --resample-config 'fixed'

fi

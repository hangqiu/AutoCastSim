EGOSPD=20
EGODIST=6
COLDIST=20
COLACCELDIST=30
COLSPD=40

#python scenario_runner.py --route srunner/data/routes_training_town03_autocast10.xml srunner/data/towns03_traffic_scenarios_autocast10.json   --reloadWorld  --agent NeuralAgents/LidarTorchAgent.py --sharing --commlog --bg 0 --file --repetition 1

#Sharing Minkowski
#for EGOSPD in 20
#do
#    for SPDDIFF in -10 -7 #-5 -3 -2 0 2 3 5 7 10 15 20
#    do
#        COLDSPD=$((EGOSPD+SPDDIFF))
#        python scenario_runner.py \
#            --route srunner/data/routes_training_town03_autocast10.xml \
#            srunner/data/towns03_traffic_scenarios_autocast10.json   \
#            --reloadWorld  \
#            --agent NeuralAgents/Lidar2DMinkowskiAgent.py  \
#            --eval $EGOSPD $EGODIST $COLDSPD $COLDIST $COLACCELDIST \
#            --commlog \
#            --bgtraffic 0 \
#            --file \
#            --sharing \
#            --agentConfig wandb/run-20210125_220341-4lpf88t5/files/config.yaml
#    done
#done
#Scene10 ParallelSharing Minkowski
python3 parallel_scenario_runner.py \
    --agent NeuralAgents/Lidar2DMinkowskiAgent.py \
    --reloadWorld \
    --port 2001 \
    --trafficManagerPort 3123 \
    --mqttport 4884 \
    --commlog \
    --bgtraffic 0 \
    --bgtraffic_initspd \
    --num-workers 5 \
    --file\
    --sharing \
    --agentConfig wandb/run-20210125_220341-4lpf88t5/files/config.yaml \
    --benchmark_config "benchmark/scene10.json"



#Non-Share Minkowski
#for EGOSPD in 20
#do
#    for SPDDIFF in -10 -7 #-5 -3 -2 0 2 3 5 7 10 15 20
#    do
#        COLDSPD=$((EGOSPD+SPDDIFF))
#        python scenario_runner.py \
#            --route srunner/data/routes_training_town03_autocast10.xml \
#            srunner/data/towns03_traffic_scenarios_autocast10.json   \
#            --reloadWorld  \
#            --agent NeuralAgents/Lidar2DMinkowskiAgent.py  \
#            --eval $EGOSPD $EGODIST $COLDSPD $COLDIST $COLACCELDIST \
#            --commlog \
#            --bgtraffic 0 \
#            --file \
#            --agentConfig wandb/run-20210126_002309-2c5o1byl/files/config.yaml
#    done
#done

#Scen6 share
#EGOSPD=20
#EGODIST=6
#COLSPD=35

#for COLDIST in 0 
#do
#python scenario_runner.py \
#    --route srunner/data/routes_training_town01_autocast6.xml \
#    srunner/data/towns01_traffic_scenarios_autocast6.json  \
#    --reloadWorld \
#    --agent NeuralAgents/Lidar2DMinkowskiAgent.py \
#    --eval $EGOSPD $EGODIST $COLSPD $COLDIST $COLACCELDIST \
#    --bgtraffic 0 \
#    --file \
#    --commlog \
#    --sharing \
#    --agentConfig wandb/run-20210125_014541-2im90uus/files/config.yaml
#done

#Scen6 Non-Shared
#for COLDIST in 0
#do
#python scenario_runner.py \
#    --route srunner/data/routes_training_town01_autocast6.xml \
#    srunner/data/towns01_traffic_scenarios_autocast6.json  \
#    --reloadWorld \
#    --agent NeuralAgents/Lidar2DMinkowskiAgent.py \
#    --eval $EGOSPD $EGODIST $COLSPD $COLDIST $COLACCELDIST \
#    --bgtraffic 0 \
#    --file \
#    --commlog \
#    --agentConfig wandb/run-20210125_044313-2hxetzf0/files/config.yaml
#done




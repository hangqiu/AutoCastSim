#!/bin/bash
echo -n "Enter Scenario No. [6,8,10, test]: "
read SCEN
echo -n "Enter Port: "
read PORT

TMPORT=$(($PORT+50))
MQTTPORT=$(($PORT+100))
mosquitto -p $MQTTPORT -d

#AGENT=AVR/autocast_agents/human_agent.py
AGENT=AVR/autocast_agents/simple_agent.py
#AGENT=AVR/autocast_agents/noisy_simple_agent.py
EGODIST=6
COLDIST=10
COLACCELDIST=0
EGOSPD=40
COLSPD=40
BGTRAFFIC=105

OUTPUTDIR=test_traces/

if [[ $SCEN == 6 ]]
then
python3 scenario_runner.py \
    --route srunner/data/routes_training_town01_autocast6.xml \
    srunner/data/towns01_traffic_scenarios_autocast6.json  \
    --reloadWorld \
    --agent $AGENT \
    --eval $EGOSPD $EGODIST $COLSPD $COLDIST $COLACCELDIST \
    --bgtraffic $BGTRAFFIC \
    --port $PORT \
    --trafficManagerPort $TMPORT \
    --mqttport $MQTTPORT \
    --commlog \
    --hud \
    --outputdir $OUTPUTDIR
#    --lean \
#    --emulate \
#    --sharing \
#    --full \
#    --nocollider \
#    --bgtraffic_initspd \
#    --profile \
#    --debug \
#    --passive_collider \
#        --agent AVR/autocast_agents/simple_agent.py \
fi


##============================================================Scen8===========================================================
if [[ $SCEN == 8 ]]
then
#COLSPD=20
COLDIST=0
SPDDIFF=-5
COLSPD=$((EGOSPD+SPDDIFF))

python3 scenario_runner.py \
    --route srunner/data/routes_training_town03_autocast8.xml \
    srunner/data/towns03_traffic_scenarios_autocast8.json   \
    --reloadWorld \
    --agent $AGENT \
    --eval $EGOSPD $EGODIST $COLSPD $COLDIST $COLACCELDIST \
    --bgtraffic $BGTRAFFIC \
    --port $PORT \
    --trafficManagerPort $TMPORT \
    --mqttport $MQTTPORT \
    --commlog \
    --outputdir $OUTPUTDIR
#    --sharing\
fi

#============================================================Scen10===========================================================
if [[ $SCEN == 10 ]]
then
python3 scenario_runner.py \
    --route srunner/data/routes_training_town03_autocast10.xml \
    srunner/data/towns03_traffic_scenarios_autocast10.json   \
    --reloadWorld \
    --agent $AGENT \
    --eval $EGOSPD $EGODIST $COLSPD $COLDIST $COLACCELDIST \
    --bgtraffic $BGTRAFFIC \
    --port $PORT \
    --trafficManagerPort $TMPORT \
    --mqttport $MQTTPORT \
    --seed 1300 \
    --outputdir $OUTPUTDIR
#    --noextrap \
#    --lean \
#    --hud \
#    --sharing \
#    --commlog \
#    --full \
#    --emulate \
#    --lean \
#    --profile
#    --agnostic \
#        --outputdir \.\/Scen10_Share\/ \
#        --commlog --sharing > Auto10_${EGOSPD}_${COLSPD}_${BGTRAFFIC}_ShareFull.txt
#        --outputdir \.\/Scen10_Share_${COLSPD}_Agnostic\/ \
#        --agnostic \
#        --commlog --sharing > Auto10_${EGOSPD}_${COLSPD}_${BGTRAFFIC}_Agnostic.txt
fi


#============================================================ScenTest===========================================================
if [[ $SCEN == test ]]
then
BGTRAFFIC=0
python3 scenario_runner.py \
    --route srunner/data/routes_training_town03_autocast_test.xml \
    srunner/data/towns03_traffic_scenarios_autocast_test.json   \
    --reloadWorld \
    --agent $AGENT \
    --eval $EGOSPD $EGODIST $COLSPD $COLDIST $COLACCELDIST \
    --bgtraffic $BGTRAFFIC \
    --port $PORT \
    --trafficManagerPort $TMPORT \
    --mqttport $MQTTPORT \
    --noextrap \
    --nofair \
    --commlog \
    --sharing \
    --agnostic \
    --outputdir $OUTPUTDIR
#    --profile \
#    --voronoi

fi

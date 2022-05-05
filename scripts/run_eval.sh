#!/bin/bash
echo -n "Enter Scenario No. [6,8,10]: "
read SCEN
echo -n "Enter Port: "
read PORT

TMPORT=$(($PORT+50))
MQTTPORT=$(($PORT+100))
mosquitto -p $MQTTPORT -d

EGOSPD=20
EGODIST=6
COLDIST=20
COLACCELDIST=0
COLSPD=20

OUTPUTDIR=.
#============================================================Scen6===========================================================

if [[ $SCEN == 6 ]]
then
for COLSPD in 20 30 40
do
  for COLDIST in 40
  do
    for BGTRAFFIC in 60 120 180 240
    do
#    python3 scenario_runner.py \
#        --route srunner/data/routes_training_town01_autocast6.xml \
#        srunner/data/towns01_traffic_scenarios_autocast6.json  \
#        --reloadWorld \
#        --agent AVR/autocast_agents/simple_agent.py \
#        --eval $EGOSPD $EGODIST $COLSPD $COLDIST $COLACCELDIST \
#        --bgtraffic $BGTRAFFIC \
#        --bgtraffic_initspd \
#        --port $PORT \
#        --trafficManagerPort $TMPORT \
#        --mqttport $MQTTPORT \
#        --outputdir ${OUTPUTDIR}\/Scen6_NoShare\/ \
#        --commlog > ${OUTPUTDIR}\/Auto6_${COLSPD}_${COLDIST}_${BGTRAFFIC}_NoShareFull.txt

#    python3 scenario_runner.py \
#        --route srunner/data/routes_training_town01_autocast6.xml \
#        srunner/data/towns01_traffic_scenarios_autocast6.json  \
#        --reloadWorld \
#        --agent AVR/autocast_agents/simple_agent.py \
#        --eval $EGOSPD $EGODIST $COLSPD $COLDIST $COLACCELDIST \
#        --bgtraffic $BGTRAFFIC \
#        --bgtraffic_initspd \
#        --port $PORT \
#        --trafficManagerPort $TMPORT \
#        --mqttport $MQTTPORT \
#        --passive_collider \
#        --noextrap \
#        --commlog \
#        --lean \
#        --outputdir ${OUTPUTDIR}\/Scen6_Share_${EGOSPD}_${COLSPD}_${BGTRAFFIC}_AutoCast\/ \
#        --sharing > ${OUTPUTDIR}\/Auto6_${COLSPD}_${COLDIST}_${BGTRAFFIC}_ShareFull.txt
#
#    python3 scenario_runner.py \
#        --route srunner/data/routes_training_town01_autocast6.xml \
#        srunner/data/towns01_traffic_scenarios_autocast6.json  \
#        --reloadWorld \
#        --agent AVR/autocast_agents/simple_agent.py \
#        --eval $EGOSPD $EGODIST $COLSPD $COLDIST $COLACCELDIST \
#        --bgtraffic $BGTRAFFIC \
#        --bgtraffic_initspd \
#        --port $PORT \
#        --trafficManagerPort $TMPORT \
#        --mqttport $MQTTPORT \
#        --passive_collider \
#        --noextrap \
#        --commlog \
#        --lean \
#        --agnostic \
#        --outputdir ${OUTPUTDIR}\/Scen6_Share_${EGOSPD}_${COLSPD}_${BGTRAFFIC}_Agnostic\/ \
#        --sharing > ${OUTPUTDIR}\/Auto6_${EGOSPD}_${COLSPD}_${BGTRAFFIC}_Agnostic.txt

    python3 scenario_runner.py \
        --route srunner/data/routes_training_town01_autocast6.xml \
        srunner/data/towns01_traffic_scenarios_autocast6.json  \
        --reloadWorld \
        --agent AVR/autocast_agents/simple_agent.py \
        --eval $EGOSPD $EGODIST $COLSPD $COLDIST $COLACCELDIST \
        --bgtraffic $BGTRAFFIC \
        --bgtraffic_initspd \
        --port $PORT \
        --trafficManagerPort $TMPORT \
        --mqttport $MQTTPORT \
        --passive_collider \
        --noextrap \
        --commlog \
        --lean \
        --voronoi \
        --outputdir ${OUTPUTDIR}\/Scen6_Share_${EGOSPD}_${COLSPD}_${BGTRAFFIC}_Voronoi_300mbps\/ \
        --sharing > ${OUTPUTDIR}\/Auto6_${EGOSPD}_${COLSPD}_${BGTRAFFIC}_Voronoi_300mbps.txt

    done
  done
done
fi

##============================================================Scen8===========================================================
if [[ $SCEN == 8 ]]
then
#COLSPD=20
COLDIST=0
SPDDIFF=-5
for EGOSPD in 30
do
#  for SPDDIFF in -5 0 20
#  do
    COLSPD=$((EGOSPD+SPDDIFF))
#    BGTRAFFIC=0
    for BGTRAFFIC in 60 180 240
    do
#      python3 scenario_runner.py \
#        --route srunner/data/routes_training_town03_autocast8.xml \
#        srunner/data/towns03_traffic_scenarios_autocast8.json   \
#        --reloadWorld \
#        --agent AVR/autocast_agents/simple_agent.py \
#        --eval $EGOSPD $EGODIST $COLSPD $COLDIST $COLACCELDIST \
#        --bgtraffic $BGTRAFFIC \
#        --port $PORT \
#        --trafficManagerPort $TMPORT \
#        --mqttport $MQTTPORT \
#        --outputdir ${OUTPUTDIR}\/Scen8_NoShare\/ \
#        --commlog > ${OUTPUTDIR}\/Auto8_${EGOSPD}_${COLSPD}_${COLDIST}_${BGTRAFFIC}_NoShareFull.txt
#
#      python3 scenario_runner.py \
#        --route srunner/data/routes_training_town03_autocast8.xml \
#        srunner/data/towns03_traffic_scenarios_autocast8.json   \
#        --reloadWorld \
#        --agent AVR/autocast_agents/simple_agent.py \
#        --eval $EGOSPD $EGODIST $COLSPD $COLDIST $COLACCELDIST \
#        --bgtraffic $BGTRAFFIC \
#        --port $PORT \
#        --trafficManagerPort $TMPORT \
#        --mqttport $MQTTPORT \
#        --passive_collider \
#        --noextrap \
#        --commlog \
#        --lean \
#        --outputdir ${OUTPUTDIR}\/Scen8_Share_${EGOSPD}_${COLSPD}_${BGTRAFFIC}_AutoCast\/ \
#        --sharing > ${OUTPUTDIR}\/Auto8_${EGOSPD}_${COLSPD}_${COLDIST}_${BGTRAFFIC}_ShareFull.txt
#
#      python3 scenario_runner.py \
#        --route srunner/data/routes_training_town03_autocast8.xml \
#        srunner/data/towns03_traffic_scenarios_autocast8.json   \
#        --reloadWorld \
#        --agent AVR/autocast_agents/simple_agent.py \
#        --eval $EGOSPD $EGODIST $COLSPD $COLDIST $COLACCELDIST \
#        --bgtraffic $BGTRAFFIC \
#        --port $PORT \
#        --trafficManagerPort $TMPORT \
#        --mqttport $MQTTPORT \
#        --passive_collider \
#        --noextrap \
#        --commlog \
#        --lean \
#        --agnostic \
#        --outputdir ${OUTPUTDIR}\/Scen8_Share_${EGOSPD}_${COLSPD}_${BGTRAFFIC}_Agnostic\/ \
#        --sharing > ${OUTPUTDIR}\/Auto8_${EGOSPD}_${COLSPD}_${COLDIST}_${BGTRAFFIC}_Agnostic.txt

      python3 scenario_runner.py \
        --route srunner/data/routes_training_town03_autocast8.xml \
        srunner/data/towns03_traffic_scenarios_autocast8.json   \
        --reloadWorld \
        --agent AVR/autocast_agents/simple_agent.py \
        --eval $EGOSPD $EGODIST $COLSPD $COLDIST $COLACCELDIST \
        --bgtraffic $BGTRAFFIC \
        --port $PORT \
        --trafficManagerPort $TMPORT \
        --mqttport $MQTTPORT \
        --passive_collider \
        --noextrap \
        --commlog \
        --lean \
        --voronoi \
        --outputdir ${OUTPUTDIR}\/Scen8_Share_${EGOSPD}_${COLSPD}_${BGTRAFFIC}_Voronoi_50mbps\/ \
        --sharing > ${OUTPUTDIR}\/Auto8_${EGOSPD}_${COLSPD}_${COLDIST}_${BGTRAFFIC}_Voronoi_50mbps.txt
    done
#  done
done
fi

#============================================================Scen10===========================================================
if [[ $SCEN == 10 ]]
then
for EGOSPD in 40
do
#  for SPDDIFF in -10 -5 0 5 10
#  do
  SPDDIFF=0
    COLSPD=$((EGOSPD+SPDDIFF))
    for BGTRAFFIC in 240
    do
#      python3 scenario_runner.py \
#        --route srunner/data/routes_training_town03_autocast10.xml \
#        srunner/data/towns03_traffic_scenarios_autocast10.json   \
#        --reloadWorld \
#        --agent AVR/autocast_agents/simple_agent.py \
#        --eval $EGOSPD $EGODIST $COLSPD $COLDIST $COLACCELDIST \
#        --bgtraffic $BGTRAFFIC \
#        --port $PORT \
#        --trafficManagerPort $TMPORT \
#        --mqttport $MQTTPORT \
#        --outputdir ${OUTPUTDIR}\/Scen10_NoShare\/ \
#        --commlog > ${OUTPUTDIR}\/Auto10_${EGOSPD}_${COLSPD}_${BGTRAFFIC}_NoShareFull.txt

#      python3 scenario_runner.py \
#        --route srunner/data/routes_training_town03_autocast10.xml \
#        srunner/data/towns03_traffic_scenarios_autocast10.json   \
#        --reloadWorld \
#        --agent AVR/autocast_agents/simple_agent.py \
#        --eval $EGOSPD $EGODIST $COLSPD $COLDIST $COLACCELDIST \
#        --bgtraffic $BGTRAFFIC \
#        --port $PORT \
#        --trafficManagerPort $TMPORT \
#        --mqttport $MQTTPORT \
#        --passive_collider \
#        --noextrap \
#        --commlog \
#        --lean \
#        --outputdir ${OUTPUTDIR}\/Scen10_Share_${EGOSPD}_${COLSPD}_${BGTRAFFIC}_AutoCast\/ \
#        --sharing > ${OUTPUTDIR}\/Auto10_${EGOSPD}_${COLSPD}_${BGTRAFFIC}_ShareFull.txt

      python3 scenario_runner.py \
        --route srunner/data/routes_training_town03_autocast10.xml \
        srunner/data/towns03_traffic_scenarios_autocast10.json   \
        --reloadWorld \
        --agent AVR/autocast_agents/simple_agent.py \
        --eval $EGOSPD $EGODIST $COLSPD $COLDIST $COLACCELDIST \
        --bgtraffic $BGTRAFFIC \
        --port $PORT \
        --trafficManagerPort $TMPORT \
        --mqttport $MQTTPORT \
        --passive_collider \
        --noextrap \
        --commlog \
        --lean \
        --agnostic \
        --outputdir ${OUTPUTDIR}\/Scen10_${EGOSPD}_${COLSPD}_${BGTRAFFIC}_Agnostic\/ \
        --sharing > ${OUTPUTDIR}\/Auto10_${EGOSPD}_${COLSPD}_${BGTRAFFIC}_Agnostic.txt
#
#      python3 scenario_runner.py \
#        --route srunner/data/routes_training_town03_autocast10.xml \
#        srunner/data/towns03_traffic_scenarios_autocast10.json   \
#        --reloadWorld \
#        --agent AVR/autocast_agents/simple_agent.py \
#        --eval $EGOSPD $EGODIST $COLSPD $COLDIST $COLACCELDIST \
#        --bgtraffic $BGTRAFFIC \
#        --port $PORT \
#        --trafficManagerPort $TMPORT \
#        --mqttport $MQTTPORT \
#        --passive_collider \
#        --noextrap \
#        --commlog \
#        --lean \
#        --voronoi \
#        --outputdir ${OUTPUTDIR}\/Scen10_Share_${EGOSPD}_${COLSPD}_${BGTRAFFIC}_Voronoi_300mbps\/ \
#        --sharing > ${OUTPUTDIR}\/Auto10_${EGOSPD}_${COLSPD}_${BGTRAFFIC}_Voronoi_300mbps.txt
    done
#  done
done
fi
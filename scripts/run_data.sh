#!/bin/bash
echo -n "Enter Scenario No. [6,8,10]: "
read SCEN
echo -n "Enter Port: "
read PORT

TMPORT=$(($PORT+50))
MQTTPORT=$(($PORT+100))
mosquitto -p $MQTTPORT -d
ITER=2
#OUTPUTDIR=/hdd1/AutoCast/
#OUTPUTDIR=/hdd/AutoCast/
OUTPUTDIR=CoDrive_Dataset/
#============================================================Scen6===========================================================
EGOSPD=20
EGODIST=6
COLDIST=20
COLACCELDIST=0
COLSPD=20
#
if [[ $SCEN == 6 ]]
then
for COLSPD in 30
do
  for COLDIST in 10
#  for COLDIST in 20
  do
    for BGTRAFFIC in 30
    do
      for ((i=0;i<$ITER;i++))
      do
#        python3 scenario_runner.py \
#            --route srunner/data/routes_training_town01_autocast6.xml \
#            srunner/data/towns01_traffic_scenarios_autocast6.json  \
#            --reloadWorld \
#            --agent AVR/autocast_agents/noisy_simple_agent.py \
#            --eval $EGOSPD $EGODIST $COLSPD $COLDIST $COLACCELDIST \
#            --bgtraffic $BGTRAFFIC \
#            --bgtraffic_initspd \
#            --port $PORT \
#            --trafficManagerPort $TMPORT \
#            --mqttport $MQTTPORT \
#            --full \
#            --nocollider \
#            --outputdir \.\/Scen6_DC_NoCollider_${COLSPD}_${COLDIST}_${BGTRAFFIC}\/ \
#            --commlog --sharing > Auto6_${COLSPD}_${COLDIST}_${BGTRAFFIC}_ShareFull_NoCollider.txt

        python3 scenario_runner.py \
            --route srunner/data/routes_training_town01_autocast6.xml \
            srunner/data/towns01_traffic_scenarios_autocast6.json  \
            --reloadWorld \
            --agent AVR/autocast_agents/noisy_simple_agent.py \
            --eval $EGOSPD $EGODIST $COLSPD $COLDIST $COLACCELDIST \
            --bgtraffic $BGTRAFFIC \
            --bgtraffic_initspd \
            --port $PORT \
            --trafficManagerPort $TMPORT \
            --mqttport $MQTTPORT \
            --full \
            --passive_collider \
            --outputdir ${OUTPUTDIR}/Scen6_DC_${COLSPD}_${COLDIST}_${BGTRAFFIC}/ \
            --commlog --sharing
      done
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
for EGOSPD in 40
do
  for SPDDIFF in -5 0 5 20
  do
    COLSPD=$((EGOSPD+SPDDIFF))
    for BGTRAFFIC in 0
    do
      for ((i=0;i<$ITER;i++))
      do
#        python3 scenario_runner.py \
#          --route srunner/data/routes_training_town03_autocast8.xml \
#          srunner/data/towns03_traffic_scenarios_autocast8.json   \
#          --reloadWorld \
#          --agent AVR/autocast_agents/noisy_simple_agent.py \
#          --eval $EGOSPD $EGODIST $COLSPD $COLDIST $COLACCELDIST \
#          --bgtraffic $BGTRAFFIC \
#          --port $PORT \
#          --trafficManagerPort $TMPORT \
#          --mqttport $MQTTPORT \
#          --full \
#          --nocollider \
#          --outputdir \.\/Scen8_DC_NoCollider_${EGOSPD}_${COLSPD}_${COLDIST}_${BGTRAFFIC}\/ \
#          --commlog --sharing > Auto8_${EGOSPD}_${COLSPD}_${COLDIST}_${BGTRAFFIC}_ShareFull_NoCollider.txt

        python3 scenario_runner.py \
          --route srunner/data/routes_training_town03_autocast8.xml \
          srunner/data/towns03_traffic_scenarios_autocast8.json   \
          --reloadWorld \
          --agent AVR/autocast_agents/noisy_simple_agent.py \
          --eval $EGOSPD $EGODIST $COLSPD $COLDIST $COLACCELDIST \
          --bgtraffic $BGTRAFFIC \
          --port $PORT \
          --trafficManagerPort $TMPORT \
          --mqttport $MQTTPORT \
          --full \
          --passive_collider \
          --outputdir ${OUTPUTDIR}/Scen8_DC_${EGOSPD}_${COLSPD}_${COLDIST}_${BGTRAFFIC}/ \
          --commlog --sharing > ${OUTPUTDIR}/Auto8_${EGOSPD}_${COLSPD}_${COLDIST}_${BGTRAFFIC}_${i}_ShareFull.txt
      done
    done
  done
done
fi

##============================================================Scen10===========================================================
if [[ $SCEN == 10 ]]
then
for EGOSPD in 30
do
  for SPDDIFF in -10 -5 0 5 10
#  for SPDDIFF in 0
  do
    COLSPD=$((EGOSPD+SPDDIFF))
    for BGTRAFFIC in 120
    do
      for ((i=0;i<$ITER;i++))
      do
#        python3 scenario_runner.py \
#          --route srunner/data/routes_training_town03_autocast10.xml \
#          srunner/data/towns03_traffic_scenarios_autocast10.json   \
#          --reloadWorld \
#          --agent AVR/autocast_agents/noisy_simple_agent.py \
#          --eval $EGOSPD $EGODIST $COLSPD $COLDIST $COLACCELDIST \
#          --bgtraffic $BGTRAFFIC \
#          --port $PORT \
#          --trafficManagerPort $TMPORT \
#          --mqttport $MQTTPORT \
#          --full \
#          --nocollider \
#          --outputdir \.\/Scen10_DC_NoCollider_${EGOSPD}_${COLSPD}_${BGTRAFFIC}\/ \
#          --commlog --sharing > Auto10_${EGOSPD}_${COLSPD}_${BGTRAFFIC}_ShareFull_NoCollider.txt

        python3 scenario_runner.py \
          --route srunner/data/routes_training_town03_autocast10.xml \
          srunner/data/towns03_traffic_scenarios_autocast10.json   \
          --reloadWorld \
          --agent AVR/autocast_agents/noisy_simple_agent.py \
          --eval $EGOSPD $EGODIST $COLSPD $COLDIST $COLACCELDIST \
          --bgtraffic $BGTRAFFIC \
          --port $PORT \
          --trafficManagerPort $TMPORT \
          --mqttport $MQTTPORT \
          --full \
          --passive_collider \
          --outputdir ${OUTPUTDIR}/Scen10_DC_${EGOSPD}_${COLSPD}_${BGTRAFFIC}/ \
          --commlog --sharing > Auto10_${EGOSPD}_${COLSPD}_${BGTRAFFIC}_${i}_ShareFull.txt
      done
    done
  done
done
fi

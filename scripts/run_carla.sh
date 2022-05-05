#!/bin/bash
#============================================================Run Carla===========================================================
DISPLAY=
CARLA_ROOT=/opt/carla-simulator/
export CUDA_VISIBLE_DEVICES=$1
PORT=$2
$CARLA_ROOT/CarlaUE4.sh -windowed -carla-rpc-port=$PORT -opengl
#$CARLA_ROOT/CarlaUE4.sh -windowed -carla-rpc-port=$PORT
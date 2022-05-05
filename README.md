# AutoCastSim

AutoCastSim is an end-to-end cooperative perception and collaborative driving simulation framework. 
It builds on top of the [CARLA Simulator](https://carla.org/), using vehicle-to-vehicle (V2V) communication, 
to enable sensor sharing and vehicle collaboration.
One feature of AutoCastSim is that it includes designed scenarios targeting long-tail events 
where single-vehicle-based solutions are incapable of consistently making safe decisions.   

Overtaking|Unprotected Left-turn|Red-light Violation|
---|---|---
![](docs/pics/Scen6_Share_bev.gif) | ![](docs/pics/Scen8_Share_bev.gif) | ![](docs/pics/Scen10_Share_bev.gif) 
![](docs/pics/Scen6_Share_lidar.gif) | ![](docs/pics/Scen8_Share_lidar.gif)| ![](docs/pics/Scen10_Share_lidar.gif)


[comment]: <> (https://user-images.githubusercontent.com/9672411/163308420-f0e45af0-ff2a-446a-be94-60829085e45a.mp4)

## Getting Started

### Prerequisites
Current setup: Ubuntu 20.04, Cuda 11.0, Python 3.7, PyTorch == 1.7.1, PyTorch Geometrics

[Carla 0.9.11](https://carla.readthedocs.io/en/0.9.11/start_quickstart/)

[Minkowski Engine](https://github.com/NVIDIA/MinkowskiEngine) CPU version



Others
```
sudo apt-get install mosquitto libopenblas-dev
sudo apt remove python3-networkx
pip3 install paho-mqtt scipy pygame py-trees==0.8.3 networkx==2.2 xmlschema numpy shapely imageio ray tqdm numba pandas scikit-image scikit-learn opencv-python h5py
```

### Running Demo Scenarios

Run Carla
```
cd /opt/carla_simulator/bin
./CarlaUE4.sh
```

Scenario: Overtake
```
python3 scenario_runner.py \
  --route srunner/data/routes_training_town01_autocast6.xml \
  srunner/data/towns01_traffic_scenarios_autocast6.json  \
  --reloadWorld \
  --bgtraffic 0 \
  --bgtraffic_initspd \
  --agent AVR/autocast_agents/simple_agent.py \
  [--hud --sharing]
```

Scenario: Unprotected Left Turn
```
python3 scenario_runner.py \
  --route srunner/data/routes_training_town03_autocast8.xml \
  srunner/data/towns03_traffic_scenarios_autocast8.json   \
  --reloadWorld \
  --bgtraffic 0 \
  --agent AVR/autocast_agents/simple_agent.py \
  [--hud --sharing]
```

Scenario: Red-light Violation
```
python3 scenario_runner.py \
  --route srunner/data/routes_training_town03_autocast10.xml \
  srunner/data/towns03_traffic_scenarios_autocast10.json   \
  --reloadWorld \
  --bgtraffic 0 \
  --agent AVR/autocast_agents/simple_agent.py \
  [--hud --sharing]
```


Note: 

* --hud: record the display images and compile into a video at the end
* --sharing: for extended vision

For a full list of config flags, please see [config.md](docs/Config.md). 
If running remotely, the above commands need X server forwarding. (e.g. use ssh -x)

### Running Multi-instance AutoCastSim in Parallel across Multiple GPUs
In order to run multiple AutoCast instances, there are three communication network needed to be separated to avoid interference: 1) Carla <-> Python_Client, 2) TrafficManager, 3) Pub-sub Network. By default they are using port 2000, 8000, 8883. For each instance, these three ports need to be unique. 
```
PORT=2000
TMPORT=$(($PORT+50))
MQTTPORT=$(($PORT+100))
```
Launch MQTT borker using unique MQTTPORT:
```
mosquitto -p $MQTTPORT -d
```
Launch Carla using unique PORT and GPU card:
```shell
bash run_carla.sh [GPU_ID] [PORT]
```
Run one example instance:
```
python3 scenario_runner.py \
        --route srunner/data/routes_training_town01_autocast6.xml \
        srunner/data/towns01_traffic_scenarios_autocast6.json  \
        --reloadWorld \
        --agent AVR/autocast_agents/simple_agent.py \ 
        --bgtraffic 0 \
        --bgtraffic_initspd \
        --port $PORT \
        --trafficManagerPort $TMPORT \
        --mqttport $MQTTPORT \                
        --commlog --sharing
```
Repeat above to launch a second instance using a different GPU or the same one if memory footprint permits (~3G each instance).

### Training Your Own Agent
Please refer to [Coopernaut](https://github.com/UT-Austin-RPL/Coopernaut) as an example agent trained using AutoCastSim.
We provide a [training dataset](https://utexas.box.com/v/coopernaut-dataset) for behavior cloning based on a simple rule-based agent. 
You can also collect your own data and train your own agents following these [instructions](https://github.com/UT-Austin-RPL/Coopernaut).



## Citation

If you want to use AutoCastSim to invent novel collaborative driving models, 
please refer to [Coopernaut](https://github.com/UT-Austin-RPL/Coopernaut), and cite
```bibtex
@inproceedings{coopernaut,
    title = {Coopernaut: End-to-End Driving with Cooperative Perception for Networked Vehicles},
    author = {Jiaxun Cui and Hang Qiu and Dian Chen and Peter Stone and Yuke Zhu},
    booktitle = {IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
    series = {CVPR '22},
    year = {2022},
}
```

If AutoCastSim helps your research in V2V/V2X communication, security, *etc.*, please refer to [AutoCast](https://github.com/hangqiu/AutoCast), and cite
```bibtex
@inproceedings{autocast,
  title={AutoCast: Scalable Infrastructure-less Cooperative Perception for Distributed Collaborative Driving},
  author={Hang Qiu and Pohan Huang and Namo Asavisanu and Xiaochen Liu and Konstantinos Psounis and Ramesh Govindan},
  booktitle = {Proceedings of the 20th Annual International Conference on Mobile Systems, Applications, and Services},
  series = {MobiSys '22},
  year={2022},
}

```

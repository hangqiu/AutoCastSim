# AutoCastSim Configuration

AutoCastSim builds on top of Carla Scenario Runner the networking component among vehicles 
as well as the occlusion scenarios that single vehicle based solution often fails. 
To config the scenarios, we added a lot of Autocast specific flags in addition to original flags.

To view the explanation of all flags, simple run
```shell
python3 scenario_runner.py --help
```

In the following, we list a few important flags for our considerations. 
These include all the major flags to be configured in **run_test.sh** and **run_eval.sh**.

Simulation engine and setup flags:
```
  --host HOST           IP of the host server (default: localhost)
  --port PORT           TCP port to listen to (default: 2000)
  --mqttport            Set the MQTT broker port (which should be different than port)
  --trafficManagerPort  Port to use for the TrafficManager (default: 8000)
  --trafficManagerSeed  Random seed used by the TrafficManager to config background traffic (default: 0)
  --emulate             Flag to switch between emulation (slower, higher fidelity, e.g. for communication, object detection) vs simulation (faster, use carla labels)
  
```

Scenario flags:
```
  --route               Run a route as a scenario (input: (route_file,scenario_file,[number of route]))
                        An example route file: "srunner/data/routes_training_town03_autocast10.xml"
                        A corresponding scenario file: "srunner/data/towns03_traffic_scenarios_autocast10.json"
  --reloadWorld         Reload the CARLA world before starting a scenario (default=True)
  --bgtraffic           Set the amount of background traffic in scenario
  --seed SEED           random seed used for background traffic
  --eval                Evaluation parameters (input: (ego_speed, ego_distance, collider_speed, collider_dist,[collider_acceleration_dist]))
  --nocollider          Disable colliders in scenarios
  
```
Recording flags:
```
  --full                Record full records for data logger (Lidar, camera, from all vehicles nearby)
  --lean                Record lean records (only ego BEV and Lidar) for data logger
  --hud                 Record the HUD display
  --commlog             Record communication logs among vehicles
  --outputdir           Set the recordings output directory
```

Agent flags
```
  --agent               Agent used to execute the scenario. e.g. "AVR/autocast_agents/simple_agent.py"
  --sharing             Enable sensor sharing 
  --passive_collider    Disable colliders sharing in scenarios
  --fullpc              Enable Full Point cloud sharing with 1000X bandwidth (Caution: this runs very slow)
```

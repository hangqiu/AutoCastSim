import os 
import json
import subprocess
#from scenario_runner import VERSION
from AVR import Utils
from scenario_runner import VERSION


def get_pid(name):
    #return subprocess.check_output(["pidof", name])
    child = subprocess.Popen(['pgrep', '-f', name], stdout=subprocess.PIPE, shell=False)
    response = child.communicate()[0]
    return [int(pid) for pid in response.split()]

def clean_process():
    mqtt_pids = get_pid("mosquitto")
    ray_pids = get_pid("ray")
    for p in mqtt_pids+ray_pids:
        print("Killing Mosquitto process",p)
        try:
            os.kill(p, signal.SIGTERM)
        except:
            pass

def parallel_eval(config):
    port = str(int(config.port) + 11*int(config.cuda_visible_devices))
    call = 'python3 parallel_scenario_runner.py  \
            --agent {} \
            --agentConfig {} \
            --reloadWorld \
            --port {} \
            --trafficManagerPort {} \
            --mqttport {} \
            --bgtraffic {} \
            --num-workers {} \
            --file \
            --sharing \
            --benchmark_config {} \
            --num_checkpoint {} \
            --beta {} \
            --passive_collider \
            --outputdir {} \
            --resample-config {} \
            --seed {}'.format(config.agent,
                              config.agentConfig,
                              port,
                              str(int(config.trafficManagerPort)+11*int(config.cuda_visible_devices)+7*int(config.seed)),
                              str(int(config.mqttport)+11*int(config.cuda_visible_devices)+7*int(config.seed)),
                              config.bgtraffic,
                              config.num_workers,
                              config.benchmark_config,
                              config.num_checkpoint,
                              config.beta,
                              config.outputdir,
                              config.resample_config,
                              config.seed
                                    )

    if config.commlog:
        call += " --commlog"
    if config.emulate:
        call += " --emulate"
    if config.full:
        call += " --full"
    elif config.daggerdatalog:
        call += " --daggerdatalog"
    elif config.lean:
        call += " --lean"
    elif config.hud:
        call += " --hud"
    if config.failure_task_mask:
        call += " --failure-task-mask "
        s = ' '.join(str(e) for e in config.failure_task_mask)
        print("Failure tasks: ", s)
        call += s
    #clean_process()
    p2=subprocess.Popen(call,shell=True)
    p2.communicate()
    #clean_process()


if __name__ == '__main__':
    parser = Utils.get_parser(VERSION=VERSION)

    parser.add_argument(
        '--failure-task-mask', help='Specify the experiment to rerun when evaluating fixed combination, e.g. if you need to rerun trajectory numbered 6 and 7, let --failure-task-mask=[6,7]', nargs='+',
        type=int)
    parser.add_argument('--resample-config', type=str, default='fixed',
                        help='how do we treat the config parameters')
    parser.add_argument('--num-config', type=int, default=None,
                        help='if not None, sample a subset of the configs to run')
     # Parallel related
    parser.add_argument('--num-workers', type=int, default=3)
    parser.add_argument('--routes', type=json.loads, default=[
        ['srunner/data/routes_training_town01_autocast6.xml', 'srunner/data/towns01_traffic_scenarios_autocast6.json'],
        ['srunner/data/routes_training_town03_autocast8.xml', 'srunner/data/towns03_traffic_scenarios_autocast8.json'],
        ['srunner/data/routes_training_town03_autocast10.xml', 'srunner/data/towns03_traffic_scenarios_autocast10.json'],
    ])
    parser.add_argument('--benchmark_config',type=str,default="benchmark/scene6.json")
    parser.add_argument('--cuda_visible_devices', type=str, default='0')
    config = parser.parse_args()
    
    # main execution
    p=subprocess.Popen(["./scripts/launch_carla.sh", str(config.cuda_visible_devices), str(config.num_workers), str(int(config.port)+11*int(config.cuda_visible_devices))])
    parallel_eval(config)
    

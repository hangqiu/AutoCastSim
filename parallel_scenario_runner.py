import os
import sys
import ray
import json
import time
import random
import numpy as np
import itertools
from copy import deepcopy

from AVR.DataLogger import DataLogger
from scenario_runner import ScenarioRunner, VERSION
from AVR import Utils


@ray.remote(num_gpus=1. / 10)
class WorkerScenarioRunner(ScenarioRunner):
    def __init__(self, args, worker_id):
        super().__init__(args, prefix='')


def env_config_resample(variable):  # List of config
    # This resample keeps the same size of the original size of config
    max_value = max(variable)
    min_value = min(variable)
    size = len(variable)
    variable = np.random.uniform(low=min_value, high=max_value, size=size)
    return list(variable)


class MasterScenarioRunner:
    def __init__(self, args):
        self.workers = []
        self.args = []
        self.num_workers = args.num_workers
        print("***************************")
        print("Configuration:", args.benchmark_config)
        print("Agent Config:", args.agentConfig)
        print("Agent:", args.agent)
        if args.benchmark_config is not None:
            b_conf = json.load(open(args.benchmark_config))
            route = b_conf["route"]
            e_conf = b_conf["eval"]
            bgtraffic = b_conf["BGTRAFFIC"]
            egospd = e_conf["EGOSPD"]
            egodist = e_conf["EGODIST"]
            colspd = e_conf["COLSPD"]
            coldist = e_conf["COLDIST"]
            colacceldist = e_conf["COLACCELDIST"]

        if args.resample_config == 'random_uniform':
            # Resample the environment configuration
            print("Resample uniformly")
            colspd = env_config_resample(colspd)
            coldist = env_config_resample(coldist)
            egospd = env_config_resample(egospd)
            egodist = env_config_resample(egodist)
            colacceldist = env_config_resample(colacceldist)
        elif args.resample_config == 'random_gaussian':
            pass
        elif args.resample_config == 'fixed':
            pass

        eval_combinations = []
        failure_task_mask = []
        if args.failure_task_mask:
            failure_task_mask = args.failure_task_mask
        eval_count = 0
        for eval_comb in itertools.product(egospd, egodist, colspd, coldist, colacceldist):
            if eval_count in failure_task_mask or not args.failure_task_mask:
                eval_combinations.append(eval_comb)
            else:
                print("Trajectory ", eval_count, " was a success, skipping")
            eval_count += 1

        if args.num_config is not None:
            # We need to downsample or upsample the combination of configurations
            assert args.num_config > 0
            assert args.num_config < len(eval_combinations) + 1
            eval_combinations = random.sample(eval_combinations, k=args.num_config)

        num_worker = args.num_workers
        eval_args = []
        job_count = 0

        for eval_comb in eval_combinations:
            _args = deepcopy(args)
            i = job_count % args.num_workers
            _args.port = int(args.port) * (i + 1)
            _args.route = route
            _args.mqttport = int(args.mqttport) + job_count
            _args.trafficManagerPort = int(args.trafficManagerPort) + job_count
            print("Traffic Port", _args.trafficManagerPort)
            print("Mqtt port", _args.mqttport)
            eval_comb = np.array(eval_comb, dtype='float')
            _args.eval = list(eval_comb)
            self.args.append(_args)
            job_count += 1
            print(eval_comb)

    def run(self):
        i = 0
        self.clean_comm_ports()
        while i < len(self.args):
            srs = []
            jobs = []
            for j in range(i, min(i + self.num_workers, len(self.args))):
                print(j, len(self.args))
                srs.append(WorkerScenarioRunner.remote(self.args[j], j))
                time.sleep((j + 1))
            for j in range(len(srs)):
                jobs.append(srs[j].run.remote())
                time.sleep((j + 1))
            ray.wait(jobs, num_returns=len(jobs))
            time.sleep(1.0)
            self.clean_comm_ports()
            i += self.num_workers
            del jobs
            for sr in srs:
                ray.kill(sr)
            del srs
            print("Scenario Done!", i)
            time.sleep(10.0)

    def destroy(self):
        for worker in self.workers:
            worker.destroy.remote()

    def clean_comm_ports(self):
        """
        Clean all the ports connected to carla
        """
        for _args in self.args:
            mqtt_port = _args.mqttport
            traffic_port = _args.trafficManagerPort
            os.system("fuser -k {}/tcp".format(mqtt_port))
            os.system("fuser -k {}/tcp".format(traffic_port))


def main():
    """
    main function
    """

    import argparse
    ray.init(logging_level=40, log_to_driver=True, local_mode=False, num_gpus=1)

    parser = Utils.get_parser(VERSION=VERSION)

    parser.add_argument(
        '--failure-task-mask',
        help='Specify the experiment to rerun when evaluating fixed combination, e.g. if you need to rerun trajectory numbered 6 and 7, let --failure-task-mask=[6,7]',
        nargs='+',
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
        ['srunner/data/routes_training_town03_autocast10.xml',
         'srunner/data/towns03_traffic_scenarios_autocast10.json'],
    ])
    parser.add_argument('--benchmark_config', type=str, default="benchmark/scene6.json")
    arguments = parser.parse_args()
    # pylint: enable=line-too-long

    # AVR
    Utils.mqtt_port = arguments.mqttport
    Utils.parse_config_flags(arguments)

    if arguments.list:
        print("Currently the following scenarios are supported:")
        print(*ScenarioConfigurationParser.get_list_of_scenarios(arguments.configFile), sep='\n')
        return 1

    if (arguments.route and arguments.openscenario) or (arguments.route and arguments.scenario):
        print("The route mode cannot be used together with a scenario (incl. OpenSCENARIO)'\n\n")
        parser.print_help(sys.stdout)
        return 1

    if arguments.agent and (arguments.openscenario or arguments.scenario):
        print("Agents are currently only compatible with route scenarios'\n\n")
        parser.print_help(sys.stdout)
        return 1

    Utils.BACKGROUND_TRAFFIC = arguments.bgtraffic
    print(Utils.BACKGROUND_TRAFFIC)
    scenario_runner = None
    result = True

    try:
        scenario_runner = MasterScenarioRunner(arguments)
        result = scenario_runner.run()
    except Exception as e:
        print(e)
    finally:
        if scenario_runner is not None:
            scenario_runner.destroy()
            del scenario_runner

    return not result


if __name__ == "__main__":
    sys.exit(main())

import os
import subprocess
import sys
import time
from argparse import ArgumentParser
from collections import OrderedDict
from multiprocessing import Process, Queue
from typing import Callable, Optional, Union



def run(cmd_str, gpu, idx, queue) -> str:
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu)
    # list_dataset = args.list_dataset.split(",")
    # assert len(list_dataset) > 0

    result = ""
    # for dataset in list_dataset:
        # cmd, cmd_args = config_cmd(training_script_args, dataset)
        # command = [cmd]
        # command.extend(cmd_args)

    cmd: Union[Callable, str]
    cmd_args = []
    cmd = os.getenv("PYTHON_EXEC", sys.executable)
    cmd_args.append("-u")
    cmd_args.extend(cmd_str.split())
    command = [cmd]
    command.extend(cmd_args)

    p = subprocess.run(
        command,
        stdout=subprocess.PIPE, encoding="utf-8")

    if p.returncode == 0:
        _ = p.stdout
        result += _
    queue: Queue
    queue.put((idx, result))
    return


def write(args, dict_result):
    with open(f"{args.output}", "w") as w:
        for i in dict_result.values():
            w.write(i)
            w.write("\n")
            w.flush()


def main(args):
    queue = Queue()
    list_dataset = args.list_dataset.split(",")

    dict_process = OrderedDict()
    dict_result = OrderedDict()
    list_gpus = args.gpus.split(",")

    for gpu in list_gpus:
        dict_process[gpu] = None
    for i in range(len(list_dataset)):
        dict_result[i] = ""

    idx = 0
    while idx < len(list_dataset):
        for gpu in list_gpus:
            process = dict_process[gpu]
            process: Optional[Process]
            cmd = f"{args.cmd} --dataset {list_dataset[idx]}"
            if process is None:

                process = Process(
                    target=run,
                    args=(cmd, gpu, idx, queue))

                process.start()
                dict_process[gpu] = process
                idx += 1

            elif not process.is_alive():
                process: Process
                process.close()
                process = Process(
                    target=run,
                    args=(cmd, gpu, idx, queue))
                process.start()
                dict_process[gpu] = process
                idx += 1
            if idx >= len(list_dataset):
                break

            time.sleep(2)
            while not queue.empty():
                i, result = queue.get()
                dict_result[i] = result
            write(args, dict_result)
        if idx >= len(list_dataset):
            break

        time.sleep(1)
        while not queue.empty():
            i, result = queue.get()
            dict_result[i] = result
        write(args, dict_result)

    for m in dict_process.values():
        if m is not None:
            m.join()
    for m in dict_process.values():
        if m is not None:
            m.close()
    while not queue.empty():
        i, result = queue.get()
        dict_result[i] = result
    write(args, dict_result)


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("cmd", type=str, default=None)
    parser.add_argument("--list_dataset", type=str, default=("cifar10,cifar100,dtd,ucf101,cars,sun397,flowers,caltech101,food101,aircraft,birdsnap,pets,eurosat,imagenet_10shot"))
    parser.add_argument("--output", type=str, default="output")
    parser.add_argument("--gpus", type=str, default="0,1,2,3,4,5,6,7")
    args = parser.parse_args()
    main(args)


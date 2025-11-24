from __future__ import print_function

import argparse
import os
import re
import subprocess
import sys
import traceback

import pkg_resources
import six

from .util import read_yaml
from .package import build_jar
from .status import status


def _ensure_str(b):
    if six.PY3:
        return str(b, 'ascii')
    else:
        return b


def get_storm_version():
    version_output = [s.strip() for s in _ensure_str(subprocess.check_output(['storm', 'version'])).split('\n')]
    for line in version_output:
        m = re.search(r'^(Storm )?(\d+\.\d+\.\d+)(-(\w+))?$', line)
        if m is not None:
            break
    return m.group(2)


def get_sourcejar():
    storm_version = get_storm_version()
    sourcejar = pkg_resources.resource_filename(
        pkg_resources.Requirement.parse('petrel'),
        'petrel/generated/storm-petrel-%s-SNAPSHOT-jar-with-dependencies.jar' % storm_version)
    return sourcejar


def submit(sourcejar, destjar, config, venv, name, definition, logdir, extrastormcp):
    # Build a topology jar and submit it to Storm.
    if not sourcejar:
        sourcejar = get_sourcejar()
    if venv == 'self':
        venv = os.path.dirname(os.path.dirname(sys.executable))
    build_jar(
        source_jar_path=sourcejar,
        dest_jar_path=destjar,
        config=config,
        definition=definition,
        venv=venv,
        logdir=logdir)
    storm_class_path = [ _ensure_str(subprocess.check_output(["storm","classpath"])).strip(), destjar ]
    if extrastormcp is not None:
        storm_class_path = [ extrastormcp ] + storm_class_path
    storm_home = os.path.dirname(os.path.dirname(
        subprocess.check_output(['which', 'storm'])))
    submit_args = [
        "",
        "-client",
        "-Dstorm.options=",
        "-Dstorm.home=%s" % storm_home,
        "-cp",":".join(storm_class_path),
        "-Dstorm.jar=%s" % destjar,
        "storm.petrel.GenericTopology",
    ]
    if name:
        submit_args += [name]
    os.execvp('java', submit_args)


def kill(name, config):
    config = read_yaml(config)
    
    # Read the nimbus.host setting from topology YAML so we can submit the
    # "kill" command to the correct cluster.
    nimbus_host = config.get('nimbus.host')
    kill_args = ['', 'kill', name]
    if nimbus_host:
        kill_args += ['-c', 'nimbus.host=%s' % nimbus_host]
    os.execvp('storm', kill_args)


def main():
    parser = argparse.ArgumentParser(prog='petrel', description='Petrel command line')
    subparsers = parser.add_subparsers()
    parser_submit = subparsers.add_parser('submit')
    parser_submit.add_argument('--sourcejar', dest='sourcejar', help='source JAR path')
    parser_submit.add_argument('--destjar', dest='destjar', default='topology.jar',
                        help='destination JAR path')
    parser_submit.add_argument('--config', dest='config', required=True,
                        help='YAML file with the topology configuration')
    parser_submit.add_argument('--definition', dest='definition',
                        help='python module and function defining the topology (must be in current directory)')
    parser_submit.add_argument('--venv', dest='venv',
                        help='An existing virtual environment to reuse on the server ("self" uses the current virtual environment)')
    parser_submit.add_argument('--logdir', dest='logdir',
                        help='Root directory for logfiles (default: the storm supervisor directory)')
    parser_submit.add_argument('--extrastormcp', dest='extrastormcp',
                        help='Extra jars on the storm classpath, useful for controlling log4j')
    parser_submit.add_argument('name', const=None, nargs='?',
        help='name of the topology. If provided, the topology is submitted to the cluster. ' +
        'If omitted, the topology runs in local mode.')
    parser_submit.set_defaults(func=submit)

    parser_status = subparsers.add_parser('status', help='Report status of running topologies')
    parser_status.add_argument(
        'nimbus',
        help='Nimbus host address')
    parser_status.add_argument('--worker', help='Only list tasks on this worker')
    parser_status.add_argument('--port', help='Only list tasks on this port number')
    parser_status.add_argument('--topology', help='Only list information on this topology')
    parser_status.set_defaults(func=status)
    
    parser_kill = subparsers.add_parser('kill', help='kill a topology running on a cluster')
    parser_kill.add_argument('name', help='name of the topology')
    parser_kill.add_argument('--config', dest='config', required=True,
                        help='YAML file with the topology configuration')
    parser_kill.set_defaults(func=kill)

    try:
        args = parser.parse_args()
        func = args.__dict__.pop('func')
        func(**args.__dict__)
    except Exception as e:
        print(str(e))
        traceback.print_exc()
        sys.exit(1)

if __name__ == '__main__':
    main()

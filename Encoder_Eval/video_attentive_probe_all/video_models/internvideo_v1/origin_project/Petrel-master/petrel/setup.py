#!/usr/bin/env python
from __future__ import print_function

import os
import re
import shutil
import subprocess
import sys

from setuptools import setup, find_packages
if sys.version_info.major == 2:
    from urllib2 import urlopen
else:
    from urllib.request import urlopen


README = os.path.join(os.path.dirname(__file__), 'README.txt')
long_description = open(README).read() + '\n\n'

PACKAGE = "petrel"

PETREL_VERSION = '0.3'


def ensure_str(b):
    if sys.version_info.major == 3:
        return str(b, 'ascii')
    else:
        return b


def get_storm_version():
    version_output = [s.strip() for s in ensure_str(subprocess.check_output(['storm', 'version'])).split('\n')]
    for line in version_output:
        m = re.search(r'^(Storm )?(\d+\.\d+\.\d+)(-(\w+))?$', line)
        if m is not None:
            break
    return m.group(0), m.group(2)


def get_version(argv):
    """ Dynamically calculate the version based on VERSION."""
    return '%s.%s' % (get_storm_version()[1], PETREL_VERSION)


def build_petrel():
    version_string, version_number = get_storm_version()
    
    # Generate Thrift Python wrappers.
    if os.path.isdir('petrel/generated'):
        shutil.rmtree('petrel/generated')
    os.mkdir('petrel/generated')
    version_tuple = tuple([int(s) for s in version_number.split('.')[:3]])
    if version_tuple <= (0, 9, 0):
        path = '%s/src/storm.thrift' % version_string
        f_url = urlopen('https://raw.github.com/apache/incubator-storm/%s' % path)
    elif version_tuple <= (0, 9, 1):
        path = 'apache-%s/storm-core/src/storm.thrift' % version_number
        f_url = urlopen('https://raw.github.com/apache/incubator-storm/%s' % path)
    elif version_tuple == (0, 9, 2):
        f_url = urlopen('https://raw.githubusercontent.com/apache/storm/v0.9.2-incubating/storm-core/src/storm.thrift')
    elif version_tuple[:2] == (1, 0):
        f_url = urlopen('https://raw.githubusercontent.com/HeartSaVioR/storm/ad46a470c173c8daf2bd76bc10ecbfc6fd08865b/storm-core/src/storm.thrift')
    else:
        f_url = urlopen('https://raw.githubusercontent.com/apache/storm/v%s/storm-core/src/storm.thrift' % version_number)

    with open('storm.thrift', 'w') as f:
        print('namespace py petrel.generated.storm', file=f)
        f.write(ensure_str(f_url.read()))
    f_url.close()

    subprocess.check_call(['thrift', '-gen', 'py', '-out', '.', 'storm.thrift'])
    os.remove('storm.thrift')

    # :HACK: Thrift generates code that is not compatible with Python 3. To fix
    # this, we patch some of the generated files to use relative imports.
    for filename in ['constants.py', 'Nimbus.py']:
        path = os.path.join('petrel/generated/storm', filename)
        with open(path, 'r') as f:
            lines = f.readlines()
            for i, line in enumerate(lines):
                if line == 'from ttypes import *\n':
                    lines[i] = 'from .ttypes import *\n'
        with open(path, 'w') as f:
            for line in lines:
                print(line, file=f, end='')

    # Build JVMPetrel.
    old_cwd = os.getcwd()
    os.chdir('../jvmpetrel')
    subprocess.check_call(['mvn', '-Dstorm_version=%s' % version_number, 'assembly:assembly'])
    os.chdir(old_cwd)
    shutil.copyfile(
        '../jvmpetrel/target/storm-petrel-%s-SNAPSHOT-jar-with-dependencies.jar' % version_number,
        'petrel/generated/storm-petrel-%s-SNAPSHOT-jar-with-dependencies.jar' % version_number)

if 'bdist_egg' in sys.argv or 'bdist_wheel' in sys.argv or 'develop' in sys.argv:
    build_petrel()

setup(name=PACKAGE
    ,version=get_version(sys.argv)
    ,description=("Storm Topology Builder")
    ,long_description=long_description
    ,classifiers=[
         "Programming Language :: Python"
        ,"Operating System :: POSIX :: Linux"
        ,"Development Status :: 5 - Production/Stable"
        ,"Intended Audience :: Developers"
        ,"License :: OSI Approved :: BSD License"
        ,"Topic :: Software Development :: Libraries :: Python Modules"
        ,"Topic :: System :: Distributed Computing"
    ]
    ,keywords='Storm Topology Builder'
    ,author='bhart'
    ,url='https://github.com/AirSage/Petrel'
    ,scripts=[
         'bin/petrel',
        ]
    ,packages=find_packages()
    ,package_data={'': ['*.jar']}
    ,license='BSD 3-clause'
    ,install_requires=[
        # Request specific Thrift version. Storm is in Java and may be sensitive to version incompatibilities.
        'thrift==0.9.3',
        'PyYAML==3.10',
        'six==1.10.0',
    ]
    # Setting this flag makes Petrel easier to debug within a running topology.
    ,zip_safe=False)

from __future__ import print_function

import io
import os
import sys
import shutil
import getpass
import socket
import zipfile
import glob
import pkg_resources

from itertools import chain

import six

from .topologybuilder import TopologyBuilder
from .util import read_yaml

MANIFEST = 'manifest.txt'


def add_to_jar(jar, name, data):
    path = 'resources/%s' % name
    print('Adding {}'.format(path))
    jar.writestr(path, data)


def add_file_to_jar(jar, directory, script=None, required=True, strip_dir=True):
    if script is not None:
        path = os.path.join(directory, script)
    else:
        path = directory
    
    # Use glob() to allow for wildcards, e.g. in manifest.txt.
    path_list = glob.glob(path)

    if len(path_list) == 0 and required:
        raise ValueError('No files found matching: %s' % path)
    #elif len(path_list) > 1:
    #    raise ValueError("Wildcard '%s' matches multiple files: %s" % (path, ', '.join(path_list)))
    for this_path in path_list:
        with open(this_path, 'rb') as f:
            if strip_dir:
                # Drop the path when adding to the jar.
                name = os.path.basename(this_path)
            else:
                name = os.path.relpath(this_path)
            add_to_jar(jar, name, f.read())


def add_dir_to_jar(jar, directory, required=True):
    dir_path_list = glob.glob(directory)

    if len(dir_path_list) == 0 and required:
        raise ValueError('No directory found matching: %s' % directory)
    for dir_path in dir_path_list:
        for dirpath, dirnames, filenames in os.walk(dir_path):
            for filename in filenames:
                add_file_to_jar(jar, dirpath, filename, strip_dir=False)


def add_item_to_jar(jar, item):
    path_list = glob.glob(item)
    for this_path in path_list:
        if os.path.isdir(this_path):
            add_dir_to_jar(jar, this_path)
        elif os.path.isfile(this_path):
            add_file_to_jar(jar, this_path)
        else:
            raise ValueError("No file or directory found matching: %s" % this_path)


def build_jar(source_jar_path, dest_jar_path, config, venv=None, definition=None, logdir=None):
    """Build a StormTopology .jar which encapsulates the topology defined in
    topology_dir. Optionally override the module and function names. This
    feature supports the definition of multiple topologies in a single
    directory."""

    if definition is None:
        definition = 'create.create'

    # Prepare data we'll use later for configuring parallelism.
    config_yaml = read_yaml(config)
    parallelism = dict((k.split('.')[-1], v) for k, v in six.iteritems(config_yaml)
        if k.startswith('petrel.parallelism'))

    pip_options = config_yaml.get('petrel.pip_options', '')

    module_name, dummy, function_name = definition.rpartition('.')
    
    topology_dir = os.getcwd()

    # Make a copy of the input "jvmpetrel" jar. This jar acts as a generic
    # starting point for all Petrel topologies.
    source_jar_path = os.path.abspath(source_jar_path)
    dest_jar_path = os.path.abspath(dest_jar_path)
    if source_jar_path == dest_jar_path:
        raise ValueError("Error: Destination and source path are the same.")
    shutil.copy(source_jar_path, dest_jar_path)
    jar = zipfile.ZipFile(dest_jar_path, 'a', compression=zipfile.ZIP_DEFLATED)
    
    added_path_entry = False
    try:
        # Add the files listed in manifest.txt to the jar.
        with open(os.path.join(topology_dir, MANIFEST), 'r') as f:
            for fn in f.readlines():
                # Ignore blank and comment lines.
                fn = fn.strip()
                if len(fn) and not fn.startswith('#'):

                    add_item_to_jar(jar, os.path.expandvars(fn.strip()))

        # Add user and machine information to the jar.
        add_to_jar(jar, '__submitter__.yaml', '''
petrel.user: %s
petrel.host: %s
''' % (getpass.getuser(),socket.gethostname()))
        
        # Also add the topology configuration to the jar.
        with open(config, 'r') as f:
            config_text = f.read()
        add_to_jar(jar, '__topology__.yaml', config_text)
    
        # Call module_name/function_name to populate a Thrift topology object.
        builder = TopologyBuilder()
        module_dir = os.path.abspath(topology_dir)
        if module_dir not in sys.path:
            sys.path[:0] = [ module_dir ]
            added_path_entry = True
        module = __import__(module_name)
        getattr(module, function_name)(builder)

        # Add the spout and bolt Python scripts to the jar. Create a
        # setup_<script>.sh for each Python script.

        # Add Python scripts and any other per-script resources.
        for k, v in chain(six.iteritems(builder._spouts), six.iteritems(builder._bolts)):
            add_file_to_jar(jar, topology_dir, v.script)

            # Create a bootstrap script.
            if venv is not None:
                # Allow overriding the execution command from the "petrel"
                # command line. This is handy if the server already has a
                # virtualenv set up with the necessary libraries.
                v.execution_command = os.path.join(venv, 'bin/python')

            # If a parallelism value was specified in the configuration YAML,
            # override any setting provided in the topology definition script.
            if k in parallelism:
                builder._commons[k].parallelism_hint = int(parallelism.pop(k))

            v.execution_command, v.script = \
                intercept(venv, v.execution_command, os.path.splitext(v.script)[0],
                          jar, pip_options, logdir)

        if len(parallelism):
            raise ValueError(
                'Parallelism settings error: There are no components named: %s' %
                ','.join(parallelism.keys()))

        # Build the Thrift topology object and serialize it to the .jar. Must do
        # this *after* the intercept step above since that step may modify the
        # topology definition.
        buf = io.BytesIO()
        builder.write(buf)
        add_to_jar(jar, 'topology.ser', buf.getvalue())
    finally:
        jar.close()
        if added_path_entry:
            # Undo our sys.path change.
            sys.path[:] = sys.path[1:]


def intercept(venv, execution_command, script, jar, pip_options, logdir):
    #create_virtualenv = 1 if execution_command == EmitterBase.DEFAULT_PYTHON else 0
    create_virtualenv = 1 if venv is None else 0
    script_base_name = os.path.splitext(script)[0]
    intercept_script = 'setup_%s.sh' % script_base_name

    # Bootstrap script that sets up the worker's Python environment.
    add_to_jar(jar, intercept_script, '''#!/bin/bash
set -e
SCRIPT=%(script)s
LOG=%(logdir)s/petrel$$_$SCRIPT.log
VENV_LOG=%(logdir)s/petrel$$_virtualenv.log
echo "Beginning task setup" >>$LOG 2>&1

# I've seen Storm Supervisor crash repeatedly if we create any new
# subdirectories (e.g. Python virtualenvs) in the worker's "resources" (i.e.
# startup) directory. So we put new directories in /tmp. It seems okay to create
# individual files though, e.g. the log.
PYVER=%(major)d.%(minor)d
CWDIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &&  pwd )"
# Should we also allow dots in topology names?
TOPOLOGY_ID_REGEX="([+A-Za-z0-9_\-]+)/resources$"
[[ $CWDIR =~ $TOPOLOGY_ID_REGEX ]] && TOPOLOGY_ID="${BASH_REMATCH[1]}"
WRKDIR=/tmp/petrel-$TOPOLOGY_ID

VENV=%(venv)s
CREATE_VENV=%(create_virtualenv)d

START=$SECONDS
mkdir -p $WRKDIR/egg_cache >>$LOG 2>&1
export PYTHON_EGG_CACHE=$WRKDIR/egg_cache

set +e
python$PYVER -c "print" >>/dev/null 2>&1
RETVAL=$?
set -e
if [ $RETVAL -ne 0 ]; then
    # If desired Python is not found, run the user's .bashrc. Maybe it will
    # add the desired Python to the path.
    source ~/.bashrc >>$LOG 2>&1
fi
# Now the desired Python *must* be available. This line ensures we detect the
# error and fail before continuing.
python$PYVER -c "pass" >>$LOG 2>&1


unamestr=`uname`
if [[ "$unamestr" != 'Darwin' ]]; then
    # Create at most ONE virtualenv for the topology. Put the lock file in /tmp
    # because when running Storm in local mode, each task runs in a different
    # subdirectory. Thus they have different lock files but are creating the same
    # virtualenv. This causes multiple tasks to get into the lock before the
    # virtualenv has all the libraries installed.
    set +e
    which flock >>$LOG 2>&1
    has_flock=$?
    set -e
    LOCKFILE="/tmp/petrel-$TOPOLOGY_ID.lock"
    LOCKFD=99
    # PRIVATE
    _lock()             { flock -$1 $LOCKFD; }
    _no_more_locking()  { _lock u; _lock xn && rm -f $LOCKFILE; }
    _prepare_locking()  { eval "exec $LOCKFD>\\"$LOCKFILE\\""; trap _no_more_locking EXIT; }
    # ON START
    if [ "$has_flock" -eq "0" ]
    then
        _prepare_locking
    fi
    # PUBLIC
    exlock_now()        { _lock xn; }  # obtain an exclusive lock immediately or fail
    exlock()            { _lock x; }   # obtain an exclusive lock
    shlock()            { _lock s; }   # obtain a shared lock
    unlock()            { _lock u; }   # drop a lock

    if [ $CREATE_VENV -ne 0 ]; then
        # On Mac OS X, the "flock" command is not available
        create_new=1
        if [ "$has_flock" -eq "0" ]
        then 
            if [ -d $VENV ];then
                echo "Using existing venv: $VENV" >>$LOG 2>&1
                shlock
                source $VENV/bin/activate >>$LOG 2>&1
                unlock
                create_new=0
            elif ! exlock_now;then
                echo "Using existing venv: $VENV" >>$LOG 2>&1
                shlock
                source $VENV/bin/activate >>$LOG 2>&1
                unlock
                create_new=0
            fi
        fi
        if [ "$create_new" -eq "1" ]
        then
            echo "Creating new venv: $VENV" >>$LOG 2>&1
            virtualenv --system-site-packages --python python$PYVER $VENV >>$VENV_LOG 2>&1
            source $VENV/bin/activate >>$VENV_LOG 2>&1

            # Ensure the version of Thrift on the worker matches our version.
            # This may not matter since Petrel only uses Thrift for topology build
            # and submission, but I've had some odd version problems with Thrift
            # and Storm/Java so I want to be safe.
            for f in thrift==%(thrift_version)s PyYAML==3.10
            do
                echo "Installing $f" >>$VENV_LOG 2>&1
                pip install %(pip_options)s $f >>$VENV_LOG 2>&1
            done

            # Install Petrel. Two possibilities:
            # 1. If a Petrel egg was provided with the topology, install it.
            # This is useful primarily for those who want direct control over
            # the specific version of Petrel being installed. This is typically
            # useful only if you're modifying and building Petrel from source.
            # 2. Otherwise, install using easy_install, typically from a mirror.
            # Because it does not require a local copy of the Petrel egg, this
            # is simpler for most Petrel users, especially new ones.
            # :TRICKY: Use bash glob expansion to see if a local file exists. See http://stackoverflow.com/questions/6363441/check-if-a-file-exists-with-wildcard-in-shell-script
            for f in petrel-*-py$PYVER.egg
            do
                # Check if the glob gets expanded to existing files.
                # If not, f here will be exactly the pattern above
                # and the exists test will evaluate to false.
                if [ -e "$f" ]
                then
                    # Case 1
                    echo "Installing Petrel from local file $f" >>$VENV_LOG 2>&1
                    easy_install $f >>$VENV_LOG 2>&1
                else
                    # Case 2
                    echo "Installing petrel==%(petrel_version)s" >>$VENV_LOG 2>&1
                    easy_install petrel==%(petrel_version)s >>$VENV_LOG 2>&1
                fi
            
                # We can break after the first iteration.
                break
            done

            if [ -f ./setup.sh ]; then
                /bin/bash ./setup.sh $CREATE_VENV >>$VENV_LOG 2>&1
            fi
            if [ "$has_flock" -eq "0" ]
            then 
                unlock
            fi
        fi
    else
        # This is a prototype feature where the topology specifies a virtualenv
        # that already exists. Could be useful in some cases, since this means the
        # topology is up and running more quickly.
        echo "Using existing venv: $VENV" >>$LOG 2>&1
        source $VENV/bin/activate >>$LOG 2>&1
    fi
fi

ELAPSED=$(($SECONDS-$START))
echo "Task setup took $ELAPSED seconds" >>$LOG 2>&1
echo "Launching: python -m petrel.run $SCRIPT $LOG" >>$LOG 2>&1
# We use exec to avoid creating another process. Creating a second process is
# not only less efficient but also confuses the way Storm monitors processes.
exec python -m petrel.run $SCRIPT $LOG
''' % dict(
        major=sys.version_info.major,
        minor=sys.version_info.minor,
        script=script,
        venv='$WRKDIR/venv' if venv is None else venv,
        logdir='$PWD' if logdir is None else logdir,
        create_virtualenv=create_virtualenv,
        thrift_version=pkg_resources.get_distribution("thrift").version,
        petrel_version=pkg_resources.get_distribution("petrel").version,
        pip_options=pip_options,
    ))

    return '/bin/bash', intercept_script

from __future__ import print_function

import os
import sys
import socket
import logging.config
import traceback

from petrel import storm

LOG_CONFIG_FILE = 'logconfig.ini'

log_file_path = None
log_initialized = False
module_name = ''


def open_log():
    return open(log_file_path, 'a')


def handle_exception(type, value, tb):
    message = 'E_RUNFAILED_%s_%s_%d_%s' % (module_name,
                                        socket.gethostname(),
                                        os.getpid(),
                                        type.__name__)
    if log_initialized:
        log = logging.getLogger('petrel.run')
        log.error(message)
    storm.sendFailureMsgToParent(message)
   
    with open_log() as f:
        print('Exception occurred in %s. Worker exiting.' % module_name, file=f)
        f.write(''.join(traceback.format_exception(type, value, tb)))


def log_config():
    assert 'PETREL_LOG_PATH' in os.environ
    from subprocess import check_output
    
    # Set an environment variable that points to the Nimbus server.
    # logconfig.ini may use this to direct SysLogHandler to this machine.
    try:
        os.environ['NIMBUS_HOST'] = check_output(['storm', 'remoteconfvalue', 'nimbus.host']).split(':')[1].strip()
    except Exception as e:
        # It's not worth crashing if we can't set this.
        pass
    
    if os.path.exists(LOG_CONFIG_FILE):
        logging.config.fileConfig(LOG_CONFIG_FILE)


# This code is still a work in progress. It may have bugs that cause
# topologies to be unstable. I've seen it cause ShellSpout.querySubprocess()
# in Java to receive a JSONObject with a null "command" value.
class StormHandler(logging.Handler):
    def __init__(self, *l, **kw):
        super(StormHandler, self).__init__(*l, **kw)
        hostname = socket.gethostname().split('.')[0]
        script_name = os.getenv('SCRIPT') # Should be passed by setup_*.sh.
        if script_name is None:
            script_name = '<unknown>'
        process_id = os.getpid()
        self.format_string = '[%s][%s][%d] %%s' % (hostname, script_name, process_id)
    
    def emit(self, record):
        msg = self.format(record)
        for line in msg.split('\n'):
            formatted_line = self.format_string % line
            #print >> sys.stderr, "Calling storm.log with: %s" % formatted_line
            storm.log('%s' % formatted_line)

# Comment this out until logging to Storm proves to be stable.
#logging.StormHandler = StormHandler


def main():
    if len(sys.argv) != 3:
        print("Usage: %s <module> <log file>" % os.path.splitext(os.path.basename(sys.argv[0]))[0], file=sys.stderr)
        sys.exit(1)

    try:
        global log_file_path, log_initialized, module_name
        os.environ['PETREL_LOG_PATH'] = log_file_path = os.path.abspath(sys.argv[2])
        os.environ['SCRIPT'] = module_name = sys.argv[1]
        sys.excepthook = handle_exception
    
        with open_log() as f:
            print('%s invoked with the following arguments: %s' % (sys.argv[0], repr(sys.argv[1:])), file=f)
            ver_info = sys.version_info
            print("python version: %d.%d.%d" % (ver_info.major, ver_info.minor, ver_info.micro), file=f)
            import getpass
            print('user=%s' % getpass.getuser(), file=f)
            print('PATH=%s' % os.getenv('PATH'), file=f)
            print('LD_LIBRARY_PATH=%s' % os.getenv('LD_LIBRARY_PATH'), file=f)
            print('PYTHON_EGG_CACHE=%s' % os.getenv('PYTHON_EGG_CACHE'), file=f)

        # Initialize logging. Redirect stderr to the log as well.
        log_config()
        log_initialized = True
        
        storm.initialize_profiling()
        
        sys.path[:0] = [ os.getcwd() ]
        module = __import__(module_name)
        getattr(module, 'run')()
        with open_log() as f:
            print('Worker %s exiting normally.' % module_name, file=f)
    except:
        # Here we explicitly catch exceptions from the worker. This is a "belt
        # and suspenders" approach in case our sys.excepthook gets overwritten
        # by another library.
        handle_exception(*sys.exc_info())

# When invoked as a main program, invoke Petrel on a spout or bolt module.
if __name__ == '__main__':
    main()

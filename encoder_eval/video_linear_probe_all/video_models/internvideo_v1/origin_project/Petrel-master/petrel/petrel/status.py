# Based on this code: http://tutorials.github.com/pages/retrieving-storm-data-from-nimbus.html
from __future__ import print_function

import datetime

from six import StringIO

from thrift.transport import TSocket
from thrift.transport import TTransport
from thrift.protocol import TBinaryProtocol

import petrel.topologybuilder # Install the __hash__ monkeypatch
from petrel.generated.storm import Nimbus
from petrel.generated.storm.ttypes import *
from petrel.generated.storm.constants import *

def get_statistic(es, name):
    if hasattr(es, 'stats') and es.stats is not None:
        if es.component_id == '__acker':
            return sum(es.stats.specific.bolt.acked[':all-time'].values())
        elif hasattr(es.stats, name):
            tmp = getattr(es.stats, name).get(':all-time', {})
            if 'default' in tmp:
                return tmp['default']

    return '-'

def print_topology_status(client, topology, worker, port):
    records = []
    print('id,uptime,host,port,component,emitted,transferred,acked,failed,num_errors')
    info = client.getTopologyInfo(topology.id)
    for i, es in enumerate(info.tasks):
        emit = True
        host = es.host.split('.')[0]
        if worker is not None and host != worker:
            emit = False
        if port is not None and es.port != port:
            emit = False
        
        if emit:
            record = {}
            record['columns'] = [
                es.task_id,
                es.uptime_secs,
                es.host.split('.')[0],
                es.port,
                es.component_id,
                get_statistic(es, 'emitted'),
                get_statistic(es, 'transferred'),
                get_statistic(es, 'acked'),
                get_statistic(es, 'failed'),
                len(es.errors)
            ]
            if len(es.errors):
                msg = StringIO()
                for j, e in enumerate(es.errors):
                    print(
                        'Error #{} ({})'.format(
                            j + 1,
                            datetime.datetime.fromtimestamp(es.errors[j].error_time_secs).strftime('%Y/%m/%d %H:%M:%S')),
                        file=msg)
                    print(file=msg)
                    print(es.errors[j].error, file=msg)
                record['error'] = msg.getvalue()
                
            records.append(record)

    records.sort(key=lambda r: (r['columns'][0], r['columns'][1]))
    for record in records:
        print(record['columns'], ', ')
        if 'error' in record:
            print(record['error'])


def status(nimbus, topology, worker, port):
    socket = TSocket.TSocket(nimbus, 6627)
    transport = TTransport.TFramedTransport(socket)
    protocol = TBinaryProtocol.TBinaryProtocol(transport)
    client = Nimbus.Client(protocol)

    transport.open()
    try:
        summary = client.getClusterInfo()
        #print(summary)
        
        assert len(summary.topologies) == 1
        for running_topology in summary.topologies:
            if topology is None or topology == running_topology.name:
                print_topology_status(client, running_topology, worker, port)
    finally:
        transport.close()

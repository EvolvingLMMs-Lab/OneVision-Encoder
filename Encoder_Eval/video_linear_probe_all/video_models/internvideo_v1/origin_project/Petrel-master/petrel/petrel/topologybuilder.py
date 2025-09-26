import io
import json

from thrift.transport import TTransport
from thrift.protocol import TBinaryProtocol

import six

from petrel.generated.storm.ttypes import ComponentCommon, Grouping, NullStruct, GlobalStreamId
from petrel.generated.storm.ttypes import StreamInfo, Bolt, SpoutSpec, ShellComponent
from petrel.generated.storm.ttypes import ComponentObject, StormTopology

# Storm uses GlobalStreamId as a dict key, but the Thrift Python binding doesn't
# support this. As a simple workaround, add a hash function that always returns
# 0. The resulting map won't be efficient at storing large amounts of data, but
# it should work in all cases.
# https://issues.apache.org/jira/browse/THRIFT-162.
GlobalStreamId.__hash__ = lambda self: 0

##########################################################################


class TMemoryBuffer(TTransport.TMemoryBuffer):
    """
    This class is a workaround for a Python 3 incompatibility in thrift --
    its implementation of TMemoryBuffer uses a StringIO object, but it should
    use bytes.
    """
    def __init__(self, value=None):
        """value -- a value to read from for stringio

        If value is set, this will be a transport for reading,
        otherwise, it is for writing"""
        if value is not None:
            self._buffer = io.BytesIO(value)
        else:
            self._buffer = io.BytesIO()

    def write(self, buf):
        if six.PY3:
            if not isinstance(buf, six.binary_type):
                buf = six.binary_type(buf, 'ascii')
        self._buffer.write(buf)


class TopologyBuilder(object):
    def __init__(self):
        self._bolts = {}
        self._spouts = {}
        self._commons = {}
    
    #/**
    # * Define a new bolt in this topology with the specified amount of parallelism.
    # *
    # * @param id the id of this component. This id is referenced by other components that want to consume this bolt's outputs.
    # * @param bolt the bolt
    # * @param parallelism_hint the number of tasks that should be assigned to execute this bolt. Each task will run on a thread in a process somewhere around the cluster.
    # * @return use the returned object to declare the inputs to this component
    # */
    def setBolt(self, id, bolt, parallelism_hint=None):
        self._validateUnusedId(id);
        self._initCommon(id, bolt, parallelism_hint);
        self._bolts[id] = bolt

        return _BoltGetter(self, id)

    #/**
    # * Define a new spout in this topology.
    # *
    # * @param id the id of this component. This id is referenced by other components that want to consume this spout's outputs.
    # * @param spout the spout
    # */
    def setSpout(self, id, spout, parallelism_hint=None):
        self._validateUnusedId(id);
        self._initCommon(id, spout, parallelism_hint)
        self._spouts[id] = spout

    def addOutputStream(self, id, streamId, output_fields, direct=False):
        self._commons[id].streams[streamId] = StreamInfo(output_fields, direct=direct)

    def createTopology(self):
        boltSpecs = {}
        spoutSpecs = {}
        for boltId, bolt in six.iteritems(self._bolts):
            t_bolt = Bolt()
            shell_object = ShellComponent()
            shell_object.execution_command = bolt.execution_command
            shell_object.script = bolt.script
            t_bolt.bolt_object = ComponentObject()
            t_bolt.bolt_object.shell = shell_object
            t_bolt.common = self._getComponentCommon(boltId, bolt)
            boltSpecs[boltId] = t_bolt

        for spoutId, spout in six.iteritems(self._spouts):
            spout_spec = SpoutSpec()
            shell_object = ShellComponent()
            shell_object.execution_command = spout.execution_command
            shell_object.script = spout.script
            spout_spec.spout_object = ComponentObject()
            spout_spec.spout_object.shell = shell_object
            spout_spec.common = self._getComponentCommon(spoutId, spout)
            spoutSpecs[spoutId] = spout_spec

        topology = StormTopology()
        topology.spouts = spoutSpecs
        topology.bolts = boltSpecs
        topology.state_spouts = {} # Not supported yet, I think
        return topology

    def write(self, stream):
        """Writes the topology to a stream or file."""
        topology = self.createTopology()
        def write_it(stream):
            transportOut = TMemoryBuffer()
            protocolOut = TBinaryProtocol.TBinaryProtocol(transportOut)
            topology.write(protocolOut)
            bytes = transportOut.getvalue()
            stream.write(bytes)

        if isinstance(stream, six.string_types):
            with open(stream, 'wb') as f:
                write_it(f)
        else:
            write_it(stream)
            
        return topology

    def read(self, stream):
        """Reads the topology from a stream or file."""
        def read_it(stream):
            bytes = stream.read()
            transportIn = TMemoryBuffer(bytes)
            protocolIn = TBinaryProtocol.TBinaryProtocol(transportIn)
            topology = StormTopology()
            topology.read(protocolIn)
            return topology
            
        if isinstance(stream, six.string_types):
            with open(stream, 'rb') as f:
                return read_it(f)
        else:
            return read_it(stream)

    def _validateUnusedId(self, id):
        if id in self._bolts:
            raise KeyError("Bolt has already been declared for id " + id);
        elif id in self._spouts:
            raise KeyError("Spout has already been declared for id " + id);

    def _getComponentCommon(self, id, component):
        common = self._commons[id]
        stream_info = StreamInfo()
        stream_info.output_fields = component.declareOutputFields()
        stream_info.direct = False # Appears to be unused by Storm
        common.streams['default'] = stream_info
        
        return common

    def _initCommon(self, id, component, parallelism):
        common = ComponentCommon()
        
        common.inputs = {}
        common.streams = {}
        if parallelism is not None:
            common.parallelism_hint = parallelism
        conf = component.getComponentConfiguration()
        if conf is not None:
            common.json_conf = json.dumps(conf)
        self._commons[id] = common


class _BoltGetter(object):
    def __init__(self, owner, boltId):
        self._owner = owner
        self._boltId = boltId

    def allGrouping(self, componentId, streamId='default'):
        return self.grouping(componentId, 'all', NullStruct(), streamId)

    def fieldsGrouping(self, componentId, fields, streamId='default'):
        return self.grouping(componentId, 'fields', fields, streamId)

    def globalGrouping(self, componentId, streamId='default'):
        return self.fieldsGrouping(componentId, [], streamId)

    def shuffleGrouping(self, componentId, streamId='default'):
        return self.grouping(componentId, 'shuffle', NullStruct(), streamId)

    def localOrShuffleGrouping(self, componentId, streamId='default'):
        return self.grouping(componentId, 'local_or_shuffle', NullStruct(), streamId)

    def noneGrouping(self, componentId, streamId='default'):
        return self.grouping(componentId, 'none', NullStruct(), streamId)

    def directGrouping(self, componentId, streamId='default'):
        return self.grouping(componentId, 'direct', NullStruct(), streamId)

    def grouping(self, componentId, attr, grouping, streamId):
        o_grouping = Grouping()
        setattr(o_grouping, attr, grouping)
        self._owner._commons[self._boltId].inputs[GlobalStreamId(componentId, streamId)] = o_grouping
        return self

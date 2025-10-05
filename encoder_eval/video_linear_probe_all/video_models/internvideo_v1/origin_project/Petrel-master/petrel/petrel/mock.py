from collections import deque, defaultdict, namedtuple

import storm

python_id = id

STORM_TUPLE = 0
LIST = 1
TUPLE = 2
NAMEDTUPLE = 3

class MockSpout(storm.Spout):
    def __init__(self, output_fields, data):
        self.output_fields = output_fields
        self.data = data
        self.index = 0

    def declareOutputFields(self):
        return self.output_fields

    def nextTuple(self):
        if self.index < len(self.data):
            storm.emit(self.data[self.index])
            self.index += 1

class Mock(object):
    def __init__(self):
        self.output_type = {}
        self.pending = defaultdict(deque)
        self.processed = defaultdict(deque)
        self.emitter = None
    
    def __enter__(self):
        self.old_emit = storm.emit
        storm.emit = self.emit
        self.old_emitMany = storm.emitMany
        storm.emitMany = self.emitMany
        return self

    def __exit__(self, type, value, traceback):
        storm.emit = self.old_emit
        storm.emitMany = self.old_emitMany
    
    def activate(self, emitter):
        self.emitter = emitter
        if isinstance(emitter, storm.Spout):
            storm.MODE = storm.Spout
        elif isinstance(emitter, (storm.Bolt, storm.BasicBolt)):
            storm.MODE = storm.Bolt
        else:
            assert False, "Neither a spout nor a bolt!"
    
    def emit(self, *args, **kwargs):
        self.__emit(*args, **kwargs)
        #return readTaskIds()
    
    def __emit(self, *args, **kwargs):
        if storm.MODE == storm.Bolt:
            self.emitBolt(*args, **kwargs)
        elif storm.MODE == storm.Spout:
            self.emitSpout(*args, **kwargs)

    def emitMany(self, *args, **kwargs):
        if storm.MODE == storm.Bolt:
            self.emitManyBolt(*args, **kwargs)
        elif storm.MODE == storm.Spout:
            self.emitManySpout(*args, **kwargs)

    def emitManyBolt(self, tuples, stream=None, anchors = [], directTask=None):
        for t in tuples:
            self.emitBolt(t, stream, anchors, directTask)
    
    def emitManySpout(self, tuples, stream=None, anchors = [], directTask=None):
        for t in tuples:
            self.emitSpout(t, stream, id, directTask)

    def emitter_id(self, emitter=None):
        if emitter is None:
            emitter = self.emitter
        return type(emitter).__name__, python_id(emitter)
    
    def emitBolt(self, tup, stream=None, anchors = [], directTask=None):
        # Nice idea, but throws off profiling
        #assert len(tup) == len(self.emitter.declareOutputFields())
        # TODO: We should probably be capturing "anchors" so tests can verify
        # the topology is anchoring output tuples correctly.
        self.pending[self.emitter_id()].append(storm.Tuple(id=None, component=None, stream=stream, task=directTask, values=tup))
        
    def emitSpout(self, tup, stream=None, id=None, directTask=None):
        # Nice idea, but throws off profiling
        #assert len(tup) == len(self.emitter.declareOutputFields())
        self.pending[self.emitter_id()].append(storm.Tuple(id=id, component=None, stream=stream, task=directTask, values=tup))

    def read(self, source_emitter):
        emitter_id = self.emitter_id(source_emitter)
        result = self.pending[emitter_id].popleft()
        self.processed[emitter_id].append(result)
        return result
    
    def get_output_type(self, emitter):
        emitter_id = self.emitter_id(emitter)
        if emitter_id not in self.output_type:
            self.output_type[emitter_id] = namedtuple('%sTuple' % type(emitter).__name__, emitter.declareOutputFields())
            
        return self.output_type[emitter_id]

    @classmethod
    def run_simple_topology(cls, config, emitters, result_type=NAMEDTUPLE, max_spout_emits=None):
        """Tests a simple topology. "Simple" means there it has no branches
        or cycles. "emitters" is a list of emitters, starting with a spout
        followed by 0 or more bolts that run in a chain."""
        
        # The config is almost always required. The only known reason to pass
        # None is when calling run_simple_topology() multiple times for the
        # same components. This can be useful for testing spout ack() and fail()
        # behavior.
        if config is not None:
            for emitter in emitters:
                emitter.initialize(config, {})

        with cls() as self:
            # Read from the spout.
            spout = emitters[0]
            spout_id = self.emitter_id(spout)
            old_length = -1
            length = len(self.pending[spout_id])
            while length > old_length and (max_spout_emits is None or length < max_spout_emits):
                old_length = length 
                self.activate(spout)
                spout.nextTuple()
                length = len(self.pending[spout_id])

            # For each bolt in the sequence, consume all upstream input.
            for i, bolt in enumerate(emitters[1:]):
                previous = emitters[i]
                self.activate(bolt)
                while len(self.pending[self.emitter_id(previous)]) > 0:
                    bolt.process(self.read(previous))

        def make_storm_tuple(t, emitter):
            return t
        
        def make_python_list(t, emitter):
            return list(t.values)
        
        def make_python_tuple(t, emitter):
            return tuple(t.values)

        def make_named_tuple(t, emitter):
            return self.get_output_type(emitter)(*t.values)

        if result_type == STORM_TUPLE:
            make = make_storm_tuple
        elif result_type == LIST:
            make = make_python_list
        elif result_type == NAMEDTUPLE:
            make = make_named_tuple
        else:
            assert False, 'Invalid result type specified: %s' % result_type

        result_values = \
            [ [ make(t, emitter) for t in self.processed[self.emitter_id(emitter)]] for emitter in emitters[:-1] ] + \
            [ [ make(t, emitters[-1]) for t in self.pending[self.emitter_id(emitters[-1])] ] ]
        return dict((k, v) for k, v in zip(emitters, result_values))
        
def run_simple_topology(*l, **kw):
    return Mock.run_simple_topology(*l, **kw)

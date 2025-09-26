import time
import random
import logging

from petrel import storm
from petrel.emitter import Spout

log = logging.getLogger('randomsentence')

log.debug('randomsentence loading')

class RandomSentenceSpout(Spout):
    def __init__(self):
        super(RandomSentenceSpout, self).__init__(script=__file__)
        #self._index = 0

    @classmethod
    def declareOutputFields(cls):
        return ['sentence']

    sentences = [
        "the cow jumped over the moon",
        "an apple a day keeps the doctor away",
        "four score and seven years ago",
        "snow white and the seven dwarfs",
        "i am at two with nature"
    ]
        
    def nextTuple(self):
        #if self._index == len(self.sentences):
        #    # This is just a demo; keep sleeping and returning None after we run
        #    # out of data. We can't just sleep forever or Storm will hang.
        #    time.sleep(1)
        #    return None

        time.sleep(0.25);
        sentence = self.sentences[random.randint(0, len(self.sentences) - 1)]
        #sentence = self.sentences[self._index]
        #self._index += 1
        log.debug('randomsentence emitting: %s', sentence)
        storm.emit([sentence])

# TODO: Revisit this. Currently the spout runs forever, so it's not suitable
# for run_simple_topology(). We could modify the spout so it stops after
# emitting 'n' tuples.
#def test():
#    # To run this:
#    # pip install nose
#    # nosetests wordcount.py
#    from nose.tools import assert_true
#    from petrel import mock
#    spout = RandomSentenceSpout()
#    result = mock.run_simple_topology(None, [spout])
#    assert_true(isinstance(result[spout][0].sentence, str))

def run():
    RandomSentenceSpout().run()

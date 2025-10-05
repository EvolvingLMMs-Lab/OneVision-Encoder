import logging

from petrel import storm
from petrel.emitter import BasicBolt

log = logging.getLogger('splitsentence')

log.debug('splitsentence loading')

class SplitSentenceBolt(BasicBolt):
    def __init__(self):
        super(SplitSentenceBolt, self).__init__(script=__file__)

    def declareOutputFields(self):
        return ['word']

    def process(self, tup):
        log.debug('SplitSentenceBolt.process() called with: %s', tup)
        words = tup.values[0].split(" ")
        for word in words:
          log.debug('SplitSentenceBolt.process() emitting: %s', word)
          storm.emit([word])

def test():
    # To run this:
    # pip install nose
    # nosetests splitsentence.py
    from nose.tools import assert_equal
    bolt = SplitSentenceBolt()
    from petrel import mock
    from randomsentence import RandomSentenceSpout
    mock_spout = mock.MockSpout(RandomSentenceSpout.declareOutputFields(), [
        ["Madam, I'm Adam."],
    ])
    
    result = mock.run_simple_topology(None, [mock_spout, bolt], result_type=mock.LIST)
    assert_equal([['Madam,'], ["I'm"], ['Adam.']], result[bolt])

def run():
    SplitSentenceBolt().run()

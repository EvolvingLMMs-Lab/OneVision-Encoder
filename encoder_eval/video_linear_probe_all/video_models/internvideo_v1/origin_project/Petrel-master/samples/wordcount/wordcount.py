import logging
from collections import defaultdict

from petrel import storm
from petrel.emitter import BasicBolt

log = logging.getLogger('wordcount')

log.debug('wordcount loading')

class WordCountBolt(BasicBolt):
    def __init__(self):
        super(WordCountBolt, self).__init__(script='wordcount.py')
        self._count = defaultdict(int)

    @classmethod
    def declareOutputFields(cls):
        return ['word', 'count']

    def process(self, tup):
        log.debug('WordCountBolt.process() called with: %s', tup)
        word = tup.values[0]
        self._count[word] += 1
        log.debug('WordCountBolt.process() emitting: %s', [word, self._count[word]])
        storm.emit([word, self._count[word]])

def test():
    # To run this:
    # pip install nose
    # nosetests wordcount.py
    from nose.tools import assert_equal
    
    bolt = WordCountBolt()
    
    from petrel import mock
    from randomsentence import RandomSentenceSpout
    mock_spout = mock.MockSpout(RandomSentenceSpout.declareOutputFields(), [
        ['word'],
        ['other'],
        ['word'],
    ])
    
    result = mock.run_simple_topology(None, [mock_spout, bolt], result_type=mock.LIST)
    assert_equal(2, bolt._count['word'])
    assert_equal(1, bolt._count['other'])
    assert_equal([['word', 1], ['other', 1], ['word', 2]], result[bolt])

def run():
    WordCountBolt().run()

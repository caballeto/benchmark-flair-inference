import time
import json
import flair

from tqdm import tqdm
from flair.models import SequenceTagger
from flair.tokenization import SegtokSentenceSplitter

splitter = SegtokSentenceSplitter()
tagger = SequenceTagger.load('ner-fast')


def process_doc(content):
    sentences = splitter.split(content)
    tagger.predict(sentences, mini_batch_size=16)
    entities = []
    for sentence in sentences:
        for entity in sentence.get_spans('ner'):
            entities.append(entity.text)
    return '|'.join([x.lower() for x in entities])


if __name__ == '__main__':
    docs = json.load(open('data.json', 'r'))

    print('Starting benchmark')

    print(f"Using device {flair.device}")

    start = time.time()

    result = [process_doc(x['content']) for x in docs]

    end = time.time()

    print('Finished benchmark')

    total_docs = len(docs)
    total_len = sum((len(x['content']) for x in docs))
    total_time = end - start

    print(f"Total documents processed: {total_docs}")
    print(f"Total chars processed: {total_len}")
    print(f"Total time taken: {total_time}s")
    print(f"Time per char: {total_time/total_len} s/char")
    print(f"Time per 1000 char: {total_time/total_len * 1000}s/1k chars")

    print(result)

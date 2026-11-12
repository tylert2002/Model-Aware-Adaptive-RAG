"""Read-only verification against the delivered manifest and actual retriever."""
import json
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

def call(base, path, data=None):
    request = urllib.request.Request(base + path,
        data=json.dumps(data).encode() if data is not None else None,
        headers={'Content-Type': 'application/json'})
    with urllib.request.urlopen(request, timeout=60) as response:
        return json.load(response)

def main():
    base = 'http://127.0.0.1:9201'
    manifest = json.loads((Path(__file__).resolve().parent.parent /
        'hotpotqa_elasticsearch_7.10.2_snapshot_20261110t0630z.manifest.json').read_text())
    evidence = {}
    for key, path in {
        'server': '/', 'health': '/_cluster/health/hotpotqa',
        'count': '/hotpotqa/_count', 'mapping': '/hotpotqa/_mapping',
        'settings': '/hotpotqa/_settings', 'recovery': '/hotpotqa/_recovery',
    }.items():
        evidence[key] = call(base, path)
    assert evidence['server']['version']['number'] == '7.10.2'
    assert evidence['server']['version']['build_flavor'] == 'oss'
    assert evidence['count']['count'] == manifest['source']['document_count']
    assert evidence['health']['active_primary_shards'] == 1
    assert evidence['health']['initializing_shards'] == 0
    assert all(s['stage'] == 'DONE' for s in evidence['recovery']['hotpotqa']['shards'])
    settings = evidence['settings']['hotpotqa']['settings']['index']
    evidence['uuid_comparison'] = {
        'source': manifest['source']['index_uuid'], 'restored': settings['uuid'],
        'matches': settings['uuid'] == manifest['source']['index_uuid'],
    }
    assert settings['version']['created'] == '7100299'
    assert settings['number_of_shards'] == '1' and settings['number_of_replicas'] == '1'
    props = evidence['mapping']['hotpotqa']['mappings']['properties']
    for field in ['title', 'paragraph_text', 'url']:
        assert props[field]['type'] == 'text' and props[field]['analyzer'] == 'english'
    assert props['id']['type'] == 'text' and props['id']['fields']['keyword']['type'] == 'keyword'
    assert props['is_abstract']['type'] == 'boolean'
    assert props['paragraph_index']['type'] == 'integer'
    evidence['samples'] = []
    for sample in manifest['known_sample_documents']:
        doc = call(base, '/hotpotqa/_doc/' + sample['_id'])
        assert doc['found']
        for field in ['title', 'paragraph_index', 'is_abstract']:
            assert doc['_source'][field] == sample[field]
        evidence['samples'].append(doc)
    evidence['bm25'] = call(base, '/hotpotqa/_search', {
        'size': 3, 'query': {'multi_match': {'query': 'Luxembourg Football Federation',
        'fields': ['title^1.25', 'paragraph_text']}}})
    hits = evidence['bm25']['hits']['hits']
    assert [h['_id'] for h in hits] == [h['_id'] for h in manifest['known_sample_retrieval']['top_hits']]
    assert [h['_source']['title'] for h in hits] == [h['title'] for h in manifest['known_sample_retrieval']['top_hits']]
    evidence['retriever'] = call('http://127.0.0.1:8000', '/retrieve/', {
        'retrieval_method': 'retrieve_from_elasticsearch',
        'query_text': 'Luxembourg Football Federation', 'max_hits_count': 5,
        'document_type': 'title_paragraph_text', 'corpus_name': 'hotpotqa'})
    assert evidence['retriever']['retrieval']
    assert all(hit.get('paragraph_text') and hit.get('title') for hit in evidence['retriever']['retrieval'])
    filename = 'restore_verification_' + datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ') + '.json'
    with Path(__file__).with_name(filename).open('x') as output:
        json.dump(evidence, output, indent=2)
    print('PASS: OSS version, exact count, recovery, mapping/settings, samples, BM25 order, actual retriever')
    print('Evidence:', filename)
    print(json.dumps(evidence['retriever'], indent=2))

if __name__ == '__main__':
    main()

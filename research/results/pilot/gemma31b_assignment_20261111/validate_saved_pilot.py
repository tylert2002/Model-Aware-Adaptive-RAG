"""Offline validation of the committed pilot; never calls a model or retriever.

Use --local-evidence to additionally verify the intentionally uncommitted traces.
"""
import argparse
import hashlib
import json
import math
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT))
from research.experiments.pilot import pipeline as p

RUN = ROOT / 'research/results/pilot/20261112T054914Z_gemma_b89bcbfd'
IDS = ['5a737e4655429908901be2cc', '5a746a4455429974ef308bf6']
MODEL = 'google/gemma-4-31B-it-qat-w4a16-ct'


def read_jsonl(path):
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def validate(local_evidence=False):
    split = ROOT / 'research/data/splits/hotpotqa_v1'
    expected_hashes = {
        'calibration.jsonl': 'd387583b768319c91edb832502cfa78349a4957e6596c433b51d8c6179d59d9f',
        'inference/calibration.jsonl': '0917e208cdaea7a429e3f922a12557c93982ccf1b1a1ab83f89df3ae67e9f585',
    }
    for name, expected in expected_hashes.items():
        assert p.file_sha256(split / name) == expected, name
    safe = list(p.read_inference_safe_examples(split / 'inference/calibration.jsonl'))
    assert [x['qid'] for x in safe[:2]] == IDS
    assert all(not x['paras'] and not x['titles'] and 'answer' not in x for x in safe)
    evaluation = {x['question_id']: x for x in read_jsonl(split / 'calibration.jsonl')}
    rows = read_jsonl(RUN / 'predictions.jsonl')
    keys = [(x['question_id'], x['strategy']) for x in rows]
    assert keys == [(qid, strategy) for qid in IDS for strategy in 'AB']
    assert len(keys) == len(set(keys)) == 4
    metadata = json.loads((RUN / 'run_metadata.json').read_text())
    assert metadata['selected_question_ids'] == IDS
    assert metadata['input_schema'] == 'inference_safe'
    for code, name in metadata['strategy_configs'].items():
        assert p.file_sha256(ROOT / name) == metadata['strategy_config_sha256'][code]
    for name, expected in metadata['prompt_file_sha256'].items():
        assert p.file_sha256(ROOT / metadata['prompt_directory'] / name) == expected
    assert p.file_sha256(ROOT / metadata['retriever_address_config']) == metadata['retriever_address_config_sha256']
    with p.upstream_import_context(ROOT / 'adaptive-rag'):
        for row in rows:
            assert row['model_alias'] == 'gemma' and row['resolved_model_id'] == MODEL
            assert row['dataset'] == 'hotpotqa' and row['split'] == 'Calibration'
            assert row['completion_status'] == 'complete' and row['error_category'] is None
            assert row['llm_request_count'] == row['attempts'] == 1
            assert row['retrieval_calls'] == int(row['strategy'] == 'B')
            gold = evaluation[row['question_id']]['answers_objects']
            assert len(gold) == 1 and row['reference_answers'] == gold[0]['spans']
            f1, em = p.score_answer('hotpotqa', row['generated_answer'], row['reference_answers'])
            assert (f1, em) == (row['f1'], row['exact_match'])
            assert row['input_tokens'] + row['output_tokens'] == row['total_tokens']
            assert row['latency'] > 0 and row['end_to_end_latency'] >= row['latency']
            docs = row['retrieved_documents']
            assert len(docs) == (5 if row['strategy'] == 'B' else 0)
            if docs:
                assert all(x['included_in_fitted_prompt'] for x in docs)
                assert [x['id'] for x in docs] == row['context_verified_document_ids']
    summary = json.loads((RUN / 'summary_metrics.json').read_text())
    assert summary['valid_pilot_records'] == 4
    assert all(v == {'A': 0, 'B': 0} for v in summary['formal_completed'].values())
    previous = json.loads((RUN / 'validation_summary.json').read_text())
    for strategy in 'AB':
        selected = [x for x in rows if x['strategy'] == strategy]
        actual = summary['strategies'][strategy]
        assert actual['count'] == len(selected) == 2
        for score, field in [('em', 'exact_match'), ('f1', 'f1')]:
            mean = sum(x[field] for x in selected) / 2
            assert math.isclose(actual['mean_' + score], mean)
            assert math.isclose(previous['strategies'][strategy][score], mean)
        for field in ['retrieval_calls', 'attempts', 'input_tokens', 'output_tokens', 'total_tokens']:
            assert actual[field] == sum(x[field] for x in selected)
        assert actual['llm_requests'] == sum(x['llm_request_count'] for x in selected)
        for key, field in [('api_latency_seconds', 'latency'), ('end_to_end_latency_seconds', 'end_to_end_latency')]:
            assert math.isclose(actual[key], sum(x[field] for x in selected))
    if local_evidence:
        inventory = json.loads((RUN / 'local_evidence_manifest.json').read_text())
        for name, expected in inventory['files'].items():
            path = RUN / name
            assert path.stat().st_size == expected['bytes']
            assert p.file_sha256(path) == expected['sha256']
        raw = read_jsonl(RUN / 'scored_results.jsonl')
        for row, original in zip(rows, raw):
            for key, value in original.items():
                if key in row:
                    assert row[key] == value, key
            trace = read_jsonl(RUN / original['trace_file'])
            assert len(trace) == 1 and trace[0]['status'] == 'ok'
            assert trace[0]['model'] == MODEL and trace[0]['attempts'] == 1
            prompt = (RUN / original['trace_file']).with_suffix('.prompt.txt').read_text()
            test_example = prompt.split('\n\n\n')[-1]
            assert evaluation[row['question_id']]['question_text'] in test_example
            retrieval = read_jsonl(RUN / original['retrieval_trace_file'])
            if row['strategy'] == 'B':
                assert len(retrieval) == 1
                docs = retrieval[0]['response']['retrieval']
                assert [x['id'] for x in docs] == row['context_verified_document_ids']
                assert all(x['paragraph_text'] in test_example for x in docs)
            else:
                assert not retrieval
    print('PASS: 4 paired pilot records; frozen IDs/hashes, Gemma 31B, A/B, references, EM/F1, metrics, provenance' +
          ('; original trace hashes and fitted B contexts verified' if local_evidence else ''))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--local-evidence', action='store_true')
    validate(parser.parse_args().local_evidence)

"""Portable offline check of compact E1 exports; no raw traces or API access needed."""
import argparse
import json
import math
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[3]
sys.path.insert(0,str(ROOT))
from research.experiments.pilot import pipeline as p

OUT=ROOT/'research/results/hotpotqa_v1/gemma31b'
DATA=ROOT/'research/data/splits/hotpotqa_v1'
SPLITS={'calibration':50,'validation':25,'test':150}
MODEL='google/gemma-4-31B-it-qat-w4a16-ct'


def readl(path):
    return [json.loads(x) for x in path.read_text().splitlines() if x.strip()]


def validate(write_report=False):
    # Require completed exports before reading any evaluation references.
    assert all((OUT/(split+'_AB.jsonl')).is_file() for split in SPLITS)
    meta=json.loads((OUT/'preflight.json').read_text())
    summary=json.loads((OUT/'summary_metrics.json').read_text())
    assert summary['status']=='COMPLETE' and summary['valid_records']==450
    combined=[]
    with p.upstream_import_context(ROOT/'adaptive-rag'):
        for split,n in SPLITS.items():
            for prefix in ['', 'inference/']:
                name=prefix+split+'.jsonl'
                assert p.file_sha256(DATA/name)==meta['manifests'][name]['sha256']
            safe=readl(DATA/'inference'/(split+'.jsonl'))
            refs={x['question_id']:x for x in readl(DATA/(split+'.jsonl'))}
            ids=[x['question_id'] for x in safe]
            rows=readl(OUT/(split+'_AB.jsonl'))
            assert len(ids)==len(set(ids))==n
            assert [(x['question_id'],x['strategy']) for x in rows]==[(qid,s) for qid in ids for s in 'AB']
            for row in rows:
                assert row['split']==split and row['model_alias']=='gemma' and row['resolved_model_id']==MODEL
                assert row['completion_status']=='complete' and row['error_category'] is None
                assert row['prompt_evidence_verified']
                assert row['llm_request_count']==1 and row['retrieval_calls']==int(row['strategy']=='B')
                assert row['retry_count']==row['attempts']-1
                assert row['input_tokens']+row['output_tokens']==row['total_tokens']
                gold=refs[row['question_id']]['answers_objects']
                # Frozen HotpotQA references are single span-list answer objects.
                assert len(gold)==1 and row['reference_answers']==gold[0]['spans']
                f1,em=p.score_answer('hotpotqa',row['generated_answer'],row['reference_answers'])
                assert (f1,em)==(row['f1'],row['exact_match'])
                if row['strategy']=='B':
                    assert row['retrieved_documents'] and any(d['included_in_fitted_prompt'] for d in row['retrieved_documents'])
                else:
                    assert not row['retrieved_documents']
            for strategy in 'AB':
                selected=[x for x in rows if x['strategy']==strategy]
                stats=summary['splits'][split][strategy]
                assert stats['n']==n==len(selected)
                for label,field in [('f1','f1'),('em','exact_match'),('latency_seconds','latency'),('end_to_end_latency_seconds','end_to_end_latency'),('retrieval_calls','retrieval_calls'),('llm_api_calls','llm_request_count')]:
                    assert math.isclose(stats['mean_'+label],sum(x[field] for x in selected)/n,abs_tol=1e-12)
                for label,field in [('api_calls','llm_request_count'),('input_tokens','input_tokens'),('output_tokens','output_tokens'),('tokens','total_tokens'),('retries','retry_count'),('http_attempts','attempts'),('retrieval_calls','retrieval_calls'),('latency_seconds','latency'),('end_to_end_latency_seconds','end_to_end_latency')]:
                    assert math.isclose(stats['total_'+label],sum(x[field] for x in selected),abs_tol=1e-9)
            combined+=rows
    assert len(combined)==450 and len({x['question_id'] for x in combined})==225
    for strategy in 'AB':
        assert readl(OUT/('predictions_'+strategy+'.jsonl'))==[x for x in combined if x['strategy']==strategy]
    phase=json.loads((ROOT/'research/results/hotpotqa_500_question_manifest_seed100.json').read_text())
    assert not {x['question_id'] for x in combined}&set(phase['question_ids'])
    assert summary['official_test_n_per_strategy']==150
    result={'status':'COMPLETE','records':450,'unique_questions':225,'A':225,'B':225,'official_test_n_per_strategy':150,'checks':'Frozen split/order/model alignment, A/B pairing, reference join, original-score recomputation, exact export equality, independent aggregate recomputation, zero Phase 0 overlap. Full raw-prompt provenance is certified separately by validation_report.json.','validator_sha256':p.file_sha256(Path(__file__))}
    if write_report:
        path=OUT/'compact_validation.json'
        text=json.dumps(result,indent=2)+'\n'
        if path.exists():
            assert path.read_text()==text
        else:
            with path.open('x') as f:f.write(text)
    print(json.dumps(result,indent=2))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--write-report',action='store_true')
    validate(parser.parse_args().write_report)

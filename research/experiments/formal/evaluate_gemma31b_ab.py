"""Separate, offline evaluation and independent saved-evidence validation for E1.

Refuses to read gold evaluation records until all 450 generation checkpoints exist.
No model/retriever calls or router training are performed.
"""
import argparse
from collections import Counter
import hashlib
import json
import math
import os
from pathlib import Path
import re
import sys

ROOT=Path(__file__).resolve().parents[3]
sys.path.insert(0,str(ROOT))
from research.experiments.pilot import pipeline as p
from research.experiments.pilot.adaptive_rag_bridge import load_local_gpt2_tokenizer

OUT=ROOT/'research/results/hotpotqa_v1/gemma31b'
DATA=ROOT/'research/data/splits/hotpotqa_v1'
SPLITS={'calibration':50,'validation':25,'test':150}
MODEL='google/gemma-4-31B-it-qat-w4a16-ct'


def readl(path):
    return [json.loads(x) for x in path.read_text().splitlines() if x.strip()]


def exclusive_or_identical(path, content):
    if path.exists():
        assert path.read_text()==content, 'Refuse to overwrite different export: '+str(path)
    else:
        with path.open('x') as f:
            f.write(content)


def dump(path,value):
    exclusive_or_identical(path,json.dumps(value,indent=2,ensure_ascii=False)+'\n')


def lines(path,rows):
    exclusive_or_identical(path,''.join(json.dumps(x,ensure_ascii=False)+'\n' for x in rows))


def references(record):
    # Same best-reference/multispan representation as MultiParaRCReader, independently joined.
    formatted=[]
    for answer in record['answers_objects']:
        if answer['number']:
            value=answer['number']
        elif answer['spans']:
            value=answer['spans']
        else:
            date=answer['date']
            value=' '.join(date[k] for k in ['day','month','year'] if date[k])
        formatted.append(tuple(value))
    return list(Counter(formatted).most_common(1)[0][0])


def extracted(raw):
    # Independently reproduce LLMQAModel + original AnswerExtractor + decomposer JSON parsing.
    value=raw.strip()
    try:
        value=json.loads(value)
    except ValueError:
        pass
    # Participant stores the model answer as JSON before AnswerExtractor strips quotes.
    query=json.dumps(value)
    if query.startswith('"') and query.endswith('"'):
        query=query[1:-1]
    match=re.match(r'.* answer is:? (.*)\.?',query)
    if not match:
        match=re.match(r'(.*)',query)
    answer=match.group(1)
    if answer.endswith('.'):
        answer=answer[:-1]
    try:
        json.loads(answer)
    except ValueError:
        try:
            answer=json.dumps(json.loads(answer.encode('utf-8').decode('unicode_escape')))
        except (ValueError,UnicodeError):
            pass
    try:
        result=json.loads(answer)
        if isinstance(result,(str,list)):
            answer=result
    except ValueError:
        pass
    return answer


def selected_documents(documents):
    from rapidfuzz import fuzz
    kept=[]
    for doc in documents:
        if len(doc['paragraph_text'].split(' '))>600:
            continue
        if any((doc['title']==x['title'] and doc['paragraph_text']==x['paragraph_text']) or
               (fuzz.ratio(doc['title'],x['title'])>=90 and fuzz.ratio(doc['paragraph_text'],x['paragraph_text'])>=90) for x in kept):
            continue
        if len(kept)<15:
            kept.append(doc)
    return kept


def formatted_document(doc):
    para=' '.join(doc['paragraph_text'].split(' ')[:350]).strip()
    return para if para.startswith('Wikipedia Title: ') else 'Wikipedia Title: '+doc['title']+'\n'+para


def evaluate():
    paths=list((OUT/'checkpoints').glob('*/*.json'))
    assert len(paths)==450, 'Evaluation requires 450 checkpoints; currently '+str(len(paths))
    checkpoints=[json.loads(x.read_text()) for x in paths]
    meta=json.loads((OUT/'preflight.json').read_text())
    expected=[]
    safe={}
    for split,n in SPLITS.items():
        for kind in ['', 'inference/']:
            path=DATA/(kind+split+'.jsonl')
            assert p.file_sha256(path)==meta['manifests'][kind+split+'.jsonl']['sha256']
        examples=list(p.read_inference_safe_examples(DATA/'inference'/(split+'.jsonl')))
        assert len(examples)==n
        for ex in examples:
            assert not ex['paras'] and not ex['titles'] and 'answer' not in ex
            safe[(split,ex['qid'])]=ex
            expected.extend((split,ex['qid'],s) for s in 'AB')
    keys=[(x['split'],x['question_id'],x['strategy']) for x in checkpoints]
    assert len(set(keys))==450 and set(keys)==set(expected)
    assert len({x[1] for x in keys})==225
    records={k:v for k,v in zip(keys,checkpoints)}
    # Only now load gold records in this separate evaluation process.
    gold={split:{x['question_id']:x for x in readl(DATA/(split+'.jsonl'))} for split in SPLITS}
    phase=json.loads((ROOT/'research/results/hotpotqa_500_question_manifest_seed100.json').read_text())
    assert not {k[1] for k in keys}&set(phase['question_ids'])
    for name, expected_hash in meta['protocol']['implementation_sha256'].items():
        assert p.file_sha256(ROOT/name)==expected_hash, name
    for code, expected_hash in meta['protocol']['strategy_config_sha256'].items():
        prefix='nor' if code=='A' else 'oner'
        assert p.file_sha256(ROOT/'adaptive-rag/base_configs'/(prefix+'_qa_gpt_hotpotqa.jsonnet'))==expected_hash
    provenance_hash=p.file_sha256(OUT/'preflight.json')
    raw_hashes={}
    exports={split:[] for split in SPLITS}
    from research.experiments.pilot import run  # Forces tokenizer offline/cache-only environment.
    tokenizer=load_local_gpt2_tokenizer()
    with p.upstream_import_context(ROOT/'adaptive-rag'):
        import _jsonnet
        from commaqa.inference import prompt_reader
        from metrics.drop_answer_em_f1 import DropAnswerEmAndF1
        original_tokenizer=prompt_reader.get_tokenizer
        prompt_reader.get_tokenizer=lambda name:tokenizer
        demos={}
        address=json.loads(_jsonnet.evaluate_file('.retriever_address.jsonnet'))
        for code,prefix in [('A','nor'),('B','oner')]:
            config=json.loads(_jsonnet.evaluate_file('base_configs/'+prefix+'_qa_gpt_hotpotqa.jsonnet',ext_vars={'RETRIEVER_HOST':str(address['host']),'RETRIEVER_PORT':str(address['port'])}))
            qa=next(x for x in config['models'].values() if x['name']=='llmqa')
            assert not {key[1] for key in keys}&set(qa['prompt_reader_args']['filter_by_key_values']['qid']), 'Demonstration overlap'
            demos[code]=prompt_reader.read_prompt(file_path=qa['prompt_file'],**qa['prompt_reader_args'])
        for split,qid,strategy in expected:
            row=records[(split,qid,strategy)]
            assert row['model_alias']=='gemma' and row['resolved_model_id']==MODEL
            assert row['completion_status']=='complete' and row['error_category'] is None
            assert row['dataset']=='hotpotqa' and row['provenance_sha256']==provenance_hash
            assert row['retrieval_calls']==int(strategy=='B') and row['llm_request_count']==1
            assert row['attempts']>=1 and row['retry_count']==row['attempts']-1
            assert row['total_tokens']==row['input_tokens']+row['output_tokens']
            assert row['latency']>0 and row['end_to_end_latency']>=row['latency']
            raw=ROOT/row['evidence_directory']
            trace=readl(raw/'llm.jsonl')
            assert len(trace)==1 and trace[0]['model']==MODEL and trace[0]['status']=='ok'
            assert trace[0]['attempts']==row['attempts']
            for field in ['input_tokens','output_tokens','total_tokens']:
                assert trace[0][field]==row[field]
            assert trace[0]['latency_seconds']==row['latency']
            assert extracted(trace[0]['text'])==row['generated_answer'], ('answer extraction',qid,strategy)
            example=safe[(split,qid)]
            if (raw/'input_example.json').exists():
                assert json.loads((raw/'input_example.json').read_text())==example
            retrieval=readl(raw/'retrieval.jsonl')
            context=''
            if strategy=='A':
                assert not retrieval and not row['retrieved_documents']
            else:
                assert len(retrieval)==1 and retrieval[0]['http_status']==200
                request=retrieval[0]['request']
                assert request['max_hits_count']==5 and request['document_type']=='title_paragraph_text' and request['corpus_name']=='hotpotqa'
                removed={'who','what','when','where','why','which','how','does','is'}
                query=' '.join(word for word in example['question'].split(' ') if word.strip().lower() not in removed)
                assert request['query_text']==query
                docs=retrieval[0]['response']['retrieval']
                assert [x['id'] for x in docs]==[x['id'] for x in row['retrieved_documents']]
                kept=selected_documents(docs)
                assert kept and all(x['corpus_name']=='hotpotqa' for x in docs)
                context='\n\n'.join(formatted_document(x) for x in kept)
            original=demos[strategy]+'\n'
            if context:
                original+='\n\n'+context
            original+='\n\nQ: '+example['question'].strip()+'\nA:'
            original=original.rstrip()
            if (raw/'original_prompt.txt').exists():
                assert (raw/'original_prompt.txt').read_text()==original, ('Original prompt provenance',qid,strategy)
            fitted=prompt_reader.fit_prompt_into_given_limit(original_prompt=original,model_length_limit=3500,estimated_generation_length=300,demonstration_delimiter='\n\n\n',shuffle=False,remove_method='first',tokenizer_model_name='gpt2',last_is_test_example=True)
            assert (raw/'fitted_prompt.txt').read_text()==fitted, ('Fitted prompt provenance',qid,strategy)
            # Exact reconstruction above proves all context is retrieved evidence and fixed
            # demonstrations, rather than searching for answers which may occur naturally.
            for doc in row['retrieved_documents']:
                source=next(x for x in docs if x['id']==doc['id'])
                assert doc['included_in_fitted_prompt']==(formatted_document(source) in fitted)
            refs=references(gold[split][qid])
            prediction=row['generated_answer']
            prediction=prediction[0] if isinstance(prediction,list) and prediction else prediction
            metric=DropAnswerEmAndF1()
            metric([str(prediction or '')],[[str(x) for x in refs]])
            score=metric.get_metric()
            out=dict(row,reference_answers=refs,exact_match=float(score['em']),f1=float(score['f1']),prompt_evidence_verified=True,evaluation_implementation='DropAnswerEmAndF1; independent saved-prediction recomputation')
            exports[split].append(out)
            for path in raw.iterdir():
                if path.is_file():
                    raw_hashes[str(path.relative_to(ROOT))]={'sha256':p.file_sha256(path),'bytes':path.stat().st_size}
        prompt_reader.get_tokenizer=original_tokenizer
    # Verify pilot scores were not changed while importing reusable predictions.
    pilot=readl(ROOT/'research/results/pilot/20261112T054914Z_gemma_b89bcbfd/predictions.jsonl')
    for x in pilot:
        current=next(r for r in exports['calibration'] if r['question_id']==x['question_id'] and r['strategy']==x['strategy'])
        assert (current['f1'],current['exact_match'])==(x['f1'],x['exact_match'])
    failures=readl(OUT/'failures.jsonl') if (OUT/'failures.jsonl').exists() else []
    summary={'status':'COMPLETE','expected_records':450,'valid_records':450,'unique_questions':225,'model':MODEL,'reused_pilot_records':4,'new_records':446,'official_test_n_per_strategy':150,'splits':{}}
    for split in SPLITS:
        summary['splits'][split]={}
        for strategy in 'AB':
            selected=[x for x in exports[split] if x['strategy']==strategy]
            n=len(selected)
            assert n==SPLITS[split]
            stats={'n':n}
            for label,field in [('f1','f1'),('em','exact_match'),('latency_seconds','latency'),('end_to_end_latency_seconds','end_to_end_latency'),('retrieval_calls','retrieval_calls'),('llm_api_calls','llm_request_count')]:
                stats['mean_'+label]=sum(x[field] for x in selected)/n
            for label,field in [('api_calls','llm_request_count'),('input_tokens','input_tokens'),('output_tokens','output_tokens'),('tokens','total_tokens'),('retries','retry_count'),('http_attempts','attempts'),('retrieval_calls','retrieval_calls'),('latency_seconds','latency'),('end_to_end_latency_seconds','end_to_end_latency')]:
                stats['total_'+label]=sum(x[field] for x in selected)
            stats['failed_executions']=sum(x['split']==split and x['strategy']==strategy for x in failures)
            stats['finish_reasons']=dict(Counter(readl(ROOT/x['evidence_directory']/'llm.jsonl')[0]['finish_reason'] for x in selected))
            summary['splits'][split][strategy]=stats
    http_ledgers=list((OUT/'raw').glob('*/*/*/http_attempts.jsonl'))
    http=[x for path in http_ledgers for x in readl(path)]
    summary['execution_statistics']={'failed_executions':len(failures),'formal_http_attempts_observed':len(http),'failed_http_attempts_observed':sum('error_type' in x for x in http),
        'total_included_api_calls':sum(x['llm_request_count'] for values in exports.values() for x in values),'total_included_http_attempts':sum(x['attempts'] for values in exports.values() for x in values),
        'retries_in_successful_records':sum(x['retry_count'] for values in exports.values() for x in values),
        'input_tokens':sum(x['input_tokens'] for values in exports.values() for x in values),'output_tokens':sum(x['output_tokens'] for values in exports.values() for x in values),'total_tokens':sum(x['total_tokens'] for values in exports.values() for x in values),
        'total_recorded_api_latency_seconds':sum(x['latency'] for values in exports.values() for x in values),'total_recorded_end_to_end_latency_seconds':sum(x['end_to_end_latency'] for values in exports.values() for x in values),
        'preflight_connectivity_excluded_from_scored_records':meta['connectivity']}
    for split,rows in exports.items():
        lines(OUT/(split+'_AB.jsonl'),rows)
    all_rows=[row for values in exports.values() for row in values]
    for strategy in 'AB':
        lines(OUT/('predictions_'+strategy+'.jsonl'),[x for x in all_rows if x['strategy']==strategy])
    dump(OUT/'summary_metrics.json',summary)
    dump(OUT/'raw_evidence_hashes.json',raw_hashes)
    validation={'evaluator_sha256':p.file_sha256(Path(__file__)), 'status':'COMPLETE','valid_records':450,'checks':['frozen hashes, unique IDs, order and disjoint 50/25/150','zero Phase 0 overlap','A/B key alignment; no duplicates/unexpected/missing records','model alias and resolved identity','A zero retrieval; B one depth-five call with retrieved context','exact original/fitted prompt reconstruction from fixed demonstrations + safe question + retrieved evidence only','original answer extraction independently recomputed from saved API response','all EM/F1 recomputed through original metric independently of runner','pilot scores identical','tokens, attempts, latency, split-specific n and aggregates checked'],'missing':[],'duplicates':[],'failures':failures,'files':{p.name:{'sha256':hashlib.sha256(p.read_bytes()).hexdigest(),'bytes':p.stat().st_size} for p in OUT.glob('*.jsonl') if p.name not in ['events.jsonl','failures.jsonl']}}
    dump(OUT/'validation_report.json',validation)
    print('COMPLETE: 450 records independently validated and exported; official Test n=150 per strategy.')
    print(json.dumps(summary,indent=2))


if __name__=='__main__':
    evaluate()

"""Frozen Gemma 31B A/B execution with immutable checkpoints; no evaluation imports.

Preflight is a separate process. `run` consumes only inference-safe manifests.
"""
import argparse
import copy
import fcntl
import hashlib
import importlib.metadata
import json
import os
import platform
import shutil
import subprocess
import sys
import time
import urllib.request
import uuid
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
from research.experiments.pilot import pipeline as p
from research.experiments.pilot.adaptive_rag_bridge import AdaptiveRAGNRPGenerator, fit_prompt_with_upstream
from research.experiments.pilot.run import load_env_file
from research.experiments.pilot.nrp_adapter import NRPConfig, NRPChatAdapter, NRPRequestError, _urlopen_transport

SPLITS = {'calibration': 50, 'validation': 25, 'test': 150}
DATA = ROOT / 'research/data/splits/hotpotqa_v1'
OUT = ROOT / 'research/results/hotpotqa_v1/gemma31b'
PILOT = ROOT / 'research/results/pilot/20261112T054914Z_gemma_b89bcbfd'
MODEL = 'google/gemma-4-31B-it-qat-w4a16-ct'
DECODING = dict(max_tokens=300, temperature=0, top_p=1, stop=['\n'], enable_thinking=False)


def now():
    return datetime.now(timezone.utc).isoformat()


def readl(path):
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('x') as f:
        json.dump(value, f, indent=2, ensure_ascii=False)
        f.write('\n')
        f.flush()
        os.fsync(f.fileno())


def append(path, value):
    with path.open('a') as f:
        f.write(json.dumps(value, ensure_ascii=False) + '\n')
        f.flush()
        os.fsync(f.fileno())


def relative(path):
    return str(path.relative_to(ROOT))


def protocol():
    paths = p.default_paths(ROOT, 'hotpotqa')
    pilot = json.loads((PILOT / 'run_metadata.json').read_text())
    for name, expected in pilot['strategy_config_sha256'].items():
        assert p.file_sha256(ROOT / pilot['strategy_configs'][name]) == expected
    assert p.directory_sha256s(paths.upstream_root / 'prompts/hotpotqa') == pilot['prompt_file_sha256']
    assert p.file_sha256(paths.retriever_address_path) == pilot['retriever_address_config_sha256']
    # Prove the adapter, fitting, scoring, and upstream graphs still match pilot code.
    code = {}
    names = ['research/experiments/pilot/' + f for f in ['pipeline.py', 'nrp_adapter.py', 'adaptive_rag_bridge.py', 'inference_safe_reader.py']]
    names += subprocess.check_output(['git', 'ls-files', 'adaptive-rag/commaqa', 'adaptive-rag/metrics', 'adaptive-rag/retriever_server'], cwd=ROOT, text=True).splitlines()
    for name in names:
        path = ROOT / name
        old = subprocess.check_output(['git', 'show', pilot['git_commit'] + ':' + name], cwd=ROOT)
        assert path.read_bytes() == old, 'Pilot implementation drift: ' + name
        code[name] = p.file_sha256(path)
    return {'model_alias': 'gemma', 'resolved_model_id': MODEL, 'decoding': DECODING,
            'strategy_config_sha256': pilot['strategy_config_sha256'],
            'prompt_sha256': pilot['prompt_file_sha256'], 'retriever_address_sha256': pilot['retriever_address_config_sha256'],
            'implementation_sha256': code, 'prompt_fitting': {'model_length_limit':3500,'reserved_tokens':300,'tokenizer':'gpt2','remove_method':'first','demonstration_delimiter':'\n\n\n'},
            'retrieval': {'type':'bm25','depth':5,'corpus':'hotpotqa','query_source':'original_question','document_type':'title_paragraph_text','max_paragraphs':15,'max_paragraph_words':350}}


def preflight():
    assert not (OUT / 'preflight.json').exists(), 'Existing preflight retained; use run to resume'
    metadata = json.loads((DATA / 'split_metadata.json').read_text())
    phase = ROOT / metadata['phase0_exclusion']['manifest_path']
    assert p.file_sha256(phase) == metadata['phase0_exclusion']['manifest_sha256']
    phase_ids = set(json.loads(phase.read_text())['question_ids'])
    all_ids = []
    files = {}
    for split, n in SPLITS.items():
        safe = list(p.read_inference_safe_examples(DATA / 'inference' / (split + '.jsonl')))
        # Dataset inspection is isolated from the generation process.
        gold = readl(DATA / (split + '.jsonl'))
        assert len(safe) == len(gold) == n
        assert [(x['qid'], x['question']) for x in safe] == [(x['question_id'], x['question_text']) for x in gold]
        ids = [x['qid'] for x in safe]
        assert not set(ids) & phase_ids
        all_ids += ids
        for name in [split + '.jsonl', 'inference/' + split + '.jsonl']:
            path = DATA / name
            assert p.file_sha256(path) == metadata['files'][name]['sha256']
            shared = subprocess.check_output(['git', 'show', 'team/main:' + relative(path)], cwd=ROOT)
            assert path.read_bytes() == shared, name
            files[name] = {'sha256':p.file_sha256(path),'records':n,'matches_team_main':True}
    assert len(all_ids) == len(set(all_ids)) == 225
    assert [x['question_id'] for x in readl(PILOT / 'predictions.jsonl')][::2] == all_ids[:2]
    proto = protocol()
    server = json.load(urllib.request.urlopen('http://127.0.0.1:9201/', timeout=20))
    count = json.load(urllib.request.urlopen('http://127.0.0.1:9201/hotpotqa/_count', timeout=20))
    health = json.load(urllib.request.urlopen('http://127.0.0.1:9201/_cluster/health/hotpotqa', timeout=20))
    assert server['version']['number'] == '7.10.2' and server['version']['build_flavor'] == 'oss'
    assert count['count'] == 5233329 and health['active_primary_shards'] == 1 and health['initializing_shards'] == 0
    load_env_file(ROOT / 'env.example')
    config = NRPConfig.from_env()
    assert config.timeout_seconds == 300 and config.max_retries == 2 and config.max_concurrency == 1
    response = NRPChatAdapter(config).generate('Reply with OK.', model='gemma', max_tokens=16, temperature=0, top_p=1, enable_thinking=False)
    assert response.model == MODEL and response.status == 'ok'
    versions = {}
    for name in ['jsonnet','diskcache','nltk','transformers','requests','torch','spacy','openai','elasticsearch','fastapi','uvicorn','ftfy','rapidfuzz']:
        versions[name] = importlib.metadata.version(name)
    result = {'timestamp_utc':now(),'branch':subprocess.check_output(['git','branch','--show-current'],cwd=ROOT,text=True).strip(),
              'git_commit':p.git_commit(ROOT),'shared_main':subprocess.check_output(['git','rev-parse','team/main'],cwd=ROOT,text=True).strip(),
              'manifests':files,'unique_ids':225,'phase0_overlap':0,'protocol':proto,'python':sys.version,'platform':platform.platform(),'dependencies':versions,
              'elasticsearch':{'endpoint':'http://127.0.0.1:9201','server':server,'count':count,'health':health},'connectivity':response.to_dict(),
              'adapter':{'api_base':config.api_base,'timeout_seconds':config.timeout_seconds,'max_retries':config.max_retries,'max_concurrency':config.max_concurrency,'backoff_initial_seconds':config.backoff_initial_seconds,'backoff_max_seconds':config.backoff_max_seconds,'cache_salt_sha256':hashlib.sha256(config.cache_salt.encode()).hexdigest() if config.cache_salt else None}}
    write(OUT / 'preflight.json', result)
    print('PASS: frozen 50/25/150, 225 unique, zero Phase 0 overlap; shared GitHub bytes; pilot protocol unchanged; live OSS index and Gemma 31B', flush=True)


def evidence(attempt, question, strategy):
    prompt = (attempt / 'fitted_prompt.txt').read_text()
    # Use the known suffix, not delimiter splitting (paragraphs can contain newlines).
    suffix = 'Q: ' + question.strip() + '\nA:'
    assert prompt.endswith(suffix), 'Test example suffix drift'
    calls = readl(attempt / 'retrieval.jsonl')
    documents = []
    if strategy == 'A':
        assert not calls
        assert prompt.split('\n\n\n')[-1] == suffix, 'A test context must be empty'
    else:
        assert len(calls) == 1 and calls[0]['http_status'] == 200
        request = calls[0]['request']
        assert request['max_hits_count'] == 5 and request['corpus_name'] == 'hotpotqa'
        assert request['document_type'] == 'title_paragraph_text'
        for doc in calls[0]['response']['retrieval']:
            para = ' '.join(doc['paragraph_text'].split(' ')[:350]).strip()
            formatted = para if para.startswith('Wikipedia Title: ') else 'Wikipedia Title: ' + doc['title'] + '\n' + para
            documents.append({'id':doc['id'],'title':doc['title'],'included_in_fitted_prompt':formatted in prompt})
        assert any(d['included_in_fitted_prompt'] for d in documents), 'B evidence absent'
    return documents


def import_pilot(meta):
    for original in readl(PILOT / 'predictions.jsonl'):
        qid, strategy = original['question_id'], original['strategy']
        destination = OUT / 'checkpoints' / 'calibration' / (qid + '_' + strategy + '.json')
        if destination.exists():
            continue
        stem = hashlib.sha256(qid.encode()).hexdigest()[:16] + '_' + strategy
        attempt = OUT / 'raw' / 'pilot_reuse' / stem
        attempt.mkdir(parents=True, exist_ok=False)
        for source, name in [(PILOT/'traces'/(stem+'.prompt.txt'),'fitted_prompt.txt'),(PILOT/'traces'/(stem+'.jsonl'),'llm.jsonl'),(PILOT/'traces'/(stem+'_retrieval.jsonl'),'retrieval.jsonl')]:
            shutil.copyfile(source, attempt/name)
        row = {k:v for k,v in original.items() if k not in ['reference_answers','f1','exact_match','record_scope','context_verified_document_ids','retrieved_documents']}
        assert row['resolved_model_id'] == MODEL and row['completion_status'] == 'complete'
        row.update(split='calibration',source='reused_compatible_pilot',pilot_source=relative(PILOT/'predictions.jsonl'),
                   recorded_at_utc=now(),retry_count=original['attempts']-1,evidence_directory=relative(attempt),provenance_sha256=p.file_sha256(OUT/'preflight.json'))
        question = next(x['question'] for x in p.read_inference_safe_examples(DATA/'inference/calibration.jsonl') if x['qid']==qid)
        row['retrieved_documents'] = evidence(attempt, question, strategy)
        write(destination,row)
        append(OUT/'events.jsonl',{'event':'pilot_reused','question_id':qid,'strategy':strategy,'time':now()})


def run():
    OUT.mkdir(parents=True,exist_ok=True)
    with (OUT/'run.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        meta = json.loads((OUT/'preflight.json').read_text())
        assert protocol() == meta['protocol'], 'Protocol drift'
        for name, expected in meta['manifests'].items():
            assert p.file_sha256(DATA/name) == expected['sha256']
        import_pilot(meta)
        import diskcache
        import nltk
        nltk.data.path.insert(0,str(ROOT/'.pilot-cache/nltk_data'))
        original_cache = diskcache.Cache
        diskcache.Cache = lambda directory,*a,**kw: original_cache(str(ROOT/'.pilot-cache'/Path(directory).name),*a,**kw)
        p.STRATEGIES = p.STRATEGIES[:2]
        load_env_file(ROOT/'env.example')
        config = NRPConfig.from_env()
        for key in ['api_base','timeout_seconds','max_retries','max_concurrency','backoff_initial_seconds','backoff_max_seconds']:
            assert getattr(config,key) == meta['adapter'][key], key
        salt_hash = hashlib.sha256(config.cache_salt.encode()).hexdigest() if config.cache_salt else None
        assert salt_hash == meta['adapter']['cache_salt_sha256']
        paths = p.default_paths(ROOT,'hotpotqa')
        consecutive_failures = 0
        with p.upstream_import_context(paths.upstream_root):
            backends = p.build_upstream_backends(paths,'hotpotqa')
            for split,n in SPLITS.items():
                examples = list(p.read_inference_safe_examples(DATA/'inference'/(split+'.jsonl')))
                assert len(examples)==n
                for example in examples:
                    assert not example['paras'] and not example['titles'] and 'answer' not in example
                    for strategy in 'AB':
                        qid = example['qid']
                        destination = OUT/'checkpoints'/split/(qid+'_'+strategy+'.json')
                        if destination.exists():
                            existing = json.loads(destination.read_text())
                            assert existing['completion_status']=='complete' and existing['resolved_model_id']==MODEL
                            continue
                        attempt = OUT/'raw'/split/(qid+'_'+strategy)/uuid.uuid4().hex[:12]
                        attempt.mkdir(parents=True,exist_ok=False)
                        write(attempt/'input_example.json',example)
                        # Keep fitting unchanged and record the exact prompt prior to sending.
                        def fitter(**kwargs):
                            prompt = fit_prompt_with_upstream(**kwargs)
                            (attempt/'original_prompt.txt').write_text(kwargs['original_prompt'])
                            (attempt/'fitted_prompt.txt').write_text(prompt)
                            return prompt
                        def transport(request,timeout):
                            start = time.perf_counter()
                            entry = {'started_at_utc':now()}
                            try:
                                result = _urlopen_transport(request,timeout)
                                entry['http_status'] = result[0]
                                return result
                            except Exception as error:
                                entry.update(error_type=type(error).__name__,http_status=getattr(error,'code',None))
                                raise
                            finally:
                                entry['latency_seconds'] = time.perf_counter()-start
                                append(attempt/'http_attempts.jsonl',entry)
                        generator = AdaptiveRAGNRPGenerator(NRPChatAdapter(config,transport=transport),model='gemma',metadata_path=attempt/'llm.jsonl',prompt_fitter=fitter,**DECODING)
                        start = time.perf_counter()
                        try:
                            with p.capture_retrieval_trace(attempt/'retrieval.jsonl'):
                                answer,chain,retrieval_calls = backends[strategy].execute(copy.deepcopy(example),generator)
                            elapsed = time.perf_counter()-start
                            traces = readl(attempt/'llm.jsonl')
                            assert len(traces)==1 and traces[0]['status']=='ok', 'Unexpected LLM call count'
                            trace = traces[0]
                            assert trace['model']==MODEL, 'MODEL ID MISMATCH'
                            assert retrieval_calls==int(strategy=='B'), 'Unexpected retrieval count'
                            docs = evidence(attempt,example['question'],strategy)
                            (attempt/'reasoning_chain.txt').write_text(chain)
                            row = dict(question_id=qid,split=split,dataset='hotpotqa',model_alias='gemma',resolved_model_id=trace['model'],strategy=strategy,
                                       generated_answer=answer,completion_status='complete',error_category=None,retrieval_calls=retrieval_calls,retrieved_documents=docs,
                                       llm_request_count=len(traces),input_tokens=trace['input_tokens'],output_tokens=trace['output_tokens'],total_tokens=trace['total_tokens'],
                                       latency=trace['latency_seconds'],end_to_end_latency=elapsed,attempts=trace['attempts'],retry_count=trace['attempts']-1,
                                       finish_reason=trace['finish_reason'],source='formal_inference',recorded_at_utc=now(),evidence_directory=relative(attempt),provenance_sha256=p.file_sha256(OUT/'preflight.json'))
                            write(destination,row)
                            append(OUT/'events.jsonl',{'event':'complete','question_id':qid,'strategy':strategy,'split':split,'time':now()})
                            consecutive_failures=0
                        except Exception as error:
                            category=error.category if isinstance(error,NRPRequestError) else p.classify_error(error)
                            failure={'question_id':qid,'strategy':strategy,'split':split,'time':now(),'category':category,'exception_type':type(error).__name__,'http_status':getattr(error,'status_code',None),'evidence_directory':relative(attempt)}
                            write(attempt/'failure.json',failure)
                            append(OUT/'failures.jsonl',failure)
                            consecutive_failures+=1
                            print('FAIL',split,qid,strategy,category,flush=True)
                            # Fail closed on protocol/integration problems; bound API outage retries.
                            if not isinstance(error,NRPRequestError) or not error.retryable or consecutive_failures>=3:
                                raise
                            # Failed graph call may leave participant counters mid-question.
                            backends = p.build_upstream_backends(paths,'hotpotqa')
                        completed=len(list((OUT/'checkpoints').glob('*/*.json')))
                        if completed % 10==0:
                            print(now(),split,'completed',completed,'/450',flush=True)
                print('SPLIT FINISHED',split,flush=True)
                # Do not proceed to a later split if this split has unresolved failures.
                assert len(list((OUT/'checkpoints'/split).glob('*.json')))==2*n, 'Split incomplete; resume after investigating failures'
        print('GENERATION COMPLETE: 450 immutable checkpoints; evaluate separately.',flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command',choices=['preflight','run'])
    args=parser.parse_args()
    (preflight if args.command=='preflight' else run)()

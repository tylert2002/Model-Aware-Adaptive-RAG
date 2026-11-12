"""Execute only the authorized frozen two-question A/B pilot."""
import json
import sys
from pathlib import Path
import diskcache
import nltk

ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT))
from research.experiments.pilot import pipeline as p
from research.experiments.pilot import adaptive_rag_bridge as bridge
from research.experiments.pilot.run import load_env_file
from research.experiments.pilot.nrp_adapter import NRPConfig, NRPChatAdapter

IDS = ['5a737e4655429908901be2cc', '5a746a4455429974ef308bf6']
SPLIT = ROOT / 'research/data/splits/hotpotqa_v1'

def main():
    nltk.data.path.insert(0, str(ROOT / '.pilot-cache/nltk_data'))
    for name, expected in [
        ('calibration.jsonl', 'd387583b768319c91edb832502cfa78349a4957e6596c433b51d8c6179d59d9f'),
        ('inference/calibration.jsonl', '0917e208cdaea7a429e3f922a12557c93982ccf1b1a1ab83f89df3ae67e9f585')]:
        assert p.file_sha256(SPLIT / name) == expected
    safe = list(p.read_inference_safe_examples(SPLIT / 'inference/calibration.jsonl'))
    assert [x['qid'] for x in safe[:2]] == IDS
    # Relocate only the upstream disk caches; preserve their behavior and HOME.
    original_cache = diskcache.Cache
    def local_cache(directory, *args, **kwargs):
        return original_cache(str(ROOT / '.pilot-cache' / Path(directory).name), *args, **kwargs)
    diskcache.Cache = local_cache
    p.STRATEGIES = p.STRATEGIES[:2]
    def frozen_selection(available, limit, seed):
        assert limit == 2 and set(IDS) <= set(available)
        return IDS.copy()
    p.select_question_ids = frozen_selection
    # Save the exact fitted model prompts without changing their contents.
    original_generator = p.AdaptiveRAGNRPGenerator
    class AuditedGenerator(original_generator):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            fitter = self.prompt_fitter
            def audited_fitter(**options):
                prompt = fitter(**options)
                self.metadata_path.with_suffix('.prompt.txt').write_text(prompt)
                return prompt
            self.prompt_fitter = audited_fitter
    p.AdaptiveRAGNRPGenerator = AuditedGenerator
    load_env_file(ROOT / 'env.example')
    paths = p.default_paths(ROOT, 'hotpotqa', SPLIT / 'inference/calibration.jsonl')
    run_dir = p.run_pilot(paths=paths, dataset='hotpotqa', model='gemma', limit=2, seed=100,
                          adapter=NRPChatAdapter(NRPConfig.from_env()))
    # Gold is loaded only after all model executions, solely for original scoring.
    with p.upstream_import_context(paths.upstream_root):
        from commaqa.inference.dataset_readers import MultiParaRCReader
        references = {x['qid']: x['answer'] for x in MultiParaRCReader().read_examples(str(SPLIT / 'calibration.jsonl')) if x['qid'] in IDS}
        rows = [json.loads(line) for line in (run_dir / 'results.jsonl').read_text().splitlines()]
        for row in rows:
            row['reference_answers'] = references[row['question_id']]
            row['f1'], row['exact_match'] = p.score_answer('hotpotqa', row['generated_answer'], row['reference_answers'])
            assert row['completion_status'] == 'complete', row
            assert row['resolved_model_id'] == p.RESOLVED_MODEL_IDS['gemma'], row
            if row['strategy'] == 'B':
                retrieval = [json.loads(x) for x in (run_dir / row['retrieval_trace_file']).read_text().splitlines()]
                prompt = (run_dir / row['trace_file']).with_suffix('.prompt.txt').read_text()
                docs = retrieval[0]['response']['retrieval']
                row['context_verified_document_ids'] = [doc['id'] for doc in docs if doc['paragraph_text'] in prompt]
                assert row['context_verified_document_ids'], 'No retrieved passage in fitted prompt'
        assert len(rows) == 4
        with (run_dir / 'scored_results.jsonl').open('x') as out:
            for row in rows:
                out.write(json.dumps(row) + '\n')
    print('Completed:', run_dir, flush=True)

if __name__ == '__main__':
    main()

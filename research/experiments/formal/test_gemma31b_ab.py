"""Offline failure-closed checks for evidence validation and immutable storage."""
import json
import tempfile
import unittest
from pathlib import Path
from .gemma31b_ab import evidence, write


class EvidenceTests(unittest.TestCase):
    def test_checkpoints_cannot_overwrite_success(self):
        with tempfile.TemporaryDirectory() as d:
            path=Path(d)/'record.json'
            write(path,{'answer':'original'})
            with self.assertRaises(FileExistsError):
                write(path,{'answer':'replacement'})
            self.assertEqual(json.loads(path.read_text())['answer'],'original')

    def test_a_rejects_context_and_retrieval(self):
        with tempfile.TemporaryDirectory() as d:
            path=Path(d)
            (path/'retrieval.jsonl').write_text('')
            (path/'fitted_prompt.txt').write_text('demo\n\n\nQ: Question?\nA:')
            self.assertEqual(evidence(path,'Question?','A'),[])
            (path/'fitted_prompt.txt').write_text('demo\n\n\nGold text\n\nQ: Question?\nA:')
            with self.assertRaises(AssertionError):
                evidence(path,'Question?','A')
            (path/'retrieval.jsonl').write_text('{}\n')
            with self.assertRaises(AssertionError):
                evidence(path,'Question?','A')

    def test_b_requires_actual_prompt_evidence_and_depth(self):
        with tempfile.TemporaryDirectory() as d:
            path=Path(d)
            call={'http_status':200,'request':{'max_hits_count':5,'corpus_name':'hotpotqa','document_type':'title_paragraph_text'},'response':{'retrieval':[{'id':'doc','title':'Title','paragraph_text':'actual retrieved evidence'}]}}
            (path/'retrieval.jsonl').write_text(json.dumps(call)+'\n')
            (path/'fitted_prompt.txt').write_text('demo\n\n\nWikipedia Title: Title\nactual retrieved evidence\n\nQ: Question?\nA:')
            self.assertTrue(evidence(path,'Question?','B')[0]['included_in_fitted_prompt'])
            (path/'fitted_prompt.txt').write_text('demo\n\n\nQ: Question?\nA:')
            with self.assertRaises(AssertionError):
                evidence(path,'Question?','B')
            call['request']['max_hits_count']=1
            (path/'retrieval.jsonl').write_text(json.dumps(call)+'\n')
            with self.assertRaises(AssertionError):
                evidence(path,'Question?','B')

if __name__=='__main__':
    unittest.main()


class EvaluationBoundaryTests(unittest.TestCase):
    def test_incomplete_generation_cannot_open_gold_or_export(self):
        from unittest.mock import patch
        from . import evaluate_gemma31b_ab as evaluator
        with tempfile.TemporaryDirectory() as d:
            with patch.object(evaluator,'OUT',Path(d)), patch.object(evaluator,'readl',side_effect=RuntimeError('must not read any records')):
                with self.assertRaisesRegex(AssertionError,'450 checkpoints'):
                    evaluator.evaluate()
            self.assertFalse(list(Path(d).iterdir()))

    def test_existing_export_must_match_exactly(self):
        from .evaluate_gemma31b_ab import exclusive_or_identical
        with tempfile.TemporaryDirectory() as d:
            path=Path(d)/'results.jsonl'
            exclusive_or_identical(path,'original\n')
            exclusive_or_identical(path,'original\n')
            with self.assertRaises(AssertionError):
                exclusive_or_identical(path,'replacement\n')
            self.assertEqual(path.read_text(),'original\n')

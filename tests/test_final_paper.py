import csv
import importlib.util
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

ROOT=Path(__file__).resolve().parents[1]
spec=importlib.util.spec_from_file_location('paper_audit',ROOT/'scripts/paper/audit_final_paper.py')
audit=importlib.util.module_from_spec(spec)
spec.loader.exec_module(audit)


class PaperAuditTests(unittest.TestCase):
    def test_final_em_budget_changes_with_tolerance(self):
        coefficient=4.582886471559473
        self.assertEqual(audit.budget(coefficient,'em',.01),265)
        self.assertEqual(audit.budget(coefficient,'em',.015),177)
        self.assertEqual(audit.budget(5.46059264204673,'em',.015),211)
        with self.assertRaises(ValueError):audit.budget(coefficient,'em',0)

    def test_heun_counts_steps_before_nfe(self):
        self.assertEqual(audit.budget(90.39755038452148,'heun',.01),28)
        self.assertEqual(2*audit.budget(90.39755038452148,'heun',.01),56)

    def test_malformed_fid_row_is_not_silently_shifted(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);path=root/'fid.csv'
            path.write_text('k,nfe,fid,num_samples,seed\n177,177,15.9,10000,0\n179,15.4,10000,0\n')
            with patch.object(audit,'ROOT',root):
                rows,issues=audit.read_fids(path,'k',10000)
            self.assertEqual(set(rows),{177})
            self.assertEqual(len(issues),1)
            self.assertIn('column count',issues[0])

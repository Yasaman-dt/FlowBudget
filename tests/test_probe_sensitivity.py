import unittest
import tempfile
from pathlib import Path
from probe_sensitivity import summarize_pool, aggregate, write_reports

class ProbeTests(unittest.TestCase):
    def pool(self, seed):
        return {'probe_pool':dict(ratios=[1.0+seed]*25+[4.0]*475,
                cumulative_seconds=[float(i+1) for i in range(500)],
                seed=seed,epsilon=1e-8,batch_size=1)}
    def test_prefixes_and_nfe(self):
        rows=summarize_pool(self.pool(0),'test','heun',.01,1e-8,0)
        self.assertEqual(rows[0]['C_hat'],1)
        self.assertEqual(rows[1]['C_hat'],2.5)
        self.assertEqual(rows[1]['runtime_seconds'],50)
        self.assertEqual(rows[-1]['relative_deviation_500'],0)
        for row in rows:self.assertEqual(row['predicted_NFE'],2*row['K_hat'])
    def test_summary_and_plot(self):
        rows=[]
        for seed in (0,1,2):rows+=summarize_pool(self.pool(seed),'test','euler',.01,1e-8,seed)
        result=aggregate(rows)
        self.assertEqual(result[0]['mean_K_hat'],100)
        self.assertEqual(result[0]['std_K_hat'],50)
        self.assertEqual(result[0]['CV_K_hat'],.5)
        self.assertAlmostEqual(result[0]['CI95_high'],224.206886,places=4)
        self.assertEqual(result[-1]['mean_absolute_relative_deviation_500'],0)
        with tempfile.TemporaryDirectory() as temp:
            write_reports(rows,Path(temp))
            self.assertTrue((Path(temp)/'probe_sensitivity.svg').exists())
    def test_reject_mismatched_pool(self):
        with self.assertRaises(ValueError):summarize_pool(self.pool(0),'test','euler',.01,1e-7,0)
        rows=summarize_pool(self.pool(0),'test','euler',.01,1e-8,0)
        with self.assertRaises(ValueError):aggregate(rows*3)

    def test_extended_prefixes_keep_500_baseline(self):
        pool=self.pool(0)
        pool['probe_pool']['ratios'] += [8.0]*500
        pool['probe_pool']['cumulative_seconds'] += [float(i+1) for i in range(500,1000)]
        rows=summarize_pool(pool,'test','euler',.01,1e-8,0,[400,500,700,1000])
        self.assertEqual([r['N'] for r in rows],[400,500,700,1000])
        self.assertEqual(rows[1]['relative_deviation_500'],0)
        self.assertAlmostEqual(rows[2]['C_hat'],(25+475*4+200*8)/700)
        self.assertGreater(rows[-1]['relative_deviation_500'],0)
        self.assertEqual(rows[-1]['runtime_seconds'],1000)
        with self.assertRaises(ValueError):
            summarize_pool(self.pool(0),'test','euler',.01,1e-8,0,[700])

if __name__=='__main__':unittest.main()

import json, unittest
from pathlib import Path
import numpy as np
from run import true_step, dataset, Model, rollout, mean_ci

ROOT=Path(__file__).parent
class StudyTests(unittest.TestCase):
    def test_free_motion(self):
        np.testing.assert_allclose(true_step([[.5,.1]],[.2]),[[.626,.126]],atol=1e-14)
    def test_right_contact(self):
        np.testing.assert_allclose(true_step([[.9,.2]],[1]),[[1,-.216]],atol=1e-14)
    def test_left_contact(self):
        np.testing.assert_allclose(true_step([[.1,-.2]],[-1]),[[0,.216]],atol=1e-14)
    def test_boundary_equality_has_no_reflection(self):
        np.testing.assert_allclose(true_step([[1.,0.]],[0]),[[1,0]],atol=1e-14)
    def test_reflection_symmetry(self):
        X,Y=dataset(7654321,1000)
        reverse=np.column_stack([1-X[:,0],-X[:,1]])
        np.testing.assert_allclose(true_step(reverse,-X[:,2]),np.column_stack([1-Y[:,0],-Y[:,1]]),atol=1e-14)
    def test_dataset_replay_and_disjoint_seed_ranges(self):
        A,B=dataset(123456,100); C,D=dataset(123456,100)
        np.testing.assert_array_equal(A,C); np.testing.assert_array_equal(B,D)
        sets=[set(range(x,x+10)) for x in [1000,2000,3000,4000]]
        for i in range(4):
            for j in range(i): self.assertFalse(sets[i]&sets[j])
    def test_hybrid_identifies_correct_family(self):
        X,Y=dataset(876543,6000); T,U=dataset(987654,1000)
        model=Model('hybrid').fit(X,Y)
        np.testing.assert_allclose(model.predict(T),U,atol=1e-12)
    def test_ci_requires_replicates_and_finite(self):
        with self.assertRaises(ValueError): mean_ci([1])
        with self.assertRaises(ValueError): mean_ci([1,np.nan])
    def test_results_finite_and_key_uniqueness(self):
        result=json.loads((ROOT/'results.json').read_text())
        keys={(r['seed'],r['model'],r['K']) for r in result['rows']}
        self.assertEqual(len(keys),180)
        for r in result['rows']:
            for v in r.values():
                if isinstance(v,(int,float)): self.assertTrue(np.isfinite(v))
    def test_oracle_hybrid_k1_and_nonnegative_regret(self):
        rows=json.loads((ROOT/'results.json').read_text())['rows']
        lookup={(r['seed'],r['model'],r['K']):r for r in rows}
        names={r['model'] for r in rows}
        for seed in range(1000,1010):
            for name in names:
                self.assertEqual(lookup[seed,name,1]['actual_cost'],lookup[seed,'oracle',1]['actual_cost'])
                for k in [1,16,256]: self.assertGreaterEqual(lookup[seed,name,k]['regret'],-1e-14)
            for k in [1,16,256]: self.assertEqual(lookup[seed,'hybrid',k]['actual_cost'],lookup[seed,'oracle',k]['actual_cost'])
            self.assertGreaterEqual(lookup[seed,'oracle',1]['actual_cost'],lookup[seed,'oracle',16]['actual_cost'])
            self.assertGreaterEqual(lookup[seed,'oracle',16]['actual_cost'],lookup[seed,'oracle',256]['actual_cost'])
    def test_weighted_contact_decomposition(self):
        rows=json.loads((ROOT/'results.json').read_text())['rows']
        for r in rows:
            p=r['test_contact_fraction']; reconstructed=p*r['contact_mse']+(1-p)*r['noncontact_mse']
            self.assertAlmostEqual(reconstructed,r['test_mse'],places=14)
    def test_stored_aggregates(self):
        result=json.loads((ROOT/'results.json').read_text())
        for summary in result['summary']:
            group=[r for r in result['rows'] if r['model']==summary['model'] and r['K']==summary['K']]
            for key in ['test_mse','contact_mse','noncontact_mse','actual_cost','regret','optimism']:
                self.assertEqual(mean_ci([r[key] for r in group]),summary[key])

if __name__=='__main__': unittest.main(verbosity=2)

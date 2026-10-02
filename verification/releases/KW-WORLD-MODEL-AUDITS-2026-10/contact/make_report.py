#!/usr/bin/env python3
"""Render all quantitative report claims from checked results."""
import argparse, json
from pathlib import Path
import numpy as np
from scipy import stats
from run import mean_ci
ROOT=Path(__file__).parent
NAMES={'blind_ridge':'Action-blind ridge','ridge':'Action-conditioned ridge','poly2':'Quadratic ridge','mlp':'MLP','hybrid':'Learned hybrid (privileged)','oracle':'Same-candidate oracle'}
ORDER=list(NAMES)
def fmt(x):
    return f'{x:.6g}'
def interval(c): return f"{fmt(c['mean'])} [{fmt(c['low'])}, {fmt(c['high'])}]"
def render():
    j=json.loads((ROOT/'results.json').read_text()); rows=j['rows']
    sm={(r['model'],r['K']):r for r in j['summary']}
    lookup={(r['seed'],r['model'],r['K']):r for r in rows}
    seeds=sorted({r['seed'] for r in rows})
    support=[lookup[s,'oracle',1] for s in seeds]
    ct=[r['train_contacts'] for r in support]; tf=[r['test_contact_fraction'] for r in support]
    supporttext=(f"Training contacts ranged from {min(ct)} to {max(ct)} per 6,000 transitions. "
                 f"The mean held-out contact fraction was {100*np.mean(tf):.4f}% "
                 f"(range {100*min(tf):.4f}% to {100*max(tf):.4f}%). "
                 f"Across the ten equally sized test sets, {round(sum(tf)*20000):,} of 200,000 transitions were contacts.")
    table='At K=256. Brackets show descriptive seed-level 95% t intervals; all metrics are lower-is-better except signed optimism.\n\n| Model | Ordinary MSE | Contact MSE | Executed cost | Candidate regret | Optimism |\n|---|---:|---:|---:|---:|---:|\n'
    for n in ORDER:
        r=sm[n,256]; table+='| '+NAMES[n]+' | '+' | '.join(interval(r[k]) for k in ['test_mse','contact_mse','actual_cost','regret','optimism'])+' |\n'
    ratios=np.array([lookup[s,'mlp',256]['contact_mse']/lookup[s,'mlp',256]['test_mse'] for s in seeds])
    costratios=np.array([lookup[s,'mlp',256]['actual_cost']/lookup[s,'oracle',256]['actual_cost'] for s in seeds])
    table+=f"\nThe median across-seed ratio of MLP contact MSE to ordinary MSE was {np.median(ratios):.3f}; the median MLP-to-oracle executed-cost ratio was {np.median(costratios):.3f}. These are descriptive post hoc ratios, not independent inferential endpoints."
    budget='| Model | K=1 cost | K=16 cost | K=256 cost | Paired cost change 256 minus 16 |\n|---|---:|---:|---:|---:|\n'
    for n in ORDER:
        budget+='| '+NAMES[n]+' | '+' | '.join(fmt(sm[n,k]['actual_cost']['mean']) for k in [1,16,256])+' | '+interval(j['paired'][n]['cost_K256_minus_K16'])+' |\n'
    comparisons=[]
    for metric in ['test_mse','actual_cost']:
        d=np.array([lookup[s,'mlp',256][metric]-lookup[s,'ridge',256][metric] for s in seeds])
        comparisons.append((f'MLP minus ridge: {metric}',d))
    for n in ['blind_ridge','ridge','poly2','mlp']:
        d=np.array([lookup[s,n,256]['actual_cost']-lookup[s,n,16]['actual_cost'] for s in seeds])
        comparisons.append((f'{NAMES[n]}: K256 minus K16 cost',d))
    paired='Post hoc paired comparisons, all in one family of six two-sided tests. Negative differences favor the first-named model or the larger budget. These tests do not upgrade the evidence level.\n\n| Comparison | Paired mean and 95% t interval | Raw p | Bonferroni p |\n|---|---:|---:|---:|\n'
    for label,d in comparisons:
        p=float(stats.ttest_1samp(d,0).pvalue)
        paired+=f'| {label} | {interval(mean_ci(d))} | {p:.6g} | {min(1.,6*p):.6g} |\n'
    text=(ROOT/'report_template.md').read_text()
    for token,value in [('SUPPORT',supporttext),('MAIN_TABLE',table),('BUDGET_TABLE',budget),('PAIRED_TABLE',paired)]: text=text.replace('@@'+token+'@@',value)
    assert '@@' not in text
    return text
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--check',action='store_true');a=p.parse_args();target=ROOT/'REPORT.md';text=render()
    if a.check:
        assert target.read_text()==text, 'Report differs from deterministic numeric rendering'
        print('PASS report byte-for-byte generation check')
    else: target.write_text(text);print('Wrote REPORT.md')

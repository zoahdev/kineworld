"""Independent 100-digit verification of reported theta root enclosures."""
from pathlib import Path
import json
import mpmath as mp
mp.mp.dps=100
P=Path(__file__).parent
r=json.loads((P/'results.json').read_text());checks=[]
threshold=mp.log(40)
for run in r['runs']:
 for time,row in run['checkpoints'].items():
  n=row['informative_transitions'];k=row['flips'];iv=row['theta_interval']
  if n==0:continue
  logq=mp.loggamma(k+mp.mpf('.5'))+mp.loggamma(n-k+mp.mpf('.5'))-mp.loggamma(n+1)-mp.log(mp.pi)
  def f(x):
   if x==0:return mp.inf if k else logq
   return logq-k*mp.log(x)-(n-k)*mp.log1p(-x)-threshold
  mode=min(mp.mpf('.5'),mp.mpf(k)/n)
  assert f(mode)<=0
  lo=mp.mpf(str(iv[0]));hi=mp.mpf(str(iv[1]))
  assert lo<=mode<=hi
  left_margin=f(lo);right_margin=f(hi)
  assert lo==0 or left_margin>0
  assert hi==mp.mpf('.5') or right_margin>0
  checks.append(dict(seed=run['seed'],transitions=int(time),left_log_threshold_margin=float(left_margin),right_log_threshold_margin=float(right_margin)))
result=dict(decimal_digits=100,intervals_checked=len(checks),all_reported_intervals_outward=True,checks=checks,remark='Independent high-precision cross-check, not formal interval-arithmetic verification.')
(P/'high_precision.json').write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps(result,indent=2))

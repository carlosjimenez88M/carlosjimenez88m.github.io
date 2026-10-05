"""Cross-check numerical claims against independent analytical calculations."""
import json
import math
from pathlib import Path
import numpy as np
from scipy.integrate import quad
from scipy.special import beta
from scipy.stats import norm

data = json.loads((Path(__file__).parent / 'results/summary.json').read_text())
for p in (.5, .6, .7):
    exact = sum(math.comb(20,k)*p**k*(1-p)**(20-k) for k in range(15,21))
    assert math.isclose(exact, data['binomial'][str(p)]['probability_15_or_more'], rel_tol=1e-12)
predictive = sum(math.comb(20,k)*beta(k+13,29-k)/beta(13,9) for k in range(15,21))
assert math.isclose(predictive, data['uncertain_probability']['predictive_probability_15_or_more'], rel_tol=1e-11)

# Exact expected selected first score via integration of the maximum density.
for row in data['selection']['rows']:
    m = row['candidates']
    maximum, _ = quad(lambda z: z*m*norm.pdf(z)*norm.cdf(z)**(m-1), -12,12)
    expected = 70 + math.sqrt(52)*maximum
    assert abs(row['selected_score']-expected) < 4*row['mcse_selected_score']

# Ordered urn paths have the same probability as the beta-mixture likelihood.
for path in ([0,1,0,1,1], [1]*8, [0]*8):
    counts=[1,1]; probability=1.
    for color in path:
        probability *= counts[color]/sum(counts)
        counts[color] += 1
    mixture=beta(counts[0],counts[1])/beta(1,1)
    assert math.isclose(probability,mixture,rel_tol=1e-12)

cl=data['clustered_evaluation']
se=math.sqrt(.06**2/40 + .04**2/120 + .04**2/480)
assert math.isclose(se,cl['analytical_se'],rel_tol=1e-12)
user_means=np.loadtxt(Path(__file__).parent/'results/synthetic-user-means.csv',delimiter=',',skiprows=1)[:,1]
assert np.isclose(user_means.mean(),cl['observed_mean'])
assert np.isclose(user_means.std(ddof=1)/math.sqrt(40),cl['cluster_se'])
coverage=cl['cluster_95_coverage'];mcse=math.sqrt(.95*.05/4000)
assert abs(coverage-.95) < 4*mcse
for p,q in [(0.,.4),(.2,.8),(.6,.6),(1.,.5)]:
    assert math.isclose(p*(q-1)**2+(1-p)*q*q,p*(1-p)+(q-p)**2,abs_tol=1e-12)
print('Verified: exact tails, Gaussian maximum, urn equivalence, sampling variance, interval inputs, coverage, Brier identity.')

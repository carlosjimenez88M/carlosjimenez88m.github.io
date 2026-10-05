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
# Independently check the new graphical comparisons with finite sums and moments.
graphics=json.loads((Path(__file__).parent/'results/graphics.json').read_text())
m=13/22
plugin_tail=sum(math.comb(20,k)*m**k*(1-m)**(20-k) for k in range(15,21))
assert math.isclose(plugin_tail, graphics['plug_in_posterior_mean']['probability_15_or_more'], rel_tol=1e-12)
assert math.isclose(20*m*(1-m), graphics['plug_in_posterior_mean']['variance'], rel_tol=1e-12)
v=13*9/(22**2*23)
assert math.isclose(20*m*(1-m)+20*19*v,graphics['posterior_predictive']['variance'],rel_tol=1e-12)
shared_tail=sum(math.comb(20,k)*beta(k+2.4,20-k+1.6)/beta(2.4,1.6) for k in range(15,21))
assert math.isclose(shared_tail,graphics['shared_environment']['probability_15_or_more'],rel_tol=1e-11)
assert math.isclose(20*.6*.4*(1+19*.2),graphics['shared_environment']['variance'],rel_tol=1e-12)
q=data['binomial']['0.5']['probability_15_or_more']
assert math.isclose(1-(1-q)**100,.8764595887372029,rel_tol=1e-12)
# These are intervention predictions, distinct from conditioning on an observed A.
assert math.isclose((1+499/2)/500,.501,rel_tol=1e-12)
assert math.isclose((1+499*2/3)/500,.6673333333333333,rel_tol=1e-12)
# Fresh-attempt pass@k versus the nonlinear plug-in, enumerated exactly.
from fractions import Fraction
import csv
n, k, chance = 10, 3, Fraction(1, 5)
weights = [Fraction(math.comb(n,c))*chance**c*(1-chance)**(n-c) for c in range(n+1)]
subset = sum(weight*(1-Fraction(math.comb(n-c,k) if n-c>=k else 0,math.comb(n,k)))
             for c,weight in enumerate(weights))
plugin = sum(weight*(1-(1-Fraction(c,n))**k) for c,weight in enumerate(weights))
assert subset == 1-(1-chance)**k == Fraction(61,125)
assert plugin == Fraction(1408,3125)
reliability = Fraction(16,16+36)
mean_five_reliability = Fraction(16)/(16+Fraction(36,5))
assert reliability == Fraction(4,13) and mean_five_reliability == Fraction(20,29)

# A separate array-based reconstruction of the historical cohort summaries.
applied=json.loads((Path(__file__).parent/'results/applied-case.json').read_text())
with (Path(__file__).parent/'data/berkeley-admissions.csv').open() as stream:
    rows=list(csv.DictReader(stream))
counts=np.array([[[(int(next(row['admitted'] for row in rows if row['department']==dept and row['recorded_sex']==group))),
                   (int(next(row['applications'] for row in rows if row['department']==dept and row['recorded_sex']==group)))]
                  for group in ('Male','Female')] for dept in 'ABCDEF'])
assert counts[:,:,1].sum()==4526
raw=counts[:,:,0].sum(axis=0)/counts[:,:,1].sum(axis=0)
common_weights=counts[:,:,1].sum(axis=1)/4526
common=(counts[:,:,0]/counts[:,:,1]).T@common_weights
assert np.allclose(raw,[applied['observed_mix']['admission_rates'][g] for g in ('Male','Female')])
assert np.allclose(common,[applied['common_pooled_department_mix']['admission_rates'][g] for g in ('Male','Female')])
assert raw[1]-raw[0]<0<common[1]-common[0]
new_checks={'pass_at_k':{'n':n,'k':k,'p':float(chance),'expected_subset':float(subset),
                        'expected_plugin':float(plugin)},
            'population_reliability':{'single':float(reliability),'mean_of_five':float(mean_five_reliability)},
            'berkeley':{'applications':4526,'observed_female_minus_male':float(raw[1]-raw[0]),
                        'common_mix_female_minus_male':float(common[1]-common[0])},'status':'passed'}
(Path(__file__).parent/'results/design-checks.json').write_text(json.dumps(new_checks,indent=2)+'\n')
print('Verified: reference and predictive tails, selection, urn intervention, population reliability, pass@k, historical reference comparison, clustered precision and scoring identity.')

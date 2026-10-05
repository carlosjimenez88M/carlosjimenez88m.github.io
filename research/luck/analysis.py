"""Analytical checks and synthetic experiments for the three essays on luck.

No provider calls, fitted social data, or empirical estimates of human luck.
Run: MPLCONFIGDIR=/tmp/luck-mpl python3 research/luck/analysis.py
"""
from pathlib import Path
import hashlib
import json
import platform
import numpy as np
import scipy
from scipy.stats import binom, betabinom, norm, t
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "research/luck/results"
FIG = ROOT / "static/img/luck"
SEED = 20261005


def save(fig, name):
    fig.tight_layout()
    fig.savefig(FIG / f"{name}.svg", bbox_inches="tight")
    fig.savefig(FIG / f"{name}.png", dpi=170, bbox_inches="tight")
    plt.close(fig)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    FIG.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({"font.size": 10, "axes.spines.top": False,
                         "axes.spines.right": False, "svg.hashsalt": "luck-2026"})
    rng = np.random.default_rng(SEED)
    result = {"kind": "Analytical calculations and synthetic demonstrations only",
              "seed": SEED, "python": platform.python_version(),
              "numpy": np.__version__, "scipy": scipy.__version__,
              "matplotlib": matplotlib.__version__}

    x = np.arange(21)
    result["binomial"] = {
        str(p): {"mean": 20*p, "variance": 20*p*(1-p),
                 "probability_15_or_more": float(binom.sf(14, 20, p)),
                 "mid_percentile_at_15": float(binom.cdf(14, 20, p)+binom.pmf(15, 20, p)/2)}
        for p in [.5, .6, .7]}
    result["uncertain_probability"] = {
        "prior": "Beta(1,1)", "training_successes": 12, "training_trials": 20,
        "posterior": "Beta(13,9)",
        "predictive_probability_15_or_more": float(betabinom.sf(14,20,13,9)),
        "predictive_variance": float(betabinom.var(20,13,9))}
    result["shared_environment"] = {
        "alpha": 2.4, "beta": 1.6, "intraclass_correlation": .2,
        "variance_count": float(betabinom.var(20,2.4,1.6)),
        "independent_variance_count": float(binom.var(20,.6))}
    fig, ax = plt.subplots(1,2,figsize=(10,3.7))
    for p in [.5,.6,.7]: ax[0].plot(x, binom.pmf(x,20,p), marker="o", ms=3, label=f"p = {p}")
    ax[0].axvline(15,color="#555",ls=":");ax[0].set(title="One result, three reference distributions",xlabel="Successes in 20 trials",ylabel="Probability")
    ax[0].legend()
    ax[1].plot(x,binom.pmf(x,20,.6),label="Independent trials")
    ax[1].plot(x,betabinom.pmf(x,20,2.4,1.6),label="Shared environment: correlation 0.2")
    ax[1].set(title="The same mean can conceal different exposure",xlabel="Successes in 20 trials",ylabel="Probability");ax[1].legend(fontsize=8)
    save(fig,"reference-distributions")

    # Known artificial score model, unbounded score units (not accuracy).
    worlds = 20000
    mu, tau, sigma = 70., 4., 6.
    ms = [1,5,20,100]
    selection=[]
    for m in ms:
        theta=rng.normal(mu,tau,(worlds,m)); y=theta+rng.normal(0,sigma,(worlds,m))
        best=np.argmax(y,axis=1); rows=np.arange(worlds)
        observed=y[rows,best]; latent=theta[rows,best]
        future=latent+rng.normal(0,sigma,worlds)
        selection.append({"candidates":m,"selected_score":float(observed.mean()),
            "selected_latent_mean":float(latent.mean()),"independent_repeat_mean":float(future.mean()),
            "selection_optimism":float((observed-latent).mean()),
            "mcse_selected_score":float(observed.std(ddof=1)/np.sqrt(worlds))})
    result["selection"]={"worlds":worlds,"mu":mu,"skill_sd":tau,"occasion_sd":sigma,
                         "shrinkage_weight":tau*tau/(tau*tau+sigma*sigma),"rows":selection}
    fig,ax=plt.subplots(figsize=(7.5,3.8))
    for key,label in [("selected_score","Winning observed score"),("selected_latent_mean","Winner's latent mean"),("independent_repeat_mean","Independent repeat")]:
        ax.plot(ms,[v[key] for v in selection],marker="o",label=label)
    ax.axhline(mu,color="#555",ls=":",label="Population mean")
    ax.set(xscale="log",xticks=ms,xticklabels=ms,xlabel="Candidates inspected",ylabel="Mean score (synthetic units)",title="Selection improves the story more than the selected ability")
    ax.legend(fontsize=8);save(fig,"selection-and-regression")
    # Collider selection, not a causal estimate from observational data.
    skill=rng.normal(size=200000); shock=rng.normal(size=200000)
    mask=skill+shock>2
    result["collider"]={"population_correlation":float(np.corrcoef(skill,shock)[0,1]),
         "selected_correlation":float(np.corrcoef(skill[mask],shock[mask])[0,1]),
         "selected_count":int(mask.sum()),"rule":"skill + shock > 2"}

    # Exact exchangeable Polya-urn marginal; equivalent to reinforced draws.
    draws=500
    no_feedback=rng.binomial(draws,.5,worlds)/draws
    feedback_p=rng.beta(1,1,worlds)
    feedback=rng.binomial(draws,feedback_p)/draws
    conditional=(1+rng.binomial(draws-1,rng.beta(2,1,worlds)))/draws
    result["reinforcement"]={"worlds":worlds,"draws":draws,"initial_weights":[1,1],
        "independent_share_sd":float(no_feedback.std(ddof=1)),
        "reinforced_share_sd":float(feedback.std(ddof=1)),
        "expected_share_after_first_win":float((1+(draws-1)*2/3)/draws),
        "simulated_share_after_first_win":float(conditional.mean())}
    fig,ax=plt.subplots(figsize=(7.5,3.8))
    bins=np.linspace(0,1,31)
    ax.hist(no_feedback,bins,density=True,alpha=.65,label="Independent: fixed p = 0.5")
    ax.hist(feedback,bins,density=True,alpha=.55,label="Reinforced: equal initial weights")
    ax.set(xlabel="Final share of A",ylabel="Density",title="Equal beginnings do not guarantee similar endings")
    ax.legend(fontsize=8);save(fig,"reinforcement")

    # Clustered differences: user effect + task effect + generation noise.
    n,k,r=40,3,4
    d,sd_user,sd_task,sd_run=.02,.06,.04,.04
    def sample(size):
        return d+rng.normal(0,sd_user,(size,n,1,1))+rng.normal(0,sd_task,(size,n,k,1))+rng.normal(0,sd_run,(size,n,k,r))
    panel=sample(1)[0]
    by_user=panel.mean(axis=(1,2))
    mean=float(panel.mean());se_cluster=float(by_user.std(ddof=1)/np.sqrt(n))
    se_naive=float(panel.std(ddof=1)/np.sqrt(n*k*r))
    critical=float(t.ppf(.975,n-1))
    reps=4000;panels=sample(reps)
    means=panels.mean(axis=(1,2,3))
    clse=panels.mean(axis=(2,3)).std(axis=1,ddof=1)/np.sqrt(n)
    nse=panels.reshape(reps,-1).std(axis=1,ddof=1)/np.sqrt(n*k*r)
    result["clustered_evaluation"]={"users":n,"tasks_per_user":k,"generations_per_task":r,
        "true_difference":d,"sd_user":sd_user,"sd_task":sd_task,"sd_run":sd_run,
        "observed_mean":mean,"cluster_se":se_cluster,"naive_se":se_naive,
        "cluster_95_interval":[mean-critical*se_cluster,mean+critical*se_cluster],
        "naive_95_interval":[mean-1.96*se_naive,mean+1.96*se_naive],
        "repeated_panels":reps,"cluster_95_coverage":float(np.mean(np.abs(means-d)<=critical*clse)),
        "naive_95_coverage":float(np.mean(np.abs(means-d)<=1.96*nse)),
        "analytical_se":float(np.sqrt((sd_user**2+sd_task**2/k+sd_run**2/(k*r))/n))}
    np.savetxt(OUT/'synthetic-user-means.csv',np.c_[np.arange(1,n+1),by_user],delimiter=',',header='user_id,mean_paired_difference',comments='')
    fig,ax=plt.subplots(figsize=(7.5,3.6))
    ax.errorbar([mean,mean],[1,0],xerr=[critical*se_cluster,1.96*se_naive],fmt="o",capsize=5)
    ax.axvline(d,color="#2465a4",ls=":",label="Known synthetic effect: 0.02")
    ax.axvline(0,color="#555",lw=.8)
    ax.set(yticks=[0,1],yticklabels=["Treat all 480 differences as independent","40 independent user averages"],xlabel="Mean paired score difference",title="Repeating an answer is not sampling a new user")
    ax.legend(fontsize=8);save(fig,"evaluation-units")
    # Independent analytic cross-checks of the generative assumptions.
    assert np.isclose(result['shared_environment']['variance_count'],23.04)
    assert np.isclose(result['selection']['shrinkage_weight'],4/13)
    assert abs(result['reinforcement']['simulated_share_after_first_win']-result['reinforcement']['expected_share_after_first_win'])<.01
    assert result['clustered_evaluation']['cluster_95_coverage']>.93
    assert result['clustered_evaluation']['naive_95_coverage']<.80
    result["script_sha256"]=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    (OUT/'summary.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result,indent=2))


if __name__ == '__main__':
    main()

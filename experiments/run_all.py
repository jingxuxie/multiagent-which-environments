"""CPU-only fixed-seed experiment suite. Run from the repository root.

Scientific CSVs contain no wall-clock timestamps. Metrics are exact population
values unless explicitly labeled sampled. No policy is pretrained or API-backed.
"""
from __future__ import annotations
import argparse
import csv
import hashlib
import json
import platform
import time
from pathlib import Path
import numpy as np
import scipy
from coopcurriculum.core import *
from coopcurriculum.design import pair_sum_design, exact_design, coverage_design
from coopcurriculum.assignment import *
from coopcurriculum.learning import learn_batch


def save(root, name, rows):
    path = root / f'{name}.csv'
    with path.open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        for row in rows:
            writer.writerow({k: format(float(v), '.12g') if isinstance(v, (float, np.floating)) else v for k, v in row.items()})
    return rows


def finite_geometries(root):
    rows = []
    for seed in range(120):
        rng = np.random.default_rng(10000 + seed)
        m, k = 3 + seed % 3, 4 + seed % 4
        a = rng.uniform(0, 1, (m, k))
        b = [(i, j) for i in range(k) for j in range(i+1, k) if rng.random() < .5]
        if not b: b = [(0, 1)]
        lp, exact = pair_sum_design(a, b), exact_design(a, b)
        uniform = margin(a, np.ones(m)/m, b)
        assert .5 * exact.margin <= lp.margin + 2e-7 <= exact.margin + 4e-7
        rows.append(dict(seed=seed, environments=m, policies=k, bad_pairs=len(b),
                         uniform=uniform, lp=lp.margin, exact=exact.margin,
                         ratio=lp.margin/exact.margin if exact.margin > 1e-9 else 1.))
    return save(root, 'finite_geometries', rows)


def sharp_examples(root):
    rows=[]
    for small in [.1, .03, .01, .003, .001, .0001]:
        a=np.array([[1,0,1],[1,.5-small,.5-small]])
        lp,opt=pair_sum_design(a,[(1,2)]),exact_design(a,[(1,2)])
        rows.append(dict(slack=small,lp=lp.margin,exact=opt.margin,ratio=lp.margin/opt.margin))
    save(root,'approximation_sharpness',rows)
    rows=[]
    for neutral in [0,1,2,4,8,18,38,78]:
        a=np.vstack([[[1, .8, 1],[1,1,.8]],np.ones((neutral,3))])
        t=np.eye(3)
        for eta in [0,.005,.01,.02,.05,.1]:
            mu=np.ones(neutral+2)/(neutral+2)
            rows.append(dict(neutral=neutral,eta=eta,margin=margin(a,mu,bad_pairs(t,.9)),
                             near_optimal=len(near_optimal(a,mu,eta)),worst=worst_crossplay(a,mu,t,eta)))
    return save(root,'dilution',rows)


def structured_checks(root):
    rows=[]
    for seed in range(300):
        rng=np.random.default_rng(20000+seed)
        d=3+seed%4
        D=rng.dirichlet(np.ones(d),size=5)*rng.uniform(.05,.6,(5,1))
        mu,q=rng.dirichlet(np.ones(5)),rng.dirichlet(np.ones(d))
        eps=[.071,.217,.413][seed%3]
        a,t=task_tables(D,q)
        b=bad_pairs(t,1-eps)
        direct=margin(a,mu,b)
        balanced=balanced_margin(mu@D,q,eps)
        cover=subset_coverage(mu@D,q,eps)
        assert abs(direct-balanced)<1e-9
        assert cover/2-1e-9<=balanced<=cover+1e-9
        rows.append(dict(seed=seed,dimension=d,epsilon=eps,direct=direct,
                         balanced=balanced,coverage=cover,error=abs(direct-balanced)))
    save(root,'structured_checks',rows)
    # A exact-tolerance family that retains multiple compatible conventions.
    rows=[]
    for d in [3,4,5,6]:
        D=.4*np.eye(d)
        a,t=task_tables(D,np.ones(d)/d)
        for eps in [.071,.217,.413,.613]:
            design=coverage_design(D,np.ones(d)/d,eps)
            eta=.8*design.margin
            active=near_optimal(a,design.mu,eta)
            rows.append(dict(d=d,epsilon=eps,margin=design.margin,eta=eta,
                             support=int(np.sum(design.mu>1e-8)),survivors=len(active),
                             worst=worst_crossplay(a,design.mu,t,eta)))
    return save(root,'compatible_ambiguity',rows)


def interval_checks(root):
    rows=[]
    for seed in range(400):
        rng=np.random.default_rng(30000+seed)
        lo=rng.uniform(0,.8,(4,6))
        hi=lo+rng.random((4,6))*(1-lo)
        # Include narrow and heterogeneous boxes, not only uninformative ones.
        if seed%2: hi=lo+.02*(hi-lo)
        mu=rng.dirichlet(np.ones(4))
        bad=[(0,1),(2,3),(1,4),(4,5)]+([(5,5)] if seed%7==0 else [])
        cert=interval_margin(lo,hi,mu,bad)
        witness=interval_witness(lo,hi,mu,bad)
        attained=margin(witness,mu,bad)
        assert abs(cert-attained)<1e-9
        minimum=1.
        for _ in range(12):
            sample=lo+rng.random(lo.shape)*(hi-lo)
            minimum=min(minimum,margin(sample,mu,bad))
        assert minimum>=cert-1e-9
        rows.append(dict(seed=seed,certificate=cert,witness=attained,
                         sampled_minimum=minimum,error=abs(cert-attained)))
    return save(root,'interval_checks',rows)


def calibration(root):
    rows=[]
    for n in [16,64,256,1024,4096,16384]:
        for seed in range(100):
            rng=np.random.default_rng(40000+seed+1000*n)
            # A fixed target incompatibility (policies 1 and 2); no reward labels
            # or teacher training samples are used as target cross-play data.
            gap=.12+.12*(seed%10)/9
            a=np.array([[.8,.8-gap,.8], [.8,.8-gap*.55,.8-gap*.55],
                        [.8,.8,.8-gap], [.65,.65,.65]])
            b=[(1,2)]
            estimates=rng.binomial(n,a)/n
            design=pair_sum_design(estimates,b)
            lo,hi=hoeffding_box(estimates,n,.05)
            robust=interval_margin(lo,hi,design.mu,b)
            actual=margin(a,design.mu,b)
            for eta in [.075,.15]:
                rows.append(dict(n=n,seed=seed,eta=eta,plugin=design.margin,robust=robust,
                    actual=actual,plugin_accept=int(design.margin>eta),
                    robust_accept=int(robust>eta),
                    plugin_false=int(design.margin>eta and actual<=eta),
                    robust_false=int(robust>eta and actual<=eta),
                    interval_covers=int(np.all(lo<=a)&np.all(a<=hi))))
    return save(root,'calibration',rows)


def curriculum_library(neutral):
    d=6
    delta=np.array([.08,.12,.18,.25,.35,.5])
    q=np.array([.05,.08,.12,.15,.25,.35])
    D=np.vstack([np.diag(delta),np.zeros((neutral,d))])
    def extend(w): return np.r_[w,np.zeros(neutral)]
    designs={
        'uniform':np.ones(d+neutral)/(d+neutral),
        'diagnostic_uniform':extend(np.ones(d)/d),
        'inverse_gap':extend((1/delta)/np.sum(1/delta)),
        'inverse_gap_squared':extend((1/delta**2)/np.sum(1/delta**2)),
        'difficulty_only':extend(np.eye(d)[-1]),
        'coverage_lp':coverage_design(D,q,.10).mu,
    }
    # Farthest-first selection on standardized visible-to-teacher nuisance features.
    # This is a deliberately controlled nuisance-diversity baseline, not CEC/UED.
    features=np.column_stack([np.linspace(-1,1,d+neutral),np.cos(np.arange(d+neutral)),
                              np.sin(np.arange(d+neutral)*1.7)])
    chosen=[0]
    for _ in range(3):
        dist=((features[:,None]-features[chosen][None])**2).sum(axis=2).min(axis=1)
        chosen.append(int(np.argmax(dist)))
    w=np.zeros(d+neutral); w[chosen]=1/len(chosen)
    designs['feature_diversity']=w
    return delta,q,D,designs


def learning(root):
    rows, seedrows, designrows=[],[],[]
    theta=np.array([1,0,1,1,0,1])
    seeds=np.arange(50000,50024)
    for neutral in [0,24]:
        delta,q,D,designs=curriculum_library(neutral)
        a,t=task_tables(D,q,theta)
        for name,mu in designs.items():
            gm=balanced_margin(mu@D,q,.10)
            designrows.append(dict(neutral=neutral,method=name,margin=gm,
                coverage=subset_coverage(mu@D,q,.10),support=int((mu>1e-8).sum()),
                diagnostic_mass=float(mu[:6].sum()),weights=';'.join(format(x,'.8g') for x in mu)))
            for kind in ['search','reinforce']:
                for budget in [256,1024,4096]:
                    bits,gs=learn_batch(mu,delta,theta,neutral,budget,seeds,kind)
                    xp=1-np.einsum('ijk,k->ij',np.abs(bits[:,None,:]-bits[None,:,:]),q)
                    offdiag=xp[~np.eye(len(seeds),dtype=bool)]
                    eta=.8*gm if gm>1e-10 else 0.
                    qualified=gs<=eta+1e-10
                    sub=xp[np.ix_(qualified,qualified)]
                    certified_worst=float(sub.min()) if sub.size and gm>1e-10 else np.nan
                    u=float(offdiag.mean())
                    leave=[]
                    for i in range(len(seeds)):
                        keep=np.arange(len(seeds))!=i
                        smaller=xp[np.ix_(keep,keep)]
                        leave.append(smaller[~np.eye(len(seeds)-1,dtype=bool)].mean())
                    pseudo=len(seeds)*u-(len(seeds)-1)*np.array(leave)
                    jackknife_se=float(pseudo.std(ddof=1)/np.sqrt(len(seeds)))
                    if gm>1e-10 and sub.size: assert certified_worst>=.9-1e-9
                    rows.append(dict(neutral=neutral,method=name,learner=kind,budget=budget,
                        margin=gm,mean_gap=float(gs.mean()),max_gap=float(gs.max()),
                        mean_crossplay=float(offdiag.mean()),jackknife_se=jackknife_se,worst_crossplay=float(offdiag.min()),
                        bad_pair_rate=float((offdiag<.9-1e-9).mean()),qualified=int(qualified.sum()),
                        certified_count=int(qualified.sum()) if gm>1e-10 else 0,
                        conditional_worst=certified_worst,distinct=len(np.unique(bits,axis=0))))
                    for idx,seed in enumerate(seeds):
                        seedrows.append(dict(neutral=neutral,method=name,learner=kind,budget=budget,
                            seed=int(seed),gap=float(gs[idx]),qualified=int(qualified[idx]),
                            mean_partner_return=float(np.delete(xp[idx],idx).mean()),
                            bits=''.join(map(str,bits[idx]))))
    save(root,'curricula',designrows)
    save(root,'learned_seeds',seedrows)
    return save(root,'learning',rows)


def class_expansion(root):
    rows=[]
    for seed in range(100):
        rng=np.random.default_rng(60000+seed)
        d=3+seed%3
        D=rng.dirichlet(np.ones(d),size=5)*rng.uniform(.2,.5,(5,1))
        q=rng.dirichlet(np.ones(d))
        design=coverage_design(D,q,.2)
        a,t=task_tables(D,q)
        # An out-of-class policy agrees perfectly in training but adopts an
        # incompatible target convention; an observable source flag permits it.
        expanded=np.column_stack([a,a[:,0]])
        target=np.zeros((len(t)+1,len(t)+1));target[:-1,:-1]=t;target[-1,-1]=1
        old=margin(a,design.mu,bad_pairs(t,.8))
        new=margin(expanded,design.mu,bad_pairs(target,.8))
        assert old>0 and abs(new)<1e-9
        rows.append(dict(seed=seed,dimension=d,original_margin=old,expanded_margin=new))
    return save(root,'class_expansion',rows)


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--output',default='results')
    args=parser.parse_args();root=Path(args.output);root.mkdir(parents=True,exist_ok=True)
    start=time.perf_counter()
    for fn in [finite_geometries,sharp_examples,structured_checks,interval_checks,calibration,learning,class_expansion]:
        tick=time.perf_counter(); fn(root); print(f'{fn.__name__}: {time.perf_counter()-tick:.2f}s',flush=True)
    science=sorted(root.glob('*.csv'))
    manifest={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in science}
    (root/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    metadata=dict(python=platform.python_version(),numpy=np.__version__,scipy=scipy.__version__,
                  seconds=time.perf_counter()-start,platform=platform.platform(),
                  seed_protocol='fixed named streams; see run_all.py (calibration streams depend on n)',
                  limitations='finite known policy class; no LLM, human, real-robot, or deep-MARL evidence')
    (root/'run_metadata.json').write_text(json.dumps(metadata,indent=2)+'\n')
    print(json.dumps(metadata,indent=2))

if __name__=='__main__': main()

"""Generate manuscript tables, figures, and an auditable summary from scientific CSVs."""
from __future__ import annotations
import csv
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

LABELS={'uniform':'Uniform library','diagnostic_uniform':'Diagnostic uniform',
        'inverse_gap':'Inverse gap','inverse_gap_squared':'Inverse squared gap',
        'difficulty_only':'Difficulty only','coverage_lp':'Coverage LP',
        'feature_diversity':'Feature diversity'}


def load(name):
    with open(f'results/{name}.csv',newline='') as f:
        rows=list(csv.DictReader(f))
    for row in rows:
        for key,value in row.items():
            try:row[key]=float(value)
            except ValueError:pass
    return rows


def finish(fig,ax,name,xlabel,ylabel):
    ax.set_xlabel(xlabel);ax.set_ylabel(ylabel)
    ax.spines[['top','right']].set_visible(False)
    fig.tight_layout()
    path=Path('figures');path.mkdir(exist_ok=True)
    fig.savefig(path/f'{name}.pdf',bbox_inches='tight',metadata={'CreationDate':None,'ModDate':None})
    fig.savefig(path/f'{name}.png',dpi=160,bbox_inches='tight')
    plt.close(fig)


def main():
    Path('paper').mkdir(exist_ok=True)
    finite,cal,learning=[load(n) for n in ['finite_geometries','calibration','learning']]
    selected=[r for r in learning if r['neutral']==24 and r['budget']==4096 and r['learner']=='search']
    tex=[r'\begin{tabular}{lrrrr}',r'\toprule',
         r'Curriculum & Margin & Mean cross-play & Worst & Eligible \\',r'\midrule']
    for r in selected:
        tex.append(f"{LABELS[r['method']]} & {r['margin']:.4f} & ${r['mean_crossplay']:.3f}\\,\\pm\\,{r['jackknife_se']:.3f}$ & {r['worst_crossplay']:.2f} & {int(r['certified_count'])}/24 \\\\")
    tex += [r'\bottomrule',r'\end{tabular}']
    Path('paper/results_table.tex').write_text('\n'.join(tex)+'\n')
    tex=[r'\begin{tabular}{rrrrrr}',r'\toprule',
         r'$n$ per cell & Plug-in accepts & False & Box accepts & False & Trials \\',r'\midrule']
    for n in sorted(set(r['n'] for r in cal)):
        sub=[r for r in cal if r['n']==n]
        totals=[sum(r[k] for r in sub) for k in ['plugin_accept','plugin_false','robust_accept','robust_false']]
        tex.append(f'{int(n)} & {int(totals[0])} & {int(totals[1])} & {int(totals[2])} & {int(totals[3])} & {len(sub)} \\\\')
    tex += [r'\bottomrule',r'\end{tabular}']
    Path('paper/calibration_table.tex').write_text('\n'.join(tex)+'\n')
    summary={
        'finite_geometries':len(finite),
        'mean_lp_to_optimal_margin':float(np.mean([r['ratio'] for r in finite])),
        'minimum_random_lp_ratio':float(min(r['ratio'] for r in finite)),
        'lp_exact_ties':sum(abs(r['lp']-r['exact'])<1e-7 for r in finite),
        'mean_margins':{k:float(np.mean([r[k] for r in finite])) for k in ['uniform','lp','exact']},
        'structured_instances':len(load('structured_checks')),
        'interval_witnesses':len(load('interval_checks')),
        'calibration_decisions':len(cal),
        'calibration_totals':{k:int(sum(r[k] for r in cal)) for k in ['plugin_accept','plugin_false','robust_accept','robust_false']},
        'learner_runs':len(load('learned_seeds')),
        'sampled_training_episodes':int(sum(r['budget'] for r in load('learned_seeds'))),
        'class_expansions':len(load('class_expansion')),
        'learned_search_at_4096':{r['method']:{k:r[k] for k in ['margin','mean_crossplay','jackknife_se','worst_crossplay','certified_count']} for r in selected},
        'reinforce_coverage_at_4096':next(r for r in learning if r['neutral']==24 and r['budget']==4096 and r['learner']=='reinforce' and r['method']=='coverage_lp'),
    }
    Path('results/summary.json').write_text(json.dumps(summary,indent=2,allow_nan=False)+'\n')
    fig,ax=plt.subplots(figsize=(5.6,3.5))
    x=np.array([0,1,2,4,8,18,38,78])
    ax.plot(x+2,.2/(x+2),'o-',label='Uniform over expanded library')
    ax.plot(x+2,np.full(len(x),.1),'--',label='Preserve diagnostic exposure')
    ax.axhline(.02,linestyle=':',label='Optimization tolerance 0.02')
    ax.set_xscale('log');ax.set_yscale('log');ax.legend(fontsize=8)
    finish(fig,ax,'dilution','Total environments','Compatibility margin')
    fig,ax=plt.subplots(figsize=(4.7,3.8))
    ax.scatter([r['exact'] for r in finite],[r['lp'] for r in finite],s=16,alpha=.7)
    ax.plot([0,1],[0,1],linestyle='--',label='Optimal margin')
    ax.plot([0,1],[0,.5],linestyle=':',label='Proved lower guarantee')
    ax.set_xlim(0,1);ax.set_ylim(0,1);ax.legend(fontsize=8)
    finish(fig,ax,'approximation','Exact MILP margin','Pair-sum LP margin')
    for kind in ['search','reinforce']:
        fig,ax=plt.subplots(figsize=(5.6,3.6))
        for name in ['uniform','diagnostic_uniform','inverse_gap','coverage_lp']:
            sub=[r for r in learning if r['neutral']==24 and r['learner']==kind and r['method']==name]
            ax.errorbar([r['budget'] for r in sub],[r['mean_crossplay'] for r in sub],
                        yerr=[r['jackknife_se'] for r in sub],marker='o',capsize=3,label=LABELS[name])
        ax.set_xscale('log',base=2);ax.set_ylim(.4,1.025);ax.legend(fontsize=8)
        finish(fig,ax,f'learning_{kind}','Training episodes per seed','Mean cross-play (seed-jackknife SE)')
    fig,ax=plt.subplots(figsize=(5.6,3.5))
    ns=sorted(set(r['n'] for r in cal))
    for key,label in [('plugin_accept','Plug-in: accepts'),('plugin_false','Plug-in: false'),
                      ('robust_accept','Box: accepts'),('robust_false','Box: false')]:
        ax.plot(ns,[np.mean([r[key] for r in cal if r['n']==n]) for n in ns],marker='o',label=label)
    ax.set_xscale('log',base=4);ax.set_ylim(-.02,1.02);ax.legend(fontsize=8)
    finish(fig,ax,'calibration','Samples per source-policy cell','Fraction of certificate decisions')
    print(json.dumps(summary,indent=2))

if __name__=='__main__':main()

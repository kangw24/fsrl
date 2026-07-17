"""CGR-v3 held-out generalization BLIND test (synthetic, no human data).

sigma_a=0.30 is FROZEN (locked as the candidate, not re-tuned). The model is run
on held-out few-shot ranking tasks it never saw during sigma_a selection: varied
item counts (5..8), varied random support graphs, and held-out seeds. A
magnitude Q-learning control is run on the same tasks. The test asks whether the
constructive pattern (bimodality on near-diagonal pairs, high self-consistency,
self-consistent-but-wrong, LOW inter-subject similarity vs the converging
Q-learning control, distance effect) GENERALIZES beyond the Liu 8-edge graph.

This is a robustness/generalization blind test on SYNTHETIC data. It cannot
confirm reproduction of HUMANS (needs unseen human data); it checks the mechanism
is not overfit to the Liu graph.
"""
import sys, json
from pathlib import Path
from itertools import combinations
import numpy as np
from scipy.stats import kendalltau, beta as beta_dist

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from fsrl.task.liu2026 import Liu2026SymbolicSubjectTask, SymbolicMagnitudeSupportObservation, SymbolicSupportObservation, SymbolicSupportTrial, SymbolicQueryTrial
from fsrl.model.constructive_global_rank import ConstructiveGlobalRankCompression as V3, QLearningControl

def gen_cohort(n_items, n_edges, n_subjects, support_blocks, query_blocks, rng):
    all_pairs=list(combinations(range(n_items),2))
    min_edges=n_items-1
    n_edges=max(min_edges, min(n_edges, len(all_pairs)))
    # sample a connected-ish support graph: start with a random spanning tree then add random edges
    support_pairs=[]
    # random spanning tree over positions
    nodes=list(range(n_items)); rng.shuffle(nodes)
    for k in range(1,n_items):
        a,b=nodes[k-1],nodes[k]; support_pairs.append((min(a,b),max(a,b)))
    while len(support_pairs)<n_edges:
        a,b=tuple(sorted(rng.choice(n_items,2,replace=False)))
        if (a,b) not in support_pairs: support_pairs.append((a,b))
    support_pairs=support_pairs[:n_edges]
    tasks=[]
    for si in range(n_subjects):
        true_rank=tuple(int(x) for x in rng.permutation(n_items))  # strong->weak cue list
        def cue_at(pos): return true_rank[n_items-1-pos]
        sup=[]
        for blk in range(support_blocks):
            for pi in rng.permutation(len(support_pairs)):
                low,high=support_pairs[int(pi)]
                lc,hc=cue_at(low),cue_at(high)
                if rng.random()<0.5: left,right,sign=lc,hc,-1
                else: left,right,sign=hc,lc,+1
                sup.append(SymbolicSupportTrial(observation=SymbolicMagnitudeSupportObservation(left_cue=left,right_cue=right,sign=sign,magnitude=(high-low)/float(n_items-1)),block_index=blk,position_pair=(low,high)))
        qry=[]
        for blk in range(query_blocks):
            for pi in rng.permutation(len(all_pairs)):
                low,high=all_pairs[int(pi)]
                lc,hc=cue_at(low),cue_at(high)
                if rng.random()<0.5: left,right,cc=lc,hc,1
                else: left,right,cc=hc,lc,0
                qry.append(SymbolicQueryTrial(observation=SymbolicSupportObservation(left_cue=left,right_cue=right,sign=0),block_index=blk,position_pair=(low,high),correct_choice=cc))
        tasks.append(Liu2026SymbolicSubjectTask(subject_index=si,true_rank=true_rank,support_trials=tuple(sup),query_trials=tuple(qry)))
    return tasks, set(support_pairs)

def fit_beta(arr):
    arr=np.clip(np.asarray(arr,float),0.01,0.99)
    try: a,b,_,_=beta_dist.fit(arr,floc=0,fscale=1.0)
    except: return None
    return "bimodal" if (a<1 and b<1) else ("unimodal" if (a>1 and b>1) else ("high_accuracy" if (a>1 and b<1) else "other"))

def score(model, tasks, n_items, support_pairs, eps, rng):
    n=len(tasks); allpp=list(combinations(range(n_items),2))
    psa={pq:[] for pq in allpp}; orderings=[]; tau_true=[]; triads_total=0; sc_wrong=0; n_allcorrect=0; n_below=0
    max_triads=(n_items**3-4*n_items)//24 if n_items%2==0 else (n_items**3-n_items)//24
    for task in tasks:
        run=model.run_subject(task,rng)
        perc={c:(n_items-1-r) for r,c in enumerate(run.order_strong_to_weak)}  # cue->perceived pos
        true_rank=list(task.true_rank)
        cue_at={p:true_rank[n_items-1-p] for p in range(n_items)}
        pos_order=sorted(range(n_items),key=lambda p:perc[cue_at[p]])
        orderings.append(pos_order); tau_true.append(_kt(pos_order,list(range(n_items))))
        # triads from sampled prefs
        subj_acc={}
        for (p,q) in allpp:
            hi,lo=cue_at[q],cue_at[p]
            a,b=(hi,lo) if hi<lo else (lo,hi)
            p_hi=run.choice_prob[(a,b)] if hi==a else 1-run.choice_prob[(a,b)]
            pc=(1-eps)*p_hi+eps*0.5
            acc=float(rng.binomial(10,pc))/10.0; subj_acc[(p,q)]=acc; psa[(p,q)].append(acc)
        pref=[[0.5]*n_items for _ in range(n_items)]
        for (p,q) in allpp:
            pref[q][p]=subj_acc[(p,q)]; pref[p][q]=1-subj_acc[(p,q)]
        ntri=0
        for a,b,c in combinations(range(n_items),3):
            if (pref[a][b]>0.5 and pref[b][c]>0.5 and pref[c][a]>0.5) or (pref[b][a]>0.5 and pref[c][b]>0.5 and pref[a][c]>0.5): ntri+=1
        triads_total+=ntri
        if ntri==0 and pos_order!=list(range(n_items)): sc_wrong+=1
        if all(v>0.5 for v in subj_acc.values()): n_allcorrect+=1
        if float(np.mean(list(subj_acc.values())))<0.5: n_below+=1
    # inter-subject tau over included (exclude all-correct)
    inc=[o for i,o in enumerate(orderings)]
    taus=[_kt(inc[i],inc[j]) for i in range(len(inc)) for j in range(i+1,len(inc))]
    inter=float(np.mean(taus)) if taus else float('nan')
    # metrics
    overall=float(np.mean([np.mean(psa[pq]) for pq in allpp]))
    learned=float(np.mean([np.mean(psa[pq]) for pq in allpp if pq in support_pairs])) if support_pairs else float('nan')
    dist={}
    for pq in allpp:
        d=pq[1]-pq[0]; dist.setdefault(d,[]).append(np.mean(psa[pq]))
    distance={str(d):float(np.mean(v)) for d,v in sorted(dist.items())}
    nbim=0; nnear=0
    for pq in allpp:
        pr=fit_beta(psa[pq])
        if pr=='bimodal': nbim+=1
        if pr=='bimodal' and (pq[1]-pq[0])<=2: nnear+=1
    ntot=len(allpp)
    selfc=1.0-triads_total/(n*max_triads) if max_triads else float('nan')
    return dict(acc=overall,learned=learned,bim=nbim,ntot=ntot,nnear=nnear,inter_tau=inter,selfc=selfc,triads=triads_total,scwrong_frac=sc_wrong/n,tau_true=float(np.mean(tau_true)),distance=distance)

def _kt(a,b):
    t,_=kendalltau(a,b); return 0.0 if np.isnan(t) else float(t)

configs=[(5,4,971101),(6,5,971201),(6,7,971301),(7,6,971401),(7,9,971501),(8,7,971601),(8,10,971701)]
results=[]
for n_items,n_edges,seed in configs:
    rng=np.random.default_rng(seed)
    tasks,sup=gen_cohort(n_items,n_edges,120,4,10,rng)
    rngv=np.random.default_rng(seed+999); rngq=np.random.default_rng(seed+777)
    v3=score(V3(n_items,sigma_a=0.30,beta=12.0),tasks,n_items,sup,0.08,rngv)
    ql=score(QLearningControl(n_items),tasks,n_items,sup,0.08,rngq)
    results.append(dict(n_items=n_items,n_edges=n_edges,seed=seed,v3=v3,qlearning=ql))
    Path(ROOT/'outputs'/'cgr_v3_heldout_blind'/'heldout_report.json').parent.mkdir(parents=True,exist_ok=True)
    (ROOT/'outputs'/'cgr_v3_heldout_blind'/'heldout_report.json').write_text(json.dumps(results,indent=2),encoding='utf-8')
print('done')

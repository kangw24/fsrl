"""CGR-v3 vs the 77 real Liu humans: detailed pair-by-pair, person-by-person
comparison in abstract position space (the public CSV is already aligned to the
A..H low-to-high order, so film_index == abstract position). Only sigma_a was
tuned (to the bimodality COUNT 15/28); which-pairs-bimodal, per-pair accuracy,
self-consistency distribution and inter-subject similarity are NON-tuned
predictions checked here against the real humans we already have.
"""
import sys, json, csv
from pathlib import Path
from itertools import combinations
import numpy as np
from scipy.stats import kendalltau, beta as beta_dist, pearsonr

ROOT=Path('.').resolve(); sys.path.insert(0,str(ROOT)); sys.path.insert(0,str(ROOT/'scripts'))
from fsrl.task.liu2026 import build_liu2026_symbolic_subject_tasks
from fsrl.model.constructive_global_rank import ConstructiveGlobalRankCompression as V3

N=8; ALLPP=list(combinations(range(N),2))
SUP=set([(0,5),(1,2),(1,4),(2,6),(3,5),(3,6),(4,7),(0,7)])

def load_human():
    subj={}
    for fn in ['preregistered_experiment_data.csv','replication_experiment_data.csv']:
        p=ROOT/'outputs'/'liu2026_human_audit'/'raw'/fn
        with open(p,encoding='utf-8-sig',newline='') as f:
            for row in csv.DictReader(f):
                s=int(row['id']); a=int(row['film_index_1'])-1; b=int(row['film_index_2'])-1
                ch=int(row['film_choose_index'])-1; p_,q_=sorted((a,b))
                correct = (ch==max(a,b))
                subj.setdefault(s,{}).setdefault((p_,q_),[]).append(int(correct))
    # per-subject per-pair accuracy (fraction chose higher over 10 blocks)
    out={}
    for s,d in subj.items():
        out[s]={pq:float(np.mean(d[pq])) for pq in ALLPP}
    return out

def fit_beta(arr):
    arr=np.clip(np.asarray(arr,float),0.01,0.99)
    try: a,b,_,_=beta_dist.fit(arr,floc=0,fscale=1.0)
    except: return None
    return 'bimodal' if (a<1 and b<1) else ('unimodal' if (a>1 and b>1) else ('high_accuracy' if (a>1 and b<1) else 'other'))

def metrics(subj_acc):
    # subj_acc: dict subject -> {pair: accuracy}
    subjects=list(subj_acc.keys()); n=len(subjects)
    pair_acc={pq:float(np.mean([subj_acc[s][pq] for s in subjects])) for pq in ALLPP}
    bimodal=set(); near_bimodal=set()
    for pq in ALLPP:
        pr=fit_beta([subj_acc[s][pq] for s in subjects])
        if pr=='bimodal':
            bimodal.add(pq)
            if pq[1]-pq[0]<=2: near_bimodal.add(pq)
    # per-subject self-consistency (triads from 28-pair prefs), inter-subject kendall
    max_tri=(N**3-4*N)//24
    rankings=[]; selfc_list=[]; sc_wrong=0; triads_total=0
    for s in subjects:
        pref=[[0.5]*N for _ in range(N)]
        for (p,q) in ALLPP:
            acc=subj_acc[s][(p,q)]  # P(higher q beats p)
            pref[q][p]=acc; pref[p][q]=1-acc
        ntri=0
        for a,b,c in combinations(range(N),3):
            if (pref[a][b]>0.5 and pref[b][c]>0.5 and pref[c][a]>0.5) or (pref[b][a]>0.5 and pref[c][b]>0.5 and pref[a][c]>0.5): ntri+=1
        triads_total+=ntri; selfc_list.append(1-ntri/max_tri)
        # reconstruct ranking: score[p]=sum_q(pref[p][q]-0.5); strong->weak
        score=[sum(pref[p][q]-0.5 for q in range(N) if q!=p) for p in range(N)]
        ranking=tuple(np.argsort(-np.array(score),kind='mergesort'))  # positions strong->weak
        rankings.append(ranking)
        if ntri==0 and ranking!=tuple(range(N-1,-1,-1)): sc_wrong+=1  # true strong->weak = (7,6,...,0)
    inter=[kendalltau(rankings[i],rankings[j])[0] for i in range(n) for j in range(i+1,n)]
    inter=[x for x in inter if not np.isnan(x)]
    dist={}
    for pq in ALLPP: dist.setdefault(pq[1]-pq[0],[]).append(pair_acc[pq])
    distance={str(d):float(np.mean(v)) for d,v in sorted(dist.items())}
    posacc={p:[] for p in range(N)}
    for (p,q) in ALLPP: posacc[p].append(pair_acc[(p,q)]); posacc[q].append(pair_acc[(p,q)])
    serial={str(p):float(np.mean(posacc[p])) for p in range(N)}
    learned=[pair_acc[pq] for pq in ALLPP if pq in SUP]; unlearned=[pair_acc[pq] for pq in ALLPP if pq not in SUP]
    return dict(n=n, pair_acc=pair_acc, bimodal=sorted(bimodal), near_bimodal=sorted(near_bimodal),
                n_bimodal=len(bimodal), selfc_mean=float(np.mean(selfc_list)), selfc_list=selfc_list,
                triads_total=triads_total, sc_wrong_frac=sc_wrong/n, inter_mean=float(np.mean(inter)),
                inter_list=inter, distance=distance, serial=serial,
                overall=float(np.mean(list(pair_acc.values()))), learned=float(np.mean(learned)), unlearned=float(np.mean(unlearned)))

# humans
H=metrics(load_human())
# v3 virtual cohort, n=77 to match
tasks=build_liu2026_symbolic_subject_tasks(n_subjects=77,rng=np.random.default_rng(971001),nbcues=8,randomize_true_rank=True,include_magnitude=True)
m=V3(8,sigma_a=0.30,beta=12.0)
rng=np.random.default_rng(971001+999)
v3_acc={}
for t in tasks:
    run=m.run_subject(t,rng)
    perc={c:(N-1-r) for r,c in enumerate(run.order_strong_to_weak)}
    sa={}
    for (p,q) in ALLPP:
        cue_p,cue_q=t.true_rank[N-1-p],t.true_rank[N-1-q]  # cue at position p (low), q (high)
        hi,lo=cue_q,cue_p; a,b=(hi,lo) if hi<lo else (lo,hi)
        p_hi=run.choice_prob[(a,b)] if hi==a else 1-run.choice_prob[(a,b)]
        pc=(0.08)*0.5+(0.92)*p_hi
        sa[(p,q)]=float(rng.binomial(10,pc))/10.0
    v3_acc[t.subject_index]=sa
V=metrics(v3_acc)

# compare
pa_h=np.array([H['pair_acc'][pq] for pq in ALLPP]); pa_v=np.array([V['pair_acc'][pq] for pq in ALLPP])
r,_=pearsonr(pa_h,pa_v)
bim_j=len(set(H['bimodal'])&set(V['bimodal']))/len(set(H['bimodal'])|set(V['bimodal'])) if (set(H['bimodal'])|set(V['bimodal'])) else 0
def s(o):
    if isinstance(o,dict): return {str(k):s(v) for k,v in o.items()}
    if isinstance(o,(list,tuple)): return [s(x) for x in o]
    return o
rep=dict(comparison=dict(
    per_pair_accuracy_pearson_r=float(r),
    bimodal_pair_jaccard=float(bim_j),
    human_bimodal_pairs=[list(x) for x in H['bimodal']], v3_bimodal_pairs=[list(x) for x in V['bimodal']],
    human_n_bimodal=H['n_bimodal'], v3_n_bimodal=V['n_bimodal'],
    human_selfc_mean=H['selfc_mean'], v3_selfc_mean=V['selfc_mean'],
    human_inter_mean=H['inter_mean'], v3_inter_mean=V['inter_mean'],
    human_scwrong_frac=H['sc_wrong_frac'], v3_scwrong_frac=V['sc_wrong_frac'],
    human_overall=H['overall'], v3_overall=V['overall'],
    human_learned=H['learned'], v3_learned=V['learned'],
    human_unlearned=H['unlearned'], v3_unlearned=V['unlearned'],
    human_triads_total=H['triads_total'], v3_triads_total=V['triads_total'],
    human_distance=H['distance'], v3_distance=V['distance'],
    human_serial=H['serial'], v3_serial=V['serial'],
    human_pair_acc={str(pq):H['pair_acc'][pq] for pq in ALLPP},
    v3_pair_acc={str(pq):V['pair_acc'][pq] for pq in ALLPP}))
out=ROOT/'outputs'/'cgr_v3_vs_liu_humans'/'comparison_report.json'
out.parent.mkdir(parents=True,exist_ok=True)
out.write_text(json.dumps(s(rep),indent=2),encoding='utf-8')
print('=== CGR-v3 vs 77 real Liu humans (abstract position space) ===')
print('per-pair accuracy pearson r = %.3f'%r)
print('bimodal pairs: human %d %s | v3 %d %s | jaccard %.2f'%(H['n_bimodal'],H['bimodal'],V['n_bimodal'],V['bimodal'],bim_j))
print('self-consistency mean: human %.3f | v3 %.3f'%(H['selfc_mean'],V['selfc_mean']))
print('self-consistent-but-wrong frac: human %.3f | v3 %.3f'%(H['sc_wrong_frac'],V['sc_wrong_frac']))
print('inter-subject kendall mean: human %.3f | v3 %.3f'%(H['inter_mean'],V['inter_mean']))
print('overall acc: human %.3f | v3 %.3f | learned %.3f/%.3f | unlearned %.3f/%.3f'%(H['overall'],V['overall'],H['learned'],V['learned'],H['unlearned'],V['unlearned']))
print('total circular triads: human %d | v3 %d'%(H['triads_total'],V['triads_total']))
print('distance effect human:',{k:round(v,2) for k,v in H['distance'].items()})
print('distance effect v3   :',{k:round(v,2) for k,v in V['distance'].items()})
print('report',out)

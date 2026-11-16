#!/usr/bin/env python3
"""Figure 2 of the paper (dose-response, privacy families). Run scripts/reproduce_paper_numbers.py first."""
import sys,os,warnings; warnings.filterwarnings('ignore')
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
os.environ.setdefault('STB_FROZEN_CHECKOUT',str(ROOT/'vendor'/'frozen_v030'))
import numpy as np, pandas as pd, matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
from stb import analysis
from stb.bases import Workspace
from stb.plan import full_plan
from statsmodels.stats.proportion import proportion_confint
PLAN=full_plan()
plt.rcParams.update({'font.size':7,'font.family':'serif','axes.linewidth':0.6,'xtick.major.width':0.5,'ytick.major.width':0.5})
fig,axes=plt.subplots(1,2,figsize=(3.4,1.65),sharey=True)
cols={'copy_random':'#1f5fa8','copy_outlier':'#d1495b','near_copy':'#2a9d55'}
lods={'cdc':{'copy_random':812,'copy_outlier':812},'uci':{'copy_random':3,'copy_outlier':3,'near_copy':1145}}
for ax,ds,title in zip(axes,['cdc','uci'],['CDC ($\\tilde n=40{,}589$)','UCI ($\\tilde n=11{,}443$)']):
    ws=Workspace(ds,str(ROOT/'results'/'evidence'/'stb-v0.5-full'),PLAN.data_scale); o=analysis.outcomes_frame(ws.store)
    o=o[(o.matrix=='honest')&(o.purpose=='estimation')&(o.rows>0)]
    for fam,c in cols.items():
        g=o[o.family==fam].groupby('rows').detected.agg(['sum','size']).reset_index()
        lo,hi=proportion_confint(g['sum'],g['size'],method='wilson')
        r=g['sum']/g['size']
        ax.errorbar(g.rows,r,yerr=[r-lo,hi-r],color=c,marker='o',ms=2,lw=0.8,elinewidth=0.5,capsize=0,label=fam.replace('_','\\_') if False else fam.replace('_',' '))
        if fam in lods[ds]: ax.axvline(lods[ds][fam],color=c,ls=':',lw=0.7)
    ax.set_xscale('log'); ax.set_title(title,fontsize=7); ax.axhline(0.95,color='0.6',lw=0.5,ls='--')
    ax.set_xlabel('copied rows'); ax.set_ylim(-0.03,1.03)
axes[0].set_ylabel('detection rate')
axes[0].legend(frameon=False,fontsize=5.5,loc='upper left')
fig.tight_layout(pad=0.4,w_pad=0.6)
fig.savefig(str(ROOT/'results'/'figures'/'fig_dose.pdf'),bbox_inches='tight',pad_inches=0.02); fig.savefig(str(ROOT/'results'/'figures'/'fig_dose.png'),dpi=200,bbox_inches='tight',pad_inches=0.02)

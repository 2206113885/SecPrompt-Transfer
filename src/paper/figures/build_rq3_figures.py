"""RQ3 publication figures from independently audited, frozen CSVs.

uv run --isolated --no-project --with numpy==2.5.4 --with matplotlib==3.11.2 \
    python figures/build_rq3_figures.py
Add --only-transfers to preserve the separately presented error-tradeoff figure.
Use --only-behavior to rebuild only the error-tradeoff figure.
No estimation or model calls occur here. Printed width matches sn-jnl.cls.
"""
from pathlib import Path
import argparse
import csv
import json
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import orx_figstyle as style

HERE = Path(__file__).resolve().parent
DATA = HERE.parent / 'data' / 'rq3-systematic'
WIDTH = 372 / 72.27
style.TEXT = WIDTH
style.use_style()
mpl.rcParams.update({'font.size': 8, 'axes.labelsize': 8, 'xtick.labelsize': 8,
                     'ytick.labelsize': 8, 'legend.fontsize': 8, 'savefig.dpi': 300})
MODELS = ['llama3_8b','qwen2_7b','granite3_2_8b','falcon3_7b',
          'ministral_8b','granite3_3_8b','qwen3_8b']
DISPLAY = ['Llama-3', 'Qwen2', 'Granite-3.2', 'Falcon3',
           'Ministral', 'Granite-3.3', 'Qwen3']
SHORT = ['L3', 'Q2', 'G32']
FAMILIES = [(p,o) for p in MODELS[:3] for o in ('pe2','btape')]
LABELS = [SHORT[MODELS.index(p)] + ' / ' + ('PE2' if o=='pe2' else 'BT-APE') for p,o in FAMILIES]
BLUE = style.PALETTE['blue']
ADVERSE = style.PALETTE['red']
NEUTRAL = '#555555'
COLORS = ['#0072B2', '#D55E00', '#009E73']
TEXT_COLORS = ['#00649D', '#9B4400', '#007654']
TINTS = ['#F0F6FA', '#FBF4EE', '#EEF8F4']
HEADER = '#245C78'

def read(name):
    with (DATA / name).open() as f:
        return list(csv.DictReader(f))

G = {(r['view'],r['deployer'],r['producer'],r['optimizer'],r['metric']):r for r in read('group_intervals.csv')}
C = read('cell_effects.csv')
B = {(r['view'],r['deployer'],r['metric']):r for r in read('baseline_intervals.csv')}
AUDITS = {}
PLACEMENTS = []
GEOMETRY = {}

def export(fig, name):
    issues = style._audit(fig)
    AUDITS[name] = issues
    if issues:
        raise AssertionError((name, issues))
    style.save(fig, str(HERE/name), formats=('pdf','svg','png'))

def forest(view):
    # Separate optimizer pages retain a common x scale and native 8 pt type.
    # Greedy packing moves split marks ONLY vertically, below their group mean.
    left, right, top, bottom = 86, 40, 48, 31  # native physical points
    plot_width = WIDTH * 72 - left - right
    for j, o in enumerate(['pe2', 'btape']):
        groups, blocks, cursor = [], [], 4.0
        for di, d in enumerate(MODELS):
            start = cursor
            for fi, p in enumerate(MODELS[:3]):
                r = G[view, d, p, o, 'macro_f1']
                vals = sorted([c for c in C if c['view']==view and c['deployer']==d
                               and c['producer']==p and c['optimizer']==o
                               and c['metric']=='macro_f1'], key=lambda c:int(c['seed']))
                assert len(vals)==5
                assert abs(np.mean([float(c['gain_pp']) for c in vals])-float(r['mean_gain_pp']))<1e-12
                lanes, positions = [], []
                for c in vals:
                    x = float(c['gain_pp'])
                    xp = (x+7.5)*plot_width/40
                    lane = next((i for i, xs in enumerate(lanes)
                                 if all(abs(xp-z)>=4.2 for z in xs)), len(lanes))
                    if lane==len(lanes): lanes.append([])
                    lanes[lane].append(xp)
                    positions.append((c, cursor+5.2+lane*4.2))
                height = 6.4+4.2*len(lanes)
                groups.append((d,p,fi,r,cursor,height,positions))
                cursor += height
            blocks.append((di,d,start,cursor))
            cursor += 3
        body = cursor+1
        height = body+top+bottom
        fig = plt.figure(figsize=(WIDTH,height/72))
        ax = fig.add_axes([left/(WIDTH*72),bottom/height,plot_width/(WIDTH*72),body/height])
        ax.set_ylim(body,0)
        ax.set_xlim(-7.5,32.5);ax.set_xticks([-5,0,10,20,30])
        for di,d,start,end in blocks:
            base=float(B[view,d,'macro_f1']['estimate'])*100
            ypos=(bottom+body-(start+end)/2)/height
            fig.text(.009,ypos,f'{DISPLAY[di]}\n{base:.2f}%',fontsize=8,va='center')
        ys, labels, point_positions, marker_boxes = [], [], [], []
        for d,p,fi,r,y,gh,positions in groups:
            ys.append(y);labels.append(SHORT[fi])
            ax.axhspan(y-2,y+gh-1,color=TINTS[fi],zorder=0)
            for lo,hi,color,lw,alpha,z in [('simultaneous_low_pp','simultaneous_high_pp',COLORS[fi],3.5,.32,2),
                                     ('ci_low_pp','ci_high_pp',COLORS[fi],1.3,1,4)]:
                a,b=float(r[lo]),float(r[hi])
                assert -7.5<=a<=b<=32.5, (view,d,p,o,lo,a,b)
                ax.plot([a,b],[y,y],color=color,lw=lw,alpha=alpha,solid_capstyle='butt',zorder=z)
            ax.scatter(float(r['mean_gain_pp']),y,marker='s' if p==d else 'D',
                       facecolor=COLORS[fi],edgecolor='white',lw=.4,s=15,zorder=5)
            marker_boxes.append((float(r['mean_gain_pp']),y,
                                 np.sqrt(15)*(1 if p==d else np.sqrt(2))+.4))
            # Exact mean is printed in a separate aligned column; data x values,
            # intervals, scale, and five split effects are unchanged.
            fig.text(.987,(bottom+body-y)/height,f"{float(r['mean_gain_pp']):+.2f}",
                     fontsize=8,ha='right',va='center')
            for c,cy in positions:
                x=float(c['gain_pp']); assert -7.5<=x<=32.5
                ax.scatter(x,cy,s=10,facecolor='white',edgecolor=COLORS[fi],lw=.65,zorder=3)
                point_positions.append((x,cy))
                marker_boxes.append((x,cy,np.sqrt(10)+.65))
                PLACEMENTS.append({'view':view,'optimizer':o,'deployer':d,'producer':p,
                                   'seed':int(c['seed']),'gain_pp':x,'y_physical_pt':cy,
                                   'identical_to_baseline':c['identical_to_baseline'],
                                   'mean_y_physical_pt':y})
        ax.set_yticks(ys,labels);ax.tick_params(axis='y',length=0,pad=5)
        for tick,group in zip(ax.get_yticklabels(),groups):
            tick.set_color(TEXT_COLORS[group[2]])
        ax.set_xlabel('Macro-F1 gain (pp)')
        ax.grid(axis='y',visible=False);ax.grid(axis='x',visible=True,linewidth=.4,color='#DCE2E7')
        ax.spines['left'].set_visible(False)
        ax.axvline(0,c='#727A82',ls='--',lw=.75,zorder=1)
        fig.text(left/(WIDTH*72),1-40/height,'(a) PE2' if j==0 else '(b) BT-APE',fontweight='bold',fontsize=8.5,color=HEADER)
        fig.text(.009,1-40/height,'Deployer',fontweight='bold',color=HEADER)
        fig.text(62/(WIDTH*72),1-40/height,'From',fontweight='bold',color=HEADER)
        fig.text(.987,1-40/height,'Mean',ha='right',color=HEADER)
        handles=[Line2D([],[],marker='o',mfc='white',mec=NEUTRAL,ls='',markersize=3.5,label='Source split'),
                 Line2D([],[],marker='D',c=NEUTRAL,ls='',markersize=3.5,label='Model-switch mean'),
                 Line2D([],[],marker='s',c=NEUTRAL,ls='',markersize=3.5,label='Same-origin mean (ref)'),
                 Line2D([],[],c=NEUTRAL,lw=1.3,label='Pointwise 95% CI'),
                 Line2D([],[],c=NEUTRAL,lw=3.5,alpha=.32,label='Simultaneous 95% band')]
        fig.legend(handles=handles[:3],loc='upper center',bbox_to_anchor=(.53,1.005),ncol=3,
                   columnspacing=1,handlelength=1.5,handletextpad=.5)
        fig.legend(handles=handles[3:],loc='upper center',bbox_to_anchor=(.53,1-13/height),ncol=2,
                   columnspacing=1,handlelength=1.5,handletextpad=.5)
        fig.canvas.draw()
        pts=ax.transData.transform(point_positions)*72/fig.dpi
        minimum=min(np.linalg.norm(a-b) for i,a in enumerate(pts) for b in pts[i+1:])
        assert minimum>=np.sqrt(10)+.65, minimum
        assert len(point_positions)==105
        physical = ax.transData.transform([(x,y) for x,y,s in marker_boxes])*72/fig.dpi
        clearance = min(max(abs(a[0]-b[0]), abs(a[1]-b[1]))
                        -(marker_boxes[i][2]+marker_boxes[j][2])/2
                        for i,a in enumerate(physical) for j,b in enumerate(physical)
                        if j>i)
        assert clearance>=.3, clearance
        name = 'rq3_transfer_'+view.lower()+'_'+o
        GEOMETRY[name] = {'split_points':105,'summary_points':21,
                          'minimum_marker_box_clearance_pt':float(clearance)}
        export(fig,name)

def behavior():
    fig,axes=plt.subplots(1,2,figsize=(WIDTH,5.9),sharey=True)
    fig.subplots_adjust(left=.22,right=.985,bottom=.105,top=.89,wspace=.18)
    for j,(metric,xlabel) in enumerate([('security_recall','Security recall change (pp)\n(higher is better)'),
                                      ('false_positive_rate','Non-security FPR change (pp)\n(lower is better)')]):
        ax=axes[j];ys=[];labels=[]
        for di,d in enumerate(MODELS[3:]):
            offset=di*8
            ax.axhspan(offset-.45,offset+5.45,color='#F5F8FA',zorder=0)
            if j==0:
                ax.text(.01,offset-1.05,DISPLAY[di+3],transform=ax.get_yaxis_transform(),
                        fontsize=8.5,fontweight='bold',color=HEADER,va='center',
                        bbox={'facecolor':'white','edgecolor':'none','pad':.2},zorder=6)
            for fi,(p,o) in enumerate(FAMILIES):
                y=offset+fi;ys.append(y);labels.append(LABELS[fi])
                r=G['T_All',d,p,o,metric]
                gain=float(r['mean_gain_pp'])
                favorable = gain > 1e-12 if metric=='security_recall' else gain < -1e-12
                adverse = gain < -1e-12 if metric=='security_recall' else gain > 1e-12
                # This figure encodes screening direction, while the transfer
                # forests encode prompt producer. Keep its existing legend.
                color = BLUE if favorable else ADVERSE if adverse else NEUTRAL
                ax.plot([float(r['ci_low_pp']),float(r['ci_high_pp'])],[y,y],c='#555555',lw=1.3)
                ax.scatter(gain,y,facecolor=color,edgecolor='white',
                           marker='D' if favorable else 'v' if adverse else 'o',
                           s=18,linewidth=.45,zorder=5)
        ax.set_yticks(ys,labels);ax.tick_params(axis='y',length=0,labelleft=j==0)
        ax.set_ylim(30.0,-1.8);ax.set_xlabel(xlabel)
        ax.set_ylabel('Prompt producer / optimizer' if j==0 else '')
        ax.grid(axis='y',visible=False);ax.grid(axis='x',visible=True,linewidth=.4,color='#DCE2E7')
        ax.spines['left'].set_visible(False)
        ax.axvline(0,c='#727A82',ls='--',lw=.75,zorder=1)
    axes[0].set_xlim(-12,48);axes[0].set_xticks([-10,0,20,40])
    axes[1].set_xlim(-8.3,3.1);axes[1].set_xticks([-8,-4,0])
    fig.legend(handles=[Line2D([],[],marker='D',c=BLUE,ls='',label='Favorable mean'),
                        Line2D([],[],marker='v',c=ADVERSE,ls='',label='Adverse mean'),
                        Line2D([],[],c=NEUTRAL,lw=1.3,label='Pointwise 95% CI')],
               loc='upper center',bbox_to_anchor=(.53,1),ncol=3,
               handlelength=1.5,handletextpad=.5,columnspacing=1)
    for ax,label in zip(axes,['(a) Recall','(b) False positives']):
        ax.text(0,1.055,label,transform=ax.transAxes,ha='left',va='bottom',
                fontsize=8.5,fontweight='bold',color=HEADER)
    # The shared y coordinate is named on the left, as in the existing paper.
    export(fig,'rq3_error_tradeoffs')

if __name__=='__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    selection = parser.add_mutually_exclusive_group()
    selection.add_argument('--only-transfers', action='store_true',
                        help='Regenerate the two transfer views without changing the error-tradeoff figure.')
    selection.add_argument('--only-behavior', action='store_true',
                        help='Regenerate only the error-tradeoff figure, preserving both transfer views.')
    args = parser.parse_args()
    audit_path = HERE/'rq3_figure_audits.json'
    if (args.only_transfers or args.only_behavior) and audit_path.exists():
        AUDITS.update(json.loads(audit_path.read_text()))
    if not args.only_behavior:
        forest('T_All');forest('T_NFR')
    if not args.only_transfers:
        behavior()
    audit_path.write_text(json.dumps(AUDITS,indent=2)+'\n')
    if GEOMETRY:
        (HERE/'rq3_marker_checks.json').write_text(json.dumps(GEOMETRY,indent=2)+'\n')
    if PLACEMENTS:
        with (HERE/'rq3_split_placements.csv').open('w',newline='') as f:
            writer=csv.DictWriter(f,fieldnames=PLACEMENTS[0].keys())
            writer.writeheader();writer.writerows(PLACEMENTS)

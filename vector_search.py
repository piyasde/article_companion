""" Vector Search Example.
python vector_search.py [--plots] [--query '...']
Generate Plots with --plots argument
Try new query with --query 'your sentence' like - 'Can sunlight heat water?'

"""
import argparse
import csv
import re
from pathlib import Path
import numpy as np

# Fixed concept dictionary
CONCEPTS = {
    'Solar': ['sunlight','sunshine','solar','photovoltaic'],
    'Electricity': ['electricity','electrical energy','electric','power'],
    'Storage': ['battery','batteries','store','stored energy'],
    'Lighting': ['lamps','lamp','lights','lighting','bulbs','streetlights'],
    'Nighttime': ['night','after sunset','after dark','midnight'],
    'Water': ['water','rainwater'],
    'Cost': ['cost','costs','price','expense'],
    'Maintenance': ['maintenance','cleaning','inspection','maintain'],
}
SENTENCES = [
    'Photovoltaic panels convert sunlight into electricity during the day.',
    'Rechargeable batteries store electrical energy for later use.',
    'Battery banks supply electricity to lamps after sunset.',
    'Solar-powered garden lights use stored energy at night.',
    'Solar thermal collectors use sunshine to heat water.',
    'Electric pumps move water into overhead tanks.',
    'LED bulbs consume less electricity than traditional bulbs.',
    'Outdoor lighting is switched off at midnight to reduce costs.',
    'Lithium batteries need periodic inspection and maintenance.',
    'A replacement inverter can increase installation costs.',
    'Rainwater harvesting reduces demand for municipal water.',
    'Solar panels need regular cleaning to maintain output.',
    'Grid electricity can power streetlights during the night.',
    'A charge controller regulates power flowing from solar panels to batteries.',
    'A maintenance contract covers battery replacement costs.',
]
IDS = [f'S{i:02d}' for i in range(1, len(SENTENCES)+1)]
QUERY = 'How can electricity from sunshine keep lamps running after dark?'

# Encode text for binary concept presence. Word boundaries are given.
PATTERNS = [re.compile(r'\b(?:' + '|'.join(re.escape(w) for w in words) + r')\b', re.I)
            for words in CONCEPTS.values()]

def embed(text):
    return np.array([float(bool(p.search(text))) for p in PATTERNS])

# L2 normalization applied to stored documents
def unit(x):
    x = np.asarray(x, dtype=float)
    length = np.linalg.norm(x, axis=-1, keepdims=True)
    return np.divide(x, length, out=np.zeros_like(x), where=length != 0)

# Exact retrieval - cosine is dot product of unit vectors.
def search(index, q, k=5):
    if np.linalg.norm(q) == 0:
        raise ValueError('No known semantic concepts in query; cosine is undefined.')
    scores = index @ unit(q)
    order = np.argsort(-np.round(scores, 12), kind='stable')
    return scores, order[:k]

# Attention on three sentence vector
def softmax(x):
    e = np.exp(x - np.max(x))
    return e/e.sum()

def attention_demo(D, q):
    K = D[:3]  # identity key projection; three selected sentences
    V = np.eye(3)  # artificial one-hot values for the three facts
    weights = softmax((K @ q)/np.sqrt(K.shape[1]))
    return weights, weights @ V

# Simple pyplot charts, one figure per chart.
def charts(out, D, q, q2, scores, scores2, weights):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    labels = list(CONCEPTS)
    def save(name):
        plt.tight_layout(); plt.savefig(out/name, dpi=150); plt.close()

    plt.figure(figsize=(10,6))
    mat = np.vstack([D,q])
    plt.imshow(mat, cmap='Greys', vmin=0, vmax=1, aspect='auto')
    plt.xticks(range(len(labels)), labels, rotation=30, ha='right')
    plt.yticks(range(len(IDS)+1), IDS+['Query'])
    for i in range(mat.shape[0]):
        for j in range(mat.shape[1]):
            plt.text(j,i,str(int(mat[i,j])),ha='center',va='center',
                     color='white' if mat[i,j] else 'black',fontsize=8)
    plt.title('Binary semantic embedding matrix')
    save('01_embedding_matrix.png')

    order = np.argsort(-np.round(scores,12),kind='stable')
    plt.figure(figsize=(9,5)); plt.bar(np.array(IDS)[order],scores[order])
    plt.ylim(0,1); plt.ylabel('Cosine similarity'); plt.title('Exact retrieval ranking')
    save('02_similarity_ranking.png')

    plt.figure(figsize=(7,4)); angles=np.linspace(0,180,361)
    plt.plot(angles,np.cos(np.deg2rad(angles))); plt.axhline(0,linewidth=.7)
    plt.xlabel('Angle (degrees)'); plt.ylabel('Cosine similarity'); plt.title('Cosine versus angle')
    plt.grid(alpha=.2); save('03_cosine_angle.png')

    # Each document is projected onto query and residual norm.
    # This preserves query angle
    plt.figure(figsize=(7,7))
    for idx in [0,1,2,12]:
        x = np.clip(scores[idx], -1,1); y = np.sqrt(max(0,1-x*x))
        plt.quiver(0,0,x,y,angles='xy',scale_units='xy',scale=1)
        plt.text(x*1.04,y*1.04,IDS[idx],ha='center')
    plt.quiver(0,0,1,0,angles='xy',scale_units='xy',scale=1)
    plt.text(1.02,.02,'Query'); plt.xlim(0,1.22); plt.ylim(0,1.22)
    plt.gca().set_aspect('equal'); plt.xlabel('Projection onto query'); plt.ylabel('Residual magnitude')
    plt.title('Query-relative geometry (not a common 2D embedding)'); plt.grid(alpha=.2)
    save('04_query_geometry.png')

    plt.figure(figsize=(10,5)); x=np.arange(len(IDS)); w=.36
    plt.bar(x-w/2,scores,w,label='Original query'); plt.bar(x+w/2,scores2,w,label='Storage expansion')
    plt.xticks(x,IDS); plt.ylim(0,1); plt.ylabel('Cosine similarity'); plt.legend(); plt.title('Before and after query expansion')
    save('05_query_expansion.png')

    plt.figure(figsize=(7,4)); plt.bar(IDS[:3],weights); plt.ylim(0,1)
    plt.ylabel('Attention weight'); plt.title('QKV: softmax weights')
    save('06_attention.png')

    n=np.array([1000,10000,100000,1000000,10000000],dtype=np.int64)
    plt.figure(figsize=(7,4)); plt.loglog(n,n*384,marker='o'); plt.xlabel('Stored vectors'); plt.ylabel('Coordinate multiplications'); plt.title('Exact scan work, d=384'); plt.grid(alpha=.2)
    save('07_exact_scan_scaling.png')


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--query',default=QUERY); parser.add_argument('--plots',action='store_true')
    parser.add_argument('--out',type=Path,default=Path('vector_lab_output'))
    args=parser.parse_args()
    D=np.stack([embed(s) for s in SENTENCES]); index=unit(D)
    q=embed(args.query)
    try: scores, order=search(index,q,len(IDS))
    except ValueError as e: parser.error(str(e))
    # Controlled query expansion - inject storage as a missing concept.
    q2=q.copy(); q2[list(CONCEPTS).index('Storage')]=1
    scores2, order2=search(index,q2,len(IDS))
    print('Dimensions:', ', '.join(CONCEPTS)); print('Query:',args.query)
    print('Query vector:',q.astype(int)); print('Matrix shape:',D.shape)
    for i in range(len(IDS)): print(IDS[i],D[i].astype(int),SENTENCES[i])
    print('\nOriginal cosine ranking:')
    for i in order: print(f'{IDS[i]}  {scores[i]:.4f}  {SENTENCES[i]}')
    print('\nExpanded query:',q2.astype(int))
    for i in order2[:5]: print(f'{IDS[i]}  {scores2[i]:.4f}  {SENTENCES[i]}')
    weights, output=attention_demo(D,q)
    print('\nSample attention weights:',np.round(weights,4),'output:',np.round(output,4))
    args.out.mkdir(parents=True,exist_ok=True)
    with (args.out/'embeddings_and_scores.csv').open('w',newline='',encoding='utf-8') as f:
        writer=csv.writer(f); writer.writerow(['id','text',*CONCEPTS,'original_cosine','expanded_cosine'])
        for i in range(len(IDS)):
            writer.writerow([IDS[i],SENTENCES[i],*D[i].astype(int),float(scores[i]),float(scores2[i])])
    if args.plots: charts(args.out,D,q,q2,scores,scores2,weights)
    print('\nOutput directory:',args.out.resolve())

if __name__=='__main__':
    main()

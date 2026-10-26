"""Experiment 6: Wholesale customer single-linkage; 500 generated customers."""
import random,math
random.seed(60); centers=[[.15,.65,.1,.18,.2,.12],[.6,.15,.2,.35,.1,.18],[.2,.2,.65,.12,.55,.4]]; rows=[]
for i in range(500):
 c=centers[i%3]; rows.append([max(1,int(random.lognormvariate(math.log(v*1000+1),.35))) for v in c])
cols=['Fresh','Milk','Grocery','Frozen','Detergents_Paper','Delicassen']; n=len(rows); means=[sum(r[j] for r in rows)/n for j in range(6)]; sd=[math.sqrt(sum((r[j]-means[j])**2 for r in rows)/n) or 1 for j in range(6)]; x=[[(v-means[j])/sd[j] for j,v in enumerate(row)] for row in rows]
# Single linkage is equivalent to building the Euclidean minimum spanning tree (Prim), then cutting its longest edges.
used={0}; edges=[]; best=[float('inf')]*n; parent=[-1]*n
for i in range(1,n): best[i]=sum((x[0][k]-x[i][k])**2 for k in range(6)); parent[i]=0
while len(used)<n:
 v=min((i for i in range(n) if i not in used),key=lambda i:best[i]); edges.append((math.sqrt(best[v]),parent[v],v)); used.add(v)
 for j in range(n):
  if j not in used:
   d=sum((x[v][k]-x[j][k])**2 for k in range(6))
   if d<best[j]:best[j]=d;parent[j]=v
# cut two largest MST edges for three clusters
cut={(a,b) for _,a,b in sorted(edges,reverse=True)[:2]}; graph=[[] for _ in range(n)]
for _,a,b in edges:
 if (a,b) not in cut:graph[a].append(b);graph[b].append(a)
labels=[0]*n; cid=0
for s in range(n):
 if labels[s]:continue
 cid+=1; stack=[s];labels[s]=cid
 while stack:
  u=stack.pop()
  for v in graph[u]:
   if not labels[v]:labels[v]=cid;stack.append(v)
print('Wholesale-customer style synthetic data:',n,'rows; features:',', '.join(cols)); print('Single linkage / Euclidean distance; inputs standardized; clusters:',cid)
for c in range(1,cid+1):
 ids=[i for i,z in enumerate(labels) if z==c]; print('Cluster',c,'size',len(ids),'means',[round(sum(rows[i][j] for i in ids)/len(ids),1) for j in range(6)])



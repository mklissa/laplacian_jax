import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
G=nx.Graph()

  

distances = np.load('data/distances.npy')
stickers = np.load('data/index_to_sticker.npy')
action_matrices = np.load('data/full_matrices.npy').astype(int)

# import pdb;pdb.set_trace()
dist = 3
eigen_id = 10
indices = np.where(distances<=dist)[0]

# values = np.load(f'data/eigenvectors_distance{dist}.npy')[:, eigen_id]
# print(values)

def to_str(number):
  return f'{number:.2}'
labels = dict(enumerate(list(map(to_str, values))))

all_states = {}
for i, ind in enumerate(indices):
  G.add_node(i, at=distances[ind])
  all_states[ind] = i

for i, ind in enumerate(indices):
  neighbours = action_matrices[:,ind]
  for nei in neighbours:
    if nei in all_states:
      G.add_edge(i, all_states[nei]) 


plt.figure(figsize=(8,8)) 
# nx.draw(G);plt.show()

# pos = nx.spring_layout(G)
pos = nx.nx_pydot.pydot_layout(G, prog="twopi")

nx.draw_networkx(G, pos=pos, with_labels=0,
  cmap='Reds', node_size=50, font_size=20, font_weight='bold', node_color=values, labels=labels) 


# nx.draw_networkx(G.subgraph(1), pos=pos, with_labels=1,
#   cmap='YlOrRd', node_size=100, font_size=1, font_weight='bold', node_color=values[1])#labels=labels, 


# nx.draw_networkx(G.subgraph(most_distant), pos=pos, with_labels=1,
#   cmap='YlOrRd', node_size=100, font_size=1, font_weight='bold', node_color='red')#labels=labels, 


# # # draw subgraph for highlights
# nx.draw_networkx(G.subgraph(0), pos=pos, node_color='red', node_size=400, labels={0:'yo'}, font_size=7)


# colors = ['orange','blue','grey']
# for dist in range(3,-1,-1):
#   # import pdb;pdb.set_trace()
#   nodes = [x for x,y in G.nodes(data=True) if y['at']==dist]
#   # draw graph
#   node_size= 50 if dist != 3 else 10
#   # nx.draw_networkx(G.subgraph(nodes), pos=pos, font_size=0, node_size=node_size)

#   nx.draw_networkx(G.subgraph(nodes), pos=pos, cmap='YlOrRd', node_size=50, font_size=8,
#            font_weight='bold', with_labels=False, node_color=values[nodes])#labels=labels, 


    
plt.show()


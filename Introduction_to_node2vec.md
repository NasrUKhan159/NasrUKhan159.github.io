# Tutorial: 'Introduction to node2vec'

# Total No of Lines of Code: 374

# Word Count: 2461 Words

## 1. Idea:

The node2vec algorithm is a semi-supervised framework for learning continuous feature representations for nodes in a network $G = (V, E)$ [1].

It was built to generalise prior work which is based on rigid motions of network neighbourhoods. The problem to solve is sampling neighbourhoods of a source node 𝑢 as a form of local search, which is a type of search based optimisation problem [1].

Generally, there are two extreme sampling strategies for generating network neighbourhood set(s): Breadth-first Sampling (BFS) and Depth-first Sampling (DFS). BFS and DFS represent extreme scenarios in terms of the search space they explore. Prediction tasks on nodes in networks often shuttle between homophily and structural equivalence. 

<u>Homophily hypothesis:</u> Highly interconnected nodes that belong to similar network clusters or communities should be embedded closely together. The neighbourhoods sampled by DFS can explore larger parts of the network as it can move further away from the source node, with sample size staying fixed. The sampled nodes more accurately reflect a macro-view of the neighbourhood which is essential in inferring communities based on homophily. However, moving to greater depths leads to complex dependencies since a sampled node may be far from the source and potentially less representative [1]. 

<u>Structural equivalence hypothesis:</u> Nodes with similar structural roles in networks should be embedded closely together. The neighbourhoods sampled by BFS lead to embeddings that correspond closely to structural equivalence. However, a very small portion of the graph is explored for any sample size. 

This tradeoff in performance between DFS and BFS represents the so-called exploration-exploitation tradeoff, which node2vec aims to balance. node2vec can smoothly interpolate between DFS and BFS, by simulating biased random walks that can explore neighbourhoods in BFS and DFS fashion [1]. There are 2 hyperparameters that control the direction of this random walk: $p$ (return hyperparameter) and $q$ (input hyperparameter). Suppose the random walk transitions from some node $t$ to another node $v$. The hyperparameters control the probability of a walk staying inward revisiting nodes (i.e. node $t$), staying close to preceding nodes (denoted by $x_{1}$) or moving outward farther away (denoted by $x_{2}$, $x_{3}$).  

Main applications of node2vec are in multi-label classification and link prediction. In multi-label classification, every node is assigned one or more labels from a countable finite set $L$. In link prediction, we are given a network with a certain fraction of edges removed, and we would like to predict these missing edges [1].  

A potential domain application for node2vec is link prediction in recommendation systems. Problems to solve in recommender systems range from product recommendations, to recommending other users that some user should follow. Since feature representations constructed by node2vec preserves network structure, hence embeddings can effectively quantify the likelihood of there existing an edge between a pair of nodes or not.  

## 2. Base implementation of node2vec, inspired from [5]:


```python
#Import all the necessary libraries and modules
import networkx as nx
import node2vec as N2V
import pandas as pd
import numpy as np
import random
import sklearn
import os
import warnings
warnings.filterwarnings('ignore')
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from itertools import product
from node2vec.edges import AverageEmbedder
from node2vec.edges import HadamardEmbedder
from node2vec.edges import WeightedL1Embedder
from node2vec.edges import WeightedL2Embedder
from sklearn.manifold import TSNE
from scipy.spatial.distance import cityblock
from datetime import datetime
```

'node2vec' is a package created by the authors of [1] that implements node2vec. Hence, this tutorial will be <b>reworking</b> with existing code on node2vec in Python, offering alternative perspectives on what is already out there.


```python
#Baseline graph: Erdos-Renyi Graph with |V| = 100 and Pr(∃ a link from node i to node j) = 1/2 
Erdos_Renyi_Graph = nx.fast_gnp_random_graph(n=100, p=0.5)
```


```python
#Set parameter values in the Node2Vec constructor with additional explanations by me
set_window = 1
set_dimensions = 128 #Embedding dimensions (default: 128)
set_walk_length = 80 #Number of nodes in each biased random walk (default: 80)
set_num_walks = 10 #Number of random walks per node (default: 10)
set_workers = 1 #Number of workers to train the model (default: 1). In multicore machines, this will provide faster training but in Windows machines, it is advised to set workers=1.
set_p = 1 #Pr(random walk goes back to previous node) (i.e. in [1], Pr(walk traverses back from node 𝑣  to node 𝑡 )) (default: 1)
set_q = 1 #probability that a random walk can pass through a previously unseen part of the graph (default: 1).
#Set parameter values in the Node2Vec.fit method - this method constructs the embeddings using the node2vec algorithm, and accepts any argument that can be used in gensim.Word2Vec.
#The argument "size" is not used in node2vec.Node2Vec.fit because "dimensions" has been supplied by the Node2Vec constructor. 
set_vector_size = 16 #Dimensionality of node vectors (in gensim.Word2Vec, is dimensionality of word vectors).
set_min_count = 5 #(Datatype = int) Ignores nodes with total absolute frequency less than this value.
set_window = 8 #(Datatype = int) The maximum distance between the current and predicted node within a walk.
set_sample = 1e-6 #(Datatype = float) The threshold for considering which nodes are randomly downsampled.
set_negative = 7 #(Datatype = int) If this is positive, then negative sampling is used. 
#Its value specifies how many "noise nodes" should be discarded. No negative sampling is used if this is set is 0.
set_batch_words = 4 #Number of words (in node2vec, is nodes instead) passed to worker threads. 'Workers' is automatically passed from the Node2Vec constructor
```


```python
from IPython import display
display.Image("Illustration_of_random_walk_in_node2vec.jpg") #Figure 2 from [1]
```




    
![jpeg](output_8_0.jpg)
    



If we set $p$ to a high value (in particular, $p > max(q, 1)$), then it is less likely to sample an already visited node in the next 2 steps. If we set it to a low value, (in particular, $p < min(q, 1)$), it would keep the walk close to the source node [1]. The parameter $q$ allows the search to differentiate between "inward" and "outward" nodes. If $q > 1$, the random walk is biased towards nodes close to node $t$. Hence, the walk approximates BFS behaviour such that samples consist of nodes in a small locality. If $q < 1$, the walk is more inclined to visit nodes further away from node $t$. Such behaviour is reflective of DFS which encourages outward exploration. 


```python
Erdos_Renyi_Graph_embedded = N2V.Node2Vec(Erdos_Renyi_Graph, dimensions=set_dimensions, walk_length = set_walk_length, num_walks = set_num_walks, p = set_p, q = set_q, workers = set_workers)
Erdos_Renyi_Graph_Node2Vecmodel = Erdos_Renyi_Graph_embedded.fit(vector_size = set_vector_size, window=set_window, min_count=set_min_count, batch_words=set_batch_words)
```


    Computing transition probabilities:   0%|          | 0/100 [00:00<?, ?it/s]


    Generating walks (CPU: 1): 100%|███████████████████████████████████████████████████████| 10/10 [00:09<00:00,  1.02it/s]
    


```python
input_node = '1' #Set the source node, denoted by u in [1]
for s in Erdos_Renyi_Graph_Node2Vecmodel.wv.most_similar(input_node, topn = 10):
    print(s)
```

    ('60', 0.9441300630569458)
    ('59', 0.9161945581436157)
    ('3', 0.9144318699836731)
    ('38', 0.9087920188903809)
    ('85', 0.905761182308197)
    ('47', 0.9054301381111145)
    ('94', 0.9043151140213013)
    ('58', 0.9032318592071533)
    ('5', 0.8981741666793823)
    ('70', 0.8967902660369873)
    

<u>Original interpretation:</u> The output above provides the respective nodes through which the random walk will go, and the second argument in each tuple is the transition probability. The transition probability is the highest for node '60'. Hence it is most probable for the walk to visit node '60', assuming node '1' is the source node. Transition probabilities have the following definition:

$$
\pi_{vx} = \alpha_{pq}(t, x).w_{vx} $$ where 
$$
\alpha_{pq}(t, x) = 
\begin{cases} 
      \frac{1}{p} & d_{tx} = 0 \\
      1 & d_{tx} = 1 \\
      \frac{1}{q} & d_{tx} = 2
   \end{cases} 
$$

$d_{tx} = $ shortest path distance between nodes $t$ and $x$;  $w_{vx} = $ static edge weights. In the case of unweighted graphs, $w_{vx}$ = 1. Parameters $p$ and $q$ control how fast the biased 2nd order random walk constructed, explores and leaves the neighbourhood of source node $u$.

Let's attempt to embed the edges using different choices of embedders suggested in [1] (Original Work). The different embedding methods suggested are:

<u> Average Embedder: <u> 
$$
\frac{f_{i}(u) + f_{i}(v)}{2} 
$$
    
<u> Hadamard Embedder: <u> 
$$
f_{i}(u) * f_{i}(v) 
$$
    
<u> Weighted-L1 Embedder: <u> 
$$
|f_{i}(u) - f_{i}(v)|
$$

<u> Weighted-L2 Embedder: <u> 
$$
|f_{i}(u) - f_{i}(v)|^{2}
$$    


```python
#Embed edges using different types of embedders
edges_embs_average = AverageEmbedder(keyed_vectors=Erdos_Renyi_Graph_Node2Vecmodel.wv)
edges_embs_hadamard = HadamardEmbedder(keyed_vectors=Erdos_Renyi_Graph_Node2Vecmodel.wv)
edges_embs_weightedl1 = WeightedL1Embedder(keyed_vectors=Erdos_Renyi_Graph_Node2Vecmodel.wv)
edges_embs_weightedl2 = WeightedL2Embedder(keyed_vectors=Erdos_Renyi_Graph_Node2Vecmodel.wv)
```


```python
edges_embs_average[('1', '2')] #find embeddings between node '1' and node '2' using AverageEmbedder
```




    array([-0.13727346,  0.09686343,  0.38122067,  0.29982412,  0.32796603,
            0.3625239 ,  0.4214275 ,  0.08991597, -0.04503382,  0.17428799,
           -0.12346756, -0.03556038, -0.01880949, -0.11449624, -0.19013706,
            0.27078483], dtype=float32)




```python
edges_embs_hadamard[('1', '2')] #find embeddings between node '1' and node '2' using HadamardEmbedder
```




    array([ 0.01307615, -0.00204751,  0.13946629,  0.08952308,  0.10756172,
            0.12462053,  0.1696693 ,  0.00799231, -0.00590736,  0.02463935,
            0.01524305, -0.03058972, -0.00321151,  0.01292198,  0.02546   ,
            0.06422546], dtype=float32)




```python
edges_embs_weightedl1[('1', '2')] #find embeddings between node '1' and node '2' using WeightedL1Embedder
```




    array([1.51892811e-01, 2.13822722e-01, 1.53139353e-01, 3.85444462e-02,
           1.53034925e-04, 1.64961368e-01, 1.78121775e-01, 1.92424208e-02,
           1.78161755e-01, 1.51485354e-01, 2.17889249e-03, 3.56955260e-01,
           1.19420364e-01, 2.73791105e-02, 2.06805244e-01, 1.90776899e-01],
          dtype=float32)




```python
edges_embs_weightedl2[('1', '2')] #find embeddings between node '1' and node '2' using WeightedL2Embedder
```




    array([2.3071427e-02, 4.5720156e-02, 2.3451662e-02, 1.4856743e-03,
           2.3419688e-08, 2.7212253e-02, 3.1727366e-02, 3.7027075e-04,
           3.1741612e-02, 2.2947812e-02, 4.7475723e-06, 1.2741706e-01,
           1.4261223e-02, 7.4961566e-04, 4.2768408e-02, 3.6395825e-02],
          dtype=float32)



<u> Interpretation (Original):</u> Node embeddings are used to map nodes to $\mathbb{R}^{d}$ where $d$ = set_vector_size variable defined above (i.e. dim(node vectors)). These embeddings are continuous feature representations of nodes in $G$, and preserve knowledge gained in the network. With the help of knowledge from these embeddings, if these embeddings are plotted in a 2-dimensional graph using dimensionality reduction, the distance of 2 nodes in the actual graph would be the same as the distance between 2 nodes in the low-dimensional graph. Plus, we can use these embeddings for node/link prediction tasks.

## 3. Training runs on real-world network data and analysis of output:

### a. Application of node2vec on a real-world network dataset (Original Work):

Instead of checking on a simulated, standard network, let's run node2vec on a real-world network dataset. Let's use the network of American football games between Division IA colleges during regular season [8]. This has labelled nodes, with the label being a certain college. Here, $|V| = 115$, $|E| = 613$.


```python
football_network = nx.read_gml('football.gml')
```


```python
football_network.nodes() #Let's view what the set of nodes inside the NodeView() object looks like
```




    NodeView(('BrighamYoung', 'FloridaState', 'Iowa', 'KansasState', 'NewMexico', 'TexasTech', 'PennState', 'SouthernCalifornia', 'ArizonaState', 'SanDiegoState', 'Baylor', 'NorthTexas', 'NorthernIllinois', 'Northwestern', 'WesternMichigan', 'Wisconsin', 'Wyoming', 'Auburn', 'Akron', 'VirginiaTech', 'Alabama', 'UCLA', 'Arizona', 'Utah', 'ArkansasState', 'NorthCarolinaState', 'BallState', 'Florida', 'BoiseState', 'BostonCollege', 'WestVirginia', 'BowlingGreenState', 'Michigan', 'Virginia', 'Buffalo', 'Syracuse', 'CentralFlorida', 'GeorgiaTech', 'CentralMichigan', 'Purdue', 'Colorado', 'ColoradoState', 'Connecticut', 'EasternMichigan', 'EastCarolina', 'Duke', 'FresnoState', 'OhioState', 'Houston', 'Rice', 'Idaho', 'Washington', 'Kansas', 'SouthernMethodist', 'Kent', 'Pittsburgh', 'Kentucky', 'Louisville', 'LouisianaTech', 'LouisianaMonroe', 'Minnesota', 'MiamiOhio', 'Vanderbilt', 'MiddleTennesseeState', 'Illinois', 'MississippiState', 'Memphis', 'Nevada', 'Oregon', 'NewMexicoState', 'SouthCarolina', 'Ohio', 'IowaState', 'SanJoseState', 'Nebraska', 'SouthernMississippi', 'Tennessee', 'Stanford', 'WashingtonState', 'Temple', 'Navy', 'TexasA&M', 'NotreDame', 'TexasElPaso', 'Oklahoma', 'Toledo', 'Tulane', 'Mississippi', 'Tulsa', 'NorthCarolina', 'UtahState', 'Army', 'Cincinnati', 'AirForce', 'Rutgers', 'Georgia', 'LouisianaState', 'LouisianaLafayette', 'Texas', 'Marshall', 'MichiganState', 'MiamiFlorida', 'Missouri', 'Clemson', 'NevadaLasVegas', 'WakeForest', 'Indiana', 'OklahomaState', 'OregonState', 'Maryland', 'TexasChristian', 'California', 'AlabamaBirmingham', 'Arkansas', 'Hawaii'))




```python
#Applying the node2vec algorithm
#Set parameter values in the Node2Vec constructor
set_window = 1
set_dimensions = 128
set_walk_length = 80
set_num_walks = 10
set_workers = 1
set_p = 1
set_q = 1
#Set parameter values in the Node2Vec.fit method
set_vector_size = 16
set_min_count = 5
set_window = 8
set_sample = 1e-6
set_negative = 7
set_batch_words = 4
```


```python
football_network_embedded = N2V.Node2Vec(football_network, dimensions=set_dimensions, walk_length = set_walk_length, num_walks = set_num_walks, p = set_p, q = set_q, workers = set_workers)
football_network_embedded_Node2Vecmodel = football_network_embedded.fit(vector_size = set_vector_size, window=set_window, min_count=set_min_count, batch_words=set_batch_words)
input_node = 'FloridaState' #Set the source node, denoted by u in [1]
for s in football_network_embedded_Node2Vecmodel.wv.most_similar(input_node, topn = 2):
    print(s)
```


    Computing transition probabilities:   0%|          | 0/115 [00:00<?, ?it/s]


    Generating walks (CPU: 1): 100%|███████████████████████████████████████████████████████| 10/10 [00:09<00:00,  1.04it/s]
    

    ('Virginia', 0.9596551060676575)
    ('WakeForest', 0.9404214024543762)
    

We can see that biased random walks constructed by node2vec suggest Virginia and WakeForest as most similar nodes to Florida State University. Next, let's find the embeddings generated. Empirically, it has been shown that the Hadamard embedder does better than the other embedders [1] (it provides the highest AUC (Area Under Curve) score, which is a popular model evaluation metric in network science). The Hadamard embedder, when used with node2vec is highly stable and gives the best average performance across all networks [1]. Thus, let's use the Hadamard embedder:


```python
edges_embs_football_network_hadamard = HadamardEmbedder(keyed_vectors=football_network_embedded_Node2Vecmodel.wv)
edges_embs_football_network_hadamard[('FloridaState', 'Virginia')] #Embeddings between similar nodes 'FloridaState' and 'Virginia'
```




    array([0.24256386, 0.00271838, 0.16682932, 0.6778495 , 0.53370416,
           0.16753244, 0.24878754, 0.21180706, 0.03213182, 0.12138209,
           0.04554601, 0.00193814, 0.02445841, 0.02480539, 0.19350165,
           0.04409627], dtype=float32)




```python
edges_embs_football_network_hadamard[('FloridaState', 'Maryland')] #Embeddings between similar nodes 'FloridaState' and 'Maryland'
```




    array([ 0.21493095, -0.03250636,  0.11127021,  0.75095475,  0.5513734 ,
            0.13248496,  0.30655277,  0.15666462,  0.12820445,  0.0962764 ,
            0.01018562, -0.00302646,  0.03363754,  0.03376811,  0.18214588,
            0.03349097], dtype=float32)



<u>Next step:</u> Visualise the embeddings using a dimensionality reduction procedure since set_dimensions = 128. Let's use the t-SNE procedure to visualise embeddings in 2 dimensions. We use t-SNE instead of PCA because PCA is a linear dimensionality reduction technique, while t-SNE is non-linear. While PCA attempts to maintain global structure, t-SNE tries to preserve the local structure of data (preserves neighbourhoods of points), and this is more important when dealing with biased random walks and node2vec.


```python
# Retrieve node embeddings and corresponding subjects
node_ids_football_network = football_network_embedded_Node2Vecmodel.wv.index_to_key  # list of node IDs
node_embeddings_football_network = football_network_embedded_Node2Vecmodel.wv.vectors  # numpy.ndarray of size number of nodes times embeddings dimensionality
node_targets_football_network = [list(football_network.nodes()).index(node_id) for node_id in node_ids_football_network]
```


```python
#Applying t-SNE transformation on node embeddings
tsne_football_network = TSNE(n_components=2)
node_embeddings_2d_football_network = tsne_football_network.fit_transform(node_embeddings_football_network)
alpha = 0.9
label_map_football_network = {l: i for i, l in enumerate(np.unique(node_targets_football_network))}
node_colours_football_network = [label_map_football_network[target] for target in node_targets_football_network]
plt.figure(figsize=(10,8))
plt.scatter(node_embeddings_2d_football_network[:,0],
            node_embeddings_2d_football_network[:,1],
            c=node_colours_football_network, cmap="jet", alpha=alpha)
```




    <matplotlib.collections.PathCollection at 0x1ead7e80888>




    
![png](output_35_1.png)
    



```python
#Check model accuracy by first getting training and target data
football_network_unique_nodes = list(football_network.nodes())
football_network_all_possible_edges = [(x,y) for (x,y) in product(football_network_unique_nodes, football_network_unique_nodes)]
football_network_edge_features = [(football_network_embedded_Node2Vecmodel.wv.get_vector(str(i)) + football_network_embedded_Node2Vecmodel.wv.get_vector(str(j))) for i,j in football_network_all_possible_edges]
football_network_edges = list(football_network.edges())
football_network_is_con = [1 if e in football_network_edges else 0 for e in football_network_all_possible_edges]
football_network_X = np.array(football_network_edge_features)
football_network_y = football_network_is_con
#Carry out the train and test procedure
football_network_X_train, football_network_X_test, football_network_y_train, football_network_y_test = train_test_split(football_network_X, football_network_y, train_size=0.75, test_size=None, random_state=42)
football_network_clf = LogisticRegression().fit(football_network_X_train, football_network_y_train, sample_weight=None) #Fit a logistic regression classifier
football_network_y_pred = football_network_clf.predict(football_network_X_test)
accuracy_score(football_network_y_test, football_network_y_pred) #print out the accuracy score of embeddings
```




    0.9567583912912004



We can see a high model accuracy value for the embeddings construcuted.


```python
#Applying embeddings to the link prediction problem by first constructing an embedding dataframe (Original work)
embs_football_network_df = (pd.DataFrame([football_network_embedded_Node2Vecmodel.wv.get_vector(str(n)) for n in football_network.nodes()], index = football_network.nodes))
```


```python
embs_football_network_df 
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>0</th>
      <th>1</th>
      <th>2</th>
      <th>3</th>
      <th>4</th>
      <th>5</th>
      <th>6</th>
      <th>7</th>
      <th>8</th>
      <th>9</th>
      <th>10</th>
      <th>11</th>
      <th>12</th>
      <th>13</th>
      <th>14</th>
      <th>15</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>BrighamYoung</th>
      <td>-0.920171</td>
      <td>-0.240420</td>
      <td>-0.178673</td>
      <td>0.169650</td>
      <td>0.420931</td>
      <td>0.217141</td>
      <td>0.624761</td>
      <td>0.402967</td>
      <td>-0.572512</td>
      <td>-0.150548</td>
      <td>-0.036374</td>
      <td>-0.478842</td>
      <td>-0.327270</td>
      <td>-0.100827</td>
      <td>-0.043706</td>
      <td>0.037766</td>
    </tr>
    <tr>
      <th>FloridaState</th>
      <td>-0.259201</td>
      <td>-0.260852</td>
      <td>0.392552</td>
      <td>-0.007128</td>
      <td>-0.040312</td>
      <td>0.497239</td>
      <td>0.412986</td>
      <td>0.276857</td>
      <td>-0.608607</td>
      <td>-0.056045</td>
      <td>-0.570975</td>
      <td>-1.154523</td>
      <td>0.022227</td>
      <td>-0.175624</td>
      <td>0.159941</td>
      <td>0.079085</td>
    </tr>
    <tr>
      <th>Iowa</th>
      <td>-0.415826</td>
      <td>0.447975</td>
      <td>0.098402</td>
      <td>0.080347</td>
      <td>0.426913</td>
      <td>0.195986</td>
      <td>0.366542</td>
      <td>-0.539529</td>
      <td>0.913464</td>
      <td>0.197252</td>
      <td>-0.049430</td>
      <td>-0.579480</td>
      <td>0.077162</td>
      <td>-0.271537</td>
      <td>0.086198</td>
      <td>0.248290</td>
    </tr>
    <tr>
      <th>KansasState</th>
      <td>-0.200583</td>
      <td>-0.018226</td>
      <td>-0.247143</td>
      <td>0.649198</td>
      <td>0.822826</td>
      <td>-0.021364</td>
      <td>0.402380</td>
      <td>-0.364053</td>
      <td>0.321298</td>
      <td>0.560104</td>
      <td>-0.287676</td>
      <td>-0.269711</td>
      <td>0.546703</td>
      <td>0.085001</td>
      <td>-0.145931</td>
      <td>0.612907</td>
    </tr>
    <tr>
      <th>NewMexico</th>
      <td>-0.934580</td>
      <td>-0.327070</td>
      <td>-0.545245</td>
      <td>0.325193</td>
      <td>0.563539</td>
      <td>-0.143328</td>
      <td>0.700958</td>
      <td>0.227876</td>
      <td>-0.350130</td>
      <td>0.176721</td>
      <td>0.296234</td>
      <td>-0.265379</td>
      <td>-0.367628</td>
      <td>-0.251309</td>
      <td>-0.036770</td>
      <td>0.247793</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>TexasChristian</th>
      <td>-0.180049</td>
      <td>0.244441</td>
      <td>0.476253</td>
      <td>0.553349</td>
      <td>0.137665</td>
      <td>-0.112399</td>
      <td>0.803288</td>
      <td>-0.028006</td>
      <td>0.114726</td>
      <td>0.808035</td>
      <td>0.237937</td>
      <td>-0.280270</td>
      <td>-0.172389</td>
      <td>-0.518422</td>
      <td>-0.428804</td>
      <td>-0.351169</td>
    </tr>
    <tr>
      <th>California</th>
      <td>-0.409662</td>
      <td>-0.336332</td>
      <td>-0.250450</td>
      <td>0.474633</td>
      <td>0.161987</td>
      <td>-0.219038</td>
      <td>0.471442</td>
      <td>0.542208</td>
      <td>0.599851</td>
      <td>-0.062048</td>
      <td>0.473700</td>
      <td>-0.368282</td>
      <td>-0.175428</td>
      <td>-1.113861</td>
      <td>-0.010982</td>
      <td>0.220088</td>
    </tr>
    <tr>
      <th>AlabamaBirmingham</th>
      <td>0.061806</td>
      <td>-0.132032</td>
      <td>0.348997</td>
      <td>0.473199</td>
      <td>-0.116197</td>
      <td>0.646392</td>
      <td>0.854453</td>
      <td>-0.569560</td>
      <td>-0.198293</td>
      <td>-0.120403</td>
      <td>0.469515</td>
      <td>-0.371444</td>
      <td>0.436671</td>
      <td>0.015648</td>
      <td>-0.172295</td>
      <td>0.407605</td>
    </tr>
    <tr>
      <th>Arkansas</th>
      <td>0.053908</td>
      <td>0.255762</td>
      <td>0.462571</td>
      <td>0.227927</td>
      <td>0.559906</td>
      <td>0.027401</td>
      <td>0.769173</td>
      <td>0.240693</td>
      <td>-0.625720</td>
      <td>-0.354405</td>
      <td>-0.052150</td>
      <td>-0.162078</td>
      <td>0.091654</td>
      <td>-0.402100</td>
      <td>0.409875</td>
      <td>0.777843</td>
    </tr>
    <tr>
      <th>Hawaii</th>
      <td>-0.693254</td>
      <td>0.104831</td>
      <td>0.679338</td>
      <td>0.397309</td>
      <td>0.175059</td>
      <td>-0.058778</td>
      <td>0.613137</td>
      <td>0.325301</td>
      <td>0.406809</td>
      <td>0.596632</td>
      <td>0.115858</td>
      <td>-0.057187</td>
      <td>0.334597</td>
      <td>-0.496361</td>
      <td>-0.302833</td>
      <td>-0.154878</td>
    </tr>
  </tbody>
</table>
<p>115 rows × 16 columns</p>
</div>




```python
#An example of link prediction for a specified college - the generalised function is defined in the next codeblock (Original work)
college = embs_football_network_df.loc['BrighamYoung']
# other colleges are the colleges with which the given college doesn't have a connection
all_nodes = football_network.nodes()
other_nodes = [n for n in all_nodes if n not in list(football_network.adj['BrighamYoung']) + ['BrighamYoung']]
other_colleges = embs_football_network_df[embs_football_network_df.index.isin(other_nodes)]
manhattan_norm = []
for i in range(len(other_colleges)):
    manhattan_norm.append(cityblock(college, other_colleges.iloc[i]))
idx = other_colleges.index.tolist()
idx_sim = dict(zip(idx, manhattan_norm))
idx_sim = sorted(idx_sim.items(), key=lambda x: x[1], reverse=False)
similar_colleges = idx_sim[:4]
colleges = [art[0] for art in similar_colleges]
colleges
```




    ['Nevada', 'NorthCarolinaState', 'Navy', 'WakeForest']



Now that we have an embedding vector for each college, we can use a distance metric to compute the "distance" between colleges. 

<u>Interpretation:</u> Colleges closer to each other should have an edge connecting them. This is a reasonable argument since node2vec preserves the local structure in a network. 

One distance metric that is not viable to use here is the Euclidean distance metric. This is because with high dimensional data (which is the case here), the $L_{1}$ distance metric (Manhattan norm) is preferred and so let's use that (Original):


```python
def predict_links_for_college(G, df, college_name, N):
    '''
    This function will predict the top N links a college should be connected with
    which it is not already connected with in the football network.
    
    params:
        G (NetworkX Graph) : The network used to create the embeddings
        df (DataFrame) : The dataframe with embeddings associated to each college
        college_name (String) : The college interested in
        N (Integer) : The number of recommended links for a given college
        
    returns:
        This function will return a list of colleges the input college should be connected with.
    '''
    #separate target college
    college = embs_football_network_df.loc[college_name]
    # other colleges are the colleges with which the given college doesn't have a connection
    all_nodes = football_network.nodes()
    other_nodes = [n for n in all_nodes if n not in list(football_network.adj[college_name]) + [college_name]]
    other_colleges = embs_football_network_df[embs_football_network_df.index.isin(other_nodes)]
    # compute the Manhattan norm for given college and all other colleges
    manhattan_norm = []
    for i in range(len(other_colleges)):
        manhattan_norm.append(cityblock(college, other_colleges.iloc[i]))
    idx = other_colleges.index.tolist()
    # create a dictionary of distances for this college w.r.t. other colleges
    idx_sim = dict(zip(idx, manhattan_norm))
    idx_sim = sorted(idx_sim.items(), key=lambda x: x[1], reverse=False)
    #Pick the top N colleges that have the smallest Manhattan distance from the specified college
    similar_colleges = idx_sim[:N]
    colleges = [art[0] for art in similar_colleges]
    return colleges
```


```python
#Use predict_links_for_college() to find the top 7 colleges in order, with which Florida State should share a link
predict_links_for_college(G = football_network, df = embs_football_network_df, college_name = 'FloridaState', N = 7)
```




    ['Vanderbilt',
     'Navy',
     'EastCarolina',
     'WestVirginia',
     'Temple',
     'Syracuse',
     'VirginiaTech']



Since $(p,q) = (1,1)$ corresponds to the DeepWalk algorithm case, let's analyse output for different choices of $\{(p,q)\}$: (Original work). Denote the following constraints $\{p > max(q,1), p < min(q,1), q > 1, q < 1\}$ as $\{c_{i}\}_{i=1}^{4}$. Let's consider 1 case where constraints $\{c_{i}\}_{i=3}^{4}$ do not hold, 1 case where constraints $\{c_{i}\}_{i=1}^{2}$ do not hold, and 2 cases where any 1 selection from $\{c_{i}\}_{i=1}^{2}$ and any 1 selection from $\{c_{i}\}_{i=3}^{4}$ can hold, for parsimonious arguments.

Case 2: $p = 0.5, q = 1 (p < min(q,1), q=1)$


```python
set_p = 0.5
set_q = 1
football_network_embedded = N2V.Node2Vec(football_network, dimensions=set_dimensions, walk_length = set_walk_length, num_walks = set_num_walks, p = set_p, q = set_q, workers = set_workers)
football_network_embedded_Node2Vecmodel = football_network_embedded.fit(vector_size = set_vector_size, window=set_window, min_count=set_min_count, batch_words=set_batch_words)
input_node = 'FloridaState' #Set the source node, denoted by u in [1]
most_similar_nodes = []
for s in football_network_embedded_Node2Vecmodel.wv.most_similar(input_node, topn = 2):
    most_similar_nodes.append(s)
#Next, construct embeddings
edges_embs_football_network_hadamard = HadamardEmbedder(keyed_vectors=football_network_embedded_Node2Vecmodel.wv)
FloridaState_Virginia = edges_embs_football_network_hadamard[('FloridaState', 'Virginia')]
FloridaState_Maryland = edges_embs_football_network_hadamard[('FloridaState', 'Maryland')]
# Retrieve node embeddings and corresponding subjects
node_ids_football_network = football_network_embedded_Node2Vecmodel.wv.index_to_key  # list of node IDs
node_embeddings_football_network = football_network_embedded_Node2Vecmodel.wv.vectors  # numpy.ndarray of size number of nodes times embeddings dimensionality
node_targets_football_network = [list(football_network.nodes()).index(node_id) for node_id in node_ids_football_network]
#Apply t-SNE transformation on node embeddings
tsne_football_network = TSNE(n_components=2)
node_embeddings_2d_football_network = tsne_football_network.fit_transform(node_embeddings_football_network)
alpha = 0.9
label_map_football_network = {l: i for i, l in enumerate(np.unique(node_targets_football_network))}
node_colours_football_network = [label_map_football_network[target] for target in node_targets_football_network]
plt.figure(figsize=(10,8))
plt.scatter(node_embeddings_2d_football_network[:,0],
            node_embeddings_2d_football_network[:,1],
            c=node_colours_football_network, cmap="jet", alpha=alpha)
```


    Computing transition probabilities:   0%|          | 0/115 [00:00<?, ?it/s]


    Generating walks (CPU: 1): 100%|███████████████████████████████████████████████████████| 10/10 [00:10<00:00,  1.04s/it]
    




    <matplotlib.collections.PathCollection at 0x1ead41d0848>




    
![png](output_45_3.png)
    



```python
#Link prediction in Case 2
embs_football_network_df = (pd.DataFrame([football_network_embedded_Node2Vecmodel.wv.get_vector(str(n)) for n in football_network.nodes()], index = football_network.nodes))
predict_links_for_college(G = football_network, df = embs_football_network_df, college_name = 'FloridaState', N = 7)
```




    ['MississippiState',
     'Georgia',
     'SouthCarolina',
     'Syracuse',
     'Mississippi',
     'Temple',
     'VirginiaTech']



Case 3: $p = 1, q = 0.5 (p = max(q,1), q < 1)$


```python
set_p = 1
set_q = 0.5
football_network_embedded = N2V.Node2Vec(football_network, dimensions=set_dimensions, walk_length = set_walk_length, num_walks = set_num_walks, p = set_p, q = set_q, workers = set_workers)
football_network_embedded_Node2Vecmodel = football_network_embedded.fit(vector_size = set_vector_size, window=set_window, min_count=set_min_count, batch_words=set_batch_words)
input_node = 'FloridaState' #Set the source node, denoted by u in [1]
most_similar_nodes = []
for s in football_network_embedded_Node2Vecmodel.wv.most_similar(input_node, topn = 2):
    most_similar_nodes.append(s)
#Next, construct embeddings
edges_embs_football_network_hadamard = HadamardEmbedder(keyed_vectors=football_network_embedded_Node2Vecmodel.wv)
FloridaState_Virginia = edges_embs_football_network_hadamard[('FloridaState', 'Virginia')]
FloridaState_Maryland = edges_embs_football_network_hadamard[('FloridaState', 'Maryland')]
# Retrieve node embeddings and corresponding subjects
node_ids_football_network = football_network_embedded_Node2Vecmodel.wv.index_to_key  # list of node IDs
node_embeddings_football_network = football_network_embedded_Node2Vecmodel.wv.vectors  # numpy.ndarray of size number of nodes times embeddings dimensionality
node_targets_football_network = [list(football_network.nodes()).index(node_id) for node_id in node_ids_football_network]
#Apply t-SNE transformation on node embeddings
tsne_football_network = TSNE(n_components=2)
node_embeddings_2d_football_network = tsne_football_network.fit_transform(node_embeddings_football_network)
alpha = 0.9
label_map_football_network = {l: i for i, l in enumerate(np.unique(node_targets_football_network))}
node_colours_football_network = [label_map_football_network[target] for target in node_targets_football_network]
plt.figure(figsize=(10,8))
plt.scatter(node_embeddings_2d_football_network[:,0],
            node_embeddings_2d_football_network[:,1],
            c=node_colours_football_network, cmap="jet", alpha=alpha)
```


    Computing transition probabilities:   0%|          | 0/115 [00:00<?, ?it/s]


    Generating walks (CPU: 1): 100%|███████████████████████████████████████████████████████| 10/10 [00:11<00:00,  1.19s/it]
    




    <matplotlib.collections.PathCollection at 0x1ead41d1548>




    
![png](output_48_3.png)
    



```python
#Link prediction in Case 3
embs_football_network_df = (pd.DataFrame([football_network_embedded_Node2Vecmodel.wv.get_vector(str(n)) for n in football_network.nodes()], index = football_network.nodes))
predict_links_for_college(G = football_network, df = embs_football_network_df, college_name = 'FloridaState', N = 7)
```




    ['MississippiState',
     'Georgia',
     'SouthCarolina',
     'Syracuse',
     'Mississippi',
     'Temple',
     'VirginiaTech']



Case 4: $p = 4, q = 2 (p > max(q,1), q > 1)$


```python
set_p = 4
set_q = 2
football_network_embedded = N2V.Node2Vec(football_network, dimensions=set_dimensions, walk_length = set_walk_length, num_walks = set_num_walks, p = set_p, q = set_q, workers = set_workers)
football_network_embedded_Node2Vecmodel = football_network_embedded.fit(vector_size = set_vector_size, window=set_window, min_count=set_min_count, batch_words=set_batch_words)
input_node = 'FloridaState' #Set the source node, denoted by u in [1]
most_similar_nodes = []
for s in football_network_embedded_Node2Vecmodel.wv.most_similar(input_node, topn = 2):
    most_similar_nodes.append(s)
#Next, construct embeddings
edges_embs_football_network_hadamard = HadamardEmbedder(keyed_vectors=football_network_embedded_Node2Vecmodel.wv)
FloridaState_Virginia = edges_embs_football_network_hadamard[('FloridaState', 'Virginia')]
FloridaState_Maryland = edges_embs_football_network_hadamard[('FloridaState', 'Maryland')]
# Retrieve node embeddings and corresponding subjects
node_ids_football_network = football_network_embedded_Node2Vecmodel.wv.index_to_key  # list of node IDs
node_embeddings_football_network = football_network_embedded_Node2Vecmodel.wv.vectors  # numpy.ndarray of size number of nodes times embeddings dimensionality
node_targets_football_network = [list(football_network.nodes()).index(node_id) for node_id in node_ids_football_network]
#Apply t-SNE transformation on node embeddings
tsne_football_network = TSNE(n_components=2)
node_embeddings_2d_football_network = tsne_football_network.fit_transform(node_embeddings_football_network)
alpha = 0.9
label_map_football_network = {l: i for i, l in enumerate(np.unique(node_targets_football_network))}
node_colours_football_network = [label_map_football_network[target] for target in node_targets_football_network]
plt.figure(figsize=(10,8))
plt.scatter(node_embeddings_2d_football_network[:,0],
            node_embeddings_2d_football_network[:,1],
            c=node_colours_football_network, cmap="jet", alpha=alpha)
```


    Computing transition probabilities:   0%|          | 0/115 [00:00<?, ?it/s]


    Generating walks (CPU: 1): 100%|███████████████████████████████████████████████████████| 10/10 [00:09<00:00,  1.03it/s]
    




    <matplotlib.collections.PathCollection at 0x1ead82ebdc8>




    
![png](output_51_3.png)
    



```python
#Link prediction in Case 4
embs_football_network_df = (pd.DataFrame([football_network_embedded_Node2Vecmodel.wv.get_vector(str(n)) for n in football_network.nodes()], index = football_network.nodes))
predict_links_for_college(G = football_network, df = embs_football_network_df, college_name = 'FloridaState', N = 7)
```




    ['Navy',
     'WestVirginia',
     'Temple',
     'Vanderbilt',
     'Rutgers',
     'Pittsburgh',
     'Syracuse']



Case 5: $p = 0.5, q = 4  (p < min(q,1), q > 1)$


```python
set_p = 0.5
set_q = 4
football_network_embedded = N2V.Node2Vec(football_network, dimensions=set_dimensions, walk_length = set_walk_length, num_walks = set_num_walks, p = set_p, q = set_q, workers = set_workers)
football_network_embedded_Node2Vecmodel = football_network_embedded.fit(vector_size = set_vector_size, window=set_window, min_count=set_min_count, batch_words=set_batch_words)
input_node = 'FloridaState' #Set the source node, denoted by u in [1]
most_similar_nodes = []
for s in football_network_embedded_Node2Vecmodel.wv.most_similar(input_node, topn = 2):
    most_similar_nodes.append(s)
#Next, construct embeddings
edges_embs_football_network_hadamard = HadamardEmbedder(keyed_vectors=football_network_embedded_Node2Vecmodel.wv)
FloridaState_Virginia = edges_embs_football_network_hadamard[('FloridaState', 'Virginia')]
FloridaState_Maryland = edges_embs_football_network_hadamard[('FloridaState', 'Maryland')]
# Retrieve node embeddings and corresponding subjects
node_ids_football_network = football_network_embedded_Node2Vecmodel.wv.index_to_key  # list of node IDs
node_embeddings_football_network = football_network_embedded_Node2Vecmodel.wv.vectors  # numpy.ndarray of size number of nodes times embeddings dimensionality
node_targets_football_network = [list(football_network.nodes()).index(node_id) for node_id in node_ids_football_network]
#Apply t-SNE transformation on node embeddings
tsne_football_network = TSNE(n_components=2)
node_embeddings_2d_football_network = tsne_football_network.fit_transform(node_embeddings_football_network)
alpha = 0.9
label_map_football_network = {l: i for i, l in enumerate(np.unique(node_targets_football_network))}
node_colours_football_network = [label_map_football_network[target] for target in node_targets_football_network]
plt.figure(figsize=(10,8))
plt.scatter(node_embeddings_2d_football_network[:,0],
            node_embeddings_2d_football_network[:,1],
            c=node_colours_football_network, cmap="jet", alpha=alpha)
```


    Computing transition probabilities:   0%|          | 0/115 [00:00<?, ?it/s]


    Generating walks (CPU: 1): 100%|███████████████████████████████████████████████████████| 10/10 [00:09<00:00,  1.06it/s]
    




    <matplotlib.collections.PathCollection at 0x1ead8c31208>




    
![png](output_54_3.png)
    



```python
#Link prediction in Case 5
embs_football_network_df = (pd.DataFrame([football_network_embedded_Node2Vecmodel.wv.get_vector(str(n)) for n in football_network.nodes()], index = football_network.nodes))
predict_links_for_college(G = football_network, df = embs_football_network_df, college_name = 'FloridaState', N = 7)
```




    ['Vanderbilt',
     'MiddleTennesseeState',
     'Navy',
     'MississippiState',
     'Georgia',
     'Syracuse',
     'Pittsburgh']



<u>Explanation (Original):</u> With a smaller $p$, we can see that the clusters formed are more compact and less spread out. With a smaller $q$ (especially when $q < 1$), we see the range of values taken on the y-axis in the embedding visualisations to be much higher. Analogously, if $q$ is large (in particular, $q > 1$), we see the range on the y axis becoming smaller. This is because larger $q$ enables the walk to obtain a local view of the graph, whilst smaller walk enables the walk to obtain a global view of the graph. This local and global view created by the walk as a result of choosing different $(p,q)$ also impacts the choice of colleges for link prediction.

### b. Extension of node2vec to weighted directed networks (Original Work):

A potential direction for research suggested by [1] is weighted and signed-edge networks. Let's run node2vec on the Bitcoin Alpha trust weighted signed network [9]. This is a trust network of people who trade using Bitcoin on a platform called Bitcoin Alpha. Members of Bitcoin Alpha rate other members on a scale of -10 (total distrust) to +10 (total trust). Here, $|V| = 3783$, $|E| = 24,186$, with the edge weight $\in {(i)}_{i=-10}^{10}$


```python
bitcoin_alpha_trust_data = pd.read_csv("soc-sign-bitcoinalpha.csv")
bitcoin_alpha_trust_data
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Source</th>
      <th>Target</th>
      <th>Rating</th>
      <th>Time</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>7188</td>
      <td>1</td>
      <td>10</td>
      <td>1407470400</td>
    </tr>
    <tr>
      <th>1</th>
      <td>430</td>
      <td>1</td>
      <td>10</td>
      <td>1376539200</td>
    </tr>
    <tr>
      <th>2</th>
      <td>3134</td>
      <td>1</td>
      <td>10</td>
      <td>1369713600</td>
    </tr>
    <tr>
      <th>3</th>
      <td>3026</td>
      <td>1</td>
      <td>10</td>
      <td>1350014400</td>
    </tr>
    <tr>
      <th>4</th>
      <td>3010</td>
      <td>1</td>
      <td>10</td>
      <td>1347854400</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>24181</th>
      <td>7604</td>
      <td>7601</td>
      <td>10</td>
      <td>1364270400</td>
    </tr>
    <tr>
      <th>24182</th>
      <td>7601</td>
      <td>7604</td>
      <td>10</td>
      <td>1364270400</td>
    </tr>
    <tr>
      <th>24183</th>
      <td>7604</td>
      <td>7602</td>
      <td>10</td>
      <td>1364270400</td>
    </tr>
    <tr>
      <th>24184</th>
      <td>7602</td>
      <td>7604</td>
      <td>10</td>
      <td>1364270400</td>
    </tr>
    <tr>
      <th>24185</th>
      <td>7604</td>
      <td>7603</td>
      <td>-10</td>
      <td>1364270400</td>
    </tr>
  </tbody>
</table>
<p>24186 rows × 4 columns</p>
</div>




```python
#Time column above provides time in seconds due to epochs - converting this to human readable time:
human_readable_time_bitcoin_data = []
for i in range(len(bitcoin_alpha_trust_data)):
    human_readable_time_bitcoin_data.append(datetime.fromtimestamp(bitcoin_alpha_trust_data['Time'][i]).strftime('%Y-%m-%d %H:%M:%S'))
bitcoin_alpha_trust_data.insert(4, "Human Readable Time", human_readable_time_bitcoin_data, True)
bitcoin_alpha_trust_data
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Source</th>
      <th>Target</th>
      <th>Rating</th>
      <th>Time</th>
      <th>Human Readable Time</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>7188</td>
      <td>1</td>
      <td>10</td>
      <td>1407470400</td>
      <td>2014-08-08 05:00:00</td>
    </tr>
    <tr>
      <th>1</th>
      <td>430</td>
      <td>1</td>
      <td>10</td>
      <td>1376539200</td>
      <td>2013-08-15 05:00:00</td>
    </tr>
    <tr>
      <th>2</th>
      <td>3134</td>
      <td>1</td>
      <td>10</td>
      <td>1369713600</td>
      <td>2013-05-28 05:00:00</td>
    </tr>
    <tr>
      <th>3</th>
      <td>3026</td>
      <td>1</td>
      <td>10</td>
      <td>1350014400</td>
      <td>2012-10-12 05:00:00</td>
    </tr>
    <tr>
      <th>4</th>
      <td>3010</td>
      <td>1</td>
      <td>10</td>
      <td>1347854400</td>
      <td>2012-09-17 05:00:00</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>24181</th>
      <td>7604</td>
      <td>7601</td>
      <td>10</td>
      <td>1364270400</td>
      <td>2013-03-26 04:00:00</td>
    </tr>
    <tr>
      <th>24182</th>
      <td>7601</td>
      <td>7604</td>
      <td>10</td>
      <td>1364270400</td>
      <td>2013-03-26 04:00:00</td>
    </tr>
    <tr>
      <th>24183</th>
      <td>7604</td>
      <td>7602</td>
      <td>10</td>
      <td>1364270400</td>
      <td>2013-03-26 04:00:00</td>
    </tr>
    <tr>
      <th>24184</th>
      <td>7602</td>
      <td>7604</td>
      <td>10</td>
      <td>1364270400</td>
      <td>2013-03-26 04:00:00</td>
    </tr>
    <tr>
      <th>24185</th>
      <td>7604</td>
      <td>7603</td>
      <td>-10</td>
      <td>1364270400</td>
      <td>2013-03-26 04:00:00</td>
    </tr>
  </tbody>
</table>
<p>24186 rows × 5 columns</p>
</div>




```python
#before converting above Pandas dataframe into a NetworkX graph, take a random sample of 2000 edges
#This is because runtime for Node2Vec constructor and Node2Vec.fit method code below: 1 min 52 seconds if we use 1000 edges
#Runtime for Node2Vec constructor and Node2Vec.fit method code below: 3 min 45 seconds if we use 2000 edges
#Runtime for Node2Vec constructor and Node2Vec.fit method code if we use entire network: more than 10 minutes, hence not feasible to use the entire network
bitcoin_alpha_trust_data_random_sample = bitcoin_alpha_trust_data.sample(n=2000)
bitcoin_alpha_trust_data_random_sample
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Source</th>
      <th>Target</th>
      <th>Rating</th>
      <th>Time</th>
      <th>Human Readable Time</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>11549</th>
      <td>68</td>
      <td>166</td>
      <td>1</td>
      <td>1348718400</td>
      <td>2012-09-27 05:00:00</td>
    </tr>
    <tr>
      <th>19844</th>
      <td>339</td>
      <td>508</td>
      <td>5</td>
      <td>1305086400</td>
      <td>2011-05-11 05:00:00</td>
    </tr>
    <tr>
      <th>3461</th>
      <td>8</td>
      <td>883</td>
      <td>-1</td>
      <td>1396238400</td>
      <td>2014-03-31 05:00:00</td>
    </tr>
    <tr>
      <th>5687</th>
      <td>16</td>
      <td>367</td>
      <td>2</td>
      <td>1307419200</td>
      <td>2011-06-07 05:00:00</td>
    </tr>
    <tr>
      <th>7506</th>
      <td>58</td>
      <td>27</td>
      <td>3</td>
      <td>1405915200</td>
      <td>2014-07-21 05:00:00</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>21644</th>
      <td>571</td>
      <td>1474</td>
      <td>2</td>
      <td>1315281600</td>
      <td>2011-09-06 05:00:00</td>
    </tr>
    <tr>
      <th>1820</th>
      <td>683</td>
      <td>4</td>
      <td>4</td>
      <td>1321074000</td>
      <td>2011-11-12 05:00:00</td>
    </tr>
    <tr>
      <th>16490</th>
      <td>489</td>
      <td>165</td>
      <td>1</td>
      <td>1337227200</td>
      <td>2012-05-17 05:00:00</td>
    </tr>
    <tr>
      <th>3347</th>
      <td>8</td>
      <td>869</td>
      <td>2</td>
      <td>1340164800</td>
      <td>2012-06-20 05:00:00</td>
    </tr>
    <tr>
      <th>3416</th>
      <td>8</td>
      <td>33</td>
      <td>1</td>
      <td>1343102400</td>
      <td>2012-07-24 05:00:00</td>
    </tr>
  </tbody>
</table>
<p>2000 rows × 5 columns</p>
</div>




```python
bitcoin_alpha_trust_network = nx.from_pandas_edgelist(bitcoin_alpha_trust_data_random_sample, source = 'Source', target = 'Target', edge_attr = 'Rating')
```

Using node2vec, we want to discover which users have the same structural roles, so we set $(p,q) = (1,2)$:


```python
#Set parameter values in the Node2Vec constructor
set_window = 1
set_dimensions = 128
set_walk_length = 80
set_num_walks = 10
set_weight_key = 'weight' #The key for the weight attribute on weighted graphs
set_workers = 1
set_p = 1
set_q = 2
set_quiet = False 
set_seed = None
#Set parameter values in the Node2Vec.fit method
set_vector_size = 16
set_min_count = 5
set_window = 8
set_sample = 1e-6
set_negative = 7
set_batch_words = 4
```


```python
bitcoin_alpha_trust_network_embedded = N2V.Node2Vec(bitcoin_alpha_trust_network, dimensions=set_dimensions, walk_length = set_walk_length, num_walks = set_num_walks, p = set_p, q = set_q, weight_key = set_weight_key, workers = set_workers)
bitcoin_alpha_trust_network_embedded_Node2Vecmodel = bitcoin_alpha_trust_network_embedded.fit(vector_size = set_vector_size, window=set_window, min_count=set_min_count, batch_words=set_batch_words)
input_node = '1' #Set the source node, denoted by u in [1] 
for s in bitcoin_alpha_trust_network_embedded_Node2Vecmodel.wv.most_similar(input_node, topn = 2):
    print(s)
```


    Computing transition probabilities:   0%|          | 0/1461 [00:00<?, ?it/s]


    Generating walks (CPU: 1): 100%|███████████████████████████████████████████████████████| 10/10 [01:47<00:00, 10.79s/it]
    

    ('2844', 0.9968326091766357)
    ('7557', 0.9964576959609985)
    


```python
input_node = '234' #Let's now change the source node and number, labels of most similar nodes to source node (Original work)
for s in bitcoin_alpha_trust_network_embedded_Node2Vecmodel.wv.most_similar(input_node, topn = 4):
    print(s)
```

    ('537', 0.9073918461799622)
    ('989', 0.9054194688796997)
    ('40', 0.9026824235916138)
    ('770', 0.8986309170722961)
    

Users 537, 989, 40 and 770 are most similar (in order) to user 234 in the Bitcoin Alpha Trust Network. Next, let's find embeddings using the Hadamard embedder:


```python
edges_embs_bitcoin_alpha_trust_network_hadamard = HadamardEmbedder(keyed_vectors=bitcoin_alpha_trust_network_embedded_Node2Vecmodel.wv)
```


```python
#Find embeddings between different nodes
edges_embs_bitcoin_alpha_trust_network_hadamard[('1', '2')]
```




    array([ 0.00951553,  0.12892376,  1.3518488 ,  0.07254826,  0.42692184,
           -0.03737859, -0.01484667, -0.20738432, -0.21985842, -0.3513598 ,
           -0.30401057,  0.43097275,  0.01367862, -0.11400049,  0.0233879 ,
           -0.02645725], dtype=float32)



Let's visualise embeddings using dimensionality reduction and t-SNE as done previously:


```python
#Retrieve node embeddings and corresponding subjects
node_ids_bitcoin_alpha_trust_network = bitcoin_alpha_trust_network_embedded_Node2Vecmodel.wv.index_to_key  # list of node IDs
node_embeddings_bitcoin_alpha_trust_network = bitcoin_alpha_trust_network_embedded_Node2Vecmodel.wv.vectors  # numpy.ndarray of size number of nodes times embeddings dimensionality
node_targets_bitcoin_alpha_trust_network = [list(bitcoin_alpha_trust_network.nodes()).index(int(node_id)) for node_id in node_ids_bitcoin_alpha_trust_network]
```


```python
#t-SNE transformation on node embeddings
tsne_bitcoin_alpha_trust_network = TSNE(n_components=2)
node_embeddings_2d_bitcoin_alpha_trust_network = tsne_bitcoin_alpha_trust_network.fit_transform(node_embeddings_bitcoin_alpha_trust_network)
alpha = 0.9
label_map_bitcoin_alpha_trust_network = {l: i for i, l in enumerate(np.unique(node_targets_bitcoin_alpha_trust_network))}
node_colours_bitcoin_alpha_trust_network = [label_map_bitcoin_alpha_trust_network[target] for target in node_targets_bitcoin_alpha_trust_network]
plt.figure(figsize=(10,8))
plt.scatter(node_embeddings_2d_bitcoin_alpha_trust_network[:,0],
            node_embeddings_2d_bitcoin_alpha_trust_network[:,1],
            c=node_colours_bitcoin_alpha_trust_network, cmap="jet", alpha=alpha)
```




    <matplotlib.collections.PathCollection at 0x1eade783588>




    
![png](output_72_1.png)
    



```python
#Link prediction as done for the football network
embs_bitcoin_alpha_trust_network_df = (pd.DataFrame([bitcoin_alpha_trust_network_embedded_Node2Vecmodel.wv.get_vector(str(n)) for n in bitcoin_alpha_trust_network.nodes()], index = bitcoin_alpha_trust_network.nodes))
def predict_links_for_user(G, df, user_id, N):
    '''
    This function will predict the top N links a user should be connected with
    which it is not already connected with in the user network.
    
    params:
        G (NetworkX Graph) : The network used to create the embeddings
        df (DataFrame) : The dataframe with embeddings associated to each user
        user_id (Integer) : The user id interested in
        N (Integer) : The number of recommended links for a given user
        
    returns:
        This function will return a list of users the input user should be connected with.
    '''
    #separate target user
    user = embs_bitcoin_alpha_trust_network_df.loc[user_id]
    # other users are the users with which the given user doesn't have a connection
    all_nodes = bitcoin_alpha_trust_network.nodes()
    other_nodes = [n for n in all_nodes if n not in list(bitcoin_alpha_trust_network.adj[user_id]) + [user_id]]
    other_users = embs_bitcoin_alpha_trust_network_df[embs_bitcoin_alpha_trust_network_df.index.isin(other_nodes)]
    # compute the Manhattan norm for given user and all other users
    manhattan_norm = []
    for i in range(len(other_users)):
        manhattan_norm.append(cityblock(user, other_users.iloc[i]))
    idx = other_users.index.tolist()
    # create a dictionary of distances for this user w.r.t. other users
    idx_sim = dict(zip(idx, manhattan_norm))
    idx_sim = sorted(idx_sim.items(), key=lambda x: x[1], reverse=False)
    #Pick the top N users that have the smallest Manhattan distance from the specified user
    similar_users = idx_sim[:N]
    users = [art[0] for art in similar_users]
    return users

predict_links_for_user(G = bitcoin_alpha_trust_network, df = embs_bitcoin_alpha_trust_network_df, user_id = 1, N = 7)
```




    [1288, 1324, 22, 869, 1285, 413, 1023]



The embeddings visualisation suggests difficulty in constructing well-defined clusters for this network using node2vec. Using link prediction, node2vec recommends that user '1' should connect with users '1288', '1324', '22', '869', '1285', '413', '1023' in order of trust.

## 4. Related Work

Other models previously used for search based optimisation are Spectral Clustering, DeepWalk and LINE [1]. Let’s mathematically display the differences between them and node2vec:

i. <u> Spectral Clustering/Embedding: <u>
    
It is an unsupervised learning algorithm with the following procedure [12, 13]:
    
Step 1: Compute the diagonal degree matrix $D$ and normalise the adjacency matrix $A$ of the graph using the following: $D^{-\frac{1}{2}}AD^{-\frac{1}{2}}$
    
Step 2: Take the top $k$ eigenvectors from the following eigenvalue equation: $k \in \mathbb{R}, D^{-\frac{1}{2}}AD^{-\frac{1}{2}}v_{k} = \lambda_{k}v_{k}$ ; $\lambda_{k} = $  $k$-th largest eigenvalue ; $v_{k} = $ corresponding eigenvector.
    
Step 3: Take each eigenvector and $\forall n \in V$, construct feature vectors $u_{n}$ where $u_{ni} = v_{in}$ (the $i$th component of the feature vector associated with node $n = $ the nth component of the ith eigenvector satisfying the above eigenvalue equation). 

So in the eigenvector matrix, each row is a feature embedding for the corresponding node. 
    
<u>Difference from node2vec:</u> Here, the context graph is degree and adjacency matrix based.
    
ii. <u> DeepWalk: <u>
    
Like node2vec, this is a random walk based embedding method. Other similarities include a symmetric adjacency matrix, same structure for learnt and used embeddings and having the same functional form for the loss function to optimise [14]. 
    
However, optimisation in node2vec takes place using negative sampling whilst DeepWalk uses hierarchical softmax. Secondly, DeepWalk learns d-dimensional feature representations by simulating uniform random walks (hence the random walks are unbiased). Plus, node2vec with $(p,q) = (1,1)$ simplifies to DeepWalk.
    
iii. <u> LINE: <u>
    
Unlike DeepWalk and node2vec, LINE uses an adjacency based context graph. Like DeepWalk, LINE learns d-dimensional feature representations but unlike DeepWalk, does so in 2 separate phases [1]. While node2vec smoothly interpolates over BFS and DFS sampling, in the first phase LINE learns $\frac{d}{2}$ dimensions by BFS-based sampling over 1-hop neighbours of nodes. In phase 2, it learns next $\frac{d}{2}$ dimensions by sampling nodes using a 2-hop distance from a given source node $u$. Like node2vec, it takes 2nd order into account but is not a random walk embedding approach and also solely involves optimisation by first-order and second-order proximity together. However, like node2vec it involves optimisation by negative sampling [14].
    
There are other matrix factorisation approaches similar to node2vec but are extremely inferior to node2vec in performance and so have been excluded from [1] and this tutorial.

## 5. Summary and Future Directions

<u>Summary of node2vec:</u> node2vec adopts a biased random walk procedure to learn continuous node embeddings, smoothly interpolates between BFS and DFS sampling to trade off between local and global structure of the network, and uses skip-gram training as also used by the celebrated word2vec model. 

<u>Assumptions:</u> In extending the skip-gram approach to networks, 2 assumptions are made by [1] to make the optimisation problem for node2vec analytically solvable: 

a. Conditional independence: Assume the likelihood of observing a neighbourhood node is independent of observing any other neighbourhood node given the feature representation of the source: $Pr(N_{S}(u)|f(u)) = \prod_{n_{i} \in N_{S}(u)} Pr(n_{i}|f(u))$

b. Symmetry in feature space: A source node and neighbourhood node have a symmetric effect over each other in feature space. So the conditional likelihood of every source-neighbourhood node pair is modelled as a softmax unit parametrised by the dot product of features: $Pr(n_{i}|f(u)) = \frac{exp(f(n_{i}).f(u))}{\sum_{v \in V}(exp(f(u).f(v))}$

<u>Limitations:</u> node2vec can only perform representation learning in homogeneous networks, i.e. where nodes and relationships are of the same type. [16] presents *metapath2vec* and *metapath2vec++*. metapath2vec maximises the likelihood of preserving the structures and semantics of a given heterogeneous network [16]. metapath2vec++ builds on metapath2vec, but where the network probability in terms of local structures $p(c_{t}|v;\theta)$, in softmax functional form, is normalised with respect to the node type of the context $c_{t}$. However, like node2vec, they face the challenge of large intermediate output data when sampling a network into paths, hence optimisation of sample space is something that requires future work [16].

<u>Future work:</u> Theoretical analysis of why the Hadamard embedder works best relative to other embedding approaches is an interesting direction of future work. Secondly, networks with explicit domain features for nodes and edges can be looked into. Another interesting direction of research is to generalise node2vec to a wider class of objects, i.e. simplical complexes [17]. Mathematically, nodes and edges are 0-simplices and 1-simplices but k-simplex2vec, suggested by [17], extends random walks to k-simplices. However, this method only considers clique complexes of graphs and so future work could involve extending node2vec to other simplical complexes. 

<u>Summary of tutorial:</u> This tutorial builds on existing work done in node2vec in the following manner:

a. Data selection and preprocessing: Relevant network datasets from literature are selected on which node2vec has not been used yet. The Bitcoin network was preprocessed before identification of similar users, embedding visualisation and link prediction could be done. 

b. Model design: Different cases for $(p,q)$ in the model are explored, with relevant explanations.

c. Predictions: Apart from Hadamard, other embedding approaches are also checked. Link prediction and embedding visualisations are done differently to other tutorials in literature, that allow for deeper insights into the workings of node2vec. 

This tutorial can be used to explore how changing $(p,q)$ influences the transition probabilities, ordering of similar nodes, embedding visualisations and link predictions. The user can change parameter values in the Node2Vec constructor and Node2Vec.fit method, and change the number of most similar nodes interested in, to further build on understanding of node2vec. Instead of the Manhattan norm, the user can use cosine similarity or other similarity measure for link prediction tasks. The user can construct embedding visualisations in 3D-space instead of 2D-space. The user can extend this tutorial to carry out node classification and community detection tasks using node2vec.

## 6. References

- [1]: (Source Paper) Grover A, Leskovec J. node2vec: Scalable feature learning for networks. InProceedings of the 22nd ACM SIGKDD international conference on Knowledge discovery and data mining 2016 Aug 13 (pp. 855-864).

- [2]: https://developers.google.com/machine-learning/crash-course/classification/roc-and-auc

- [3]: https://networkx.org/

- [4]: https://towardsdatascience.com/node2vec-explained-db86a319e9ab

- [5]: https://github.com/eliorc/node2vec

- [6]: https://www.kaggle.com/code/pierremegret/gensim-word2vec-tutorial/notebook

- [7]: https://radimrehurek.com/gensim/models/word2vec.html

- [8]: M.Girvan and M. E. J. Newman, Proc. Natl. Acad. Sci. USA 99, 7821-7826 (2002).

- [9]: https://snap.stanford.edu/data/soc-sign-bitcoin-alpha.html

- [10]: S. Kumar, F. Spezzano, V.S. Subrahmanian, C. Faloutsos. Edge Weight Prediction in Weighted Signed Networks. IEEE International Conference on Data Mining (ICDM), 2016.

- [11]: S. Kumar, B. Hooi, D. Makhija, M. Kumar, V.S. Subrahmanian, C. Faloutsos. REV2: Fraudulent User Prediction in Rating Platforms. 11th ACM International Conference on Web Searchand Data Mining (WSDM), 2018.

- [12]: https://people.csail.mit.edu/dsontag/courses/ml14/notes/Luxburg07_tutorial_spectral_clustering.pdf

- [13]: https://snap.stanford.edu/class/cs224w-2017/projects/cs224w-38-final.pdf

- [14]: Khosla M, Setty V, Anand A. A comparative study for unsupervised network representation learning. IEEE Transactions on Knowledge and Data Engineering. 2019 Nov 4;33(5):1807-18.

- [15]: https://www.geeksforgeeks.org/node2vec-algorithm/

- [16]: Dong Y, Chawla NV, Swami A. metapath2vec: Scalable representation learning for heterogeneous networks. InProceedings of the 23rd ACM SIGKDD international conference on knowledge discovery and data mining 2017 Aug 4 (pp. 135-144).

- [17]: Hacker C. k-simplex2vec: a simplicial extension of node2vec. arXiv preprint arXiv:2010.05636. 2020 Oct 12.

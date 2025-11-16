# Campus Navigation & Utility Planner

A **Python-based data-structure project** that models university buildings, manages building information using trees, and performs navigation and utility planning using graph algorithms.

---

## Features

### Tree Structures
- **Binary Search Tree (BST)**  
  - Insert building records  
  - Search buildings  
  - Inorder, Preorder, Postorder traversals  
  - Compare height with AVL  

- **AVL Tree (Self-Balancing)**  
  - Auto-balances using rotations (LL, RR, LR, RL)  
  - Stores rotation history  
  - Demonstrates difference from BST  

---

## Campus Graph & Navigation

- **Graph Representation**
  - Each building is a node  
  - Paths (roads) are weighted edges  
  - Represented using **Adjacency List**

- **Graph Algorithms**
  - **BFS** → Explore campus level-wise  
  - **DFS** → Deep traversal  
  - **Dijkstra** → Shortest path between buildings  
  - **Kruskal (MST)** → Minimum cost utility layout using Union-Find  

---

## Expression Tree

- Converts **infix → postfix → expression tree**
- Evaluates expressions like:
  100 + rate * consumption + (peak_hours * peak_rate)
- Useful for computing:
- Electricity bills  
- Utility cost evaluations  

---

## How It Works

1. **Building ADT**  
 Stores:  
 - Building ID  
 - Name  
 - Location  
 - Connections  

2. **BST & AVL Tree:**  
 - Buildings stored based on building ID  
 - AVL ensures balanced height  
 - Traversals used to list campus buildings  

3. **Graph:**  
 - Weighted edges represent distances  
 - BFS & DFS show exploration patterns  
 - Dijkstra finds shortest navigation path  
 - Kruskal builds minimum spanning tree  

4. **Expression Tree:**  
 - Uses tokenization and postfix conversion  
 - Builds a binary expression tree  
 - Recursively evaluates the expression  

5. **Example output:**

BST inorder: [2-Hostel, 5-Cafe, 10-Library, 15-SciLab, 20-Admin Block, 25-Gym]
BST height: 3

AVL inorder: [1, 2, 3, 4, 5]
AVL rotations: ['LL at 1', 'RR at 3', 'LR at 4']

BFS from Library: [10, 20, 5, 15, 2, 25]
Shortest path Library -> Gym: distance = 140, path = [10, 20, 15, 25]

MST edges:
10 - 5 : 30
10 - 20 : 50
20 - 15 : 40
15 - 25 : 60
Total weight = 180

Expression Value = 1000.0

6. **Technologies Used:**
 - Python 3.x
 - Data Structures (Trees, Graphs, Union-Find)
 - Algorithms (Dijkstra, Kruskal, BFS, DFS)
 - Expression Trees

7. **Learning Outcomes:**
 - Understanding BST & AVL trees
 - Implementing graph algorithms in real scenarios
 - Using Dijkstra & Kruskal for optimization
 - Expression tree construction & evaluation
 - Applying data structure concepts to a real-world model
 - 
---




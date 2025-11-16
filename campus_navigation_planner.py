class Building:
    def __init__(self, building_id, building_name, building_location):
        self.id = building_id
        self.name = building_name
        self.loc = building_location
        self.connections = {}

    def __repr__(self):
        return f"{self.id}-{self.name}"


class BstNode:
    def __init__(self, data):
        self.data = data
        self.left_child = None
        self.right_child = None


class BST:
    def __init__(self):
        self.root = None

    def add(self, building):
        def insert(parent, value):
            if parent is None:
                return BstNode(value)
            if value.id < parent.data.id:
                parent.left_child = insert(parent.left_child, value)
            elif value.id > parent.data.id:
                parent.right_child = insert(parent.right_child, value)
            else:
                parent.data = value
            return parent
        
        self.root = insert(self.root, building)

    def inorder(self):
        output = []
        def traverse(node):
            if node:
                traverse(node.left_child)
                output.append(node.data)
                traverse(node.right_child)
        traverse(self.root)
        return output

    def preorder(self):
        output = []
        def traverse(node):
            if node:
                output.append(node.data)
                traverse(node.left_child)
                traverse(node.right_child)
        traverse(self.root)
        return output

    def postorder(self):
        output = []
        def traverse(node):
            if node:
                traverse(node.left_child)
                traverse(node.right_child)
                output.append(node.data)
        traverse(self.root)
        return output

    def height(self):
        def get_height(node):
            if not node:
                return 0
            return 1 + max(get_height(node.left_child), get_height(node.right_child))
        return get_height(self.root)


class AvlNode:
    def __init__(self, data):
        self.data = data
        self.left_child = None
        self.right_child = None
        self.height = 1


class AVL:
    def __init__(self):
        self.root = None
        self.rotations = []

    def add(self, building):
        def node_height(node):
            return node.height if node else 0

        def update_height(node):
            node.height = 1 + max(node_height(node.left_child), node_height(node.right_child))

        def balance_factor(node):
            return node_height(node.left_child) - node_height(node.right_child)

        def rotate_right(parent):
            new_root = parent.left_child
            temp_child = new_root.right_child
            new_root.right_child = parent
            parent.left_child = temp_child
            update_height(parent)
            update_height(new_root)
            self.rotations.append(f"RR rotation at {parent.data.id}")
            return new_root

        def rotate_left(parent):
            new_root = parent.right_child
            temp_child = new_root.left_child
            new_root.left_child = parent
            parent.right_child = temp_child
            update_height(parent)
            update_height(new_root)
            self.rotations.append(f"LL rotation at {parent.data.id}")
            return new_root

        def insert(node, value):
            if not node:
                return AvlNode(value)
            if value.id < node.data.id:
                node.left_child = insert(node.left_child, value)
            elif value.id > node.data.id:
                node.right_child = insert(node.right_child, value)
            else:
                node.data = value
                return node

            update_height(node)
            bf = balance_factor(node)

            if bf > 1 and value.id < node.left_child.data.id:
                return rotate_right(node)
            if bf < -1 and value.id > node.right_child.data.id:
                return rotate_left(node)
            if bf > 1 and value.id > node.left_child.data.id:
                node.left_child = rotate_left(node.left_child)
                self.rotations.append(f"LR rotation at {node.data.id}")
                return rotate_right(node)
            if bf < -1 and value.id < node.right_child.data.id:
                node.right_child = rotate_right(node.right_child)
                self.rotations.append(f"RL rotation at {node.data.id}")
                return rotate_left(node)
            return node

        self.root = insert(self.root, building)

    def inorder(self):
        result = []
        def traverse(node):
            if node:
                traverse(node.left_child)
                result.append(node.data)
                traverse(node.right_child)
        traverse(self.root)
        return result

    def height(self):
        return self.root.height if self.root else 0


class CampusGraph:
    def __init__(self):
        self.adjacency = {}
        self.building_nodes = {}

    def add_building(self, building):
        self.building_nodes[building.id] = building
        if building.id not in self.adjacency:
            self.adjacency[building.id] = []

    def link(self, from_node, to_node, weight, undirected=True):
        self.adjacency[from_node].append((to_node, weight))
        if undirected:
            self.adjacency[to_node].append((from_node, weight))

    def bfs(self, start_node):
        if start_node not in self.adjacency:
            return []
        visited = {start_node}
        queue = [start_node]
        result = []
        while queue:
            current = queue.pop(0)
            result.append(current)
            for next_node, _ in self.adjacency[current]:
                if next_node not in visited:
                    visited.add(next_node)
                    queue.append(next_node)
        return result

    def dfs(self, start_node):
        visited = set()
        result = []
        def explore(node):
            visited.add(node)
            result.append(node)
            for next_node, _ in self.adjacency[node]:
                if next_node not in visited:
                    explore(next_node)
        explore(start_node)
        return result

    def dijkstra(self, start_node):
        dist = {i: float('inf') for i in self.building_nodes}
        parent = {i: None for i in self.building_nodes}
        if start_node not in self.building_nodes:
            return dist, parent
        dist[start_node] = 0
        heap = [(0, start_node)]
        import heapq
        while heap:
            distance, node = heapq.heappop(heap)
            if distance > dist[node]:
                continue
            for neighbor, weight in self.adjacency[node]:
                new_distance = distance + weight
                if new_distance < dist[neighbor]:
                    dist[neighbor] = new_distance
                    parent[neighbor] = node
                    heapq.heappush(heap, (new_distance, neighbor))
        return dist, parent

    def shortest(self, start_node, end_node):
        dist, parent = self.dijkstra(start_node)
        if dist[end_node] == float('inf'):
            return float('inf'), []
        path = []
        node = end_node
        while node is not None:
            path.append(node)
            node = parent[node]
        return dist[end_node], path[::-1]

    def all_edges(self):
        used = set()
        edges = []
        for from_node in self.adjacency:
            for to_node, weight in self.adjacency[from_node]:
                if (to_node, from_node) not in used:
                    used.add((from_node, to_node))
                    edges.append((weight, from_node, to_node))
        return edges


class UnionFind:
    def __init__(self, items):
        self.parent = {x: x for x in items}
        self.rank = {x: 0 for x in items}

    def find(self, x):
        if self.parent[x] != x:
            self.parent[x] = self.find(self.parent[x])
        return self.parent[x]

    def unite(self, a, b):
        root_a = self.find(a)
        root_b = self.find(b)
        if root_a == root_b:
            return False
        if self.rank[root_a] < self.rank[root_b]:
            self.parent[root_a] = root_b
        elif self.rank[root_a] > self.rank[root_b]:
            self.parent[root_b] = root_a
        else:
            self.parent[root_b] = root_a
            self.rank[root_a] += 1
        return True


def kruskal(graph):
    edges = sorted(graph.all_edges(), key=lambda x: x[0])
    uf = UnionFind(graph.building_nodes.keys())
    result = []
    total = 0
    for weight, node1, node2 in edges:
        if uf.unite(node1, node2):
            result.append((node1, node2, weight))
            total += weight
    return result, total


def tokenize(expr_string):
    tokens = []
    index = 0
    while index < len(expr_string):
        char = expr_string[index]
        if char.isspace():
            index += 1
            continue
        if char.isdigit() or char == '.':
            j = index
            dot = 0
            while j < len(expr_string) and (expr_string[j].isdigit() or (expr_string[j] == '.' and dot == 0)):
                if expr_string[j] == '.':
                    dot = 1
                j += 1
            tokens.append(expr_string[index:j])
            index = j
            continue
        if char.isalpha():
            j = index
            while j < len(expr_string) and (expr_string[j].isalnum() or expr_string[j] == '_'):
                j += 1
            tokens.append(expr_string[index:j])
            index = j
            continue
        tokens.append(char)
        index += 1
    return tokens


def infix_to_postfix(tokens):
    precedence = {'+':1,'-':1,'*':2,'/':2,'^':3}
    stack = []
    output = []
    for token in tokens:
        if token.replace('.','',1).isdigit() or token.isidentifier():
            output.append(token)
        elif token == '(':
            stack.append(token)
        elif token == ')':
            while stack and stack[-1] != '(':
                output.append(stack.pop())
            stack.pop()
        else:
            while stack and stack[-1] != '(' and precedence.get(stack[-1],0) >= precedence.get(token,0):
                output.append(stack.pop())
            stack.append(token)
    while stack:
        output.append(stack.pop())
    return output


class ExpressionNode:
    def __init__(self, value):
        self.val = value
        self.left = None
        self.right = None


def build_expression_tree(postfix_list):
    stack = []
    for token in postfix_list:
        if token.isidentifier() or token.replace('.','',1).isdigit():
            stack.append(ExpressionNode(token))
        else:
            right_node = stack.pop()
            left_node = stack.pop()
            new = ExpressionNode(token)
            new.left = left_node
            new.right = right_node
            stack.append(new)
    return stack[0]


def evaluate_expression(node, value_map):
    if node.val not in ['+','-','*','/','^']:
        if node.val.replace('.','',1).isdigit():
            return float(node.val)
        return float(value_map.get(node.val,0))

    a = evaluate_expression(node.left, value_map)
    b = evaluate_expression(node.right, value_map)

    if node.val == '+': return a + b
    if node.val == '-': return a - b
    if node.val == '*': return a * b
    if node.val == '/': return a / b
    return a ** b


def run_demo():
    building1 = Building(10,"Library","A")
    building2 = Building(20,"Admin","B")
    building3 = Building(5,"Cafe","C")
    building4 = Building(15,"SciLab","B2")
    building5 = Building(25,"Gym","D")
    building6 = Building(2,"Hostel","North")

    data_list = [building1,building2,building3,building4,building5,building6]

    bst_tree = BST()
    for building in data_list:
        bst_tree.add(building)

    print("BST inorder:", bst_tree.inorder())
    print("BST preorder:", bst_tree.preorder())
    print("BST postorder:", bst_tree.postorder())
    print("BST height:", bst_tree.height())

    avl_tree = AVL()
    for index in [1,2,3,4,5]:
        avl_tree.add(Building(index,f"B{index}",f"L{index}"))

    print("AVL inorder:", avl_tree.inorder())
    print("AVL height:", avl_tree.height())
    print("AVL rotations:", avl_tree.rotations)

    campus_graph = CampusGraph()
    for building in data_list:
        campus_graph.add_building(building)

    campus_graph.link(10,20,50)
    campus_graph.link(10,5,30)
    campus_graph.link(20,15,40)
    campus_graph.link(5,2,120)
    campus_graph.link(15,25,60)
    campus_graph.link(2,25,200)

    print("BFS:", campus_graph.bfs(10))
    print("DFS:", campus_graph.dfs(10))

    distance, path = campus_graph.shortest(10,25)
    print("Shortest path:", distance, path)

    mst_result, mst_total = kruskal(campus_graph)
    print("MST:", mst_result, "Total:", mst_total)

    expression = "100 + rate * units + (peak_hours * peak_rate)"
    tokens = tokenize(expression)
    postfix_list = infix_to_postfix(tokens)
    expr_tree = build_expression_tree(postfix_list)
    value_map = {"rate":5,"units":200,"peak_hours":10,"peak_rate":2}

    print("Expression value:", evaluate_expression(expr_tree, value_map))


run_demo()

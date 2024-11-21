from __future__ import annotations
from matplotlib import pyplot as plt
import sympy as sp
import networkx as nx

# def execute(arg_one: RefinedNode, arg_two: RefinedNode, operation: str):

#     val_one = arg_one.value if arg_one.value_symbol is None else arg_one.value_symbol
#     val_two = arg_two.value if arg_two.value_symbol is None else arg_two.value_symbol

#     unc_one = arg_one.uncertainty if arg_one.uncertainty_symbol is None else arg_one.uncertainty_symbol
#     unc_two = arg_two.uncertainty if arg_two.uncertainty_symbol is None else arg_two.uncertainty_symbol

#     # Calculate new value
#     if(operation == "add"): 
#         out = val_one + val_two
#         combined_unc = sp.sqrt(unc_one**2 + unc_two**2)
#     elif(operation == "sub"): 
#         out = val_one - val_two
#         combined_unc = sp.sqrt(unc_one**2 + unc_two**2)
#     elif(operation == "mul"): 
#         out = val_one * val_two
#         combined_unc = sp.sqrt((unc_one/val_one)**2 + (unc_two/val_one)**2)*out**2
#     elif(operation == "div"): 
#         out = val_one / val_two
#         combined_unc = sp.sqrt((unc_one/val_one)**2 + (unc_two/val_one)**2)*out**2
#     elif(operation == "neg"): 
#         out = -val_one
#         combined_unc = unc_one
#     elif(operation == "pow"): raise ValueError("Undefined")

#     # Return as a RefinedNode
#     return RefinedNode(value=out, uncertainty=combined_unc)

def execute(arg_one: ExpressionGraphNode, arg_two: ExpressionGraphNode, operation: str):

    val_one = arg_one.value if arg_one.value_symbol is None else arg_one.value_symbol
    val_two = arg_two.value if arg_two.value_symbol is None else arg_two.value_symbol

    unc_one = arg_one.uncertainty if arg_one.uncertainty_symbol is None else arg_one.uncertainty_symbol
    unc_two = arg_two.uncertainty if arg_two.uncertainty_symbol is None else arg_two.uncertainty_symbol

    # Calculate new value
    if(operation == "add"): out = val_one + val_two
    elif(operation == "sub"): out = val_one - val_two
    elif(operation == "mul"): out = val_one * val_two
    elif(operation == "div"): out = val_one / val_two
    elif(operation == "neg"): out = -val_one
    elif(operation == "pow"): raise ValueError("Undefined")

    # Calculate combined uncertainty    
    vals = [val_one, val_two]
    uncs = [unc_one, unc_two]
    combined_unc = 0
    for i in range(2):
        temp_sym = sp.symbols('temp')
        temp_expr = out.subs(vals[i], temp_sym)
        df_di = sp.diff(temp_expr, temp_sym).subs(temp_sym, vals[i])
        combined_unc += (df_di * uncs[i])**2

    combined_unc = sp.sqrt(combined_unc)

    # Return as a RefinedNode
    return ExpressionGraphNode(value=out, uncertainty=combined_unc)

def execute_simple(arg_one: ExpressionGraphNode, arg_two: int | float, operation: str):

    val_one = arg_one.value if arg_one.value_symbol is None else arg_one.value_symbol
    unc_one = arg_one.uncertainty if arg_one.uncertainty_symbol is None else arg_one.uncertainty_symbol

    if(operation == "add"): 
        out = val_one + arg_two
        unc = unc_one
    elif(operation == "sub"): 
        out = val_one - arg_two
        unc = unc_one
    elif(operation == "mul"): 
        out = val_one * arg_two
        unc = unc_one * abs(arg_two)
    elif(operation == "div"): 
        out = val_one / arg_two
        unc = unc_one / abs(arg_two)
    elif(operation == "pow"): 
        out = val_one ** arg_two
        unc = val_one ** (arg_two - 1) * sp.Abs(arg_two) * unc_one
    
    return ExpressionGraphNode(value=out, uncertainty=unc)

class ExpressionGraphNode:
    """
    First the node will be defined with an expression. This happens in __init__
        I want the expression to be an expression of RefinedNodes. How do I make that possible. I need to delegate every construction to sympy, then wrap it in RefinedNode.

    The node can then be labelled. It can only ever be labelled once. Once labelled, the expression within is associated with the label and is immutable (well. I guess its always immutable. Sometimes nodes are anonymous and transient)
    A node's expression consists of a sp.Expr with variables that are named with the child nodes' labels.

    When a child node takes on a value, the value is propogated through its parents. That means we have to track both children and parents of nodes
    
    """

    node_value_dict: dict[str, ExpressionGraphNode] = {}
    node_uncertainty_dict: dict[str, ExpressionGraphNode] = {}

    def __init__(self, value: sp.Expr = None, uncertainty: sp.Expr | int | float = None):
        self.value_symbol: sp.Symbol = None
        self.uncertainty_symbol: sp.Symbol = None

        self.children: list[ExpressionGraphNode] = []
        self.parents: list[ExpressionGraphNode] = []

        self.uncertainty_children: list[ExpressionGraphNode] = []
        self.uncertainty_parents: list[ExpressionGraphNode] = []
        
        self.value: sp.Expr = value
        self.uncertainty: sp.Expr = uncertainty
        self.is_leaf = False

    def get_name(self):
        if(self.value_symbol is None): raise ValueError("This node has not been labelled yet!")
        return self.value_symbol.name
    
    def label(self, name: str):
        if(self.value_symbol is not None): raise ValueError("Only able to label a Node once!")
        self.value_symbol = sp.symbols(name)
        self.uncertainty_symbol = sp.symbols("U_{" + name + "}")
        if(self.value is None):
            self.value = self.value_symbol # Only happens for leaf nodes
            self.uncertainty = self.uncertainty_symbol # Only happens for leaf nodes
            self.is_leaf = True
        ExpressionGraphNode.node_value_dict[self.value_symbol.name] = self # Register node globally
        ExpressionGraphNode.node_uncertainty_dict[self.uncertainty_symbol.name] = self # Register node globally

        for arg in self.value.free_symbols: # Connect the node to its children and set it as a parent of those children
            if isinstance(arg, sp.Symbol) and arg.name != self.value_symbol.name:
                self.children.append([n for label, n in ExpressionGraphNode.node_value_dict.items() if n.value_symbol == arg][0])
                self.children[-1].parents.append(self)
        
        for arg in self.uncertainty.free_symbols: # Connect the node to its children and set it as a parent of those children
            if isinstance(arg, sp.Symbol) and arg.name != self.uncertainty_symbol.name and arg.name in ExpressionGraphNode.node_uncertainty_dict.keys():
                self.uncertainty_children.append([n for label, n in ExpressionGraphNode.node_uncertainty_dict.items() if n.uncertainty_symbol == arg][0])
                # if(self.uncertainty_children[-1].uncertainty_symbol.name == "U_{M}"): print("ARG",arg.name)
                self.uncertainty_children[-1].uncertainty_parents.append(self)

        return self

    def leaf(self) -> bool:
        return self.is_leaf
    
    def set_numeric_measured_value(self, value, uncertainty):
        if(not self.leaf()): raise ValueError("Only able to call this function on measured values!")
        self._substitute_numeric_into_value_expression(self.value_symbol, value)
        self._substitute_numeric_into_uncertainty_expression(self.uncertainty_symbol, uncertainty)

    def _substitute_numeric_into_value_expression(self, symbol, numeric):
        # print("In", self.value_symbol, "substituting", symbol, "for", numeric)
        if(symbol not in self.value.free_symbols): 
            raise ValueError(f"{symbol} is not a part of the value expression: {self.value}")
            print(f"{symbol} is not a part of {self.value}")
            return
        self.value = self.value.subs(symbol, numeric)
        if(len(self.value.free_symbols) == 0): self._propagate_value_numerics()

    def _substitute_numeric_into_uncertainty_expression(self, symbol, numeric):
        # print("In", self.uncertainty_symbol, "substituting", symbol, "for", numeric)
        if(symbol not in self.uncertainty.free_symbols): 
            raise ValueError(f"{symbol} is not a part of the uncertainty expression: {self.uncertainty}")
            print(f"{symbol} is not a part of {self.uncertainty}")
            return 
        self.uncertainty = self.uncertainty.subs(symbol, numeric)
        if(len(self.uncertainty.free_symbols) == 0): self._propagate_uncertainty_numerics()

    def _propagate_value_numerics(self):
        # print("Value: ", self.value_symbol)
        if(len(self.value.free_symbols) == 0):
            for p in self.parents:
                # print(p.value_symbol)
                p._substitute_numeric_into_value_expression(self.value_symbol, self.value)
                if(self.value_symbol in p.uncertainty.free_symbols): p._substitute_numeric_into_uncertainty_expression(self.value_symbol, self.value)

    def _propagate_uncertainty_numerics(self):
        # print("Uncertainty parents: ", [p.value_symbol for p in self.uncertainty_parents])
        if(len(self.uncertainty.free_symbols) == 0):
            for p in self.uncertainty_parents:
                p._substitute_numeric_into_uncertainty_expression(self.uncertainty_symbol, self.uncertainty)

    def __add__(self, other):
        if isinstance(other, ExpressionGraphNode):
            return execute(self, other, "add")
        return execute_simple(self, other, "add")
    
    def __radd__(self, other):
        return self.__add__(other)
    
    def __sub__(self, other):
        if isinstance(other, ExpressionGraphNode):
            return execute(self, other, "sub")
        return execute_simple(self, other, "sub")
    
    def __rsub__(self, other):
        return -self.__sub__(other)

    def __mul__(self, other):
        if isinstance(other, ExpressionGraphNode):
            return execute(self, other, "mul")
        return execute_simple(self, other, "mul")
    
    def __rmul__(self,other):
        return self.__mul__(other)

    def __truediv__(self, other):
        if isinstance(other, ExpressionGraphNode):
            return execute(self, other, "div")
        return execute_simple(self, other, "div")
    
    def __rtruediv__(self, other):
        return (self ** (-1)) * other

    def __pow__(self, other):
        if isinstance(other, ExpressionGraphNode):
            return execute(self, other, "pow")
        return execute_simple(self, other, "pow")

    def __neg__(self):
        return execute(self, ExpressionGraphNode().label("Anon"), "neg")
    
    def __str__(self):
        s = ""
        if(self.value_symbol is not None): s += f"Symbol: {self.value_symbol}"
        s += f"\nValue: {self.value}"
        s += f"\nUncertainty: {self.uncertainty}"

        return s
    
    def get_value_graph(self):
        _graph = nx.DiGraph()
        generate_graph(self, _graph)
        systematically_rename_nodes(_graph, generate_value_expression)
        show_graph(_graph)

    def get_uncertainty_graph(self):
        _graph = nx.DiGraph()
        generate_graph(self, _graph)
        systematically_rename_nodes(_graph, generate_uncertainty_expression)
        show_graph(_graph)


    # Workflow: 
    #   a = RefinedNode().label("a"); b = RefinedNode().label("b");
    #   c = a + b -> RefinedNode w. value of symbols of "a" and "b"

def generate_graph(root: ExpressionGraphNode, graph: nx.DiGraph, count=0):
    color_map = {
        True: 'lightgreen',
        False: 'lightblue'
    }

    if(graph.nodes.get(root.get_name()) is not None):
        if(graph.nodes()[root.get_name()]["subset"] > count):
            graph.add_node(root.get_name(), subset=count, color=color_map[root.leaf()])
    
    else:
        graph.add_node(root.get_name(), color=color_map[root.leaf()])
        graph.nodes()[root.get_name()]['subset'] = count

    for child in root.children:
        generate_graph(child, graph, count-1)

    for child in root.children:
        graph.add_edge(child.get_name(), root.get_name(), label=child.get_name())

def change_node_name(graph: nx.DiGraph, oldname, newname):
    graph.add_node(newname, **graph.nodes[oldname])

    for pred in graph.predecessors(oldname):
        graph.add_edge(pred, newname, **graph.get_edge_data(pred, oldname))

    for succ in graph.successors(oldname):
        graph.add_edge(newname, succ, **graph.get_edge_data(oldname, succ))

    graph.remove_node(oldname)

def generate_value_expression(nodename):
    node = ExpressionGraphNode.node_value_dict[nodename]
    if(len(node.value.free_symbols) == 0): val = node.value.evalf(4)
    else: val = node.value
    s = f"${node.get_name()}= {sp.latex(val)}$"
    if(len(s) > 200): return f"${node.get_name()}$"
    return s

def generate_uncertainty_expression(nodename):
    node = ExpressionGraphNode.node_value_dict[nodename]
    if(len(node.uncertainty.free_symbols) == 0): unc = (node.uncertainty.evalf() / node.value.evalf()).evalf(4)
    else: unc = node.uncertainty
    return "$U_{" + node.get_name() + "}" + f" = {sp.latex(unc)}$"

def systematically_rename_nodes(graph: nx.DiGraph, expr_generating_function: function):
    for node, nodedata in dict(graph.nodes.items()).items():
        change_node_name(graph, node, expr_generating_function(node))

def show_graph(graph: nx.DiGraph):
    plt.figure(figsize=(12,12))
    pos = nx.multipartite_layout(graph, align="horizontal")

    node_colors = [graph.nodes[node]['color'] for node in graph.nodes()]

    # Draw the graph with node labels
    nx.draw(graph, pos, node_color=node_colors, with_labels=True, node_size=10000, font_size=10)

    # Draw edge labels
    edge_labels = nx.get_edge_attributes(graph, 'label')
    nx.draw_networkx_edge_labels(graph, pos, edge_labels=edge_labels)

    plt.show()
    

if __name__ == '__main__':

    a = ExpressionGraphNode().label("a")
    b = ExpressionGraphNode().label("b")
    c = a/b

    c.label("c")

    d = c*2
    d.label("d")

    print()
    a.set_numeric_measured_value(1,0.1)
    b.set_numeric_measured_value(1,0.1)
    # a._propagate_numeric_value()
    # print(a)
    # print()
    # print(c)

    # d.get_value_graph()
    d.get_uncertainty_graph()

    # graph = nx.DiGraph()
    # generate_graph(d, graph)
    # systematically_rename_nodes(graph, generate_value_expression)
    # show_graph(graph)
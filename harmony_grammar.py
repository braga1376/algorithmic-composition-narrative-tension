from typing import List, Tuple, Set, Optional
import random
from dataclasses import dataclass
from enum import Enum
from utils import Key

class Symbol(Enum):
    # Non-terminals
    TR = "TR"  # Tonic Region
    DR = "DR"  # Dominant Region
    SR = "SR"  # Subdominant Region
    
    # Terminals
    t = "t"   # tonic chord
    d = "d"   # dominant chord
    s = "s"   # subdominant chord
    tp = "tp" # tonic parallel chord
    dp = "dp" # dominant parallel chord
    sp = "sp" # subdominant parallel chord
    tcp = "tcp" # tonic cadential parallel chord
    
    def is_terminal(self) -> bool:
        return self in {Symbol.t, Symbol.d, Symbol.s, Symbol.tp, Symbol.dp, Symbol.sp, Symbol.tcp}
    
    def is_non_terminal(self) -> bool:
        return self in {Symbol.TR, Symbol.DR, Symbol.SR}
    
    def FunctionScaleInterface(self, key: Key) -> str:
        if self == Symbol.t:
            return 'I' 
        elif self == Symbol.s:
            return 'IV'
        elif self == Symbol.d:
            return 'V'
        elif self == Symbol.tp:
            return 'VI' if key.is_major else 'III'
        elif self == Symbol.dp:
            return 'VII'
        elif self == Symbol.sp:
            return 'II' if key.is_major else 'VI'
        elif self == Symbol.tcp:
            return 'III' if key.is_major else 'VI'
        else:
            return None

@dataclass
class Rule:
    left: Symbol
    right: List[Symbol]

    def __hash__(self):
        return hash((self.left, *self.right))
    
    def __eq__(self, other):
        if not isinstance(other, Rule):
            return False
        return (self.left == other.left) and all(s1 == s2 for s1, s2 in zip(self.right, other.right))

@dataclass
class TreeNode:
    symbol: Symbol
    children: List['TreeNode']
    rule_used: Optional[Rule] = None
    
    def get_terminals(self) -> List[Symbol]:
        """Get terminal symbols in order from this subtree"""
        if self.symbol.is_terminal():
            return [self.symbol]
        
        terminals = []
        for child in self.children:
            terminals.extend(child.get_terminals())
        return terminals
    
    def to_tuple(self) -> Tuple:
        """Convert tree to tuple format for comparison and deduplication"""
        if not self.children:
            return (self.symbol.value,)
        return (self.symbol.value, tuple(child.to_tuple() for child in self.children))
    
    def copy(self) -> 'TreeNode':
        """Create a deep copy of the tree"""
        new_children = [child.copy() for child in self.children]
        return TreeNode(self.symbol, new_children, self.rule_used)
    
    def get_terminals(self) -> List[Symbol]:
        """Get terminal symbols in order from this subtree"""
        if self.symbol.is_terminal():
            return [self.symbol]
        
        terminals = []
        for child in self.children:
            terminals.extend(child.get_terminals())
        return terminals
    
    def print_tree(self, prefix: str = "", is_last: bool = True) -> None:
        """
        Print tree structure using ASCII characters
        Args:
            prefix: Current prefix for the line (manages previous branches)
            is_last: Whether this is the last child of its parent
        Example output:
        TR
        ├── DR
        │   ├── SR
        │   │   └── s
        │   └── d
        └── t
        """
        connector = "└── " if is_last else "├── "
        print(prefix + connector + self.symbol.value)
        
        child_prefix = prefix + ("    " if is_last else "│   ")
        
        for i, child in enumerate(self.children):
            is_last_child = (i == len(self.children) - 1)
            child.print_tree(child_prefix, is_last_child)
    
    def to_structural_key(self) -> tuple:
        """Convert tree structure to hashable tuple for comparison"""
        if self.symbol.is_terminal():
            return (self.symbol,)
        
        children_keys = tuple(child.to_structural_key() for child in self.children)
        return (self.symbol, self.rule_used, children_keys)

class HarmonyGrammar:
    def __init__(self):
        self.rules = [
            Rule(Symbol.TR, [Symbol.DR, Symbol.t]),     # TR -> DR t
            Rule(Symbol.DR, [Symbol.SR, Symbol.d]),     # DR -> SR d
            Rule(Symbol.TR, [Symbol.TR, Symbol.DR]),    # TR -> TR DR
            Rule(Symbol.TR, [Symbol.t]),                # TR -> t
            Rule(Symbol.DR, [Symbol.d]),                # DR -> d
            Rule(Symbol.SR, [Symbol.s]),                # SR -> s
            # XR -> XR XR is handled specially to controll recursivity and since it applies to any region
        ]
    
    def expand_symbol(self, symbol: Symbol, depth: int = 0, max_depth: int = 5) -> TreeNode:
        """
        Expand a symbol according to the grammar rules
        Returns a tree node representing the expansion
        """
        if depth > max_depth:
            # force terminal production if too deep
            if symbol.is_non_terminal():
                terminal_map = {
                    Symbol.TR: Symbol.t,
                    Symbol.DR: Symbol.d,
                    Symbol.SR: Symbol.s
                }
                return TreeNode(terminal_map[symbol], [])
            return TreeNode(symbol, [])
            
        if symbol.is_terminal():
            return TreeNode(symbol, [])
            
        applicable_rules = [rule for rule in self.rules if rule.left == symbol]
        
        # Add XR -> XR XR rule if symbol is non terminal
        if symbol.is_non_terminal():
            applicable_rules.append(Rule(symbol, [symbol, symbol]))
        
        chosen_rule = random.choice(applicable_rules)
        
        node = TreeNode(symbol, [])
        node.rule_used = chosen_rule
        
        for right_symbol in chosen_rule.right:
            child_node = self.expand_symbol(right_symbol, depth + 1, max_depth)
            node.children.append(child_node)
            
        return node
    
    def generate_progression(self, start_symbol: Symbol = Symbol.TR) -> TreeNode:
        """Generate a random valid progression starting from given symbol"""
        return self.expand_symbol(start_symbol)
    
    def generate_all_progressions(self, 
                                start_symbol: Symbol = Symbol.TR, 
                                min_length: int = 3,
                                max_length: int = 8,
                                max_depth: int = 5) -> List[TreeNode]:
        """
        Generate all possible progressions between min_length and max_length
        Args:
            start_symbol: Starting symbol for the progression
            min_length: Minimum length of progressions to generate
            max_length: Maximum length of progressions to generate
            max_depth: Maximum recursion depth to prevent infinite recursion
        Returns:
            A list of TreeNodes containing unique derivation trees
        """
        def expand_all(symbol: Symbol, current_length: int = 0, depth: int = 0, recursive: bool = True) -> List[TreeNode]:
            if depth >= max_depth:
                return []
                
            if current_length > max_length:
                return []
                
            if symbol.is_terminal():
                return [TreeNode(symbol, [])]
            
            applicable_rules = [rule for rule in self.rules if rule.left == symbol]
            recursive_rule = Rule(symbol, [symbol, symbol])
            # Only add recursive rule XR -> XR XR if not reached max_depth - 1
            if recursive and symbol.is_non_terminal() and depth < max_depth - 1:
                applicable_rules.append(recursive_rule)
            
            results = []
            seen_tuples = set() 
            
            for rule in applicable_rules:
                if rule == recursive_rule:
                    next_recursive = False
                else:
                    next_recursive = True
                if len(rule.right) == 1:

                    sub_trees = expand_all(rule.right[0], current_length, depth + 1, recursive=next_recursive)
                    for sub_tree in sub_trees:
                        new_node = TreeNode(symbol, [sub_tree])
                        new_node.rule_used = rule

                        terminals = new_node.get_terminals()
                        if len(terminals) <= max_length:
                            tree_tuple = new_node.to_tuple()
                            if tree_tuple not in seen_tuples:
                                seen_tuples.add(tree_tuple)
                                results.append(new_node)
                    
                elif len(rule.right) == 2:
                    left_trees = expand_all(rule.right[0], current_length, depth + 1, recursive=next_recursive)
                    for left in left_trees:
                        left_terminals = len(left.get_terminals())
                        if current_length + left_terminals <= max_length:
                            right_trees = expand_all(rule.right[1], 
                                                   current_length + left_terminals, 
                                                   depth + 1, recursive=next_recursive)
                            for right in right_trees:
                                new_node = TreeNode(symbol, [left, right])
                                new_node.rule_used = rule
                                terminals = new_node.get_terminals()
                                if len(terminals) <= max_length:
                                    tree_tuple = new_node.to_tuple()
                                    if tree_tuple not in seen_tuples:
                                        seen_tuples.add(tree_tuple)
                                        results.append(new_node)
            
            return results
            
        all_trees = expand_all(start_symbol)
        return [tree for tree in all_trees 
                if min_length <= len(tree.get_terminals()) <= max_length]
    
    def generate_fixed_length_progressions(self, length: int, 
                                        start_symbol: Symbol = Symbol.TR) -> List[TreeNode]:
        """
        Generate progressions with exactly the specified number of terminals 
        using breadth-first expansion.
        """
        def count_non_terminals(node: TreeNode) -> int:
            """Count non-terminal symbols in tree"""
            if node.symbol.is_terminal():
                return 0
            return 1 + sum(count_non_terminals(child) for child in node.children)

        def convert_to_terminals(node: TreeNode) -> TreeNode:
            """Convert non-terminal nodes to their corresponding terminals"""
            if node.symbol.is_terminal():
                return node
                
            terminal_map = {
                Symbol.TR: Symbol.t,
                Symbol.DR: Symbol.d,
                Symbol.SR: Symbol.s
            }
            
            if not node.children:
                return TreeNode(terminal_map[node.symbol], [])
                
            node.children = [convert_to_terminals(child) for child in node.children]
            return node

        def expand_breadth_first(start_node: TreeNode) -> List[TreeNode]:
            trees = [start_node]
            current_size = 1
            
            while current_size < length:
                new_trees = []
                for tree in trees:
                    non_terminals = []
                    stack = [(tree, [])]
                    while stack:
                        node, path = stack.pop()
                        if node.symbol.is_non_terminal() and not node.children:
                            non_terminals.append((node, path))
                        for i, child in enumerate(node.children):
                            stack.append((child, path + [i]))
                    
                    for node, path in non_terminals:
                        applicable_rules = [rule for rule in self.rules if rule.left == node.symbol]
                        if node.symbol.is_non_terminal():
                            applicable_rules.append(Rule(node.symbol, [node.symbol, node.symbol]))
                        
                        for rule in applicable_rules:
                            new_tree = tree.copy()
                            target_node = new_tree
                            for idx in path:
                                target_node = target_node.children[idx]
                                
                            target_node.children = [TreeNode(sym, []) for sym in rule.right]
                            target_node.rule_used = rule
                            new_trees.append(new_tree)
                
                trees = new_trees
                current_size += 1
            
            result_trees = []
            for tree in trees:
                remaining = length - count_non_terminals(tree)
                if remaining >= 0:
                    result_trees.append(convert_to_terminals(tree.copy()))
                    
            return result_trees

        initial_node = TreeNode(start_symbol, [])
        return expand_breadth_first(initial_node)
    
    def generate_random_tree_breadth_first(self, length: int, start_symbol: Symbol = Symbol.TR) -> Optional[TreeNode]:
        """
        Generate a random tree with exact number of terminals using breadth-first expansion.
        Returns None if unable to generate tree with exact length.
        """
        def count_leaves(node: TreeNode) -> int:
            """Count all leaf nodes"""
            if not node.children:
                return 1
            return sum(count_leaves(child) for child in node.children)

        def convert_to_terminals(node: TreeNode) -> TreeNode:
            if not node.children:
                terminal_map = {
                    Symbol.TR: Symbol.t,
                    Symbol.DR: Symbol.d,
                    Symbol.SR: Symbol.s
                }
                if node.symbol in terminal_map:
                    node.symbol = terminal_map[node.symbol]
                return node
                
            node.children = [convert_to_terminals(child) for child in node.children]
            return node

        # Try multiple times to get a tree of correct size
        max_attempts = 100
        for _ in range(max_attempts):
            root = TreeNode(start_symbol, [])
            current_leaves = 1
            
            # Keep expanding until we get exactly the right number of leaves
            while current_leaves < length:
                expandable = []
                stack = [(root, [])]
                while stack:
                    node, path = stack.pop()
                    if not node.children and node.symbol.is_non_terminal():
                        expandable.append((node, path))
                    for i, child in enumerate(node.children):
                        stack.append((child, path + [i]))
                
                if not expandable:
                    break  # Start over if we can't expand further
                    
                node, path = random.choice(expandable)
                applicable_rules = [rule for rule in self.rules if rule.left == node.symbol]
                applicable_rules.append(Rule(node.symbol, [node.symbol, node.symbol]))
                rule = random.choice(applicable_rules)
                
                # Before applying rule, check if it would create too many leaves
                new_leaves = current_leaves - 1 + len(rule.right)  # -1 for expanded node, + new nodes
                if new_leaves > length:
                    continue  # Skip this expansion if it would create too many leaves
                
                # Apply rule
                target_node = root
                for idx in path:
                    target_node = target_node.children[idx]
                target_node.children = [TreeNode(sym, []) for sym in rule.right]
                target_node.rule_used = rule
                
                current_leaves = count_leaves(root)
            
            # Only convert and return if we got exactly the right number of leaves
            if current_leaves == length:
                return convert_to_terminals(root)
        
        return None 
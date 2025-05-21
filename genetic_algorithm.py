
from dataclasses import dataclass
from typing import List, Dict, Set, Tuple

import music21
from harmony_grammar import TreeNode, Symbol
from utils import MAX_PITCH, MIN_PITCH, Note, Motif, Key, calculate_similarities, sliding_windows, split_motif_into_bars, normalize_tension, get_interval_pattern, get_melodic_contour, get_rhythm_pattern, ROOT_TO_NOTE, NOTE_TO_ROOT, SCALE_DEGREES_INDEX, KEYS_MAJOR, KEYS_MINOR
from tension_model.tension_ribbons import calculate_tension, calculate_tension_fast
from narrative_tension import TensionCurve
from motif_composer import MIN_DURATION
import random
import copy
import itertools
import numpy as np
from scipy import stats

class MusicalNode:
    def __init__(self, symbol: Symbol, key: Key, children: List['MusicalNode']=None, parent: 'MusicalNode'=None, melody: List[Note]=None):
        self.symbol = symbol
        self.key = key
        self.parent = parent
        if parent:
            self._update_parent_children()
        self.children = children or []
        if symbol.is_terminal():
            self.melody = melody or []
        else:
            self.melody = None

    def _update_parent_children(self):
        """Update parent reference"""
        if self.parent:
            self.parent.children.append(self)

    def add_child(self, child: 'MusicalNode'):
        """Add a child node"""
        self.children.append(child)
        child.parent = self

    def remove_child(self, child: 'MusicalNode'):
        """Remove a child node"""
        self.children.remove(child)
        child.parent = None
    
    @classmethod
    def from_tree_node(cls, tree_node: TreeNode, key: Key) -> 'MusicalNode':
        """Convert TreeNode and all its children to MusicalNode"""
        new_node = cls(tree_node.symbol, key)

        for child in tree_node.children:
            new_node.add_child(cls.from_tree_node(child, key))

        return new_node
    
    def get_terminal_nodes(self) -> List['MusicalNode']:
        """Get all terminal nodes"""
        if self.symbol.is_terminal():
            return [self]
        else:
            return [child for child in self.children for child in child.get_terminal_nodes()]

    def get_last_chord_in_region(self) -> Symbol:
        """Get last terminal symbol and its key in a region"""
        if self.symbol.is_terminal():
            return (self.symbol, self.key)
            
        for child in reversed(self.children):
            if child.symbol == self.symbol: 
                return child.get_last_chord_in_region()
            elif child.symbol.is_terminal():
                return child.symbol
        return None
    
    def modulate_region(self, new_key: Key, modulation_function: Symbol, terminal_chord_type: Symbol) -> None:
        """Modulate all children in a region to new key except terminal symbols"""
        
        if self.symbol.is_non_terminal():
            if self.symbol != modulation_function:
                self.key = new_key
            for child in self.children:
                child.modulate_region(new_key, modulation_function, terminal_chord_type)
        elif self.symbol.is_terminal() and self.symbol != terminal_chord_type:
            self.key = new_key
            
    def print_tree(self, prefix: str = "", is_last: bool = True) -> None:
        """Print tree structure"""
        connector = "└── " if is_last else "├── "
        print(f"{prefix}{connector}{self.symbol.value} ({ROOT_TO_NOTE[self.key.root]}, {'maj' if self.key.is_major else 'min'})")
        
        child_prefix = prefix + ("    " if is_last else "│   ")
        for i, child in enumerate(self.children):
            is_last_child = (i == len(self.children) - 1)
            child.print_tree(child_prefix, is_last_child)

    def calculate_child_key(self, child_function: Symbol) -> Key:
        """Calculate key for child based on functional relationship"""
        intervals = {
            Symbol.DR: 7,   
            Symbol.SR: 5,
        }
        
        if child_function in intervals:
            new_root = (self.key.root + intervals[child_function]) % 12
            return Key(new_root, self.key.is_major)
        return self.key
    
    def get_absolute_time(self) -> float:
        """Get absolute time by traversing up the tree"""
        time = 0
        current = self
        while current.parent:
            time += current.get_duration_before()
            current = current.parent
        return time
    
    def get_duration_before(self) -> float:
        """Get duration of all previous siblings"""
        if not self.parent:
            return 0
        duration = 0
        for sibling in self.parent.children:
            if sibling == self:
                break
            duration += sibling.get_total_duration()
        return duration

    def get_total_duration(self) -> float:
        """Get total duration of this node and its children"""
        if self.symbol.is_terminal():
            return sum(note.duration for note in self.melody)
        return sum(child.get_total_duration() for child in self.children)
    
    def find_node_at_depth(self, depth: int) -> List['MusicalNode']:
        """Find all nodes at given depth"""
        if depth == 0:
            return [self]
        
        nodes = []
        for child in self.children:
            nodes.extend(child.find_node_at_depth(depth - 1))
        return nodes
    
    def copy(self) -> 'MusicalNode':
        """Create a deep copy of this node and its subtree"""
        new_node = MusicalNode(
            symbol=self.symbol,
            key=Key(self.key.root, self.key.is_major),
            melody=copy.deepcopy(self.melody)
        )
        
        for child in self.children:
            child_copy = child.copy()
            new_node.add_child(child_copy)
            
        return new_node


    def assign_motifs_melody_to_terminals(self, motifs: List[Motif], beats_per_bar: int = 4):
        """
        Assign bars from motifs to terminal nodes in order
        Args:
            motifs: List of motifs (each motif is a list of notes)
            beats_per_bar: Number of beats per bar
        """
        terminal_nodes = self.get_terminal_nodes()
        
        all_bars = []
        for motif in motifs:
            bars = split_motif_into_bars(motif.notes, beats_per_bar)
            all_bars.extend(bars)
        
        if len(terminal_nodes) != len(all_bars):
            raise ValueError("Number of terminal nodes and bars do not match")
        
        for bar in all_bars:
            start_time = 0
            for note in bar:
                note.start_time = start_time
                start_time += note.duration

        for node, bar in zip(terminal_nodes, all_bars):
            node.melody = bar
    
    def get_melody_per_bar(self) -> List[List[Note]]:
        """Get melody split into bars"""
        terminal_nodes = self.get_terminal_nodes()
        return [node.melody for node in terminal_nodes]
    
    def get_melody(self) -> List[Note]:
        """Get melody from all terminal nodes"""
        terminal_nodes = self.get_terminal_nodes()
        return [note for node in terminal_nodes for note in node.melody]

    def get_harmony_melody_notes(self) -> List[Tuple[Set[int], float]]:
        # NEW VERSION TO HANDLE SILENCES
        """Get sliding window of notes from harmony and melody with exact durations"""
        all_bars = []
        terminal_nodes = self.get_terminal_nodes()
        for terminal_node in terminal_nodes:
            melody = terminal_node.melody
            chord = self.key.get_chord(SCALE_DEGREES_INDEX[terminal_node.symbol.FunctionScaleInterface(self.key)])
            for note in melody:
                all_bars.append((set((note.pitch%12, *chord)), note.duration))
        return all_bars

    # def calculate_tension(self) -> List[float]:
    #     """Calculate tension for all terminal nodes"""
    #     all_bars = self.get_harmony_melody_notes()
    #     cloud_diameter, cloud_momentum, tensile_strain = calculate_tension(all_bars, self.key.root, self.key.is_major)
    #     cloud_diameter_normalized = normalize_tension(cloud_diameter)
    #     cloud_momentum_normalized = normalize_tension(cloud_momentum)
    #     tensile_strain_normalized = normalize_tension(tensile_strain)
    #     tensions = np.array([cloud_diameter_normalized, cloud_momentum_normalized, tensile_strain_normalized])
    #     tension = np.average(tensions, axis=0, weights=[1./3, 1./3, 1./3])
    #     return tension
    
    # def calculate_tension(self) -> np.ndarray:
    #     """Calculate normalized tension --- FASTER (not much, ~80% faster) VERSION"""
    #     all_bars = self.get_harmony_melody_notes()
        
    #     diameter, momentum, strain = calculate_tension_fast(all_bars, self.key.root, self.key.is_major)
        
    #     components = np.array([
    #         (diameter - np.min(diameter)) / (np.max(diameter) - np.min(diameter) + 1e-10),
    #         (momentum - np.min(momentum)) / (np.max(momentum) - np.min(momentum) + 1e-10),
    #         (strain - np.min(strain)) / (np.max(strain) - np.min(strain) + 1e-10)
    #     ])
        
    #     return np.average(components, axis=0, weights=[1./3, 1./3, 1./3])
    
    def calculate_tension(self, target_resolution: int = 128) -> np.ndarray:
        """Calculate normalized tension with consistent resolution"""
        all_bars = self.get_harmony_melody_notes()
        
        notes, durations = zip(*all_bars)
        total_duration = sum(durations)
        
        diameter, momentum, strain = calculate_tension_fast(notes, self.key.root, self.key.is_major)
        
        original_times = np.cumsum([0] + list(durations))
        target_times = np.linspace(0, total_duration, target_resolution)
        
        components = []
        for tension in [diameter, momentum, strain]:
            interpolated = np.interp(target_times, original_times[:-1], tension)
            normalized = (interpolated - np.min(interpolated)) / (np.max(interpolated) - np.min(interpolated) + 1e-10)
            components.append(normalized)
        
        return np.average(components, axis=0, weights=[1./3, 1./3, 1./3])


class MotifHarmonicCombiner:
    def __init__(self, motifs: List[Motif], protagonist_index = None, plot_tension: List[float] = None, target_length_bars: int = 8, motif_length_bars: int = 2):
        self.motifs = motifs
        self.motifs_patterns = []
        self.initialize_motif_patterns()
        self.protagonist_index = protagonist_index
        self.plot_tension = plot_tension
        self.target_length_bars = target_length_bars
        self.motif_length_bars = motif_length_bars

    def create_initial_population(self,
                                harmonic_trees: List[TreeNode],
                                population_size: int = 200) -> List[MusicalNode]:
        """Create initial population combining motifs with harmonic progressions"""
        population = []
        
        harmonic_trees = random.sample(harmonic_trees, min(len(harmonic_trees), population_size))

        if self.protagonist_index != None:
            key = Key(NOTE_TO_ROOT[self.motifs[self.protagonist_index].key], self.motifs[self.protagonist_index].is_major)
        else:
            key = Key(NOTE_TO_ROOT[self.motifs[0].key], self.motifs[0].is_major)

        for motif in self.motifs:
            motif.transpose(ROOT_TO_NOTE[key.root], key.is_major)

        num_repetitions = self.target_length_bars // self.motif_length_bars
        arrangements = list(itertools.product(self.motifs, repeat=num_repetitions))

        for tree in harmonic_trees:
            node = MusicalNode.from_tree_node(tree, key)
            for arrangement in arrangements:
                node_copy = node.copy()
                node_copy.assign_motifs_melody_to_terminals(arrangement)
                population.append(node_copy)
            
        population.sort(key=self.__evaluate_fitness, reverse=True)
        return population[:population_size]
    
    def initialize_motif_patterns(self):
        """Precompute patterns for all motifs"""
        self.motif_patterns = []
        for motif in self.motifs:
            notes = motif.notes
            pitches = np.array([n.pitch for n in notes])
            durations = np.array([n.duration for n in notes])

            intervals = get_interval_pattern(pitches) 
            contour = get_melodic_contour(intervals)
            rhythm = get_rhythm_pattern(durations)
            self.motif_patterns.append((contour, intervals, rhythm))

    
    def __evaluate_fitness(self, node: MusicalNode) -> float:
        """Initial fitness evaluation for population selection"""
        fitness = 0.0
        
        key_comp_fitness = self.evaluate_key_compatibility(node)
        
        character_fitness = self.evaluate_character_motif_recognition(node)
        
        tension_fitness = self.evaluate_tension_alignment(node)

        fitness = np.average([key_comp_fitness, character_fitness, tension_fitness], weights=[1./3, 1./3, 1./3])
        
        return fitness
    
    def evaluate_key_compatibility(self, node: MusicalNode) -> float:
        terminal_nodes = node.get_terminal_nodes()

        BEAT_WEIGHTS = np.array([4, 2, 3, 1])
        DEFAULT_WEIGHT = 0.5
        
        score = 0.0
        total_weight = 0.0
        
        for terminal_node in terminal_nodes:
            chord_notes = set(terminal_node.key.get_chord(
                SCALE_DEGREES_INDEX[terminal_node.symbol.FunctionScaleInterface(terminal_node.key)]
            ))
            scale_notes = set(terminal_node.key.get_scale_degree())
            
            for i, note in enumerate(terminal_node.melody):
                beats = np.arange(note.start_time, note.start_time + note.duration, 0.25)
                beat_positions = beats % 4
                
                integer_mask = beat_positions.astype(int) == beat_positions
                max_integer_weight = np.max(BEAT_WEIGHTS[beat_positions[integer_mask].astype(int)]) if any(integer_mask) else 0
                note_weight = max(max_integer_weight, DEFAULT_WEIGHT) * note.duration
                
                pitch_class = note.pitch % 12
                if pitch_class in chord_notes:
                    score += note_weight
                elif pitch_class in scale_notes and not any((pitch_class - cn) % 12 == 1 for cn in chord_notes):
                    score += 0.7 * note_weight

                if i > 3:
                    if note.pitch == terminal_node.melody[i-1].pitch:
                        if terminal_node.melody[i-1].pitch == terminal_node.melody[i-2].pitch:
                            score = 0
                
                # Melody Range
                if note.pitch < 46 or note.pitch > 83:
                    score = score * 0.3
                
                total_weight += note_weight
        
        return score / total_weight if total_weight > 0 else 0.0

    def evaluate_character_motif_recognition(self, node: MusicalNode) -> float:
        melody = node.get_melody()
        if not melody:
            return 0.0
            
        melody_pitches = np.array([n.pitch for n in melody])
        melody_durations = np.array([n.duration for n in melody])
        protagonist_weight = 1.3
        
        motif_scores = []
        for i, (motif_contour, motif_intervals, motif_rhythm) in enumerate(self.motif_patterns):
            motif_weight = protagonist_weight if i == self.protagonist_index else 1.0

            window_size = len(self.motifs[i].notes)
            
            if window_size > len(melody):
                motif_scores.append(0.0)
                continue
                
            pitch_windows = sliding_windows(melody_pitches, window_size)
            if len(pitch_windows) == 0:
                motif_scores.append(0.0)
                continue
                
            duration_windows = sliding_windows(melody_durations, window_size)
            if len(duration_windows) == 0:
                motif_scores.append(0.0)
                continue
            denominators = duration_windows[:, :-1]
            if np.any(denominators == 0):
                motif_scores.append(0.0)
                continue

            window_rhythms = np.divide(
                duration_windows[:, 1:], 
                denominators, 
                out=np.zeros_like(duration_windows[:, 1:]),
                where=denominators != 0
)

            window_intervals = np.diff(pitch_windows, axis=1)
            window_contours = np.sign(window_intervals)
            # window_rhythms = duration_windows[:, 1:] / duration_windows[:, :-1]
            
            similarities = calculate_similarities(
                window_contours, window_intervals, window_rhythms,
                motif_contour, motif_intervals, motif_rhythm
            )
            
            if len(similarities) > 0:
                top_matches = sorted(similarities, reverse=True)[:min(4, len(similarities))]
                avg_top_matches = sum(top_matches) / len(top_matches)
                motif_scores.append(avg_top_matches * motif_weight)
            else:
                motif_scores.append(0.0)
        
        if not motif_scores:
            return 0.0
        

        weights_sum = sum(protagonist_weight if i == self.protagonist_index else 1.0 
                        for i in range(len(self.motifs)))
            
        return sum(motif_scores) / weights_sum if weights_sum > 0 else 0.0

    def evaluate_tension_alignment(self, node: MusicalNode) -> float:
        """Evaluate how well tension aligns with desired dramatic curve"""
        musical_tension = node.calculate_tension( len(self.plot_tension))
        if len(musical_tension) != len(self.plot_tension):
            print("Tension length mismatch")
            print("Musical tension:", len(musical_tension))
            print("Plot tension:", len(self.plot_tension))
        correlation = stats.pearsonr(musical_tension, self.plot_tension)[0]
        return (correlation + 1) / 2
    
    def optimize(self, harmonic_trees: List[TreeNode], population_size: int = 200, n_gen: int = 100) -> List[MusicalNode]:
        initial_population = self.create_initial_population(harmonic_trees, population_size)
        nsga2 = NSGAII(population_size=population_size, 
                    objectives=[self.evaluate_key_compatibility,
                                self.evaluate_character_motif_recognition,
                                self.evaluate_tension_alignment])
        return nsga2.evolve(initial_population, n_gen)

class Individual:
    def __init__(self, musical_node: MusicalNode):
        self.musical_node = musical_node
        self.objectives = []
        self.rank = float('inf')
        self.crowding_distance = 0
    
    def dominates(self, other: 'Individual') -> bool:
        at_least_one_better = False
        for self_obj, other_obj in zip(self.objectives, other.objectives):
            if self_obj < other_obj:
                return False
            elif self_obj > other_obj:
                at_least_one_better = True
        return at_least_one_better

    def copy(self) -> 'Individual':
        new_individual = Individual(self.musical_node.copy())
        new_individual.objectives = self.objectives.copy()
        new_individual.rank = self.rank
        new_individual.crowding_distance = self.crowding_distance
        return new_individual

class NSGAII:
    def __init__(self, population_size: int, objectives: List[callable], 
                 crossover_prob: float = 0.9, mutation_prob: float = 0.2, 
                 crossover_type_probs: Dict[str, float] = None, 
                 mutation_type_probs: Dict[str, float] = None):
        self.population_size = population_size
        self.objectives = objectives
        self.crossover_prob = crossover_prob
        self.mutation_prob = mutation_prob
        self.crossover_type_probs = crossover_type_probs or {
            'melody': 1./3,
            'harmony': 1./3,
            'melody_harmony': 1./3
        }
        self.mutation_type_probs = mutation_type_probs or {
            'melody': 0.5,
            'harmony': 0.5
        }

    def evaluate_objectives(self, individual: Individual) -> None:
        individual.objectives = []
        for objective in self.objectives:
            individual.objectives.append(objective(individual.musical_node))

    def fast_non_dominated_sort(self, population: List[Individual]) -> List[List[Individual]]:
        """Sort population into Pareto fronts"""
        domination_counts = [0] * len(population)
        dominated_solutions = [[] for _ in range(len(population))]
        fronts = [[]]
        
        for i, p in enumerate(population):
            for j, q in enumerate(population[i+1:], i+1):
                if p.dominates(q):
                    dominated_solutions[i].append(q)
                    domination_counts[j] += 1
                elif q.dominates(p):
                    dominated_solutions[j].append(p)
                    domination_counts[i] += 1
                    
            if domination_counts[i] == 0:
                p.rank = 0
                fronts[0].append(p)
        
        i = 0
        while fronts[i]:
            next_front = []
            for p in fronts[i]:
                p_idx = population.index(p)
                for q in dominated_solutions[p_idx]:
                    q_idx = population.index(q)
                    domination_counts[q_idx] -= 1
                    if domination_counts[q_idx] == 0:
                        q.rank = i + 1
                        next_front.append(q)
            i += 1
            fronts.append(next_front)
            
        return fronts[:-1]
    
    def crowding_distance_sort(self, front: List[Individual]) -> List[Individual]:
        """Calculate crowding distance for individuals in a front"""
        if len(front) <= 2:
            return front
            
        for ind in front:
            ind.crowding_distance = 0
            
        for obj_idx in range(len(self.objectives)):
            front.sort(key=lambda x: x.objectives[obj_idx])
            
            front[0].crowding_distance = float('inf')
            front[-1].crowding_distance = float('inf')
            
            obj_range = front[-1].objectives[obj_idx] - front[0].objectives[obj_idx]
            if obj_range == 0:
                continue
                
            for i in range(1, len(front)-1):
                front[i].crowding_distance += (
                    front[i+1].objectives[obj_idx] - front[i-1].objectives[obj_idx]
                ) / obj_range
        
        return sorted(front, key=lambda x: x.crowding_distance, reverse=True)
    
    def tournament_selection(self, population: List[Individual]) -> List[Individual]:
        """Binary tournament selection based on rank and crowding distance"""
        selected = []
        
        while len(selected) < self.population_size:
            i1, i2 = random.sample(range(len(population)), 2)
            p1, p2 = population[i1], population[i2]
            
            if p1.rank < p2.rank:
                selected.append(p1)
            elif p2.rank < p1.rank:
                selected.append(p2)
            else:
                if p1.crowding_distance > p2.crowding_distance:
                    selected.append(p1)
                else:
                    selected.append(p2)

        return selected
    
    def crossover(self, p1: Individual, p2: Individual, crossover_type: str) -> Tuple[Individual, Individual]:
        if crossover_type == 'melody':
            return self.melody_crossover(p1, p2)
        elif crossover_type == 'harmony':
            return self.harmony_crossover(p1, p2)
        else:  # melody and harmony
            c1, c2 = self.harmony_crossover(p1, p2)
            return self.melody_crossover(c1, c2)

    def melody_crossover(self, p1: Individual, p2: Individual) -> Tuple[Individual, Individual]:
        c1, c2 = p1.copy(), p2.copy()
        terminals1 = c1.musical_node.get_terminal_nodes()
        terminals2 = c2.musical_node.get_terminal_nodes()
        
        point = random.randint(0, len(terminals1)-1)
        for i in range(point, len(terminals1)):
            t2_copy = copy.deepcopy(terminals2[i].melody)
            t1_copy = copy.deepcopy(terminals1[i].melody)
            terminals1[i].melody = t2_copy
            terminals2[i].melody = t1_copy
        
        return c1, c2

    def harmony_crossover(self, p1: Individual, p2: Individual) -> Tuple[Individual, Individual]:
        """Crossover that ensures the number of terminals remains constant"""
        c1, c2 = p1.copy(), p2.copy()
        
        def count_terminals(node: MusicalNode) -> int:
            if node.symbol.is_terminal():
                return 1
            return sum(count_terminals(child) for child in node.children)
        
        def verify_tree(node: MusicalNode, expected_terminals: int) -> bool:
            """Verify tree has correct number of terminals"""
            actual = count_terminals(node)
            return actual == expected_terminals
        
        def find_valid_subtrees(root: MusicalNode) -> List[Tuple[MusicalNode, List[int], int]]:
            """Find all subtrees with their paths and terminal counts"""
            valid_subtrees = []
            stack = [(root, [])]
            
            while stack:
                node, path = stack.pop()
                if node.symbol.is_non_terminal():
                    terminal_count = count_terminals(node)
                    if terminal_count > 1: 
                        valid_subtrees.append((node, path, terminal_count))
                for i, child in enumerate(node.children):
                    if child.symbol.is_non_terminal():
                        stack.append((child, path + [i]))
                        
            return valid_subtrees

        initial_count1 = count_terminals(c1.musical_node)
        initial_count2 = count_terminals(c2.musical_node)
        
        subtrees1 = find_valid_subtrees(c1.musical_node)
        subtrees2 = find_valid_subtrees(c2.musical_node)
        
        compatible_pairs = []
        for node1, path1, count1 in subtrees1:
            for node2, path2, count2 in subtrees2:
                if (node1.symbol == node2.symbol and 
                    count1 == count2):
                    temp1, temp2 = c1.copy(), c2.copy()
                    
                    parent1 = temp1.musical_node
                    parent2 = temp2.musical_node
                    for i in path1[:-1]:
                        parent1 = parent1.children[i]
                    for i in path2[:-1]:
                        parent2 = parent2.children[i]
                    
                    idx1 = path1[-1] if path1 else 0
                    idx2 = path2[-1] if path2 else 0
                    
                    parent1.children[idx1], parent2.children[idx2] = \
                        parent2.children[idx2].copy(), parent1.children[idx1].copy()
                    
                    if (verify_tree(temp1.musical_node, initial_count1) and 
                        verify_tree(temp2.musical_node, initial_count2)):
                        compatible_pairs.append(((node1, path1), (node2, path2)))
        
        if compatible_pairs:
            (node1, path1), (node2, path2) = random.choice(compatible_pairs)
            
            parent1 = c1.musical_node
            parent2 = c2.musical_node
            for i in path1[:-1]:
                parent1 = parent1.children[i]
            for i in path2[:-1]:
                parent2 = parent2.children[i]
            
            idx1 = path1[-1] if path1 else 0
            idx2 = path2[-1] if path2 else 0
            
            parent1.children[idx1], parent2.children[idx2] = \
                parent2.children[idx2].copy(), parent1.children[idx1].copy()
            
            if not (verify_tree(c1.musical_node, initial_count1) and 
                    verify_tree(c2.musical_node, initial_count2)):
                print("Problem in harmony crossover")
                return p1.copy(), p2.copy()
        
        return c1, c2
    
    def prepare_population(self, population: List[MusicalNode]) -> List[Individual]:
        individuals = []
        for node in population:
            ind = Individual(node)
            self.evaluate_objectives(ind)
            individuals.append(ind)
        return individuals
    
    def evolve(self, population: List[MusicalNode], num_generations: int) -> Tuple[List[Individual]]:
        """Main NSGA-II evolution process"""
        
        tracking = {
            'objective_values': [],
            'pareto_fronts': [],
            'hypervolume': []
        }

        population = self.prepare_population(population)

        fronts = self.fast_non_dominated_sort(population)
        for front in fronts:
            self.crowding_distance_sort(front)

        for gen in range(num_generations):
            print(f"Generation {gen+1}/{num_generations}")
            offspring = []
            parents = self.tournament_selection(population)
            
            for i in range(0, len(parents), 2):
                if i + 1 < len(parents) and random.random() < self.crossover_prob:
                    crossover_type = random.choices(
                        list(self.crossover_type_probs.keys()),
                        list(self.crossover_type_probs.values())
                    )[0]
                    c1, c2 = self.crossover(parents[i], parents[i+1], crossover_type)
                else:
                    c1 = parents[i].copy()
                    if i + 1 < len(parents):
                        c2 = parents[i+1].copy()

                if (random.random() < self.mutation_prob) and c1:
                    c1 = self.mutate(c1)
                if (random.random() < self.mutation_prob) and c2:
                    c2 = self.mutate(c2)
                offspring.extend([c1, c2])

            for ind in offspring:
                self.evaluate_objectives(ind)
            
            combined = population + offspring
                
            fronts = self.fast_non_dominated_sort(combined)
            
            new_population = []
            for front in fronts:
                if len(new_population) + len(front) <= self.population_size:
                    new_population.extend(front)
                else:
                    remaining = self.population_size - len(new_population)
                    sorted_front = self.crowding_distance_sort(front)
                    new_population.extend(sorted_front[:remaining])
                    break
                    
            population = new_population

            self._update_tracking(population, tracking, gen)
            
        return population, tracking
    
    def mutate(self, individual: Individual) -> Individual:
        mutation_type = random.choices(
            list(self.mutation_type_probs.keys()),
            list(self.mutation_type_probs.values())
        )[0]

        if mutation_type == 'melody':
            melody_mutations = {
                'transpose': (self.transpose_mutation, 0.1),
                'augment': (self.augment_mutation, 0.1),
                'diminish': (self.diminish_mutation, 0.1),
                'invert': (self.invert_mutation, 0.1),
                'retrograde': (self.retrograde_mutation, 0.1),
                'intervalic_change': (self.intervalic_change_mutation, 0.3),
                # 'fragment': (self.fragment_mutation, 0.1),  # FIX REST PROBLEM
                'ornament': (self.ornamentation_mutation, 0.2)
            }
            chosen_mutation = random.choices(
                list(melody_mutations.keys()),
                [w for _, w in melody_mutations.values()]
            )[0]
            return melody_mutations[chosen_mutation][0](individual)
        else:
            harmony_mutations = {
                'parallel': (self.parallel_substitution_mutation, 0.5),
                'modulation': (self.modulation_mutation, 0.5)
            }
            chosen_mutation = random.choices(
                list(harmony_mutations.keys()),
                [w for _, w in harmony_mutations.values()]
            )[0]
            return harmony_mutations[chosen_mutation][0](individual)

    def transpose_mutation(self, individual: Individual) -> Individual:
        def get_closest_scale_note_index(pitch: int, scale: List[int]) -> int:
            return min(range(len(scale)), key=lambda i: abs(scale[i] - pitch % 12))
        
        new_ind = individual.copy()
        terminals = new_ind.musical_node.get_terminal_nodes()

        # scale = new_ind.musical_node.key.get_scale_degree()
        index_transpose = random.randint(0, 6)

        for terminal in terminals:
            scale = terminal.key.get_scale_degree()
            for note in terminal.melody:
                index = get_closest_scale_note_index(note.pitch, scale)
                new_index = (index + index_transpose) % len(scale)
                interval = scale[new_index] - scale[index]
                if abs(interval) > 6:
                    if interval < 0:
                        interval += 12
                    else:
                        interval -= 12
                new_pitch = note.pitch + interval
                if MIN_PITCH <= new_pitch <= MAX_PITCH:
                    note.pitch = new_pitch
                    
        return new_ind
    
    def augment_mutation(self, individual: Individual) -> Individual:
        new_ind = individual.copy()
        terminals = new_ind.musical_node.get_terminal_nodes()
        factor = 2
        
        for terminal in terminals:
            if len(terminal.melody) < 2 or any(note.duration >= 4 for note in terminal.melody):
                continue
                
            mid = max(1, len(terminal.melody) // 2) 
            use_first_half = random.choice([True, False])
            source_melody = terminal.melody[:mid] if use_first_half else terminal.melody[mid:]
            
            if not source_melody:
                continue
                
            new_melody = []
            start_time = 0
            
            for note in source_melody:
                if start_time >= 4:
                    break
                new_note = note.copy()
                new_note.duration *= factor
                new_note.start_time = start_time
                if start_time + new_note.duration > 4:
                    new_note.duration = 4 - start_time
                start_time += new_note.duration
                new_melody.append(new_note)
            
            if new_melody:
                terminal.melody = new_melody
            else:
                print("Empty melody in augment mutation")
        
        return new_ind

    def diminish_mutation(self, individual: Individual) -> Individual:
        new_ind = individual.copy()
        terminals = new_ind.musical_node.get_terminal_nodes()
        factor = 0.5
        measure_length = 4.0

        for terminal in terminals:
            if any(note.duration < 0.5 for note in terminal.melody):
                continue
            
            if all(note.duration == 0.5 for note in terminal.melody):
                continue
        
                
            new_melody = []
            start_time = 0.0
            iter = 0
            
            while start_time < measure_length:
                for note in terminal.melody:
                    new_note = note.copy()
                    new_note.duration *= factor
                    new_note.start_time = start_time
                    
                    if start_time + new_note.duration > measure_length:
                        new_note.duration = measure_length - start_time

                    start_time += new_note.duration
                    new_melody.append(new_note)
                    
                iter += 1
                if iter > 10:
                    print("Infinite loop in diminish mutation")
                    print("Melody:")
                    for note in terminal.melody:
                        print(note.pitch, note.duration)
                    print("New melody:")
                    for note in new_melody:
                        print(note.pitch, note.duration)
                    new_melody = terminal.melody
                    break
            
            if new_melody:
                terminal.melody = new_melody
            else:
                print("Empty melody in diminish mutation")
        
        return new_ind
    
    def intervalic_change_mutation(self, individual: Individual) -> Individual:
        def get_closest_scale_note_index(pitch: int, scale: List[int]) -> int:
            return min(range(len(scale)), key=lambda i: abs(scale[i] - pitch % 12))
        
        new_ind = individual.copy()
        # melody = new_ind.musical_node.get_melody()
        # scale = new_ind.musical_node.key.get_scale_degree()
        terminals = new_ind.musical_node.get_terminal_nodes()

        for terminal in terminals:
            melody = terminal.melody
            n_notes_to_change = random.randint(1, max(1, len(melody) // 2))
            notes_to_change = random.sample(range(len(melody)), n_notes_to_change)
            scale = terminal.key.get_scale_degree()
            
            scale_intervals = [-3, -2, -1, 1, 2, 3] 
            
            for i in notes_to_change:
                current_index = get_closest_scale_note_index(melody[i].pitch, scale)
                new_index = (current_index + random.choice(scale_intervals)) % len(scale)
                interval = scale[new_index] - scale[current_index]
                if abs(interval) > 6:
                    if interval < 0:
                        interval += 12
                    else:
                        interval -= 12
                
                new_pitch = melody[i].pitch + interval
                if MIN_PITCH <= new_pitch <= MAX_PITCH:
                    melody[i].pitch = new_pitch
                
        return new_ind

    def invert_mutation(self, individual: Individual) -> Individual:
        new_ind = individual.copy()
        terminals = new_ind.musical_node.get_terminal_nodes()
        for terminal in terminals:
            if not terminal.melody:
                continue
            axis = terminal.melody[0].pitch
            for note in terminal.melody:
                note.pitch = axis + (axis - note.pitch)
        return new_ind

    def retrograde_mutation(self, individual: Individual) -> Individual:
        new_ind = individual.copy()
        terminals = new_ind.musical_node.get_terminal_nodes()
        for terminal in terminals:
            terminal.melody.reverse()
            start_time = 0
            for note in terminal.melody:
                note.start_time = start_time + note.duration
                start_time += note.duration

        return new_ind

    def fragment_mutation(self, individual: Individual) -> Individual:
        new_ind = individual.copy()
        terminals = new_ind.musical_node.get_terminal_nodes()
        for terminal in terminals:
            if len(terminal.melody) <= 1:
                continue
            split_point = random.randint(1, len(terminal.melody)-1)
            terminal.melody = terminal.melody[:split_point] * 2
        return new_ind


    def ornamentation_mutation(self, individual: Individual) -> Individual:
        new_ind = individual.copy()
        terminals = new_ind.musical_node.get_terminal_nodes()
        
        for terminal in terminals:
            new_melody = []
            scale_notes = terminal.key.get_scale_degree()
            chord_notes = set(terminal.key.get_chord(
                SCALE_DEGREES_INDEX[terminal.symbol.FunctionScaleInterface(terminal.key)]
            ))
            
            start_time = 0
            for i, note in enumerate(terminal.melody):
                types_ornament = []
                if ((note.duration / 2) % 0.25) == 0:
                    types_ornament = ['passing', 'appoggiatura']
                    if ((note.duration / 4) % 0.25) == 0:
                        types_ornament.append('neighbor')

                if (note.duration >= 1) and (random.random() < 0.3) and types_ornament:
                    ornament_type = random.choice(types_ornament)
                    
                    current_pitch_scale = note.pitch % 12
                    scale_distances = np.abs(scale_notes - current_pitch_scale)
                    current_scale_pos = np.argmin(scale_distances)      

                    if ornament_type == 'passing':
                        if i < len(terminal.melody) - 1:
                            next_note = terminal.melody[i + 1]
                            next_pitch = next_note.pitch % 12
                            next_scale_distances = np.abs(scale_notes - next_pitch)
                            next_scale_pos = np.argmin(next_scale_distances)
                            
                            if abs(next_scale_pos - current_scale_pos) >= 2:
                                passing_scale_pos = (min(current_scale_pos, next_scale_pos) + 1)
                                passing_pitch_scale = scale_notes[passing_scale_pos]
                                interval = passing_pitch_scale - current_pitch_scale
                                if abs(interval) > 6:
                                    if interval < 0:
                                        interval += 12
                                    else:
                                        interval -= 12
                                passing_pitch = note.pitch + interval

                                half_duration = note.duration / 2
                                note.duration = half_duration
                                new_melody.append(note)
                                new_melody.append(Note(passing_pitch, half_duration, 
                                                    start_time + half_duration))
                            else:
                                new_melody.append(note)
                        else:
                            new_melody.append(note)
                    
                    elif ornament_type == 'neighbor':
                        direction = random.choice([-1, 1])
                        neighbor_scale_pos = current_scale_pos + direction
                        
                        neighbor_scale_pos = neighbor_scale_pos % len(scale_notes)
                        neighbor_pitch_scale = scale_notes[neighbor_scale_pos]
                        interval = neighbor_pitch_scale - current_pitch_scale
                        if abs(interval) > 6:
                            if interval < 0:
                                interval += 12
                            else:
                                interval -= 12

                        neighbor_pitch = note.pitch + interval
                        
                        durations = [[note.duration / 2, note.duration / 4, note.duration/4],
                                     [note.duration / 4, note.duration / 4, note.duration/2],
                                     [note.duration / 4, note.duration / 2, note.duration/4],
                                     [note.duration / 2, note.duration / 4, note.duration/4],
                                     [note.duration / 4, note.duration / 4, note.duration/2],
                                     [note.duration / 4, note.duration / 2, note.duration/4],
                                     [note.duration / 3, note.duration / 3, note.duration/3]
                                     ]
                        
                        durations = random.choice(durations)
                        note.duration = durations[0]
                        new_melody.append(note) 
                        new_melody.append(Note(neighbor_pitch, durations[1], 
                                            start_time + durations[1])) 
                        new_melody.append(Note(note.pitch, durations[2], 
                                            start_time + 2 * durations[2]))  
                    
                    elif ornament_type == 'appoggiatura':
                        if note.pitch % 12 in chord_notes:
                            for direction in [-1, 1]:
                                app_scale_pos = current_scale_pos + direction
                                app_scale_pos = app_scale_pos % len(scale_notes)

                                app_pitch_scale = scale_notes[app_scale_pos]
                                if app_pitch_scale not in chord_notes:
                                    interval = app_pitch_scale - current_pitch_scale
                                    if abs(interval) > 6:
                                        if interval < 0:
                                            interval += 12
                                        else:
                                            interval -= 12
                                    app_pitch = note.pitch + interval
                                    half_duration = note.duration / 2
                                    note.duration = half_duration
                                    new_melody.append(Note(app_pitch, half_duration, start_time))
                                    note.start_time += half_duration
                                    new_melody.append(note)
                                    break
                                elif direction == -1:
                                    continue
                                else:
                                    new_melody.append(note)
                        else:
                            new_melody.append(note)
                else:
                    new_melody.append(note)
                
                start_time += note.duration
            
            if new_melody:
                terminal.melody = new_melody
            else:
                print("Empty melody in ornamentation mutation")

        return new_ind
    
    def parallel_substitution_mutation(self, individual: Individual) -> Individual:
        new_ind = individual.copy()
        terminal_nodes = new_ind.musical_node.get_terminal_nodes()
        node = random.choice(terminal_nodes)
        if node.symbol.value == 't':
            node.symbol = random.choices([Symbol.tp, Symbol.tcp], weights=[0.7, 0.3])[0]
        elif node.symbol.value == 'd':
            node.symbol = Symbol.dp
        elif node.symbol.value == 's':
            node.symbol = Symbol.sp
        return new_ind

    def modulation_mutation(self, individual: Individual) -> Individual:
        new_ind = individual.copy()
        
        def find_modulatable_regions(node: MusicalNode) -> List[Tuple[MusicalNode, Symbol]]:
            """Find non-tonic regions that can be modulated"""
            modulatable = []
            if node.symbol.is_non_terminal():
                last_symbol = node.get_last_chord_in_region()
                if (last_symbol is not None and 
                    last_symbol != Symbol.t and 
                    not (node.key.is_major and last_symbol == Symbol.dp)):
                    modulatable.append((node, last_symbol))
                    
            for child in node.children:
                modulatable.extend(find_modulatable_regions(child))
                    
            return modulatable
        
        non_tonic_nodes = find_modulatable_regions(new_ind.musical_node)
        
        if non_tonic_nodes:
            mod_node = random.choice(non_tonic_nodes)
            key_root, is_major = (KEYS_MAJOR[mod_node[1].value] 
                                if mod_node[0].key.is_major 
                                else KEYS_MINOR[mod_node[1].value])
            mod_node[0].modulate_region(
                Key((key_root + mod_node[0].key.root) % 12, is_major),
                mod_node[0].symbol,
                mod_node[1]
            )
        
        return new_ind

    def _update_tracking(self, population: List[Individual], tracking: Dict, generation: int):
        """Update tracking metrics for current generation"""
        obj_values = np.array([ind.objectives for ind in population])
        tracking['objective_values'].append({
            'min': np.min(obj_values, axis=0),
            'max': np.max(obj_values, axis=0),
            'avg': np.mean(obj_values, axis=0),
            'std': np.std(obj_values, axis=0)
        })
        
        fronts = self.fast_non_dominated_sort(population)
        tracking['pareto_fronts'].append([len(front) for front in fronts])
        
        ref_point = np.array([1.1, 1.1, 1.1]) 
        tracking['hypervolume'].append(
            self._calculate_hypervolume(fronts[0], ref_point)
        )

    def visualize_objectives(self, tracking: Dict):
        """Visualize evolution of objective values"""
        import matplotlib.pyplot as plt
        
        generations = range(len(tracking['objective_values']))
        objectives = ['Key Compatibility', 'Character Recognition', 'Tension Alignment']
        
        fig, axes = plt.subplots(3, 1, figsize=(10, 12))
        
        for i, (ax, obj_name) in enumerate(zip(axes, objectives)):
            mins = [gen['min'][i] for gen in tracking['objective_values']]
            maxs = [gen['max'][i] for gen in tracking['objective_values']]
            avgs = [gen['avg'][i] for gen in tracking['objective_values']]
            stds = [gen['std'][i] for gen in tracking['objective_values']]
            
            ax.plot(generations, avgs, 'b-', label='Average')
            ax.fill_between(generations, 
                        np.array(avgs) - np.array(stds),
                        np.array(avgs) + np.array(stds),
                        alpha=0.2)
            ax.plot(generations, mins, 'g--', label='Min')
            ax.plot(generations, maxs, 'r--', label='Max')
            
            ax.set_xlabel('Generation')
            ax.set_ylabel(f'{obj_name} Value')
            ax.legend()
            
        plt.tight_layout()
        plt.show()

    def visualize_pareto_front(self, population: List[Individual]):
        """Visualize final Pareto front"""
        import matplotlib.pyplot as plt
        from mpl_toolkits.mplot3d import Axes3D
        
        fronts = self.fast_non_dominated_sort(population)
        pareto_front = fronts[0]
        
        obj_values = np.array([ind.objectives for ind in pareto_front])
        
        fig = plt.figure(figsize=(10, 10))
        ax = fig.add_subplot(111, projection='3d')
        
        scatter = ax.scatter(obj_values[:, 0],
                            obj_values[:, 1],
                            obj_values[:, 2],
                            c='b')
        
        ax.set_xlabel('Key Compatibility')
        ax.set_ylabel('Character Recognition')
        ax.set_zlabel('Tension Alignment')
        
        plt.show()

    def analyze_final_solutions(self, population: List[Individual]):
        """Analyze characteristics of final Pareto-optimal solutions"""
        fronts = self.fast_non_dominated_sort(population)
        pareto_front = fronts[0]
        
        print(f"Number of Pareto-optimal solutions: {len(pareto_front)}")
        
        obj_values = np.array([ind.objectives for ind in pareto_front])
        objectives = ['Key Compatibility', 'Character Recognition', 'Tension Alignment']
        
        print("\nObjective Value Statistics:")
        for i, obj_name in enumerate(objectives):
            print(f"\n{obj_name}:")
            print(f"  Min: {np.min(obj_values[:, i]):.3f}")
            print(f"  Max: {np.max(obj_values[:, i]):.3f}")
            print(f"  Mean: {np.mean(obj_values[:, i]):.3f}")
            print(f"  Std: {np.std(obj_values[:, i]):.3f}")

    def _calculate_hypervolume(self, pareto_front: List[Individual], reference_point: np.ndarray) -> float:
        """
        Calculate hypervolume indicator for a set of non-dominated solutions.
        
        The hypervolume is the volume of space dominated by the Pareto front and bounded 
        by a reference point. A larger hypervolume indicates better convergence and diversity.
        
        Args:
            pareto_front: List of non-dominated solutions
            reference_point: Point used as upper bound for volume calculation
                            Should be worse than all solutions in all objectives
        
        Returns:
            float: Hypervolume value
        """
        if not pareto_front:
            return 0.0
        
        points = np.array([ind.objectives for ind in pareto_front])
        
        sorted_indices = np.argsort(points[:, 0])
        points = points[sorted_indices]
        
        hv = np.prod(reference_point - points[0])
        
        for i in range(1, len(points)):
            dominated_volume = 1.0
            for j in range(len(reference_point)):
                update_point = min(reference_point[j], points[i][j])
                previous_value = max([p[j] for p in points[:i]])
                dominated_volume *= (update_point - previous_value)
            
            hv += dominated_volume
        
        return hv
    

def play_individual_musescore(individual: Individual):
    piano_score = music21.stream.Score()
    treble_staff = music21.stream.Part()
    bass_staff = music21.stream.Part()
    
    treble_staff.append(music21.clef.TrebleClef())
    treble_staff.append(music21.meter.TimeSignature('4/4'))
    bass_staff.append(music21.clef.BassClef())
    bass_staff.append(music21.meter.TimeSignature('4/4'))
    
    terminal_nodes = individual.musical_node.get_terminal_nodes()
    
    measure_number = 1
    
    for terminal in terminal_nodes:
        melody_measure = music21.stream.Measure(number=measure_number)
        for note in terminal.melody:
            melody_note = music21.note.Note(
                note.pitch,
                quarterLength=note.duration
            )
            melody_measure.append(melody_note)
        treble_staff.append(melody_measure)
        

        chord_measure = music21.stream.Measure(number=measure_number)
        chord_notes = terminal.key.get_chord(
            SCALE_DEGREES_INDEX[terminal.symbol.FunctionScaleInterface(terminal.key)]
        )
        base_octave = 4
        chord_notes = [music21.note.Note(pitch + (base_octave * 12), quarterLength=4.0) for pitch in chord_notes]
        chord = music21.chord.Chord(chord_notes)
        chord_measure.append(chord)
        bass_staff.append(chord_measure)
        
        measure_number += 1
    
    piano_score.insert(0, treble_staff)
    piano_score.insert(0, bass_staff)
    
    piano_score.show('musicxml')

def play_individuals_musescore(individuals: List[Individual]):
    piano_score = music21.stream.Score()
    treble_staff = music21.stream.Part()
    bass_staff = music21.stream.Part()
    
    treble_staff.append(music21.clef.TrebleClef())
    treble_staff.append(music21.meter.TimeSignature('4/4'))
    bass_staff.append(music21.clef.BassClef())
    bass_staff.append(music21.meter.TimeSignature('4/4'))
    
    terminal_nodes = []
    for individual in individuals:
        terminal_nodes.extend(individual.musical_node.get_terminal_nodes())
    
    measure_number = 1
    
    for terminal in terminal_nodes:
        melody_measure = music21.stream.Measure(number=measure_number)
        for note in terminal.melody:
            melody_note = music21.note.Note(
                note.pitch,
                quarterLength=note.duration
            )
            melody_measure.append(melody_note)
        treble_staff.append(melody_measure)
        

        chord_measure = music21.stream.Measure(number=measure_number)
        chord_notes = terminal.key.get_chord(
            SCALE_DEGREES_INDEX[terminal.symbol.FunctionScaleInterface(terminal.key)]
        )
        base_octave = 4
        chord_notes = [music21.note.Note(pitch + (base_octave * 12), quarterLength=4.0) for pitch in chord_notes]
        chord = music21.chord.Chord(chord_notes)
        chord_measure.append(chord)
        bass_staff.append(chord_measure)
        
        measure_number += 1
    
    piano_score.insert(0, treble_staff)
    piano_score.insert(0, bass_staff)
    
    piano_score.show('musicxml')

def calculate_tension_span(nodes: List[MusicalNode], key: Key, target_resolution: int) -> List[float]:
    """Calculate normalized tension with consistent resolution"""
    all_bars_span = []
    for node in nodes:
        all_bars_span.extend(node.get_harmony_melody_notes())
    
    notes, durations = zip(*all_bars_span)
    total_duration = sum(durations)
    
    diameter, momentum, strain = calculate_tension_fast(notes, key.root, key.is_major)
    
    original_times = np.cumsum([0] + list(durations))
    target_times = np.linspace(0, total_duration, target_resolution)
    
    components = []
    for tension in [diameter, momentum, strain]:
        interpolated = np.interp(target_times, original_times[:-1], tension)
        normalized = (interpolated - np.min(interpolated)) / (np.max(interpolated) - np.min(interpolated) + 1e-10)
        components.append(normalized)
    
    return np.average(components, axis=0, weights=[1./3, 1./3, 1./3])

def calculate_tension_alignment_span(nodes: List[MusicalNode], key: Key, plot_tension: List[float]) -> float:
    """Evaluate how well tension aligns with desired dramatic curve"""
    musical_tension = calculate_tension_span(nodes, key, len(plot_tension))

    correlation = stats.pearsonr(musical_tension, plot_tension)[0]
    return (correlation + 1) / 2

def interactive_selection(pareto_front: List[Individual]) -> List[Individual]:
    """Allow user to explore and select from Pareto front"""
    print("Top 5 solutions for each objective:")
    
    objectives = ['Key', 'Character', 'Tension']
    solutions = {}
    for i, obj_name in enumerate(objectives):
        print(f"\nBest solutions for {obj_name}:")
        sorted_solutions = sorted(pareto_front, 
                                key=lambda x: x.objectives[i], 
                                reverse=True)
        solutions[obj_name] = sorted_solutions[:5]
        for j, sol in enumerate(sorted_solutions[:5]):
            print(f"Solution {j+1}:")
            print(f"  {obj_name}: {sol.objectives[i]:.3f}")
            print(f"  Other objectives: {[f'{v:.3f}' for k,v in enumerate(sol.objectives) if k != i]}")
    
    return solutions

def select_by_weighted_preferences(pareto_front: List[Individual], weights: List[float]) -> Individual:
    """Select solution using weighted preferences for objectives"""
    weights = np.array(weights) / sum(weights)
    
    scores = []
    for solution in pareto_front:
        weighted_score = sum(w * obj for w, obj in zip(weights, solution.objectives))
        scores.append(weighted_score)
    
    return pareto_front[np.argmax(scores)]

def select_knee_point(pareto_front: List[Individual]) -> Individual:
    """Select knee point solution - greatest marginal improvement"""
    obj_values = np.array([ind.objectives for ind in pareto_front])
    
    ideal_point = np.max(obj_values, axis=0)
    distances = np.linalg.norm(ideal_point - obj_values, axis=1)
    
    return pareto_front[np.argmin(distances)]

def select_compromise_solution(pareto_front: List[Individual]) -> Individual:
    """Select solution closest to ideal point"""
    obj_values = np.array([ind.objectives for ind in pareto_front])
    
    ideal_point = np.max(obj_values, axis=0)
    nadir_point = np.min(obj_values, axis=0)
    
    normalized_values = (obj_values - nadir_point) / (ideal_point - nadir_point)
    
    distances = np.linalg.norm(1 - normalized_values, axis=1)
    
    return pareto_front[np.argmin(distances)]

def select_n_by_weighted_preferences(pareto_front: List[Individual], weights: List[float], n: int = 1) -> List[Individual]:
    """Select n solutions using weighted preferences for objectives"""
    weights = np.array(weights) / sum(weights)
    
    scores = []
    for solution in pareto_front:
        weighted_score = sum(w * obj for w, obj in zip(weights, solution.objectives))
        scores.append(weighted_score)
    
    top_n_indices = np.argsort(scores)[-n:]
    
    return [pareto_front[i] for i in top_n_indices]

def rank_tension_combinations(solutions_per_part: List[List[Individual]], 
                            key: Key, 
                            plot_tension: List[float]) -> List[Tuple[List[Individual], float]]:
    """Rank all combinations by tension alignment score"""
    
    combinations_with_scores = []
    for combination in itertools.product(*solutions_per_part):
        nodes = [ind.musical_node for ind in combination]
        score = calculate_tension_alignment_span(nodes, key, plot_tension)
        combinations_with_scores.append((list(combination), score))
    
    ranked_combinations = sorted(combinations_with_scores, 
                               key=lambda x: x[1], 
                               reverse=True)
    
    return ranked_combinations
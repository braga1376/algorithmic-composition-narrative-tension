from typing import List, Dict, Optional
import numpy as np
from dataclasses import dataclass
from genetic_algorithm import NSGAII, Individual, MotifHarmonicCombiner, rank_tension_combinations, select_n_by_weighted_preferences
from harmony_grammar import HarmonyGrammar, TreeNode
from narrative import PlotAtom, PlotSpan, Character, Role
from motif_composer import get_motif_llm, MarkovMotifGenerator
from utils import NOTE_TO_ROOT, Key, Note, Motif

class CharacterMotifGenerator:
    def __init__(self, character: Character, plot_schema: PlotSpan):
        self.character = character
        self.plot_schema = plot_schema
    
    def get_motif_llm(self, n_bars: float = 2) -> None:
        result = get_motif_llm(character=self.character.name, character_roles=self.character.roles, plot_description=self.plot_schema.get_description(), n_bars=n_bars)
        self.motif = result['motif']
        self.explanation = result['explanation']
        return self.motif
    
class PlotAtomMusicGenerator:
    def __init__(self, 
                 plot_atom: PlotAtom,
                 plot_schema: PlotSpan,
                 character_motifs: Dict[Character, List[Motif]],
                 harmonic_progressions: List[TreeNode],
                 protagonist_index: Optional[int] = None,
                 bars_per_atom: int = 8,
                 motif_length: int = 2,
                 population_size: int = 200) -> None:
        
        # Validate inputs
        if not isinstance(plot_atom, PlotAtom):
            raise ValueError("plot_atom must be a PlotAtom instance")
        if not isinstance(plot_schema, PlotSpan):
            raise ValueError("plot_schema must be a PlotSpan instance")
        
        self.plot_atom = plot_atom
        self.plot_schema = plot_schema
        self.character_motifs = character_motifs or {}
        self.motifs: List[Motif] = []
        self.character_motif_generators: List[CharacterMotifGenerator] = []
        self.harmonic_progressions = harmonic_progressions
        self.protagonist_index = protagonist_index
        self.bars_per_atom = max(1, bars_per_atom)
        self.motif_length = max(1, motif_length)
        self.population_size = max(10, population_size)
        self.nsga2 = None
        self.results = None

    def compose_characters_motifs(self) -> Dict[Character, List[Motif]]:
        try:
            unique_characters = set()
            
            for role in self.plot_atom.roles:
                try:
                    _, character = role
                    unique_characters.add(character)
                except (ValueError, IndexError):
                    raise ValueError(f"Invalid role format: {role}")

            for i, character in enumerate(unique_characters):
                if character == self.plot_schema.protagonist:
                    self.protagonist_index = i
                    
                if character not in self.character_motifs:
                    motif_generator = CharacterMotifGenerator(character, self.plot_schema)
                    motif_generator.get_motif_llm(self.motif_length)
                    self.character_motif_generators.append(motif_generator)
                    self.character_motifs[character] = motif_generator.motif
                    self.motifs.append(motif_generator.motif)
                    print(f"Generated motif for {character.name}")
                else:
                    self.motifs.append(self.character_motifs[character])
                    print(f"Using existing motif for {character.name}")

            return self.character_motifs

        except Exception as e:
            raise RuntimeError(f"Failed to compose character motifs: {str(e)}")

    def initialize_and_evolve_genetic_algorithm(self, num_generations: int) -> None:
        if num_generations < 1:
            raise ValueError("num_generations must be positive")
            
        try:
            if not self.motifs:
                self.compose_characters_motifs()

            tension_points = self.plot_atom.get_tension_curve().interpolate(self.bars_per_atom * 4)[1]
            
            self.combiner = MotifHarmonicCombiner(
                motifs=self.motifs,
                protagonist_index=self.protagonist_index,
                plot_tension=tension_points,
                target_length_bars=self.bars_per_atom,
                motif_length_bars=self.motif_length
            )

            initial_population = self.combiner.create_initial_population(
                self.harmonic_progressions, 
                self.population_size
            )

            self.nsga2 = NSGAII(
                population_size=self.population_size,
                objectives=[
                    self.combiner.evaluate_key_compatibility,
                    self.combiner.evaluate_character_motif_recognition,
                    self.combiner.evaluate_tension_alignment
                ]
            )

            self.results = self.nsga2.evolve(initial_population, num_generations)

        except Exception as e:
            raise RuntimeError(f"Failed to run genetic algorithm: {str(e)}")
    
    def get_pareto_solutions(self) -> List[Individual]:
        return self.nsga2.fast_non_dominated_sort(self.results[0])[0]
    
    def get_weighted_sum_solutions(self, weights=[0.1, 0.6, 0.3]) -> List[Individual]:
        return select_n_by_weighted_preferences(self.get_pareto_solutions(), weights=weights, n=10)

class PlotSpanMusicGenerator:
    def __init__(self, 
                 plot_schema: PlotSpan, 
                 bars_per_atom: int = 8, 
                 motif_length: int = 2, 
                 population_size: int = 200,
                 characters_motifs: Optional[Dict[Character, List[Motif]]] = None):
        self.plot_schema = plot_schema
        self.bars_per_atom = bars_per_atom
        self.motif_length = motif_length
        self.population_size = population_size
        self.character_motifs = characters_motifs or {}
        self.harmony_grammar = None
        self.harmonic_progressions = None
        self.plot_atom_music_generators = []
            
    def _validate_inputs(self) -> None:
        """Validate initialization parameters"""
        if self.bars_per_atom < 1:
            raise ValueError("bars_per_atom must be positive")
        if self.motif_length < 1:
            raise ValueError("motif_length must be positive")
        if self.population_size < 1:
            raise ValueError("population_size must be positive")
    
    def get_protagonist_key(self) -> Key:
        """Get key based on protagonist's motif or default"""
        if self.plot_schema.protagonist and self.character_motifs:
            motif = self.character_motifs.get(self.plot_schema.protagonist)
            if motif:
                return Key(root=NOTE_TO_ROOT[motif.key], 
                         is_major=motif.is_major)
        return Key(root=0, is_major=True)
    
    def compose_harmonic_progressions(self, max_depth: int = 7) -> None:
        """Generate harmonic progressions if not already generated"""
        if self.harmony_grammar is None:
            self.harmony_grammar = HarmonyGrammar()
        if self.harmonic_progressions is None:
            self.harmonic_progressions = self.harmony_grammar.generate_all_progressions(
                min_length=self.bars_per_atom,
                max_length=self.bars_per_atom,
                max_depth=max_depth
            )
    
    def compose_plot_span_music(self, num_generations: int = 100) -> List[Individual]:
        """Compose music for entire plot span"""
        self.compose_harmonic_progressions()
        self.plot_atom_music_generators = []
        # Compose music for each plot atom
        for atom in self.plot_schema.plot_atoms:
            generator = PlotAtomMusicGenerator(
                plot_atom=atom,
                plot_schema=self.plot_schema,
                character_motifs=self.character_motifs,
                harmonic_progressions=self.harmonic_progressions,
                bars_per_atom=self.bars_per_atom,
                motif_length=self.motif_length,
                population_size=self.population_size
            )
            
            new_motifs = generator.compose_characters_motifs()
            self.character_motifs.update(new_motifs)
            
            print(f"Composing music for {atom.name}")
            generator.initialize_and_evolve_genetic_algorithm(num_generations)
            self.plot_atom_music_generators.append(generator)
        
        # Join solutions using weighted sum approach
        return self.join_pareto_solutions_weighted_sum()
    
    def join_pareto_solutions_weighted_sum(self, 
                                         weights: List[float] = [0.1, 0.6, 0.3]
                                         ) -> List[Individual]:
        """Join solutions from all atoms using weighted sum approach"""
        if not self.plot_atom_music_generators:
            raise ValueError("No music has been generated yet")
            
        pareto_solutions = [generator.get_weighted_sum_solutions(weights=weights) for generator in self.plot_atom_music_generators]
         
        tension_points = self.plot_schema.compute_tension_curve().interpolate(
            self.bars_per_atom * len(self.plot_schema.plot_atoms)
        )[1]
        
        # Rank combinations by tension alignment
        return rank_tension_combinations(
            pareto_solutions,
            self.get_protagonist_key(),
            tension_points
        )
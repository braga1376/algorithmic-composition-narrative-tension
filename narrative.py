from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass
from enum import Enum
import itertools
from narrative_tension import *

@dataclass
class Role:
    name: str
    axis: str

    def __str__(self):
        return self.name

@dataclass
class Character:
    name: str
    character_type: str
    roles: List[Role]
    
    def __hash__(self):
        return hash((self.name, self.character_type))
    
    def __eq__(self, other):
        if not isinstance(other, Character):
            return False
        return (self.name, self.character_type) == (other.name, other.character_type)

@dataclass
class PlotAtom:
    """Basic unit of narrative action"""
    name: str
    description: str
    roles: List[Tuple[Role, Character]]
    preconditions: List[str]
    postconditions: List[str]
    tension_points: List[TensionPoint]

    def assign_character_to_role(self, role_name: str, character: Character):
        """Assign a concrete character to fill a role variable"""
        if self.role_assignments is None:
            self.role_assignments = {}
        matching_roles = [r for r in self.roles if r.name == role_name]
        if not matching_roles:
            raise ValueError(f"Role {role_name} not found in plot atom")
        self.role_assignments[role_name] = character
        for matching_role in matching_roles:
            character.roles.append(matching_role)
    
    def get_tension_curve(self) -> TensionCurve:
        return TensionCurve(self.tension_points)
    

class PlotSpanType(Enum):
    AXIS_OF_INTEREST = "axis_of_interest"
    PLOT_SCHEMA = "plot_schema"

@dataclass
class PlotLink:
    """Connection between plot atoms across different axes of interest"""
    source_axis: str
    target_axis: str
    source_atom: PlotAtom
    target_atom: PlotAtom
    shared_roles: List[Role]
    tension_impact: float = 0.0

@dataclass
class PlotSpan:
    """A sequence of plot atoms that may be non-contiguous in the final story"""
    name: str
    type: PlotSpanType
    plot_atoms: List[PlotAtom]
    role_bindings: Dict[str, Character]
    protagonist: Optional[Role] = None
    
    def __init__(self, name: str, type: PlotSpanType, plot_atoms: List[PlotAtom],
                 role_bindings: Dict[str, Character] = None, protagonist: Optional[Character] = None):
        self.name = name
        self.type = type
        self.plot_atoms = plot_atoms
        self.protagonist = protagonist
        self.role_bindings = role_bindings if role_bindings else []

    def get_unique_characters(self) -> List[Character]:
        """Get all unique characters bound to this plot span"""
        return list(set(self.role_bindings.values()))

    def validate_roles(self) -> bool:
        """Verify all required roles are bound to characters"""
        bound_roles = set(self.role_bindings.keys())
        
        # Collect all required roles from plot atoms
        required_roles = set()
        for atom in self.plot_atoms:
            required_roles.update(role.name for role in atom.required_roles)
            
        return required_roles.issubset(bound_roles)

    def get_description(self) -> str:
        description = ""
        for atom in self.plot_atoms:
            description += f"  - {atom.name}: {atom.description}\n"
        return description

    def compute_tension_curve(self, resolution: int = 100) -> TensionCurve:
        """Compute tension curve by joining atom tension points"""
        if not self.plot_atoms:
            return TensionCurve([])
            
        all_points = []
        atom_duration = 1.0 / len(self.plot_atoms)
        
        for i, atom in enumerate(self.plot_atoms):
            start_time = i * atom_duration
            atom_points = atom.get_tension_curve().points
            
            for point in atom_points:
                adjusted_time = start_time + (point.time * atom_duration)
                all_points.append(TensionPoint(
                    time=adjusted_time,
                    value=point.value
                ))
        
        all_points.sort(key=lambda p: p.time)
        
        # Remove any duplicate times (keep highest tension value)
        unique_points = {}
        for point in all_points:
            if point.time not in unique_points or point.value > unique_points[point.time].value:
                unique_points[point.time] = point
                
        return TensionCurve(list(unique_points.values()))

class NarrativeGenerator:
    def __init__(self):
        self.plot_atoms: Dict[str, PlotAtom] = {}
        self.axes_of_interest: Dict[str, PlotSpan] = {}
        self.plot_schemas: Dict[str, PlotSpan] = {}
        self.plot_links: List[PlotLink] = []
        
    def add_plot_atom(self, plot_atom: PlotAtom):
        self.plot_atoms[plot_atom.name] = plot_atom
        
    def create_axis_of_interest(self, name: str, plot_atoms: List[PlotAtom],
                              protagonist: Role, roles: List[Role]) -> PlotSpan:
        axis = PlotSpan(name, PlotSpanType.AXIS_OF_INTEREST, plot_atoms, 
                       protagonist, roles)
        self.axes_of_interest[name] = axis
        return axis
        
    def add_plot_link(self, plot_link: PlotLink):
        self.plot_links.append(plot_link)
        
    def validate_plot_schema(self, plot_schema: PlotSpan) -> bool:
        """Validate a plot schema based on plot links and role consistency"""
        # Check if all adjacent plot atoms from different axes have valid plot links
        for i in range(len(plot_schema.plot_atoms) - 1):
            current_atom = plot_schema.plot_atoms[i]
            next_atom = plot_schema.plot_atoms[i + 1]
            
            # If atoms are from different axes, check for plot link
            if current_atom != next_atom:
                valid_link = False
                for link in self.plot_links:
                    if ((link.source_atom == current_atom and link.target_atom == next_atom) or
                        (link.source_atom == next_atom and link.target_atom == current_atom)):
                        valid_link = True
                        break
                if not valid_link:
                    return False
        return True
    
    def generate_plot_schema(self, axes: List[str]) -> Optional[PlotSpan]:
        """Generate a valid plot schema from given axes of interest"""
        selected_axes = [self.axes_of_interest[axis] for axis in axes]
        
        possible_combinations = self._generate_combinations(selected_axes)
        
        for combination in possible_combinations:
            plot_schema = PlotSpan(
                name=f"Generated_Schema_{len(self.plot_schemas)}",
                type=PlotSpanType.PLOT_SCHEMA,
                plot_atoms=combination
            )
            if self.validate_plot_schema(plot_schema):
                self.plot_schemas[plot_schema.name] = plot_schema
                return plot_schema
        
        return None
    
    def _generate_combinations(self, axes: List[PlotSpan]) -> List[List[PlotAtom]]:
        """Generate all possible combinations of plot atoms from given axes"""
        # simplified version - generates all permutations
        all_atoms = []
        for axis in axes:
            all_atoms.extend(axis.plot_atoms)
        return list(itertools.permutations(all_atoms))

    def compute_story_tension(self, plot_schema: PlotSpan) -> TensionCurve:
        """Compute the overall tension curve for a complete story"""
        return plot_schema.compute_tension_curve(self.plot_links)
    
    def get_plot_description(self) -> str:
        if len(self.plot_schemas) == 0:
            return "No plot schemas generated"
        else:
            description = ""
            for schema in self.plot_schemas.values():
                # description += f"- {schema.name}:\n"
                for atom in schema.plot_atoms:
                    description += f"  - {atom.name}: {atom.description}\n"
            return description
import music21
from dataclasses import dataclass
from typing import List, Tuple, Optional
import copy
import numpy as np
from scipy import stats
from numba import jit

PITCH_CONVERSION = [60, 62, 64, 65, 67, 69, 71, 72, 74, 76, 77, 79, 81, 83, 84]
DURATION_CONVERSION = [4, 2, 1, 0.5, 0.25]
NOTE_VALUES = {'C': 0, 'D': 2, 'E': 4, 'F': 5, 'G': 7, 'A': 9, 'B': 11}
ROOT_TO_NOTE = {
    0: 'C',
    1: 'C#',
    2: 'D',
    3: 'D#',
    4: 'E',
    5: 'F',
    6: 'F#',
    7: 'G',
    8: 'G#',
    9: 'A',
    10: 'A#',
    11: 'B'
}
NOTE_TO_ROOT = {v: k for k, v in ROOT_TO_NOTE.items()}

SCALE_DEGREES_INDEX = {
    'I': 0,
    'II': 1,
    'III': 2,
    'IV': 3,
    'V': 4,
    'VI': 5,
    'VII': 6
}

KEYS_MAJOR = {
    't': (0, True),
    's': (5, True),
    'd': (7, True),
    'tp': (9, False),
    # 'dp': (11, False),
    'sp': (2, False),
    'tcp': (4, False)
}

KEYS_MINOR = {
    't': (0, False),
    's': (5, False),
    'd': (7, False),
    'tp': (3, True),
    'dp': (10, True),
    'sp': (8, True),
    'tcp': (8, True)
}

MAX_PITCH = 108
MIN_PITCH = 36

@dataclass
class Key:
    root: int
    is_major: bool

    def get_scale_degree(self) -> np.array:
        if self.is_major:
            return (np.array([0, 2, 4, 5, 7, 9, 11])+self.root)%12
        else:
            return (np.array([0, 2, 3, 5, 7, 8, 10])+self.root)%12
        
    def get_chord(self, degree: int) -> np.array:
        scale_degree = self.get_scale_degree()
        chord = scale_degree[[degree, (degree+2)%7, (degree+4)%7]]
        return chord
        
@dataclass
class Note:
    pitch: int
    duration: float
    velocity: int = 64
    start_time: float = 0.0
    continues_to_next_bar: bool = False
    continues_from_prev_bar: bool = False

    def copy(self):
        return Note(
            pitch=self.pitch,
            duration=self.duration,
            velocity=self.velocity,
            start_time=self.start_time,
            continues_to_next_bar=self.continues_to_next_bar,
            continues_from_prev_bar=self.continues_from_prev_bar
        )
    

@dataclass
class Motif:
    notes: List[Note]
    key: str
    is_major: bool
    tension_profile: Optional[List[float]] = None

    def __init__(self, notes: List[Note], key: None, is_major: None, tension_profile: Optional[List[float]] = None):
        self.notes = notes
        if key is None:
            self.key, self.is_major = self.detect_key()
        else:
            self.key = key
            self.is_major = is_major

        self.tension_profile = tension_profile

    def detect_key(self) -> Tuple[str, bool]:
        try:
            motif_stream = music21.stream.Stream()
            for note in self.notes:
                motif_stream.append(music21.note.Note(note.pitch, quarterLength=note.duration))

            key = motif_stream.analyze('key')
            return (key.tonic.name, key.mode == 'major')
        except Exception as e:
            print(f"Error detecting key: {str(e)}")
            return ('', True)
        
    def transpose(self, new_key: str, new_is_major: bool):
        if new_key == self.key and new_is_major == self.is_major:
            return
        else:
            if self.is_major != new_is_major:
                new_key_root = NOTE_TO_ROOT[new_key] - 3 if new_is_major else NOTE_TO_ROOT[new_key] + 3
                new_key_root = new_key_root % 12
            else:
                new_key_root = NOTE_TO_ROOT[new_key]

            interval = new_key_root - NOTE_TO_ROOT[self.key]
            if abs(interval) >= 6:
                interval = interval - 12*np.sign(interval)
            
            for note in self.notes:
                note.pitch = (note.pitch + interval)

            self.key = new_key
            self.is_major = new_is_major

        return

def play_motif_musecore(motif: Motif):
    """
    Play a motif using Musecore
    """
    motif_stream = music21.stream.Stream()
    for note in motif.notes:
        motif_stream.append(music21.note.Note(note.pitch, quarterLength=note.duration))
    
    motif_stream.show('musicxml')

def note_to_midi(note_name: str) -> int:
    """Convert a note name to its MIDI note number.
    
    Args:
        note_name (str): Note name in format like 'C4', 'F#3', 'Bb5'
                        Letter must be A-G
                        Accidental can be # or b
                        Octave must be 0-9
    
    Returns:
        int: MIDI note number (0-127)
    
    Raises:
        ValueError: If note_name is invalid
    
    Examples:
        >>> note_to_midi('C4')
        60
        >>> note_to_midi('A4')
        69
        >>> note_to_midi('F#3')
        54
    """
    if not isinstance(note_name, str):
        raise ValueError(f"Note name must be a string, got {type(note_name)}")
    
    try:
        if len(note_name) < 2:
            raise ValueError(f"Invalid note format: {note_name}")
            
        if note_name[1] in ['#', 'b']:
            pitch = note_name[0]
            accidental = note_name[1]
            octave = int(note_name[2:])
        else:
            pitch = note_name[0]
            accidental = ''
            octave = int(note_name[1:])
            
        if pitch not in NOTE_VALUES:
            raise ValueError(f"Invalid pitch: {pitch}")
            
        midi_note = NOTE_VALUES[pitch]
        
        if accidental == '#':
            midi_note += 1
        elif accidental == 'b':
            midi_note -= 1
            

        midi_note += (octave + 1) * 12
        
        if not 0 <= midi_note <= 127:
            raise ValueError(f"MIDI note {midi_note} out of range (0-127)")
            
        return midi_note
        
    except ValueError as e:
        raise ValueError(f"Error parsing note {note_name}: {str(e)}")
    except Exception as e:
        raise ValueError(f"Invalid note format {note_name}: {str(e)}")

def detect_key(motif: Motif) -> str:
    """
    Detect the key of a motif
    """
    motif_stream = music21.stream.Stream()
    for note in motif.notes:
        motif_stream.append(music21.note.Note(note.pitch, quarterLength=note.duration))

    key = motif_stream.analyze('key')
    return (key.tonic.name, key.mode == 'major')
    
def list_to_motif(note_list: List[Tuple[str, float]]) -> Motif:
    """
    Convert a list of note tuples to a Motif object
    """
    notes = []
    start_time = 0
    for note_name, duration in note_list:
        pitch = note_to_midi(note_name)
        notes.append(Note(pitch=pitch, duration=duration, start_time=start_time))
        start_time += duration
        
    return Motif(notes, key=None, is_major=None)

def split_motif_into_bars(motif: List[Note], beats_per_bar: int = 4) -> List[List[Note]]:
    """
    Split motif into bars, marking notes that cross boundaries
    """
    bars = []
    current_bar = []
    
    for note in motif:
        note_end_time = note.start_time + note.duration
        bar_number = int(note.start_time // beats_per_bar)
        bar_boundary = (bar_number + 1) * beats_per_bar
        
        crosses_boundary = note_end_time > bar_boundary
        
        if not crosses_boundary:
            current_bar.append(note)
            if note_end_time >= bar_boundary:
                bars.append(current_bar)
                current_bar = []
        else:
            note_copy = copy.deepcopy(note)
            note_copy.continues_to_next_bar = True
            note_copy.duration = bar_boundary - note.start_time
            current_bar.append(note_copy)

            while crosses_boundary:
                next_bar_note = copy.deepcopy(note)
                next_bar_note.continues_from_prev_bar = True
                next_bar_note.start_time = bar_boundary
                
                if note_end_time > (bar_number + 2) * beats_per_bar:
                    next_bar_note.duration = beats_per_bar 
                else:
                    next_bar_note.duration = note_end_time - bar_boundary

                bars.append(current_bar)
                current_bar = [next_bar_note]
                note = next_bar_note
                bar_number = int(note.start_time // beats_per_bar)
                bar_boundary = (bar_number + 1) * beats_per_bar
                crosses_boundary = note_end_time > bar_boundary
                if crosses_boundary:
                    note.continues_to_next_bar = True

    if current_bar:
        bars.append(current_bar)
        
    return bars

def normalize_tension(tensions: List[float]) -> List[float]:
    min_val = min(tensions)
    max_val = max(tensions)
    if max_val == min_val:
        return [0.5] * len(tensions)
    return [(t - min_val) / (max_val - min_val) for t in tensions]

def sliding_windows(arr: np.ndarray, window_size: int) -> np.ndarray:
    """Create sliding windows using rolling window approach"""
    if len(arr) < window_size:
        return np.array([]).reshape(0, window_size)
    
    return np.lib.stride_tricks.sliding_window_view(arr, window_size)

# def sliding_windows(arr: np.ndarray, window_size: int) -> np.ndarray:
#     """Create sliding windows view of array"""
#     if len(arr) < window_size:
#         return np.array([]).reshape(0, window_size)
    
#     shape = (arr.shape[0] - window_size + 1, window_size)
#     strides = (arr.strides[0], arr.strides[0])
#     return np.lib.stride_tricks.as_strided(arr, shape=shape, strides=strides)

def get_melodic_contour(intervals: np.ndarray) -> np.ndarray:
    """Vectorized contour calculation"""
    return np.sign(intervals)

def get_interval_pattern(pitches: np.ndarray) -> np.ndarray:
    """Vectorized interval calculation"""
    return np.diff(pitches)

def get_rhythm_pattern(durations: np.ndarray) -> np.ndarray:
    """Vectorized rhythm calculation"""
    return durations[1:] / durations[:-1]

@jit(nopython=True)
def pearson_correlation(x, y):
    """Calculate Pearson correlation coefficient"""
    mx = x.mean()
    my = y.mean()
    xm, ym = x - mx, y - my
    r_num = np.sum(xm * ym)
    r_den = np.sqrt(np.sum(xm * xm) * np.sum(ym * ym))
    r = r_num / r_den if r_den != 0 else 0
    return r

@jit(nopython=True)
def calculate_similarities(window_contours, window_intervals, window_rhythms,
                         motif_contour, motif_intervals, motif_rhythm):
    """Calculate similarities using explicit indexing"""
    n_windows = len(window_contours)
    similarities = np.zeros(n_windows)
    
    for i in range(n_windows):
        contour_sim = abs(pearson_correlation(motif_contour, window_contours[i]))
        interval_sim = abs(pearson_correlation(motif_intervals, window_intervals[i]))
        rhythm_sim = abs(pearson_correlation(motif_rhythm, window_rhythms[i]))
        
        similarities[i] = (contour_sim + interval_sim + rhythm_sim) / 3
    
    return similarities

import ast
import re
import numpy as np
from typing import Dict, Tuple, List
import anthropic
from constants import *
from utils import *
from api_key import ANTHROPIC_API_KEY

VALID_NOTES = {'A', 'B', 'C', 'D', 'E', 'F', 'G'}
VALID_ACCIDENTALS = {'#', 'b', ''}
VALID_OCTAVES = set(range(2, 6))  # C2 to C5
MIN_DURATION = 0.25 
MAX_DURATION = 4.0

client = anthropic.Anthropic(
    api_key=ANTHROPIC_API_KEY,
)

def send_message(system_prompt: str, user_prompt: str) -> str:
    """Send message to Claude API and return response.
    
    Args:
        system_prompt (str): System prompt for Claude
        user_prompt (str): User prompt with formatted parameters
        
    Returns:
        Tuple[str, str]: System response and user response
        
    Raises:
        anthropic.APIError: If there's an API error
        anthropic.RateLimitError: If rate limit is exceeded
        anthropic.APIConnectionError: If connection fails
        Exception: For any other unexpected errors
    """
    try:
        message = client.messages.create(
            model="claude-3-5-sonnet-20241022",
            max_tokens=8192,
            temperature=0,
            system=system_prompt,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": user_prompt
                        }
                    ]
                }
            ]
        )
        response = message.content[0].text
        return response
        
    except anthropic.RateLimitError as e:
        print(f"Rate limit exceeded: {str(e)}")
        raise
        
    except anthropic.APIConnectionError as e:
        print(f"Connection error: {str(e)}")
        raise
        
    except anthropic.APIError as e:
        print(f"API error: {str(e)}")
        raise
        
    except Exception as e:
        print(f"Unexpected error: {str(e)}")
        raise

class MotifParserError(Exception):
    """Custom exception for motif parsing errors"""
    pass

class MotifParser:
    """Parser for musical motifs in the specified format"""

    @classmethod
    def parse_motif(cls, text: str, duration_sum: float) -> List[Tuple[str, float]]:
        """Parse a motif from the given text."""
        try:
            # Extract content between MOTIF tags
            pattern = r'<MOTIF_START>\s*(\[.*?\])\s*<MOTIF_END>'
            match = re.search(pattern, text, re.DOTALL)
            if not match:
                raise MotifParserError("Could not find motif in the specified format")
            
            motif_str = match.group(1)
            
            # Parse the unquoted format using regex
            # Match pattern: (LETTER[#b]?NUMBER, NUMBER)
            pattern = r'\(([A-G][#b]?[2-5]),\s*([\d.]+)\)'
            matches = re.findall(pattern, motif_str)
            
            if not matches:
                raise MotifParserError("Could not parse motif tuples")
            
            motif = []
            for note, duration in matches:
                # Validate note format
                note_match = re.match(r'^([A-G])([#b])?([2-5])$', note)
                if not note_match:
                    raise MotifParserError(f"Invalid note format: {note}")
                
                pitch, accidental, octave = note_match.groups()
                accidental = accidental if accidental else ''
                
                if pitch not in VALID_NOTES:
                    raise MotifParserError(f"Invalid pitch: {pitch}")
                if accidental not in VALID_ACCIDENTALS:
                    raise MotifParserError(f"Invalid accidental: {accidental}")
                if int(octave) not in VALID_OCTAVES:
                    raise MotifParserError(f"Invalid octave: {octave}")
                
                try:
                    duration = float(duration)
                except ValueError:
                    raise MotifParserError(f"Invalid duration format: {duration}")
                
                if not (MIN_DURATION <= duration <= MAX_DURATION):
                    raise MotifParserError(
                        f"Duration {duration} outside valid range "
                        f"({MIN_DURATION} to {MAX_DURATION})"
                    )
                
                motif.append((note, duration))
            
            total_duration = sum(duration for _, duration in motif)
            if total_duration != duration_sum:
                raise MotifParserError(f"Total duration must be 8, got {total_duration}")
            
            return motif
            
        except Exception as e:
            if isinstance(e, MotifParserError):
                raise
            raise MotifParserError(f"Error parsing motif: {str(e)}")
    
    @classmethod
    def validate_motif(cls, motif: List[Tuple[str, float]]) -> bool:
        """Validate a motif without parsing from text."""
        try:
            motif_str = "["
            motif_str += ", ".join(f"({note}, {duration})" for note, duration in motif)
            motif_str += "]"
            
            test_text = f"<MOTIF_START>{motif_str}<MOTIF_END>"
            cls.parse_motif(test_text)
            return True
        except MotifParserError as e:
            raise MotifParserError(f"Invalid motif: {str(e)}")

def get_motif_llm(character: str, character_roles: str, plot_description: str, n_bars: float = 2) -> Dict:
    """Get motif for character using Claude.
    
    Args:
        character_role (str): Role of the character
        character_traits (str): Character traits description
        plot_description (str): Plot description with character's actions
        
    Returns:
        Dict: Contains explanation and motif list of (note, duration) tuples
        
    Raises:
        MotifParserError: If motif parsing fails
        Other exceptions from send_message()
    """
    try:
        duration_sum = n_bars * 4
        response = send_message(SYSTEM_PROMPT_MOTIF, USER_PROMPT_MOTIF.format(
            character=character,
            character_roles=', '.join(str(role) for role in character_roles),
            plot_description=plot_description,
            n_bars=n_bars,
            duration_sum=duration_sum
        ))
        # response = RESPONSE_1

        explanation = response.split("<MOTIF_START>")[0].strip()
        
        parser = MotifParser()
        motif = parser.parse_motif(response, duration_sum)

        notes = []
        start_time = 0
        for note, duration in motif:
            notes.append(Note(pitch=note_to_midi(note), duration=duration, velocity=0.8, start_time=start_time))
            start_time += duration

        result = {
            'explanation': explanation,
            'motif': Motif(notes=notes, key=None, is_major=None)
        }
        
        print("Successfully parsed response:")
        print(f"Explanation length: {len(result['explanation'])} characters")
        print(f"Motif length: {len(result['motif'].notes)} notes")
        print(f"Motif: {result['motif']}")

        return result
        
    except (anthropic.RateLimitError, anthropic.APIConnectionError, anthropic.APIError) as e:
        print(f"Claude API error: {str(e)}")
        return None
        
    except MotifParserError as e:
        print(f"Error parsing motif: {str(e)}")
        return None
        
    except Exception as e:
        print(f"Unexpected error: {str(e)}")
        return None
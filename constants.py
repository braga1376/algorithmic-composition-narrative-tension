SYSTEM_PROMPT_MOTIF = """You are assisting in generating musical motifs. When providing motifs:
1. Specify both pitch and duration clearly
2. Use notes A, B, C, D, E, F, and G with necessary alterations, and represent octaves from C2 to C5.
3. Use standard duration values (whole note - 4, half note - 2, quarter note - 1, dotted half note - 1.5, eighth note - 0.5, sixteenth note - 0.25)
4. Each melody should be formatted as a sequence of tuples: motif = [(note, duration), (note, duration), ...].
5. The duration sum should be {duration_sum}.
6. Provide clear explanations of musical decisions
7. Consider both local and global tension aspects
8. Ensure the motif is memorable and distinctive"""

USER_PROMPT_MOTIF = """
I'm creating a musical motif for a story character. Please design a motif that represents this character. Here's the context:

Character: {character}

Character Roles: {character_roles}

Character's Story Arc:
The character appears in these narrative moments, in order:
{plot_description}

Inspire yourself in the great compositions that were made for narratives for theatre, movies, and sole compositions and based on this character's role and story arc, please provide:

1. A melodic motif that captures the character's essence, specified as:
- A sequence of pitches
- Corresponding duration values for each note
- Length should be {n_bars} bars, which corresponds to duration sum of {duration_sum} (appropriate for a memorable motif)

2. A brief analysis explaining:
- How the pitch contour reflects the character's personality
- How the rhythm reflects the character's energy and movement
- How the motif can be developed to follow the character's arc
- Consideration of tension and release in the motif structure

Ensure the motif is musically interesting and recognizable.

First provide your analysis of the musical decisions.

Then provide the motif in this format:

<MOTIF_START>
[(note, duration), (note, duration), ...]
<MOTIF_END>
"""

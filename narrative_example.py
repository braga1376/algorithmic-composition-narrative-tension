from narrative import *
from narrative_tension import *

def create_example_narrative_simple():
    generator = NarrativeGenerator()
    
    # roles
    perpetrator = Role("perpetrator", "CONFLICT")
    target = Role("target", "CONFLICT")
    pursuer = Role("pursuer", "CONFLICT")
    pursued = Role("pursued", "CONFLICT")
    winner = Role("winner", "CONFLICT")
    looser = Role("looser", "CONFLICT")
    
    hero = Character("Hero", "hero", [pursuer, winner])
    villain = Character("Villain", "villain", [perpetrator, pursued, looser])
    victim = Character("Victim", "victim", [target])

    # plot atoms
    villainy = PlotAtom(
        name="villainy",
        description="Villain commits crime against victim",
        roles=[(perpetrator, villain), (target, victim)],
        preconditions=["villain_exists", "victim_exists"],
        postconditions=["crime_committed"],
        tension_points=[
            TensionPoint(time=0.0, value=0.3),  # Initial state, anticipation
            TensionPoint(time=0.4, value=0.7),  # Build up to crime
            TensionPoint(time=0.8, value=0.9),  # Crime committed
            TensionPoint(time=1.0, value=0.8)   # Aftermath
        ]
    )
    generator.add_plot_atom(villainy)

    pursuit = PlotAtom(
        name="pursuit",
        description="Hero pursues villain",
        roles=[(pursuer, hero), (pursued, villain)],
        preconditions=["crime_committed"],
        postconditions=["pursuit_in_progress"],
        tension_points=[
            TensionPoint(time=0.0, value=0.5),  # Start of pursuit
            TensionPoint(time=0.3, value=0.7),  # Chase intensifies
            TensionPoint(time=0.6, value=0.8),  # Close calls/near captures
            TensionPoint(time=1.0, value=0.85)  # Maximum pursuit tension
        ]
    )
    generator.add_plot_atom(pursuit)
    
    victory = PlotAtom(
        name="victory",
        description="Hero defeats villain",
        roles=[(winner, hero), (looser, villain)],
        preconditions=["pursuit_in_progress"],
        postconditions=["villain_defeated"],
        tension_points=[
            TensionPoint(time=0.0, value=0.85),  # Start of confrontation
            TensionPoint(time=0.3, value=0.95),  # Peak of conflict
            TensionPoint(time=0.7, value=0.9),   # Moment of victory
            TensionPoint(time=1.0, value=0.3)    # Resolution/relief
        ]
    )
    generator.add_plot_atom(victory)
    
    #  axes of interest
    conflict_axis = PlotSpan(
        name="CONFLICT",
        type=PlotSpanType.AXIS_OF_INTEREST,
        plot_atoms=[villainy, pursuit, victory],
        role_bindings={
            "perpetrator": villain,
            "target": victim,
            "pursuer": hero,
            "pursued": villain,
            "winner": hero,
            "looser": villain,
        },
        protagonist=hero
    )
    # plot links
    link_1 = PlotLink(
        source_axis="conflict",
        target_axis="conflict",
        source_atom=villainy,
        target_atom=pursuit,
        shared_roles=[villain]
    )
    generator.add_plot_link(link_1)

    link_2 = PlotLink(
        source_axis="conflict",
        target_axis="conflict",
        source_atom=pursuit,
        target_atom=victory,
        shared_roles=[villain]
    )
    generator.add_plot_link(link_2)
    
    link_3 = PlotLink(
        source_axis="conflict",
        target_axis="conflict",
        source_atom=pursuit,
        target_atom=victory,
        shared_roles=[hero]
    )
    generator.add_plot_link(link_3)
    
    return conflict_axis

def create_rising_tension_narrative():
    generator = NarrativeGenerator()
    
    # Roles
    victim = Role("victim", "VILLAINY")
    perpetrator = Role("perpetrator", "VILLAINY")
    pursuer = Role("pursuer", "PURSUIT")
    pursued = Role("pursued", "PURSUIT")
    rescuer = Role("rescuer", "VICTORY")
    recued = Role("rescued", "VICTORY")
    # looser = Role("looser", "VICTORY")
    
    hero = Character("Hero", "hero", [pursuer, rescuer])
    villain = Character("Villain", "villain", [perpetrator, pursued])
    princess = Character("Princess", "victim", [victim, recued])

    villainy = PlotAtom(
        name="villainy",
        description="Villain kidnaps princess",
        roles=[(perpetrator, villain), (victim, princess)],
        preconditions=["villain_exists", "princess_exists"],
        postconditions=["crime_committed"],
        tension_points=[
            TensionPoint(time=0.0, value=0.2),  # Initial peace
            TensionPoint(time=0.3, value=0.7),  # Villain appears
            TensionPoint(time=0.7, value=0.9),  # Kidnapping
            TensionPoint(time=1.0, value=0.7)   # Princess taken
        ]
    )
    
    pursuit = PlotAtom(
        name="pursuit",
        description="Hero pursues and defeats villain",
        roles=[(pursuer, hero), (pursued, villain)],
        preconditions=["crime_committed"],
        postconditions=["pursuit_complete"],
        tension_points=[
            TensionPoint(time=0.0, value=0.5),  # Chase begins
            TensionPoint(time=0.3, value=0.7),  # Pursuit intensifies
            TensionPoint(time=0.7, value=0.8),  # Getting closer
            TensionPoint(time=1.0, value=0.9)   # Confrontation nears
        ]
    )
    
    victory = PlotAtom(
        name="victory",
        description="Hero rescues princess",
        roles=[(rescuer, hero), (recued, princess)],
        preconditions=["pursuit_complete"],
        postconditions=["princess_rescued"],
        tension_points=[
            TensionPoint(time=0.0, value=0.9),  # Final battle begins
            TensionPoint(time=0.3, value=0.7), # Climactic moment
            TensionPoint(time=0.7, value=0.5),  # Victory
            TensionPoint(time=1.0, value=0.3)   # Rescue complete
        ]
    )
    # Plot Span
    conflict_axis = PlotSpan(
        name="CONFLICT",
        type=PlotSpanType.AXIS_OF_INTEREST,
        plot_atoms=[villainy, pursuit, victory],
        role_bindings={
            "perpetrator": villain,
            "victim": princess,
            "pursuer": hero,
            "pursued": villain,
            "rescuer": hero,
            "rescued": princess
        },
        protagonist=hero
    )
    
    # Plot Links
    link_1 = PlotLink(
        source_axis="conflict",
        target_axis="conflict",
        source_atom=villainy,
        target_atom=pursuit,
        shared_roles=[villain]
    )
    generator.add_plot_link(link_1)

    link_2 = PlotLink(
        source_axis="conflict",
        target_axis="conflict",
        source_atom=pursuit,
        target_atom=victory,
        shared_roles=[hero, villain]
    )
    generator.add_plot_link(link_2)

    link_3 = PlotLink(
        source_axis="conflict",
        target_axis="conflict",
        source_atom=villainy,
        target_atom=victory,
        shared_roles=[princess]
    )
    generator.add_plot_link(link_3)

    return conflict_axis

def create_arc_tension_narrative():
    generator = NarrativeGenerator()
    
    # Roles
    tested = Role("tested", "DONOR")
    tester = Role("tester", "DONOR")
    helper = Role("helper", "HELPER")
    recipient = Role("recipient", "GIFT")
    opponent = Role("opponent", "TASK")
    
    hero = Character("Hero", "hero", [tested, recipient])
    wizard = Character("Wizard", "donor", [tester, helper])
    dragon = Character("Dragon", "villain", [opponent])

    test = PlotAtom(
        name="test",
        description="Wizard tests hero",
        roles=[(tested, hero), (tester, wizard)],
        preconditions=["hero_exists", "wizard_exists"],
        postconditions=["test_given"],
        tension_points=[
            TensionPoint(time=0.0, value=0.3),  # Meeting wizard
            TensionPoint(time=0.3, value=0.5),  # Test begins
            TensionPoint(time=0.7, value=0.7),  # Test intensifies
            TensionPoint(time=1.0, value=0.8)   # Test climax
        ]
    )
    
    preparation = PlotAtom(
        name="preparation",
        description="Wizard prepares hero for task with a gift",
        roles=[(recipient, hero), (helper, wizard), (opponent, dragon)],
        preconditions=["test_given"],
        postconditions=["hero_prepared"],
        tension_points=[
            TensionPoint(time=0.0, value=0.8),  # Learning begins
            TensionPoint(time=0.3, value=0.9),  # Peak challenge
            TensionPoint(time=0.7, value=0.7),  # Mastery gained
            TensionPoint(time=1.0, value=0.6)   # Ready for task
        ]
    )
    
    victory = PlotAtom(
        name="victory",
        description="Hero uses gift to defeat dragon",
        roles=[(recipient, hero), (opponent, dragon)],
        preconditions=["hero_prepared"],
        postconditions=["dragon_defeated"],
        tension_points=[
            TensionPoint(time=0.0, value=0.6),  # Confrontation
            TensionPoint(time=0.3, value=0.4),  # Using gift
            TensionPoint(time=0.7, value=0.3),  # Dragon falls
            TensionPoint(time=1.0, value=0.2)   # Peace restored
        ]
    )

    # Plot Span
    donor_axis = PlotSpan(
        name="DONOR",
        type=PlotSpanType.AXIS_OF_INTEREST,
        plot_atoms=[test, preparation, victory],
        role_bindings={
            "tested": hero,
            "tester": wizard,
            "helper": wizard,
            "recipient": hero,
            "opponent": dragon
        },
        protagonist=hero
    )
    
    # Plot Links
    link_1 = PlotLink(
        source_axis="donor",
        target_axis="donor",
        source_atom=test,
        target_atom=preparation,
        shared_roles=[hero, wizard]
    )
    generator.add_plot_link(link_1)

    link_2 = PlotLink(
        source_axis="donor",
        target_axis="donor",
        source_atom=preparation,
        target_atom=victory,
        shared_roles=[hero]
    )
    generator.add_plot_link(link_2)

    link_3 = PlotLink(
        source_axis="donor",
        target_axis="donor",
        source_atom=preparation,
        target_atom=victory,
        shared_roles=[dragon]
    )
    generator.add_plot_link(link_3)

    return donor_axis

def create_oscillating_tension_narrative():
    generator = NarrativeGenerator()
    
    # Roles
    seeker = Role("seeker", "TASK")
    dispatcher = Role("dispatcher", "TASK")
    helper = Role("helper", "HELP")
    aided = Role("aided", "HELP")
    victor = Role("victor", "VICTORY")
    recognizer = Role("recognizer", "VICTORY")
    
    hero = Character("Hero", "hero", [seeker, aided, victor])
    king = Character("King", "authority", [dispatcher, recognizer])
    wise_woman = Character("WiseWoman", "helper", [helper])

    task = PlotAtom(
        name="task",
        description="King assigns difficult task to hero",
        roles=[(seeker, hero), (dispatcher, king)],
        preconditions=["hero_exists", "king_exists"],
        postconditions=["task_given"],
        tension_points=[
            TensionPoint(time=0.0, value=0.2),  # Task presented
            TensionPoint(time=0.3, value=0.4),  # Challenge revealed
            TensionPoint(time=0.7, value=0.6),  # Initial attempt
            TensionPoint(time=1.0, value=0.3)   # Seeking help
        ]
    )
    
    help = PlotAtom(
        name="help",
        description="Mysterious wise woman helps hero",
        roles=[(aided, hero), (helper, wise_woman)],
        preconditions=["task_given"],
        postconditions=["help_received"],
        tension_points=[
            TensionPoint(time=0.0, value=0.4),  # Meeting wise woman
            TensionPoint(time=0.3, value=0.6),  # Learning secrets
            TensionPoint(time=0.7, value=0.7),  # Understanding
            TensionPoint(time=1.0, value=0.5)   # Ready for task
        ]
    )
    
    success = PlotAtom(
        name="success",
        description="Hero completes task and receives recognition from king",
        roles=[(victor, hero), (recognizer, king)],
        preconditions=["help_received"],
        postconditions=["task_completed"],
        tension_points=[
            TensionPoint(time=0.0, value=0.6),  # Final attempt
            TensionPoint(time=0.3, value=0.8),  # Using wisdom
            TensionPoint(time=0.7, value=0.6),  # Triumph
            TensionPoint(time=1.0, value=0.3)   # Recognition
        ]
    )
    # Plot Span
    task_axis = PlotSpan(
        name="TASK",
        type=PlotSpanType.AXIS_OF_INTEREST,
        plot_atoms=[task, help, success],
        role_bindings={
            "seeker": hero,
            "dispatcher": king,
            "aided": hero,
            "helper": wise_woman,
            "victor": hero,
            "recognizer": king
        },
        protagonist=hero
    )
    
    # Plot Links
    link_1 = PlotLink(
        source_axis="task",
        target_axis="task",
        source_atom=task,
        target_atom=help,
        shared_roles=[hero]
    )
    generator.add_plot_link(link_1)

    link_2 = PlotLink(
        source_axis="task",
        target_axis="task",
        source_atom=help,
        target_atom=success,
        shared_roles=[hero, wise_woman]
    )
    generator.add_plot_link(link_2)

    link_3 = PlotLink(
        source_axis="task",
        target_axis="task",
        source_atom=task,
        target_atom=success,
        shared_roles=[king]
    )
    generator.add_plot_link(link_3)

    return task_axis
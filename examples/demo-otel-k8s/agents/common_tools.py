from dapr_agents import tool


@tool
def get_mission_status() -> str:
    """Get the current status of the space mission."""
    return (
        "MISSION STATUS: Mars Explorer 7 — Day 142 of transit. "
        "All systems nominal. Current velocity: 24.1 km/s. "
        "Distance from Earth: 1.2 AU. ETA Mars orbit: 68 days."
    )


tools = [get_mission_status]

"""Editing modes: the brief the editor model works from."""

MODES = {
    "highlights": (
        "Find the peak moments: big reactions, surprises, wins, fails, chases, jump scares, "
        "and anything a viewer would rewind to see again. Skip setup, menus, waiting and filler."),
    "comedy": (
        "Find the funniest moments: jokes with a clear setup and punchline, absurd situations, "
        "funny reactions, banter. Always keep the setup together with the punchline."),
    "serious": (
        "Find the most serious or dramatic moments: tension, sincere or emotional statements, "
        "confrontations, high stakes. Avoid jokes and filler."),
    "character": (
        "Find moments that show who the speaker or character is: opinions, stories about "
        "themselves, values, quirks, how they react under pressure. Each must make sense to "
        "someone who has not seen the rest."),
    "story": (
        "Find the moments that carry the story forward: the setup, each turning point, the "
        "climax and the ending. Prefer moments that explain what is happening and why."),
    "find": (
        "Find only the moments that match this request from the user: \"{query}\". "
        "If nothing in this section matches, return no moments."),
}

SYSTEM = (
    "You are a skilled short-form video editor. You read a section of a video transcript and "
    "choose the best moments to cut into clips. Each line is numbered and shows its start "
    "time in seconds. A line tagged [LOUD] had a sudden spike in volume, such as a shout, "
    "scream or laugh. Reply only with JSON.")

PROMPT = """BRIEF: {brief}

RULES:
- Pick at most {per_chunk} moments from this section. Pick none if nothing is good enough.
- A moment is a run of consecutive lines, from start_line to end_line.
- It must stand on its own: begin where the thought or setup begins, and end on the payoff \
or the end of a sentence.
- Aim for {min_len} to {max_len} seconds per moment.
- score is 1 to 10. Use 8 or more only for moments you would post. Be strict.
- title is a short label. reason is one sentence saying why it works.

TRANSCRIPT SECTION:
{lines}"""

SCHEMA = {
    "type": "object",
    "properties": {
        "moments": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "start_line": {"type": "integer"},
                    "end_line": {"type": "integer"},
                    "score": {"type": "integer"},
                    "title": {"type": "string"},
                    "reason": {"type": "string"},
                },
                "required": ["start_line", "end_line", "score", "title", "reason"],
            },
        }
    },
    "required": ["moments"],
}

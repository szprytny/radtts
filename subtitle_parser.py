from pysubparser import parser
from pysubparser.classes.subtitle import Subtitle
from typing import Tuple

def time_to_seconds(time):
    return sum(x * float(t) for x, t in zip([3600, 60, 1], str(time).split(":"))) 

def parse_subtitle(subtitle: Subtitle) -> Tuple[str, float]:
    return (subtitle.text, time_to_seconds(subtitle.start), time_to_seconds(subtitle.end))
    
def parse_subtitles_file(file: str):
    subtitles = parser.parse(file)
    return list(map(parse_subtitle, subtitles))

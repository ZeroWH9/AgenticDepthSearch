"""
UI styles module
"""
from rich.style import Style
from rich.theme import Theme

# Custom theme
theme = Theme({
    "info": "dim cyan",
    "warning": "magenta",
    "danger": "bold red",
    "success": "bold green",
    "title": "bold blue",
    "query": "bold cyan",
    "result": "dim",
    "url": "blue",
    "score": "green",
    "feedback": "yellow"
})

# Custom styles
styles = {
    "panel": Style(color="blue", bold=True),
    "table": Style(color="white"),
    "header": Style(color="cyan", bold=True),
    "row": Style(color="white"),
    "progress": Style(color="blue"),
    "error": Style(color="red", bold=True),
    "success": Style(color="green", bold=True)
} 
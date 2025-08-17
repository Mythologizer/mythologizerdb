import os
from typing import Iterator
from jinja2 import Template

def _render_template(name: str, folder: str = "schemas", **kwargs) -> str:
    path = os.path.join(folder, name)
    if not os.path.isfile(path):
        raise FileNotFoundError(f"Template '{name}' not found in folder '{folder}'.")

    with open(path, "r", encoding="utf-8") as f:
        template = Template(f.read())
    return template.render(**kwargs)

def get_schemas(folder: str, **kwargs) -> Iterator[str]:
    """
    Get all schema files from the specified folder in dependency order.
    
    Args:
        folder: Path to the folder containing schema files
        **kwargs: Template variables for .j2 files
    
    Yields:
        Rendered schema SQL strings in dependency order
    """
    # Define the dependency order for schema files
    dependency_order = [
        "init.sql.j2",           # Extensions first
        "cultures.sql.j2",       # Base tables
        "agents.sql.j2",         # Base tables
        "agent_attribute_defs.sql.j2", # Base tables
        "agent_attributes.sql.j2", # Base tables
        "mythemes.sql.j2",       # Base tables
        "myths.sql.j2",          # Base tables
        "agents_cultures.sql.j2", # Junction tables
        "agents_myths.sql.j2",   # Junction tables
        "myth_writings.sql.j2",  # Dependent tables
    ]
    
    # Process files in dependency order
    for filename in dependency_order:
        filepath = os.path.join(folder, filename)
        if os.path.isfile(filepath):
            if filename.endswith(".sql.j2"):
                yield _render_template(filename, folder, **kwargs)
            else:
                with open(filepath, "r", encoding="utf-8") as f:
                    yield f.read()


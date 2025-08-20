import os
import re
from pathlib import Path
from typing import Iterator
from jinja2 import Template
import importlib.resources as pkg_resources

# Since we're inside the package, we can import directly
import mythologizer_postgres


def _render_template(name: str, folder: str = "schemas", **kwargs) -> str:
    """Render a Jinja2 template from the schemas folder."""
    template_path = pkg_resources.files(mythologizer_postgres).joinpath(folder, name)
    template = Template(template_path.read_text())
    return template.render(**kwargs)


def get_schemas(folder: str = "schemas", **kwargs) -> Iterator[str]:
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
        "epoch.sql.j2",         # Dependent tables
    ]
    
    # Process files in dependency order
    for filename in dependency_order:
        filepath = pkg_resources.files(mythologizer_postgres).joinpath(folder, filename)
        if filepath.exists():
            if filename.endswith(".sql.j2"):
                yield _render_template(filename, folder, **kwargs)
            else:
                yield filepath.read_text()


def list_schema_files() -> list[str]:
    """
    List all available schema files in the package.
    
    Returns:
        List of schema file names
    """
    schemas_dir = pkg_resources.files(mythologizer_postgres).joinpath("schemas")
    if schemas_dir.exists():
        return [f.name for f in schemas_dir.iterdir() if f.is_file()]
    return []


def get_schema_content(filename: str) -> str:
    """
    Get the content of a specific schema file.
    
    Args:
        filename: Name of the schema file to read
    
    Returns:
        Content of the schema file
    """
    schema_path = pkg_resources.files(mythologizer_postgres).joinpath("schemas", filename)
    if schema_path.exists():
        return schema_path.read_text()
    else:
        raise FileNotFoundError(f"Schema file {filename} not found")


def _extract_schema_names(sql_content: str) -> list[str]:
    """Extract schema names from SQL content."""
    # Look for CREATE TABLE statements and extract table names
    table_pattern = r"CREATE TABLE\s+(\w+)"
    return re.findall(table_pattern, sql_content, re.IGNORECASE)

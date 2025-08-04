import os
from jinja2 import Template

def _render_template(name: str, folder: str = "schemas", **kwargs) -> str:
    path = os.path.join(folder, name)
    if not os.path.isfile(path):
        raise FileNotFoundError(f"Template '{name}' not found in folder '{folder}'.")

    with open(path, "r", encoding="utf-8") as f:
        template = Template(f.read())
    return template.render(**kwargs)

def get_myth_schema(dim: int) -> str:
    return _render_template("myths.sql.j2", dim=dim)

def get_mytheme_schema(dim: int) -> str:
    return _render_template("mythemes.sql.j2", dim=dim)

def get_init_schema() -> str:
    return _render_template("init.sql.j2")

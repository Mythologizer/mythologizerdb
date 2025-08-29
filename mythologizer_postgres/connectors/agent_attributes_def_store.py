from typing import List, Optional, Sequence, Tuple, Mapping, Any, Union
from sqlalchemy import text
from sqlalchemy.engine import Engine

from mythologizer_postgres.db import get_engine, psycopg_connection

EngineT = Engine

def get_agent_attribute_defs() -> List[Tuple[int, str, str, str, Optional[float], Optional[float], int]]:
    """
    Get all agent attribute definitions from the database.
    
    Returns:
        List of tuples (id, name, description, atype, min_val, max_val, col_idx) ordered by col_idx
    """
    with psycopg_connection() as conn:
        with conn.cursor() as cur:
            cur.execute("""
                SELECT id, name, description, atype, min_val, max_val, col_idx
                FROM agent_attribute_defs
                ORDER BY col_idx
            """)
            return cur.fetchall()


def insert_agent_attribute_defs(defs: Sequence[Union[Mapping[str, Any], Any]]) -> None:
    """
    Insert attribute definitions from a list of objects.

   defs is a list of dicts or Pydantic objects, each with the following keys:
   - name: str
   - type/atype/d_type: str or type
   - description: str
   - min_val/min/min_value: float
   - max_val/max/max_value: float
   - col_idx: int
   
    """
    engine: EngineT = get_engine()

    if not defs:
        return

    # Convert Pydantic objects to dictionaries if needed
    if defs and hasattr(defs[0], 'model_dump'):
        defs = [d.model_dump() for d in defs]

    records = []
    for idx, d in enumerate(defs):
        name = d.get("name")
        atype = d.get("type") or d.get("atype") or d.get("d_type")
        if not name or not atype:
            raise ValueError("Each definition must include 'name' and 'type'")

        # Normalize optional fields
        description = d.get("description")
        if "min_val" in d:
            min_val = d["min_val"]
        elif "min" in d:
            min_val = d["min"]
        elif "min_value" in d:
            min_val = d["min_value"]
        else:
            min_val = None

        if "max_val" in d:
            max_val = d["max_val"]
        elif "max" in d:
            max_val = d["max"]
        elif "max_value" in d:
            max_val = d["max_value"]
        else:
            max_val = None

        # Coerce numeric types where provided
        if min_val is not None:
            min_val = float(min_val)
        if max_val is not None:
            max_val = float(max_val)

        # Handle type conversion - atype can be a Python type or string
        if isinstance(atype, type):
            if atype == int:
                atype = "int"
            elif atype == float:
                atype = "float"
            else:
                raise ValueError(f"Unsupported type: {atype}. Only 'int' and 'float' are supported.")
        elif isinstance(atype, str):
            atype = atype.lower()
        else:
            raise ValueError(f"Invalid type format: {atype}. Must be 'int', 'float', or a Python type.")

        # Validate type
        if atype not in ("int", "float"):
            raise ValueError("'type' must be 'int' or 'float'")

        records.append(
            {
                "name": name,
                "description": description,
                "atype": atype,
                "min_val": min_val,
                "max_val": max_val,
                "col_idx": idx,
            }
        )

    sql_insert = text(
        """
        INSERT INTO public.agent_attribute_defs
            (name, description, atype, min_val, max_val, col_idx)
        VALUES
            (:name, :description, :atype, :min_val, :max_val, :col_idx)
        """
    )

    with engine.begin() as conn:
        conn.execute(sql_insert, records)


    
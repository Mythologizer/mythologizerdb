from typing import List, Optional, Sequence, Tuple, Mapping, Any
from sqlalchemy import text
from sqlalchemy.engine import Engine

from mythologizer_postgres.db import get_engine

EngineT = Engine

def insert_agent_attribute_defs(defs: Sequence[Mapping[str, Any]]) -> None:
    """
    Insert attribute definitions from a list of objects.

   defs is a list of dicts, each with the following keys:
   - name: str
   - type: str
   - description: str
   - min_val: float
   - max_val: float
   - col_idx: int
   
    """
    engine: EngineT = get_engine()

    if not defs:
        return

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


    
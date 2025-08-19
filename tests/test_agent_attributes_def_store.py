import pytest

from mythologizer_postgres.connectors import insert_agent_attribute_defs
from mythologizer_postgres.db import get_engine
from sqlalchemy import text


class TestAgentAttributeDefStore:
    @pytest.fixture(autouse=True)
    def cleanup_database(self):
        from mythologizer_postgres.db import clear_all_rows
        try:
            clear_all_rows()
        except Exception:
            pass
        yield
        try:
            clear_all_rows()
        except Exception:
            pass

    @pytest.mark.integration
    def test_insert_and_retrieve_agent_attribute_defs(self):
        defs = [
            {
                "name": "strength",
                "type": "float",
                "description": "Physical power",
                "min_val": 0.0,
                "max_val": 100.0,
            },
            {
                "name": "wisdom",
                "type": "int",
                "description": "Cognitive ability",
                "min_val": 0,
                "max_val": 200,
            },
            {
                "name": "luck",
                "type": "float",
                "description": None,
                "min_val": None,
                "max_val": None,
            },
        ]

        insert_agent_attribute_defs(defs)

        engine = get_engine()
        with engine.connect() as conn:
            rows = conn.execute(text(
                """
                SELECT id, name, description, atype, min_val, max_val, col_idx
                  FROM public.agent_attribute_defs
                 ORDER BY col_idx ASC
                """
            )).all()
        ids, names, descriptions, atypes, min_vals, max_vals, col_idxs = zip(*rows)
        names = list(names)
        descriptions = list(descriptions)
        atypes = list(atypes)
        col_idxs = list(col_idxs)

        assert len(ids) == 3
        assert names == [d["name"] for d in defs]
        assert descriptions == [d["description"] for d in defs]
        assert atypes == [d["type"] for d in defs]
        assert col_idxs == [0, 1, 2]

        # Convert numeric optionals to floats for robust comparison
        min_vals_f = [float(v) if v is not None else None for v in min_vals]
        max_vals_f = [float(v) if v is not None else None for v in max_vals]
        expected_min = [0.0, 0.0, None]
        expected_max = [100.0, 200.0, None]
        assert min_vals_f == expected_min
        assert max_vals_f == expected_max

    @pytest.mark.integration
    def test_insert_with_d_type_and_min_max_synonyms(self):
        defs = [
            {
                "name": "charisma",
                "d_type": "float",  # synonym for type
                "description": "Charm and appeal",
                "min": 0.5,          # synonym for min_val
                "max": 99.5,         # synonym for max_val
            },
            {
                "name": "intellect",
                "d_type": "int",
                "description": None,
                "min": 1,
                "max": 999,
            },
        ]

        insert_agent_attribute_defs(defs)

        engine = get_engine()
        with engine.connect() as conn:
            rows = conn.execute(text(
                """
                SELECT id, name, description, atype, min_val, max_val, col_idx
                  FROM public.agent_attribute_defs
                 ORDER BY col_idx ASC
                """
            )).all()
        ids, names, descriptions, atypes, min_vals, max_vals, col_idxs = zip(*rows)
        names = list(names)
        descriptions = list(descriptions)
        atypes = list(atypes)
        col_idxs = list(col_idxs)

        assert len(ids) == 2
        assert names == ["charisma", "intellect"]
        assert descriptions == ["Charm and appeal", None]
        assert atypes == ["float", "int"]
        assert col_idxs == [0, 1]

        # Coerce for comparison
        min_vals_f = [float(v) if v is not None else None for v in min_vals]
        max_vals_f = [float(v) if v is not None else None for v in max_vals]
        assert min_vals_f == [0.5, 1.0]
        assert max_vals_f == [99.5, 999.0]



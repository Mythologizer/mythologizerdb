"""
Agent attributes matrix store implementation.
"""

import numpy as np
from typing import Tuple, List, Optional, Dict
from sqlalchemy import text
from sqlalchemy.engine import Engine

from mythologizer_postgres.db import get_engine, psycopg_connection

EngineT = Engine


def get_agent_attribute_matrix() -> Tuple[np.ndarray, List[int], Dict[str, int]]:
    """
    Get all agent attribute entries as a numpy matrix.
    
    Returns:
        Tuple containing:
        - matrix: numpy array where each row is an agent and each column is an attribute
        - agent_indices: list of agent IDs corresponding to the rows
        - attribute_name_to_col: dictionary mapping attribute names to their column indices
    
    The matrix is structured as:
    - Rows: agents (ordered by agent_id)
    - Columns: attributes (ordered by col_idx from agent_attribute_defs)
    - Values: attribute values from agent_attributes.attribute_values array
    """
    
    with psycopg_connection() as conn:
        with conn.cursor() as cur:
            # First, get the attribute definitions ordered by col_idx
            cur.execute("""
                SELECT name, col_idx 
                FROM agent_attribute_defs 
                ORDER BY col_idx ASC
            """)
            attribute_defs = cur.fetchall()
            
            if not attribute_defs:
                # No attributes defined, return empty matrix
                return np.array([]), [], {}
            
            num_attributes = len(attribute_defs)
            attribute_name_to_col = {def_[0]: def_[1] for def_ in attribute_defs}
            
            # Get all agents with their attribute values
            cur.execute("""
                SELECT a.id, aa.attribute_values
                FROM agents a
                LEFT JOIN agent_attributes aa ON a.id = aa.agent_id
                ORDER BY a.id ASC
            """)
            agent_data = cur.fetchall()
            
            if not agent_data:
                # No agents, return empty matrix
                return np.array([]), [], attribute_name_to_col
            
            num_agents = len(agent_data)
            agent_indices = []
            matrix_data = []
            
            for agent_id, attribute_values in agent_data:
                agent_indices.append(agent_id)
                
                if attribute_values is None:
                    # Agent has no attributes, fill with NaN
                    row = [np.nan] * num_attributes
                else:
                    # Convert attribute_values array to list and ensure it has the right length
                    values = list(attribute_values)
                    if len(values) < num_attributes:
                        # Pad with NaN if not enough values
                        values.extend([np.nan] * (num_attributes - len(values)))
                    elif len(values) > num_attributes:
                        # Truncate if too many values
                        values = values[:num_attributes]
                    
                    row = values
                
                matrix_data.append(row)
            
            # Convert to numpy array
            matrix = np.array(matrix_data, dtype=np.float64)
            
            return matrix, agent_indices, attribute_name_to_col


def update_agent_attribute_matrix(matrix: np.ndarray, agent_indices: List[int]) -> None:
    """
    Update the agent_attributes table with a modified matrix.
    
    Args:
        matrix: numpy array where each row is an agent and each column is an attribute
        agent_indices: list of agent IDs corresponding to the rows (must match matrix rows)
    
    The matrix should have the same structure as returned by get_agent_attribute_matrix:
    - Rows: agents (in the same order as agent_indices)
    - Columns: attributes (ordered by col_idx from agent_attribute_defs)
    - Values: attribute values to be stored in agent_attributes.attribute_values array
    """
    
    if matrix.size == 0 or len(agent_indices) == 0:
        return
    
    if matrix.shape[0] != len(agent_indices):
        raise ValueError(f"Matrix has {matrix.shape[0]} rows but {len(agent_indices)} agent indices provided")
    
    with psycopg_connection() as conn:
        with conn.cursor() as cur:
            # Get the number of attributes to validate matrix shape
            cur.execute("""
                SELECT COUNT(*) 
                FROM agent_attribute_defs
            """)
            num_attributes = cur.fetchone()[0]
            
            if matrix.shape[1] != num_attributes:
                raise ValueError(f"Matrix has {matrix.shape[1]} columns but {num_attributes} attributes are defined")
            
            # Update each agent's attributes
            for agent_idx, agent_id in enumerate(agent_indices):
                # Convert numpy row to list, handling NaN values
                row_values = matrix[agent_idx, :].tolist()
                
                # Replace NaN with None for database storage
                db_values = [None if np.isnan(val) else float(val) for val in row_values]
                
                # Use UPSERT to insert or update
                cur.execute("""
                    INSERT INTO agent_attributes (agent_id, attribute_values)
                    VALUES (%s, %s)
                    ON CONFLICT (agent_id) 
                    DO UPDATE SET attribute_values = EXCLUDED.attribute_values
                """, (agent_id, db_values))
            
            conn.commit()


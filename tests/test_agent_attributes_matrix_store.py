"""
Tests for the agent attributes matrix store.
"""

import pytest
import numpy as np
from sqlalchemy import text

from mythologizer_postgres.connectors.agent_atributes_matrix_store import (
    get_agent_attribute_matrix,
    update_agent_attribute_matrix,
)
from mythologizer_postgres.connectors import insert_agent_attribute_defs
from mythologizer_postgres.db import get_engine, clear_all_rows


class TestAgentAttributesMatrixStore:
    """Test the agent attributes matrix store functions."""

    def setup_method(self):
        """Clean up before each test method."""
        clear_all_rows()

    def teardown_method(self):
        """Clean up after each test method."""
        clear_all_rows()

    @pytest.mark.integration
    def test_get_agent_attribute_matrix_empty(self):
        """Test getting agent attribute matrix when no data exists."""
        matrix, agent_indices, attribute_name_to_col = get_agent_attribute_matrix()
        
        assert matrix.size == 0, "Matrix should be empty"
        assert len(agent_indices) == 0, "Agent indices should be empty"
        assert len(attribute_name_to_col) == 0, "Attribute name to column mapping should be empty"

    @pytest.mark.integration
    def test_get_agent_attribute_matrix_no_agents(self):
        """Test getting agent attribute matrix when attributes exist but no agents."""
        # Insert attribute definitions
        defs = [
            {"name": "strength", "type": "float", "description": "Physical power"},
            {"name": "wisdom", "type": "int", "description": "Cognitive ability"},
        ]
        insert_agent_attribute_defs(defs)
        
        matrix, agent_indices, attribute_name_to_col = get_agent_attribute_matrix()
        
        assert matrix.size == 0, "Matrix should be empty"
        assert len(agent_indices) == 0, "Agent indices should be empty"
        assert attribute_name_to_col == {"strength": 0, "wisdom": 1}, "Should return attribute name to column mapping"

    @pytest.mark.integration
    def test_get_agent_attribute_matrix_no_attributes(self):
        """Test getting agent attribute matrix when agents exist but no attributes."""
        # Insert agents
        engine = get_engine()
        with engine.connect() as conn:
            conn.execute(text("INSERT INTO agents (name, memory_size) VALUES ('Agent 1', 10)"))
            conn.execute(text("INSERT INTO agents (name, memory_size) VALUES ('Agent 2', 15)"))
            conn.commit()
        
        matrix, agent_indices, attribute_name_to_col = get_agent_attribute_matrix()
        
        assert matrix.size == 0, "Matrix should be empty"
        assert len(agent_indices) == 0, "Agent indices should be empty"
        assert len(attribute_name_to_col) == 0, "Attribute name to column mapping should be empty"

    @pytest.mark.integration
    def test_get_agent_attribute_matrix_basic(self):
        """Test getting agent attribute matrix with basic data."""
        # Insert attribute definitions
        defs = [
            {"name": "strength", "type": "float", "description": "Physical power"},
            {"name": "wisdom", "type": "int", "description": "Cognitive ability"},
            {"name": "luck", "type": "float", "description": "Luck factor"},
        ]
        insert_agent_attribute_defs(defs)
        
        # Insert agents
        engine = get_engine()
        with engine.connect() as conn:
            conn.execute(text("INSERT INTO agents (name, memory_size) VALUES ('Agent 1', 10)"))
            conn.execute(text("INSERT INTO agents (name, memory_size) VALUES ('Agent 2', 15)"))
            conn.commit()
            
            # Get agent IDs
            result = conn.execute(text("SELECT id FROM agents ORDER BY id"))
            agent_ids = [row[0] for row in result.fetchall()]
            
            # Insert agent attributes
            conn.execute(text("""
                INSERT INTO agent_attributes (agent_id, attribute_values) 
                VALUES (:agent_id, :attribute_values)
            """), {"agent_id": agent_ids[0], "attribute_values": [10.5, 20.0, 0.8]})
            
            conn.execute(text("""
                INSERT INTO agent_attributes (agent_id, attribute_values) 
                VALUES (:agent_id, :attribute_values)
            """), {"agent_id": agent_ids[1], "attribute_values": [15.2, 25.0, 0.3]})
            
            conn.commit()
        
        matrix, agent_indices, attribute_name_to_col = get_agent_attribute_matrix()
        
        # Verify results
        assert matrix.shape == (2, 3), f"Matrix should be 2x3, got {matrix.shape}"
        assert agent_indices == agent_ids, "Agent indices should match"
        assert attribute_name_to_col == {"strength": 0, "wisdom": 1, "luck": 2}, "Attribute name to column mapping should match"
        
        # Verify matrix values
        expected_matrix = np.array([
            [10.5, 20.0, 0.8],  # Agent 1
            [15.2, 25.0, 0.3],  # Agent 2
        ])
        np.testing.assert_array_equal(matrix, expected_matrix)

    @pytest.mark.integration
    def test_get_agent_attribute_matrix_missing_attributes(self):
        """Test getting agent attribute matrix when some agents have missing attributes."""
        # Insert attribute definitions
        defs = [
            {"name": "strength", "type": "float", "description": "Physical power"},
            {"name": "wisdom", "type": "int", "description": "Cognitive ability"},
        ]
        insert_agent_attribute_defs(defs)
        
        # Insert agents
        engine = get_engine()
        with engine.connect() as conn:
            conn.execute(text("INSERT INTO agents (name, memory_size) VALUES ('Agent 1', 10)"))
            conn.execute(text("INSERT INTO agents (name, memory_size) VALUES ('Agent 2', 15)"))
            conn.execute(text("INSERT INTO agents (name, memory_size) VALUES ('Agent 3', 20)"))
            conn.commit()
            
            # Get agent IDs
            result = conn.execute(text("SELECT id FROM agents ORDER BY id"))
            agent_ids = [row[0] for row in result.fetchall()]
            
            # Insert agent attributes for only some agents
            conn.execute(text("""
                INSERT INTO agent_attributes (agent_id, attribute_values) 
                VALUES (:agent_id, :attribute_values)
            """), {"agent_id": agent_ids[0], "attribute_values": [10.5, 20.0]})
            
            conn.execute(text("""
                INSERT INTO agent_attributes (agent_id, attribute_values) 
                VALUES (:agent_id, :attribute_values)
            """), {"agent_id": agent_ids[2], "attribute_values": [8.0, 18.0]})
            
            conn.commit()
        
        matrix, agent_indices, attribute_name_to_col = get_agent_attribute_matrix()
        
        # Verify results
        assert matrix.shape == (3, 2), f"Matrix should be 3x2, got {matrix.shape}"
        assert agent_indices == agent_ids, "Agent indices should match"
        assert attribute_name_to_col == {"strength": 0, "wisdom": 1}, "Attribute name to column mapping should match"
        
        # Verify matrix values (Agent 2 should have NaN values)
        assert not np.isnan(matrix[0, 0]), "Agent 1 should have strength value"
        assert not np.isnan(matrix[0, 1]), "Agent 1 should have wisdom value"
        assert np.isnan(matrix[1, 0]), "Agent 2 should have NaN strength"
        assert np.isnan(matrix[1, 1]), "Agent 2 should have NaN wisdom"
        assert not np.isnan(matrix[2, 0]), "Agent 3 should have strength value"
        assert not np.isnan(matrix[2, 1]), "Agent 3 should have wisdom value"

    @pytest.mark.integration
    def test_get_agent_attribute_matrix_short_values(self):
        """Test getting agent attribute matrix when attribute values are shorter than expected."""
        # Insert attribute definitions
        defs = [
            {"name": "strength", "type": "float", "description": "Physical power"},
            {"name": "wisdom", "type": "int", "description": "Cognitive ability"},
            {"name": "luck", "type": "float", "description": "Luck factor"},
        ]
        insert_agent_attribute_defs(defs)
        
        # Insert agents
        engine = get_engine()
        with engine.connect() as conn:
            conn.execute(text("INSERT INTO agents (name, memory_size) VALUES ('Agent 1', 10)"))
            conn.commit()
            
            # Get agent ID
            result = conn.execute(text("SELECT id FROM agents"))
            agent_id = result.fetchone()[0]
            
            # Insert agent attributes with only 2 values (should be padded with NaN)
            conn.execute(text("""
                INSERT INTO agent_attributes (agent_id, attribute_values) 
                VALUES (:agent_id, :attribute_values)
            """), {"agent_id": agent_id, "attribute_values": [10.5, 20.0]})
            
            conn.commit()
        
        matrix, agent_indices, attribute_name_to_col = get_agent_attribute_matrix()
        
        # Verify results
        assert matrix.shape == (1, 3), f"Matrix should be 1x3, got {matrix.shape}"
        assert agent_indices == [agent_id], "Agent indices should match"
        assert attribute_name_to_col == {"strength": 0, "wisdom": 1, "luck": 2}, "Attribute name to column mapping should match"
        
        # Verify matrix values (third value should be NaN)
        assert matrix[0, 0] == 10.5, "First value should be 10.5"
        assert matrix[0, 1] == 20.0, "Second value should be 20.0"
        assert np.isnan(matrix[0, 2]), "Third value should be NaN"

    @pytest.mark.integration
    def test_get_agent_attribute_matrix_long_values(self):
        """Test getting agent attribute matrix when attribute values are longer than expected."""
        # Insert attribute definitions
        defs = [
            {"name": "strength", "type": "float", "description": "Physical power"},
            {"name": "wisdom", "type": "int", "description": "Cognitive ability"},
        ]
        insert_agent_attribute_defs(defs)
        
        # Insert agents
        engine = get_engine()
        with engine.connect() as conn:
            conn.execute(text("INSERT INTO agents (name, memory_size) VALUES ('Agent 1', 10)"))
            conn.commit()
            
            # Get agent ID
            result = conn.execute(text("SELECT id FROM agents"))
            agent_id = result.fetchone()[0]
            
            # Insert agent attributes with 4 values (should be truncated to 2)
            conn.execute(text("""
                INSERT INTO agent_attributes (agent_id, attribute_values) 
                VALUES (:agent_id, :attribute_values)
            """), {"agent_id": agent_id, "attribute_values": [10.5, 20.0, 0.8, 99.9]})
            
            conn.commit()
        
        matrix, agent_indices, attribute_name_to_col = get_agent_attribute_matrix()
        
        # Verify results
        assert matrix.shape == (1, 2), f"Matrix should be 1x2, got {matrix.shape}"
        assert agent_indices == [agent_id], "Agent indices should match"
        assert attribute_name_to_col == {"strength": 0, "wisdom": 1}, "Attribute name to column mapping should match"
        
        # Verify matrix values (only first 2 values should be used)
        assert matrix[0, 0] == 10.5, "First value should be 10.5"
        assert matrix[0, 1] == 20.0, "Second value should be 20.0"

    @pytest.mark.integration
    def test_update_agent_attribute_matrix_basic(self):
        """Test updating agent attribute matrix with basic data."""
        # Insert attribute definitions
        defs = [
            {"name": "strength", "type": "float", "description": "Physical power"},
            {"name": "wisdom", "type": "int", "description": "Cognitive ability"},
        ]
        insert_agent_attribute_defs(defs)
        
        # Insert agents
        engine = get_engine()
        with engine.connect() as conn:
            conn.execute(text("INSERT INTO agents (name, memory_size) VALUES ('Agent 1', 10)"))
            conn.execute(text("INSERT INTO agents (name, memory_size) VALUES ('Agent 2', 15)"))
            conn.commit()
            
            # Get agent IDs
            result = conn.execute(text("SELECT id FROM agents ORDER BY id"))
            agent_ids = [row[0] for row in result.fetchall()]
        
        # Create a matrix to update
        matrix = np.array([
            [12.5, 22.0],  # Agent 1 - updated values
            [17.2, 27.0],  # Agent 2 - updated values
        ])
        
        # Update the matrix
        update_agent_attribute_matrix(matrix, agent_ids)
        
        # Verify the update by reading back
        updated_matrix, updated_agent_indices, attribute_name_to_col = get_agent_attribute_matrix()
        
        assert updated_matrix.shape == (2, 2), f"Matrix should be 2x2, got {updated_matrix.shape}"
        assert updated_agent_indices == agent_ids, "Agent indices should match"
        assert attribute_name_to_col == {"strength": 0, "wisdom": 1}, "Attribute name to column mapping should match"
        
        # Verify matrix values were updated
        np.testing.assert_array_equal(updated_matrix, matrix)

    @pytest.mark.integration
    def test_update_agent_attribute_matrix_with_nan(self):
        """Test updating agent attribute matrix with NaN values."""
        # Insert attribute definitions
        defs = [
            {"name": "strength", "type": "float", "description": "Physical power"},
            {"name": "wisdom", "type": "int", "description": "Cognitive ability"},
        ]
        insert_agent_attribute_defs(defs)
        
        # Insert agents
        engine = get_engine()
        with engine.connect() as conn:
            conn.execute(text("INSERT INTO agents (name, memory_size) VALUES ('Agent 1', 10)"))
            conn.commit()
            
            # Get agent ID
            result = conn.execute(text("SELECT id FROM agents"))
            agent_id = result.fetchone()[0]
        
        # Create a matrix with NaN values
        matrix = np.array([
            [np.nan, 25.0],  # Agent 1 - strength is NaN, wisdom is 25
        ])
        
        # Update the matrix
        update_agent_attribute_matrix(matrix, [agent_id])
        
        # Verify the update by reading back
        updated_matrix, updated_agent_indices, attribute_name_to_col = get_agent_attribute_matrix()
        
        assert updated_matrix.shape == (1, 2), f"Matrix should be 1x2, got {updated_matrix.shape}"
        assert updated_agent_indices == [agent_id], "Agent indices should match"
        
        # Verify NaN was preserved
        assert np.isnan(updated_matrix[0, 0]), "First value should be NaN"
        assert updated_matrix[0, 1] == 25.0, "Second value should be 25.0"

    @pytest.mark.integration
    def test_update_agent_attribute_matrix_mismatched_dimensions(self):
        """Test updating agent attribute matrix with mismatched dimensions."""
        # Insert attribute definitions
        defs = [
            {"name": "strength", "type": "float", "description": "Physical power"},
            {"name": "wisdom", "type": "int", "description": "Cognitive ability"},
        ]
        insert_agent_attribute_defs(defs)
        
        # Insert agents
        engine = get_engine()
        with engine.connect() as conn:
            conn.execute(text("INSERT INTO agents (name, memory_size) VALUES ('Agent 1', 10)"))
            conn.execute(text("INSERT INTO agents (name, memory_size) VALUES ('Agent 2', 15)"))
            conn.commit()
            
            # Get agent IDs
            result = conn.execute(text("SELECT id FROM agents ORDER BY id"))
            agent_ids = [row[0] for row in result.fetchall()]
        
        # Create a matrix with wrong number of rows
        matrix = np.array([
            [12.5, 22.0],  # Only one row but two agents
        ])
        
        # Should raise ValueError
        with pytest.raises(ValueError, match="Matrix has 1 rows but 2 agent indices provided"):
            update_agent_attribute_matrix(matrix, agent_ids)
        
        # Create a matrix with wrong number of columns
        matrix = np.array([
            [12.5],        # Only one column but two attributes
            [17.2],
        ])
        
        # Should raise ValueError
        with pytest.raises(ValueError, match="Matrix has 1 columns but 2 attributes are defined"):
            update_agent_attribute_matrix(matrix, agent_ids)

    @pytest.mark.integration
    def test_update_agent_attribute_matrix_empty(self):
        """Test updating agent attribute matrix with empty data."""
        # Should not raise any errors
        update_agent_attribute_matrix(np.array([]), [])
        update_agent_attribute_matrix(np.array([[1.0, 2.0]]), [])
        update_agent_attribute_matrix(np.array([]), [1, 2])

    @pytest.mark.integration
    def test_get_agent_attribute_matrix_column_order(self):
        """Test that columns are ordered by col_idx from agent_attribute_defs."""
        # Insert attribute definitions in the order they should appear in the matrix
        defs = [
            {"name": "strength", "type": "float", "description": "Physical power"},
            {"name": "luck", "type": "float", "description": "Luck factor"},
            {"name": "wisdom", "type": "int", "description": "Cognitive ability"},
        ]
        insert_agent_attribute_defs(defs)
        
        # Insert an agent
        engine = get_engine()
        with engine.connect() as conn:
            conn.execute(text("INSERT INTO agents (name, memory_size) VALUES ('Agent 1', 10)"))
            conn.commit()
            
            # Get agent ID
            result = conn.execute(text("SELECT id FROM agents"))
            agent_id = result.fetchone()[0]
            
            # Insert agent attributes
            conn.execute(text("""
                INSERT INTO agent_attributes (agent_id, attribute_values) 
                VALUES (:agent_id, :attribute_values)
            """), {"agent_id": agent_id, "attribute_values": [10.5, 0.8, 20.0]})
            
            conn.commit()
        
        matrix, agent_indices, attribute_name_to_col = get_agent_attribute_matrix()
        
        # Verify column order matches the order they were inserted (which becomes col_idx)
        assert attribute_name_to_col == {"strength": 0, "luck": 1, "wisdom": 2}, "Columns should be ordered by col_idx"
        
        # Verify matrix values are in correct order
        expected_matrix = np.array([[10.5, 0.8, 20.0]])  # strength, luck, wisdom
        np.testing.assert_array_equal(matrix, expected_matrix)

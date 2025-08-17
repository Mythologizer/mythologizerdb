import os
import pytest
import numpy as np
from sqlalchemy import text

from mythologizer_postgres.db import (
    session_scope,
    get_table_row_counts,
    clear_all_rows,
)


class TestAgentAttributesSchema:
    """Test the agent_attributes table schema and operations."""
    
    @pytest.mark.integration
    def test_agent_attributes_table_structure(self):
        """Test that agent_attributes table has the correct structure."""
        with session_scope() as session:
            # Check agent_attributes table structure
            result = session.execute(text("""
                SELECT column_name, data_type, udt_name
                FROM information_schema.columns 
                WHERE table_name = 'agent_attributes' 
                ORDER BY ordinal_position
            """))
            columns = {row[0]: (row[1], row[2]) for row in result.fetchall()}
            
            expected_columns = {
                'agent_id': ('integer', 'int4'),
                'attribute_values': ('ARRAY', 'float8')
            }
            
            for col, (expected_type, expected_udt) in expected_columns.items():
                assert col in columns, f"Column {col} should exist in agent_attributes table"
                assert columns[col][0] == expected_type, f"Column {col} should be {expected_type}"
                # For array columns, check the UDT name to ensure it's the right array type
                if expected_type == 'ARRAY':
                    assert expected_udt in columns[col][1], f"Column {col} should have UDT containing {expected_udt}"
    
    @pytest.mark.integration
    def test_agent_attributes_table_constraints(self):
        """Test that agent_attributes table constraints work correctly."""
        with session_scope() as session:
            # First create an agent to reference
            session.execute(text("""
                INSERT INTO agents (name, memory_size) 
                VALUES (:name, :memory_size)
            """), {
                'name': 'Test Agent for Attributes',
                'memory_size': 10
            })
            
            agent_result = session.execute(text("SELECT id FROM agents WHERE name = 'Test Agent for Attributes'"))
            agent_id = agent_result.fetchone()[0]
            
            # Test valid insertions with float arrays
            session.execute(text("""
                INSERT INTO agent_attributes (agent_id, attribute_values) 
                VALUES (:agent_id, :attribute_values)
            """), {
                'agent_id': agent_id,
                'attribute_values': [1.5, 2.7, 3.14, 4.0]
            })
            
            # Test that we can't insert another row with the same agent_id (PRIMARY KEY constraint)
            with pytest.raises(Exception):
                session.execute(text("""
                    INSERT INTO agent_attributes (agent_id, attribute_values) 
                    VALUES (:agent_id, :attribute_values)
                """), {
                    'agent_id': agent_id,
                    'attribute_values': [1.0, 2.0, 3.0, 4.0]  # Use floats to avoid mixed type issues
                })
            
            # Test foreign key constraint
            with pytest.raises(Exception):
                session.execute(text("""
                    INSERT INTO agent_attributes (agent_id, attribute_values) 
                    VALUES (:agent_id, :attribute_values)
                """), {
                    'agent_id': 99999,  # Non-existent agent ID
                    'attribute_values': [1.0, 2.0, 3.0]
                })
            
            # Test NOT NULL constraint
            with pytest.raises(Exception):
                session.execute(text("""
                    INSERT INTO agent_attributes (agent_id, attribute_values) 
                    VALUES (:agent_id, :attribute_values)
                """), {
                    'agent_id': agent_id,
                    'attribute_values': None
                })
        
        # Clean up
        clear_all_rows()
    
    @pytest.mark.integration
    def test_agent_attributes_type_conversions(self):
        """Test that agent_attributes can handle different numeric types correctly."""
        with session_scope() as session:
            # Create test agent
            session.execute(text("""
                INSERT INTO agents (name, memory_size) 
                VALUES (:name, :memory_size)
            """), {
                'name': 'Type Conversion Test Agent',
                'memory_size': 8
            })
            
            agent_result = session.execute(text("SELECT id FROM agents WHERE name = 'Type Conversion Test Agent'"))
            agent_id = agent_result.fetchone()[0]
            
            # Note: PostgreSQL DOUBLE PRECISION[] can store both integers and floats,
            # but psycopg requires consistent types when sending arrays to avoid
            # "cannot dump lists of mixed types" errors. The solution is to convert
            # all values to the same type (float) before sending to PostgreSQL.
            
            # Test inserting float array
            session.execute(text("""
                INSERT INTO agent_attributes (agent_id, attribute_values) 
                VALUES (:agent_id, :attribute_values)
            """), {
                'agent_id': agent_id,
                'attribute_values': [1.5, 2.7, 3.14, 4.0]
            })
            
            # Test updating to int array (all values as floats to avoid mixed type issues)
            session.execute(text("""
                UPDATE agent_attributes 
                SET attribute_values = :new_values 
                WHERE agent_id = :agent_id
            """), {
                'agent_id': agent_id,
                'new_values': [1.0, 2.0, 3.0, 4.0]
            })
            
            # Verify the int array update
            result = session.execute(text("""
                SELECT attribute_values 
                FROM agent_attributes 
                WHERE agent_id = :agent_id
            """), {'agent_id': agent_id})
            
            int_row = result.fetchone()
            assert int_row is not None
            
            # Should have the int array (stored as doubles by PostgreSQL)
            int_array = int_row[0]
            assert len(int_array) == 4
            assert int_array == [1.0, 2.0, 3.0, 4.0]
        
        # Clean up
        clear_all_rows()
    
    @pytest.mark.integration
    def test_agent_attributes_array_operations(self):
        """Test array operations and queries on attribute_values."""
        with session_scope() as session:
            # Create test agents
            session.execute(text("""
                INSERT INTO agents (name, memory_size) 
                VALUES (:name, :memory_size)
            """), {
                'name': 'Agent Alpha',
                'memory_size': 10
            })
            
            session.execute(text("""
                INSERT INTO agents (name, memory_size) 
                VALUES (:name, :memory_size)
            """), {
                'name': 'Agent Beta',
                'memory_size': 15
            })
            
            agent_alpha_result = session.execute(text("SELECT id FROM agents WHERE name = 'Agent Alpha'"))
            agent_alpha_id = agent_alpha_result.fetchone()[0]
            
            agent_beta_result = session.execute(text("SELECT id FROM agents WHERE name = 'Agent Beta'"))
            agent_beta_id = agent_beta_result.fetchone()[0]
            
            # Insert attribute arrays
            session.execute(text("""
                INSERT INTO agent_attributes (agent_id, attribute_values) 
                VALUES (:agent_id, :attribute_values)
            """), {
                'agent_id': agent_alpha_id,
                'attribute_values': [1.1, 2.2, 3.3, 4.4]
            })
            
            session.execute(text("""
                INSERT INTO agent_attributes (agent_id, attribute_values) 
                VALUES (:agent_id, :attribute_values)
            """), {
                'agent_id': agent_beta_id,
                'attribute_values': [5.0, 6.0, 7.0, 8.0]  # Use floats to avoid mixed type issues
            })
            
            # Test array length queries
            result = session.execute(text("""
                SELECT agent_id, array_length(attribute_values, 1) as array_len
                FROM agent_attributes 
                ORDER BY agent_id
            """))
            
            lengths = {row[0]: row[1] for row in result.fetchall()}
            assert lengths[agent_alpha_id] == 4
            assert lengths[agent_beta_id] == 4
            
            # Test array element access
            result = session.execute(text("""
                SELECT agent_id, attribute_values[1] as first_element, attribute_values[4] as last_element
                FROM agent_attributes 
                ORDER BY agent_id
            """))
            
            elements = {row[0]: (row[1], row[2]) for row in result.fetchall()}
            assert elements[agent_alpha_id] == (1.1, 4.4)
            assert elements[agent_beta_id] == (5.0, 8.0)  # PostgreSQL casts ints to doubles
            
            # Test array contains operator
            result = session.execute(text("""
                SELECT agent_id 
                FROM agent_attributes 
                WHERE 3.3 = ANY(attribute_values)
            """))
            
            matching_agents = [row[0] for row in result.fetchall()]
            assert agent_alpha_id in matching_agents
            assert agent_beta_id not in matching_agents
        
        # Clean up
        clear_all_rows()
    
    @pytest.mark.integration
    def test_agent_attributes_crud_operations(self):
        """Test basic CRUD operations on agent_attributes table."""
        with session_scope() as session:
            # Create test agent
            session.execute(text("""
                INSERT INTO agents (name, memory_size) 
                VALUES (:name, :memory_size)
            """), {
                'name': 'CRUD Test Agent',
                'memory_size': 12
            })
            
            agent_result = session.execute(text("SELECT id FROM agents WHERE name = 'CRUD Test Agent'"))
            agent_id = agent_result.fetchone()[0]
            
            # Create - Insert attribute values
            session.execute(text("""
                INSERT INTO agent_attributes (agent_id, attribute_values) 
                VALUES (:agent_id, :attribute_values)
            """), {
                'agent_id': agent_id,
                'attribute_values': [10.5, 20.7, 30.1, 40.9]
            })
            
            # Read - Query the inserted data
            result = session.execute(text("""
                SELECT agent_id, attribute_values 
                FROM agent_attributes 
                WHERE agent_id = :agent_id
            """), {'agent_id': agent_id})
            
            row = result.fetchone()
            assert row is not None
            assert row[0] == agent_id
            assert len(row[1]) == 4
            assert row[1] == [10.5, 20.7, 30.1, 40.9]
            
            # Update - Modify the attribute values
            session.execute(text("""
                UPDATE agent_attributes 
                SET attribute_values = :new_values 
                WHERE agent_id = :agent_id
            """), {
                'agent_id': agent_id,
                'new_values': [15.0, 25.0, 35.0, 45.0]
            })
            
            # Verify update
            result = session.execute(text("""
                SELECT attribute_values 
                FROM agent_attributes 
                WHERE agent_id = :agent_id
            """), {'agent_id': agent_id})
            
            updated_row = result.fetchone()
            assert updated_row[0] == [15.0, 25.0, 35.0, 45.0]
            
            # Delete - Remove the attribute values
            session.execute(text("""
                DELETE FROM agent_attributes 
                WHERE agent_id = :agent_id
            """), {'agent_id': agent_id})
            
            # Verify deletion
            result = session.execute(text("""
                SELECT COUNT(*) 
                FROM agent_attributes 
                WHERE agent_id = :agent_id
            """), {'agent_id': agent_id})
            
            count = result.fetchone()[0]
            assert count == 0
        
        # Clean up
        clear_all_rows()
    
    @pytest.mark.integration
    def test_agent_attributes_cascade_delete(self):
        """Test that agent_attributes are deleted when the referenced agent is deleted."""
        with session_scope() as session:
            # Create test agent
            session.execute(text("""
                INSERT INTO agents (name, memory_size) 
                VALUES (:name, :memory_size)
            """), {
                'name': 'Cascade Test Agent',
                'memory_size': 8
            })
            
            agent_result = session.execute(text("SELECT id FROM agents WHERE name = 'Cascade Test Agent'"))
            agent_id = agent_result.fetchone()[0]
            
            # Create attribute values
            session.execute(text("""
                INSERT INTO agent_attributes (agent_id, attribute_values) 
                VALUES (:agent_id, :attribute_values)
            """), {
                'agent_id': agent_id,
                'attribute_values': [1.0, 2.0, 3.0, 4.0]
            })
            
            # Verify attribute values exist
            result = session.execute(text("""
                SELECT COUNT(*) 
                FROM agent_attributes 
                WHERE agent_id = :agent_id
            """), {'agent_id': agent_id})
            
            assert result.fetchone()[0] == 1
            
            # Delete the agent (should cascade to agent_attributes)
            session.execute(text("DELETE FROM agents WHERE id = :agent_id"), {'agent_id': agent_id})
            
            # Verify attribute values are also deleted
            result = session.execute(text("""
                SELECT COUNT(*) 
                FROM agent_attributes 
                WHERE agent_id = :agent_id
            """), {'agent_id': agent_id})
            
            assert result.fetchone()[0] == 0
        
        # Clean up
        clear_all_rows()
    
    @pytest.mark.integration
    def test_agent_attributes_array_functions(self):
        """Test PostgreSQL array functions on attribute_values."""
        with session_scope() as session:
            # Create test agent
            session.execute(text("""
                INSERT INTO agents (name, memory_size) 
                VALUES (:name, :memory_size)
            """), {
                'name': 'Array Functions Test Agent',
                'memory_size': 10
            })
            
            agent_result = session.execute(text("SELECT id FROM agents WHERE name = 'Array Functions Test Agent'"))
            agent_id = agent_result.fetchone()[0]
            
            # Insert attribute values
            session.execute(text("""
                INSERT INTO agent_attributes (agent_id, attribute_values) 
                VALUES (:agent_id, :attribute_values)
            """), {
                'agent_id': agent_id,
                'attribute_values': [3.14, 2.71, 1.41, 2.23]
            })
            
            # Test array_append
            session.execute(text("""
                UPDATE agent_attributes 
                SET attribute_values = array_append(attribute_values, 5.55)
                WHERE agent_id = :agent_id
            """), {'agent_id': agent_id})
            
            # Test array_prepend
            session.execute(text("""
                UPDATE agent_attributes 
                SET attribute_values = array_prepend(0.0, attribute_values)
                WHERE agent_id = :agent_id
            """), {'agent_id': agent_id})
            
            # Test array_remove
            session.execute(text("""
                UPDATE agent_attributes 
                SET attribute_values = array_remove(attribute_values, 2.71)
                WHERE agent_id = :agent_id
            """), {'agent_id': agent_id})
            
            # Verify final array
            result = session.execute(text("""
                SELECT attribute_values 
                FROM agent_attributes 
                WHERE agent_id = :agent_id
            """), {'agent_id': agent_id})
            
            final_array = result.fetchone()[0]
            assert len(final_array) == 5
            assert 0.0 in final_array  # Prepend worked
            assert 5.55 in final_array  # Append worked
            assert 2.71 not in final_array  # Remove worked
        
        # Clean up
        clear_all_rows()
    
    @pytest.mark.integration
    def test_agent_attributes_numpy_integration(self):
        """Test integration with numpy arrays for attribute_values."""
        with session_scope() as session:
            # Create test agent
            session.execute(text("""
                INSERT INTO agents (name, memory_size) 
                VALUES (:name, :memory_size)
            """), {
                'name': 'Numpy Test Agent',
                'memory_size': 6
            })
            
            agent_result = session.execute(text("SELECT id FROM agents WHERE name = 'Numpy Test Agent'"))
            agent_id = agent_result.fetchone()[0]
            
            # Create numpy arrays
            float_array = np.array([1.1, 2.2, 3.3, 4.4], dtype=np.float64)
            int_array = np.array([5.0, 6.0, 7.0, 8.0], dtype=np.float64)  # Convert to float64 to avoid mixed types
            
            # Insert numpy arrays
            session.execute(text("""
                INSERT INTO agent_attributes (agent_id, attribute_values) 
                VALUES (:agent_id, :attribute_values)
            """), {
                'agent_id': agent_id,
                'attribute_values': float_array.tolist()
            })
            
                        # Query and verify the inserted float array
            result = session.execute(text("""
                SELECT attribute_values
                FROM agent_attributes
                WHERE agent_id = :agent_id
            """), {'agent_id': agent_id})
            
            row = result.fetchone()
            assert row is not None
            
            # Should have the float array we inserted
            stored_array = row[0]
            assert len(stored_array) == 4
            assert stored_array == [1.1, 2.2, 3.3, 4.4]
            
            # Test updating to int array
            session.execute(text("""
                UPDATE agent_attributes 
                SET attribute_values = :new_values 
                WHERE agent_id = :agent_id
            """), {
                'agent_id': agent_id,
                'new_values': int_array.tolist()
            })
            
            # Verify the update
            result = session.execute(text("""
                SELECT attribute_values 
                FROM agent_attributes 
                WHERE agent_id = :agent_id
            """), {'agent_id': agent_id})
            
            updated_row = result.fetchone()
            assert updated_row is not None
            
            # Should now have the int array (cast to doubles by PostgreSQL)
            updated_array = updated_row[0]
            assert len(updated_array) == 4
            assert updated_array == [5.0, 6.0, 7.0, 8.0]
        
        # Clean up
        clear_all_rows()

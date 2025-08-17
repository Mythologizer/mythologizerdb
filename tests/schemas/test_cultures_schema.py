import os
import pytest
import numpy as np
from sqlalchemy import text

from mythologizer_postgres.db import (
    session_scope,
    get_table_row_counts,
    clear_all_rows,
)


def get_embedding_dim():
    """Get embedding dimension from environment variable."""
    return int(os.getenv('EMBEDDING_DIM', '4'))


class TestCulturesSchema:
    """Test the cultures table schema and operations."""
    
    @pytest.mark.integration
    def test_cultures_table_structure(self):
        """Test that cultures table has the correct structure."""
        with session_scope() as session:
            # Check cultures table structure
            result = session.execute(text("""
                SELECT column_name, data_type 
                FROM information_schema.columns 
                WHERE table_name = 'cultures' 
                ORDER BY ordinal_position
            """))
            cultures_columns = {row[0]: row[1] for row in result.fetchall()}
            
            expected_cultures_columns = {
                'id': 'integer',
                'name': 'text',
                'description': 'text'
            }
            
            for col, expected_type in expected_cultures_columns.items():
                assert col in cultures_columns, f"Column {col} should exist in cultures table"
                assert cultures_columns[col] == expected_type, f"Column {col} should be {expected_type}"
    
    @pytest.mark.integration
    def test_cultures_basic_operations(self):
        """Test basic CRUD operations on cultures table."""
        with session_scope() as session:
            # Insert cultures
            session.execute(text("""
                INSERT INTO cultures (name, description) 
                VALUES (:name, :description)
            """), {
                'name': 'Greek Mythology',
                'description': 'Ancient Greek myths and legends'
            })
            
            session.execute(text("""
                INSERT INTO cultures (name, description) 
                VALUES (:name, :description)
            """), {
                'name': 'Norse Mythology',
                'description': 'Norse and Viking myths and legends'
            })
            
            # Query cultures
            result = session.execute(text("SELECT id, name, description FROM cultures ORDER BY name"))
            cultures = result.fetchall()
            
            assert len(cultures) == 2, "Should have 2 cultures"
            assert cultures[0][1] == 'Greek Mythology', "First culture should be Greek Mythology"
            assert cultures[1][1] == 'Norse Mythology', "Second culture should be Norse Mythology"
            
            # Update culture
            session.execute(text("""
                UPDATE cultures 
                SET description = :new_description 
                WHERE name = :name
            """), {
                'new_description': 'Updated Greek mythology description',
                'name': 'Greek Mythology'
            })
            
            # Verify update
            result = session.execute(text("""
                SELECT description FROM cultures WHERE name = 'Greek Mythology'
            """))
            updated_description = result.fetchone()[0]
            assert 'Updated' in updated_description, "Description should be updated"
        
        # Clean up
        clear_all_rows()
    
    @pytest.mark.integration
    def test_cultures_constraints(self):
        """Test that cultures table constraints work correctly."""
        with session_scope() as session:
            # Test valid insertions
            session.execute(text("""
                INSERT INTO cultures (name, description) 
                VALUES (:name, :description)
            """), {
                'name': 'Test Culture 1',
                'description': 'Test description 1'
            })
            
            session.execute(text("""
                INSERT INTO cultures (name, description) 
                VALUES (:name, :description)
            """), {
                'name': 'Test Culture 2',
                'description': 'Test description 2'
            })
            
            # Test NOT NULL constraints
            with pytest.raises(Exception):
                session.execute(text("""
                    INSERT INTO cultures (name, description) 
                    VALUES (:name, :description)
                """), {
                    'name': None,  # Should fail
                    'description': 'Valid description'
                })
            
            with pytest.raises(Exception):
                session.execute(text("""
                    INSERT INTO cultures (name, description) 
                    VALUES (:name, :description)
                """), {
                    'name': 'Valid name',
                    'description': None  # Should fail
                })
        
        # Clean up
        clear_all_rows()


class TestAgentCulturesSchema:
    """Test the agent_cultures junction table schema and operations."""
    
    @pytest.mark.integration
    def test_agent_cultures_table_structure(self):
        """Test that agent_cultures table has the correct structure."""
        with session_scope() as session:
            # Check agent_cultures table structure
            result = session.execute(text("""
                SELECT column_name, data_type 
                FROM information_schema.columns 
                WHERE table_name = 'agent_cultures' 
                ORDER BY ordinal_position
            """))
            agent_cultures_columns = {row[0]: row[1] for row in result.fetchall()}
            
            expected_agent_cultures_columns = {
                'agent_id': 'integer',
                'culture_id': 'integer'
            }
            
            for col, expected_type in expected_agent_cultures_columns.items():
                assert col in agent_cultures_columns, f"Column {col} should exist in agent_cultures table"
                assert agent_cultures_columns[col] == expected_type, f"Column {col} should be {expected_type}"
    
    @pytest.mark.integration
    def test_agent_cultures_basic_operations(self):
        """Test basic operations on agent_cultures table."""
        with session_scope() as session:
            # Create test data
            embedding_dim = get_embedding_dim()
            embedding = np.random.rand(embedding_dim).tolist()
            
            # Insert a myth
            session.execute(text("""
                INSERT INTO myths (embedding, embedding_ids, offsets, weights) 
                VALUES (:embedding, ARRAY[1], ARRAY[]::vector[], ARRAY[]::double precision[])
            """), {
                'embedding': embedding
            })
            
            # Insert an agent
            session.execute(text("""
                INSERT INTO agents (name, memory_size) 
                VALUES (:name, :memory_size)
            """), {
                'name': 'Test Agent',
                'memory_size': 5
            })
            
            # Insert cultures
            session.execute(text("""
                INSERT INTO cultures (name, description) 
                VALUES (:name, :description)
            """), {
                'name': 'Greek Mythology',
                'description': 'Ancient Greek myths and legends'
            })
            
            session.execute(text("""
                INSERT INTO cultures (name, description) 
                VALUES (:name, :description)
            """), {
                'name': 'Norse Mythology',
                'description': 'Norse and Viking myths and legends'
            })
            
            # Get the IDs
            myth_result = session.execute(text("SELECT id FROM myths LIMIT 1"))
            myth_id = myth_result.fetchone()[0]
            
            agent_result = session.execute(text("SELECT id FROM agents LIMIT 1"))
            agent_id = agent_result.fetchone()[0]
            
            culture_result = session.execute(text("SELECT id FROM cultures ORDER BY name"))
            culture_ids = [row[0] for row in culture_result.fetchall()]
            
            # Insert agent_cultures records
            session.execute(text("""
                INSERT INTO agent_cultures (agent_id, culture_id) 
                VALUES (:agent_id, :culture_id)
            """), {
                'agent_id': agent_id,
                'culture_id': culture_ids[0]  # Greek Mythology
            })
            
            session.execute(text("""
                INSERT INTO agent_cultures (agent_id, culture_id) 
                VALUES (:agent_id, :culture_id)
            """), {
                'agent_id': agent_id,
                'culture_id': culture_ids[1]  # Norse Mythology
            })
            
            # Verify records exist
            result = session.execute(text("""
                SELECT COUNT(*) FROM agent_cultures WHERE agent_id = :agent_id
            """), {
                'agent_id': agent_id
            })
            count = result.fetchone()[0]
            assert count == 2, "Agent should be associated with 2 cultures"
        
        # Clean up
        clear_all_rows()
    
    @pytest.mark.integration
    def test_agent_cultures_foreign_key_constraints(self):
        """Test that agent_cultures foreign key constraints work correctly."""
        with session_scope() as session:
            # Try to insert with non-existent agent_id
            with pytest.raises(Exception):
                session.execute(text("""
                    INSERT INTO agent_cultures (agent_id, culture_id) 
                    VALUES (:agent_id, :culture_id)
                """), {
                    'agent_id': 999,  # Non-existent agent
                    'culture_id': 1
                })
            
            # Try to insert with non-existent culture_id
            with pytest.raises(Exception):
                session.execute(text("""
                    INSERT INTO agent_cultures (agent_id, culture_id) 
                    VALUES (:agent_id, :culture_id)
                """), {
                    'agent_id': 1,
                    'culture_id': 999  # Non-existent culture
                })
    
    @pytest.mark.integration
    def test_agent_cultures_primary_key_constraint(self):
        """Test that agent_cultures primary key constraint works correctly."""
        with session_scope() as session:
            # Create test data
            embedding_dim = get_embedding_dim()
            embedding = np.random.rand(embedding_dim).tolist()
            
            # Insert a myth
            session.execute(text("""
                INSERT INTO myths (embedding, embedding_ids, offsets, weights) 
                VALUES (:embedding, ARRAY[1], ARRAY[]::vector[], ARRAY[]::double precision[])
            """), {
                'embedding': embedding
            })
            
            # Insert an agent
            session.execute(text("""
                INSERT INTO agents (name, memory_size) 
                VALUES (:name, :memory_size)
            """), {
                'name': 'Test Agent',
                'memory_size': 5
            })
            
            # Insert a culture
            session.execute(text("""
                INSERT INTO cultures (name, description) 
                VALUES (:name, :description)
            """), {
                'name': 'Test Culture',
                'description': 'Test description'
            })
            
            # Get the IDs
            agent_result = session.execute(text("SELECT id FROM agents LIMIT 1"))
            agent_id = agent_result.fetchone()[0]
            
            culture_result = session.execute(text("SELECT id FROM cultures LIMIT 1"))
            culture_id = culture_result.fetchone()[0]
            
            # Insert first record
            session.execute(text("""
                INSERT INTO agent_cultures (agent_id, culture_id) 
                VALUES (:agent_id, :culture_id)
            """), {
                'agent_id': agent_id,
                'culture_id': culture_id
            })
            
            # Test primary key constraint violation (duplicate)
            with pytest.raises(Exception):
                session.execute(text("""
                    INSERT INTO agent_cultures (agent_id, culture_id) 
                    VALUES (:agent_id, :culture_id)
                """), {
                    'agent_id': agent_id,  # Same combination - should fail
                    'culture_id': culture_id
                })
        
        # Clean up
        clear_all_rows()
    
    @pytest.mark.integration
    def test_agent_cultures_cascade_delete(self):
        """Test that agent_cultures records are deleted when parent records are deleted."""
        with session_scope() as session:
            # Create test data
            embedding_dim = get_embedding_dim()
            embedding = np.random.rand(embedding_dim).tolist()
            
            # Insert a myth
            session.execute(text("""
                INSERT INTO myths (embedding, embedding_ids, offsets, weights) 
                VALUES (:embedding, ARRAY[1], ARRAY[]::vector[], ARRAY[]::double precision[])
            """), {
                'embedding': embedding
            })
            
            # Insert an agent
            session.execute(text("""
                INSERT INTO agents (name, memory_size) 
                VALUES (:name, :memory_size)
            """), {
                'name': 'Test Agent',
                'memory_size': 5
            })
            
            # Insert a culture
            session.execute(text("""
                INSERT INTO cultures (name, description) 
                VALUES (:name, :description)
            """), {
                'name': 'Test Culture',
                'description': 'Test description'
            })
            
            # Get the IDs
            agent_result = session.execute(text("SELECT id FROM agents LIMIT 1"))
            agent_id = agent_result.fetchone()[0]
            
            culture_result = session.execute(text("SELECT id FROM cultures LIMIT 1"))
            culture_id = culture_result.fetchone()[0]
            
            # Insert agent_cultures record
            session.execute(text("""
                INSERT INTO agent_cultures (agent_id, culture_id) 
                VALUES (:agent_id, :culture_id)
            """), {
                'agent_id': agent_id,
                'culture_id': culture_id
            })
            
            # Verify record exists
            result = session.execute(text("SELECT COUNT(*) FROM agent_cultures"))
            count_before = result.fetchone()[0]
            assert count_before == 1, "Should have 1 agent_cultures record"
            
            # Delete the agent
            session.execute(text("DELETE FROM agents WHERE id = :agent_id"), {
                'agent_id': agent_id
            })
            
            # Verify agent_cultures record was deleted
            result = session.execute(text("SELECT COUNT(*) FROM agent_cultures"))
            count_after = result.fetchone()[0]
            assert count_after == 0, "agent_cultures record should be deleted when agent is deleted"
            
            # Re-insert for culture deletion test
            session.execute(text("""
                INSERT INTO agents (name, memory_size) 
                VALUES (:name, :memory_size)
            """), {
                'name': 'Test Agent 2',
                'memory_size': 10
            })
            
            agent_result = session.execute(text("SELECT id FROM agents LIMIT 1"))
            agent_id = agent_result.fetchone()[0]
            
            session.execute(text("""
                INSERT INTO agent_cultures (agent_id, culture_id) 
                VALUES (:agent_id, :culture_id)
            """), {
                'agent_id': agent_id,
                'culture_id': culture_id
            })
            
            # Delete the culture
            session.execute(text("DELETE FROM cultures WHERE id = :culture_id"), {
                'culture_id': culture_id
            })
            
            # Verify agent_cultures record was deleted
            result = session.execute(text("SELECT COUNT(*) FROM agent_cultures"))
            count_after_culture_delete = result.fetchone()[0]
            assert count_after_culture_delete == 0, "agent_cultures record should be deleted when culture is deleted"
        
        # Clean up
        clear_all_rows()

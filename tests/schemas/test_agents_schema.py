import os
import pytest
import numpy as np
from sqlalchemy import text

from mythologizer_postgres.db import (
    session_scope,
    get_table_row_counts,
    clear_all_rows,
)
from mythologizer_postgres.connectors import insert_agent_myth_safe_with_session


def get_embedding_dim():
    """Get embedding dimension from environment variable."""
    return int(os.getenv('EMBEDDING_DIM', '4'))


class TestAgentsSchema:
    """Test the agents table schema and operations."""
    
    @pytest.mark.integration
    def test_agents_table_structure(self):
        """Test that agents table has the correct structure."""
        with session_scope() as session:
            # Check agents table structure
            result = session.execute(text("""
                SELECT column_name, data_type 
                FROM information_schema.columns 
                WHERE table_name = 'agents' 
                ORDER BY ordinal_position
            """))
            agents_columns = {row[0]: row[1] for row in result.fetchall()}
            
            expected_agents_columns = {
                'id': 'integer',
                'name': 'text',
                'memory_size': 'integer'
            }
            
            for col, expected_type in expected_agents_columns.items():
                assert col in agents_columns, f"Column {col} should exist in agents table"
                assert agents_columns[col] == expected_type, f"Column {col} should be {expected_type}"
    
    @pytest.mark.integration
    def test_agents_table_constraints(self):
        """Test that agents table constraints work correctly."""
        with session_scope() as session:
            # Test valid insertions
            session.execute(text("""
                INSERT INTO agents (name, memory_size) 
                VALUES (:name, :memory_size)
            """), {
                'name': 'Test Agent 1',
                'memory_size': 10
            })
            
            session.execute(text("""
                INSERT INTO agents (name, memory_size) 
                VALUES (:name, :memory_size)
            """), {
                'name': 'Test Agent 2',
                'memory_size': 5
            })
            
            # Test memory_size constraint (must be >= 3)
            with pytest.raises(Exception):
                session.execute(text("""
                    INSERT INTO agents (name, memory_size) 
                    VALUES (:name, :memory_size)
                """), {
                    'name': 'Invalid Memory Agent',
                    'memory_size': 2  # Should fail
                })
        
        # Clean up
        clear_all_rows()
    
    @pytest.mark.integration
    def test_agents_basic_operations(self):
        """Test basic CRUD operations on agents table."""
        with session_scope() as session:
            # Insert agents
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
            
            # Query agents
            result = session.execute(text("SELECT id, name, memory_size FROM agents ORDER BY name"))
            agents = result.fetchall()
            
            assert len(agents) == 2, "Should have 2 agents"
            assert agents[0][1] == 'Agent Alpha', "First agent should be Agent Alpha"
            assert agents[1][1] == 'Agent Beta', "Second agent should be Agent Beta"
            
            # Update agent
            session.execute(text("""
                UPDATE agents 
                SET memory_size = :new_memory_size 
                WHERE name = :name
            """), {
                'new_memory_size': 20,
                'name': 'Agent Alpha'
            })
            
            # Verify update
            result = session.execute(text("""
                SELECT memory_size FROM agents WHERE name = 'Agent Alpha'
            """))
            updated_memory = result.fetchone()[0]
            assert updated_memory == 20, "Memory size should be updated to 20"
        
        # Clean up
        clear_all_rows()


class TestAgentMythsSchema:
    """Test the agent_myths junction table schema and operations."""
    
    @pytest.mark.integration
    def test_agent_myths_table_structure(self):
        """Test that agent_myths table has the correct structure."""
        with session_scope() as session:
            # Check agent_myths table structure
            result = session.execute(text("""
                SELECT column_name, data_type 
                FROM information_schema.columns 
                WHERE table_name = 'agent_myths' 
                ORDER BY ordinal_position
            """))
            agent_myths_columns = {row[0]: row[1] for row in result.fetchall()}
            
            expected_agent_myths_columns = {
                'myth_id': 'integer',
                'agent_id': 'integer',
                'position': 'integer',
                'retention': 'double precision'
            }
            
            for col, expected_type in expected_agent_myths_columns.items():
                assert col in agent_myths_columns, f"Column {col} should exist in agent_myths table"
                assert agent_myths_columns[col] == expected_type, f"Column {col} should be {expected_type}"
    
    @pytest.mark.integration
    def test_agent_myths_constraints(self):
        """Test that agent_myths table constraints work correctly."""
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
            
            # Get the IDs
            myth_result = session.execute(text("SELECT id FROM myths LIMIT 1"))
            myth_id = myth_result.fetchone()[0]
            
            agent_result = session.execute(text("SELECT id FROM agents LIMIT 1"))
            agent_id = agent_result.fetchone()[0]
            
            # Test valid insertion - use safe function instead of trigger
            success = insert_agent_myth_safe_with_session(
                session=session,
                myth_id=myth_id,
                agent_id=agent_id,
                retention=0.8
            )
            assert success, f"Failed to insert myth {myth_id} into agent {agent_id}"
            
            # Verify the trigger worked
            result = session.execute(text("""
                SELECT position FROM agent_myths WHERE myth_id = :myth_id
            """), {
                'myth_id': myth_id
            })
            position = result.fetchone()[0]
            assert position == 0, "Trigger should assign position 0 for first myth"
        
        # Clean up
        clear_all_rows()
    
    @pytest.mark.integration
    def test_agent_myths_unique_constraint(self):
        """Test that agent_myths unique constraint works correctly."""
        with session_scope() as session:
            # Create test data
            embedding_dim = get_embedding_dim()
            embedding1 = np.random.rand(embedding_dim).tolist()
            embedding2 = np.random.rand(embedding_dim).tolist()
            
            # Insert two myths
            session.execute(text("""
                INSERT INTO myths (embedding, embedding_ids, offsets, weights) 
                VALUES (:embedding, ARRAY[1], ARRAY[]::vector[], ARRAY[]::double precision[])
            """), {
                'embedding': embedding1
            })
            
            session.execute(text("""
                INSERT INTO myths (embedding, embedding_ids, offsets, weights) 
                VALUES (:embedding, ARRAY[2], ARRAY[]::vector[], ARRAY[]::double precision[])
            """), {
                'embedding': embedding2
            })
            
            # Insert an agent
            session.execute(text("""
                INSERT INTO agents (name, memory_size) 
                VALUES (:name, :memory_size)
            """), {
                'name': 'Test Agent',
                'memory_size': 5
            })
            
            # Get the IDs
            myth_result = session.execute(text("SELECT id FROM myths ORDER BY id"))
            myth_ids = [row[0] for row in myth_result.fetchall()]
            
            agent_result = session.execute(text("SELECT id FROM agents LIMIT 1"))
            agent_id = agent_result.fetchone()[0]
            
            # Insert first record
            success = insert_agent_myth_safe_with_session(
                session=session,
                myth_id=myth_ids[0],
                agent_id=agent_id,
                retention=0.8
            )
            assert success, f"Failed to insert myth {myth_ids[0]} into agent {agent_id}"
            
            # Test unique constraint violation (same myth_id - myth can only belong to one agent)
            success = insert_agent_myth_safe_with_session(
                session=session,
                myth_id=myth_ids[0],  # Same myth - should fail
                agent_id=agent_id,
                retention=0.9
            )
            assert not success, "Should fail when inserting same myth twice"
        
        # Clean up
        clear_all_rows()
    
    @pytest.mark.integration
    def test_agent_myths_foreign_key_constraints(self):
        """Test that agent_myths foreign key constraints work correctly."""
        with session_scope() as session:
            # Try to insert with non-existent myth_id
            with pytest.raises(Exception):
                session.execute(text("""
                    INSERT INTO agent_myths (myth_id, agent_id, position, retention) 
                    VALUES (:myth_id, :agent_id, :position, :retention)
                """), {
                    'myth_id': 999,  # Non-existent myth
                    'agent_id': 1,
                    'position': 1,
                    'retention': 0.8
                })
            
            # Try to insert with non-existent agent_id
            with pytest.raises(Exception):
                session.execute(text("""
                    INSERT INTO agent_myths (myth_id, agent_id, position, retention) 
                    VALUES (:myth_id, :agent_id, :position, :retention)
                """), {
                    'myth_id': 1,
                    'agent_id': 999,  # Non-existent agent
                    'position': 1,
                    'retention': 0.8
                })
    
    @pytest.mark.integration
    def test_agent_myths_cascade_delete(self):
        """Test that agent_myths records are deleted when parent records are deleted."""
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
            
            # Get the IDs
            myth_result = session.execute(text("SELECT id FROM myths LIMIT 1"))
            myth_id = myth_result.fetchone()[0]
            
            agent_result = session.execute(text("SELECT id FROM agents LIMIT 1"))
            agent_id = agent_result.fetchone()[0]
            
            # Insert agent_myths record - let trigger handle position
            session.execute(text("""
                INSERT INTO agent_myths (myth_id, agent_id, position, retention) 
                VALUES (:myth_id, :agent_id, :position, :retention)
            """), {
                'myth_id': myth_id,
                'agent_id': agent_id,
                'position': 1,  # Will be overridden by trigger
                'retention': 0.8
            })
            
            # Verify record exists
            result = session.execute(text("SELECT COUNT(*) FROM agent_myths"))
            count_before = result.fetchone()[0]
            assert count_before == 1, "Should have 1 agent_myths record"
            
            # Delete the myth
            session.execute(text("DELETE FROM myths WHERE id = :myth_id"), {
                'myth_id': myth_id
            })
            
            # Verify agent_myths record was deleted
            result = session.execute(text("SELECT COUNT(*) FROM agent_myths"))
            count_after = result.fetchone()[0]
            assert count_after == 0, "agent_myths record should be deleted when myth is deleted"
        
        # Clean up
        clear_all_rows()

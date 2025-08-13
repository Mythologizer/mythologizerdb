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


class TestAgentMythsComprehensive:
    """Comprehensive tests for agent_myths table with stack-like behavior and triggers."""
    
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
    def test_push_agent_myth_basic_insertion(self):
        """Test basic insertion with push semantics - position should be auto-assigned."""
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
            
            # Insert an agent with memory_size = 5
            session.execute(text("""
                INSERT INTO agents (name, age, memory_size) 
                VALUES (:name, :age, :memory_size)
            """), {
                'name': 'Test Agent',
                'age': 25,
                'memory_size': 5
            })
            
            # Get the IDs
            myth_result = session.execute(text("SELECT id FROM myths LIMIT 1"))
            myth_id = myth_result.fetchone()[0]
            
            agent_result = session.execute(text("SELECT id FROM agents LIMIT 1"))
            agent_id = agent_result.fetchone()[0]
            
            # Insert agent_myth record - trigger should assign position = 1
            session.execute(text("""
                INSERT INTO agent_myths (myth_id, agent_id, position, retention) 
                VALUES (:myth_id, :agent_id, :position, :retention)
            """), {
                'myth_id': myth_id,
                'agent_id': agent_id,
                'position': 999,  # This should be overridden by trigger
                'retention': 0.8
            })
            
            # Verify the trigger worked
            result = session.execute(text("""
                SELECT position FROM agent_myths WHERE myth_id = :myth_id
            """), {
                'myth_id': myth_id
            })
            position = result.fetchone()[0]
            assert position == 1, "First myth should get position 1 (bottom of stack)"
        
        # Clean up
        clear_all_rows()
    
    @pytest.mark.integration
    def test_push_agent_myth_stack_behavior(self):
        """Test stack behavior: new myths go to top, eviction from bottom when full."""
        with session_scope() as session:
            # Create test data
            embedding_dim = get_embedding_dim()
            
            # Insert an agent with memory_size = 3
            session.execute(text("""
                INSERT INTO agents (name, age, memory_size) 
                VALUES (:name, :age, :memory_size)
            """), {
                'name': 'Test Agent',
                'age': 25,
                'memory_size': 3
            })
            
            agent_result = session.execute(text("SELECT id FROM agents LIMIT 1"))
            agent_id = agent_result.fetchone()[0]
            
            # Insert 3 myths
            myth_ids = []
            for i in range(3):
                embedding = np.random.rand(embedding_dim).tolist()
                session.execute(text("""
                    INSERT INTO myths (embedding, embedding_ids, offsets, weights) 
                    VALUES (:embedding, ARRAY[:id], ARRAY[]::vector[], ARRAY[]::double precision[])
                """), {
                    'embedding': embedding,
                    'id': i + 1
                })
                
                # Get the myth ID using a different approach
                myth_result = session.execute(text("SELECT id FROM myths ORDER BY id DESC LIMIT 1"))
                myth_id = myth_result.fetchone()[0]
                myth_ids.append(myth_id)
                
                # Insert agent_myth record - trigger will assign position
                session.execute(text("""
                    INSERT INTO agent_myths (myth_id, agent_id, position, retention) 
                    VALUES (:myth_id, :agent_id, :position, :retention)
                """), {
                    'myth_id': myth_id,
                    'agent_id': agent_id,
                    'position': 999,  # Will be overridden by trigger
                    'retention': 0.8
                })
            
            # Verify positions: should be 1, 2, 3 (bottom to top)
            result = session.execute(text("""
                SELECT myth_id, position FROM agent_myths 
                WHERE agent_id = :agent_id 
                ORDER BY position
            """), {
                'agent_id': agent_id
            })
            positions = result.fetchall()
            
            assert len(positions) == 3, "Should have 3 myths"
            assert positions[0][1] == 1, "First myth should be at position 1 (bottom)"
            assert positions[1][1] == 2, "Second myth should be at position 2"
            assert positions[2][1] == 3, "Third myth should be at position 3 (top)"
            
            # Now insert a 4th myth - should evict the bottom one (position 1)
            embedding = np.random.rand(embedding_dim).tolist()
            session.execute(text("""
                INSERT INTO myths (embedding, embedding_ids, offsets, weights) 
                VALUES (:embedding, ARRAY[4], ARRAY[]::vector[], ARRAY[]::double precision[])
            """), {
                'embedding': embedding
            })
            
            myth_result = session.execute(text("SELECT id FROM myths ORDER BY id DESC LIMIT 1"))
            new_myth_id = myth_result.fetchone()[0]
            
            session.execute(text("""
                INSERT INTO agent_myths (myth_id, agent_id, position, retention) 
                VALUES (:myth_id, :agent_id, :position, :retention)
            """), {
                'myth_id': new_myth_id,
                'agent_id': agent_id,
                'position': 999,  # Should be overridden
                'retention': 0.8
            })
            
            # Verify the stack behavior: old position 1 should be gone, others shifted down
            result = session.execute(text("""
                SELECT myth_id, position FROM agent_myths 
                WHERE agent_id = :agent_id 
                ORDER BY position
            """), {
                'agent_id': agent_id
            })
            new_positions = result.fetchall()
            
            assert len(new_positions) == 3, "Should still have 3 myths (memory_size)"
            
            # Check that the old bottom myth (myth_ids[0]) is gone
            old_bottom_myth_id = myth_ids[0]
            result = session.execute(text("""
                SELECT COUNT(*) FROM agent_myths 
                WHERE agent_id = :agent_id AND myth_id = :myth_id
            """), {
                'agent_id': agent_id,
                'myth_id': old_bottom_myth_id
            })
            count = result.fetchone()[0]
            assert count == 0, "Old bottom myth should be evicted"
            
            # Check that new myth is at top (position 3)
            result = session.execute(text("""
                SELECT position FROM agent_myths 
                WHERE agent_id = :agent_id AND myth_id = :myth_id
            """), {
                'agent_id': agent_id,
                'myth_id': new_myth_id
            })
            new_myth_position = result.fetchone()[0]
            assert new_myth_position == 3, "New myth should be at top position"
        
        # Clean up
        clear_all_rows()
    
    @pytest.mark.integration
    def test_myth_unique_constraint(self):
        """Test that a myth can only belong to one agent globally."""
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
            
            # Insert two agents
            session.execute(text("""
                INSERT INTO agents (name, age, memory_size) 
                VALUES (:name, :age, :memory_size)
            """), {
                'name': 'Agent 1',
                'age': 25,
                'memory_size': 5
            })
            
            session.execute(text("""
                INSERT INTO agents (name, age, memory_size) 
                VALUES (:name, :age, :memory_size)
            """), {
                'name': 'Agent 2',
                'age': 30,
                'memory_size': 5
            })
            
            # Get the IDs
            myth_result = session.execute(text("SELECT id FROM myths LIMIT 1"))
            myth_id = myth_result.fetchone()[0]
            
            agent_result = session.execute(text("SELECT id FROM agents ORDER BY name"))
            agent_ids = [row[0] for row in agent_result.fetchall()]
            
            # Insert myth into first agent
            session.execute(text("""
                INSERT INTO agent_myths (myth_id, agent_id, position, retention) 
                VALUES (:myth_id, :agent_id, :position, :retention)
            """), {
                'myth_id': myth_id,
                'agent_id': agent_ids[0],
                'position': 1,
                'retention': 0.8
            })
            
            # Try to insert same myth into second agent - should fail
            with pytest.raises(Exception):
                session.execute(text("""
                    INSERT INTO agent_myths (myth_id, agent_id, position, retention) 
                    VALUES (:myth_id, :agent_id, :position, :retention)
                """), {
                    'myth_id': myth_id,
                    'agent_id': agent_ids[1],
                    'position': 1,
                    'retention': 0.8
                })
        
        # Clean up
        clear_all_rows()
    

    
    @pytest.mark.integration
    def test_trim_agent_memory_trigger(self):
        """Test the trim_agent_memory trigger when agent memory_size is reduced."""
        with session_scope() as session:
            # Create test data
            embedding_dim = get_embedding_dim()
            
            # Insert an agent with memory_size = 5
            session.execute(text("""
                INSERT INTO agents (name, age, memory_size) 
                VALUES (:name, :age, :memory_size)
            """), {
                'name': 'Test Agent',
                'age': 25,
                'memory_size': 5
            })
            
            agent_result = session.execute(text("SELECT id FROM agents LIMIT 1"))
            agent_id = agent_result.fetchone()[0]
            
            # Insert 5 myths
            myth_ids = []
            for i in range(5):
                embedding = np.random.rand(embedding_dim).tolist()
                session.execute(text("""
                    INSERT INTO myths (embedding, embedding_ids, offsets, weights) 
                    VALUES (:embedding, ARRAY[:id], ARRAY[]::vector[], ARRAY[]::double precision[])
                """), {
                    'embedding': embedding,
                    'id': i + 1
                })
                
                # Get the myth ID using a different approach
                myth_result = session.execute(text("SELECT id FROM myths ORDER BY id DESC LIMIT 1"))
                myth_id = myth_result.fetchone()[0]
                myth_ids.append(myth_id)
                
                # Insert agent_myth record - trigger will assign position
                session.execute(text("""
                    INSERT INTO agent_myths (myth_id, agent_id, position, retention) 
                    VALUES (:myth_id, :agent_id, :position, :retention)
                """), {
                    'myth_id': myth_id,
                    'agent_id': agent_id,
                    'position': 999,  # Will be overridden by trigger
                    'retention': 0.8
                })
            
            # Verify we have 5 myths
            result = session.execute(text("""
                SELECT COUNT(*) FROM agent_myths WHERE agent_id = :agent_id
            """), {
                'agent_id': agent_id
            })
            count_before = result.fetchone()[0]
            assert count_before == 5, "Should have 5 myths initially"
            
            # Reduce memory_size to 3 - should keep top 3, delete bottom 2
            session.execute(text("""
                UPDATE agents SET memory_size = :new_size WHERE id = :agent_id
            """), {
                'new_size': 3,
                'agent_id': agent_id
            })
            
            # Verify only 3 myths remain
            result = session.execute(text("""
                SELECT COUNT(*) FROM agent_myths WHERE agent_id = :agent_id
            """), {
                'agent_id': agent_id
            })
            count_after = result.fetchone()[0]
            assert count_after == 3, "Should have 3 myths after reducing memory_size"
            
            # Verify positions are still contiguous starting from 1
            result = session.execute(text("""
                SELECT position FROM agent_myths 
                WHERE agent_id = :agent_id 
                ORDER BY position
            """), {
                'agent_id': agent_id
            })
            positions = [row[0] for row in result.fetchall()]
            assert positions == [1, 2, 3], "Positions should be contiguous 1, 2, 3"
            
            # Test reducing to minimum allowed value (3) - should keep top 3
            session.execute(text("""
                UPDATE agents SET memory_size = :new_size WHERE id = :agent_id
            """), {
                'new_size': 3,
                'agent_id': agent_id
            })
            
            result = session.execute(text("""
                SELECT COUNT(*) FROM agent_myths WHERE agent_id = :agent_id
            """), {
                'agent_id': agent_id
            })
            count_min = result.fetchone()[0]
            assert count_min == 3, "Should have 3 myths when memory_size = 3"
        
        # Clean up
        clear_all_rows()
    
    @pytest.mark.integration
    def test_agent_myths_constraints(self):
        """Test various constraints on agent_myths table."""
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
                INSERT INTO agents (name, age, memory_size) 
                VALUES (:name, :age, :memory_size)
            """), {
                'name': 'Test Agent',
                'age': 25,
                'memory_size': 5
            })
            
            # Get the IDs
            myth_result = session.execute(text("SELECT id FROM myths LIMIT 1"))
            myth_id = myth_result.fetchone()[0]
            
            agent_result = session.execute(text("SELECT id FROM agents LIMIT 1"))
            agent_id = agent_result.fetchone()[0]
            
            # Test retention constraint (must be > 0) - position will be overridden by trigger
            with pytest.raises(Exception):
                session.execute(text("""
                    INSERT INTO agent_myths (myth_id, agent_id, position, retention) 
                    VALUES (:myth_id, :agent_id, :position, :retention)
                """), {
                    'myth_id': myth_id,
                    'agent_id': agent_id,
                    'position': 1,
                    'retention': 0.0  # Should fail
                })
        
        # Clean up
        clear_all_rows()

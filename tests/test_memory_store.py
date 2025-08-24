import pytest
import numpy as np
from sqlalchemy import text
from mythologizer_postgres.connectors import get_myth_ids_and_retention_from_agents_memory
from mythologizer_postgres.db import get_engine, clear_all_rows


class TestMemoryStore:
    """Test the memory store functions."""
    
    def setup_method(self):
        """Clear all rows before each test."""
        clear_all_rows()
    
    def teardown_method(self):
        """Clear all rows after each test."""
        clear_all_rows()
    
    @pytest.mark.integration
    def test_get_myth_ids_and_retention_from_agents_memory_empty(self):
        """Test getting memory from an agent with no myths."""
        # Insert an agent
        engine = get_engine()
        with engine.connect() as conn:
            conn.execute(text("INSERT INTO agents (name, memory_size) VALUES ('Test Agent', 5)"))
            conn.commit()
            
            # Get agent ID
            result = conn.execute(text("SELECT id FROM agents"))
            agent_id = result.fetchone()[0]
        
        # Get memory - should return empty lists
        myth_ids, retentions = get_myth_ids_and_retention_from_agents_memory(agent_id)
        
        assert myth_ids == [], "Should return empty list for myth IDs"
        assert retentions == [], "Should return empty list for retentions"
    
    @pytest.mark.integration
    def test_get_myth_ids_and_retention_from_agents_memory_basic(self):
        """Test getting memory from an agent with myths."""
        # Insert an agent
        engine = get_engine()
        with engine.connect() as conn:
            conn.execute(text("INSERT INTO agents (name, memory_size) VALUES ('Test Agent', 5)"))
            conn.commit()
            
            # Get agent ID
            result = conn.execute(text("SELECT id FROM agents"))
            agent_id = result.fetchone()[0]
            
            # Insert myths
            embedding_dim = 4
            embedding1 = np.random.rand(embedding_dim).tolist()
            embedding2 = np.random.rand(embedding_dim).tolist()
            embedding3 = np.random.rand(embedding_dim).tolist()
            
            conn.execute(text("""
                INSERT INTO myths (embedding, embedding_ids, offsets, weights) 
                VALUES (:embedding, ARRAY[1], ARRAY[]::vector[], ARRAY[]::double precision[])
            """), {"embedding": embedding1})
            
            conn.execute(text("""
                INSERT INTO myths (embedding, embedding_ids, offsets, weights) 
                VALUES (:embedding, ARRAY[2], ARRAY[]::vector[], ARRAY[]::double precision[])
            """), {"embedding": embedding2})
            
            conn.execute(text("""
                INSERT INTO myths (embedding, embedding_ids, offsets, weights) 
                VALUES (:embedding, ARRAY[3], ARRAY[]::vector[], ARRAY[]::double precision[])
            """), {"embedding": embedding3})
            
            conn.commit()
            
            # Get myth IDs
            result = conn.execute(text("SELECT id FROM myths ORDER BY id"))
            myth_ids = [row[0] for row in result.fetchall()]
            
            # Insert agent myths (let triggers handle position assignment)
            conn.execute(text("""
                INSERT INTO agent_myths (myth_id, agent_id, position, retention) 
                VALUES (:myth_id, :agent_id, :position, :retention)
            """), {"myth_id": myth_ids[0], "agent_id": agent_id, "position": 1, "retention": 0.8})
            
            conn.execute(text("""
                INSERT INTO agent_myths (myth_id, agent_id, position, retention) 
                VALUES (:myth_id, :agent_id, :position, :retention)
            """), {"myth_id": myth_ids[1], "agent_id": agent_id, "position": 1, "retention": 0.9})
            
            conn.execute(text("""
                INSERT INTO agent_myths (myth_id, agent_id, position, retention) 
                VALUES (:myth_id, :agent_id, :position, :retention)
            """), {"myth_id": myth_ids[2], "agent_id": agent_id, "position": 1, "retention": 0.7})
            
            conn.commit()
        
        # Get memory
        retrieved_myth_ids, retrieved_retentions = get_myth_ids_and_retention_from_agents_memory(agent_id)
        
        # Should have 3 myths
        assert len(retrieved_myth_ids) == 3, "Should return 3 myth IDs"
        assert len(retrieved_retentions) == 3, "Should return 3 retention values"
        
        # Should be ordered by position (ASC)
        # The trigger assigns positions based on insertion order, so the last inserted should be at position 3
        assert retrieved_myth_ids[0] == myth_ids[0], "First myth should be the first inserted"
        assert retrieved_myth_ids[1] == myth_ids[1], "Second myth should be the second inserted"
        assert retrieved_myth_ids[2] == myth_ids[2], "Third myth should be the third inserted"
        
        # Check that retentions correspond to the correct myths
        # We need to check the actual retention values from the database
        with engine.connect() as conn:
            result = conn.execute(text("""
                SELECT myth_id, retention FROM agent_myths 
                WHERE agent_id = :agent_id 
                ORDER BY position ASC
            """), {"agent_id": agent_id})
            expected_data = result.fetchall()
        
        expected_myth_ids = [row[0] for row in expected_data]
        expected_retentions = [row[1] for row in expected_data]
        
        assert retrieved_myth_ids == expected_myth_ids, "Myth IDs should match expected order"
        assert retrieved_retentions == expected_retentions, "Retentions should match expected values"
    
    @pytest.mark.integration
    def test_get_myth_ids_and_retention_from_agents_memory_nonexistent_agent(self):
        """Test getting memory from a non-existent agent."""
        # Should return empty lists for non-existent agent
        myth_ids, retentions = get_myth_ids_and_retention_from_agents_memory(999)
        
        assert myth_ids == [], "Should return empty list for non-existent agent"
        assert retentions == [], "Should return empty list for non-existent agent"
    
    @pytest.mark.integration
    def test_get_myth_ids_and_retention_from_agents_memory_position_ordering(self):
        """Test that memory is correctly ordered by position."""
        # Insert an agent
        engine = get_engine()
        with engine.connect() as conn:
            conn.execute(text("INSERT INTO agents (name, memory_size) VALUES ('Test Agent', 5)"))
            conn.commit()
            
            # Get agent ID
            result = conn.execute(text("SELECT id FROM agents"))
            agent_id = result.fetchone()[0]
            
            # Insert a myth
            embedding_dim = 4
            embedding = np.random.rand(embedding_dim).tolist()
            
            conn.execute(text("""
                INSERT INTO myths (embedding, embedding_ids, offsets, weights) 
                VALUES (:embedding, ARRAY[1], ARRAY[]::vector[], ARRAY[]::double precision[])
            """), {"embedding": embedding})
            
            conn.commit()
            
            # Get myth ID
            result = conn.execute(text("SELECT id FROM myths"))
            myth_id = result.fetchone()[0]
            
            # Insert with specific position and retention
            conn.execute(text("""
                INSERT INTO agent_myths (myth_id, agent_id, position, retention) 
                VALUES (:myth_id, :agent_id, :position, :retention)
            """), {"myth_id": myth_id, "agent_id": agent_id, "position": 1, "retention": 0.85})
            
            conn.commit()
        
        # Get memory
        retrieved_myth_ids, retrieved_retentions = get_myth_ids_and_retention_from_agents_memory(agent_id)
        
        # Should have 1 myth
        assert len(retrieved_myth_ids) == 1, "Should return 1 myth ID"
        assert len(retrieved_retentions) == 1, "Should return 1 retention value"
        
        # Check values
        assert retrieved_myth_ids[0] == myth_id, "Myth ID should match"
        assert retrieved_retentions[0] == 0.85, "Retention should match"

    @pytest.mark.integration
    def test_get_myth_ids_and_retention_stack_behavior_with_eviction(self):
        """Test that memory correctly implements stack behavior with eviction when at capacity."""
        # Insert an agent with small memory size to test eviction
        engine = get_engine()
        with engine.connect() as conn:
            conn.execute(text("INSERT INTO agents (name, memory_size) VALUES ('Stack Agent', 3)"))
            conn.commit()
            
            # Get agent ID
            result = conn.execute(text("SELECT id FROM agents"))
            agent_id = result.fetchone()[0]
            
            # Insert 5 myths (more than memory capacity)
            embedding_dim = 4
            myth_ids = []
            retentions = [0.1, 0.2, 0.3, 0.4, 0.5]  # Different retentions to track each myth
            
            for i in range(5):
                embedding = np.random.rand(embedding_dim).tolist()
                conn.execute(text("""
                    INSERT INTO myths (embedding, embedding_ids, offsets, weights) 
                    VALUES (:embedding, ARRAY[:id], ARRAY[]::vector[], ARRAY[]::double precision[])
                """), {"embedding": embedding, "id": i + 1})
                
                conn.commit()
                
                # Get myth ID
                result = conn.execute(text("SELECT id FROM myths ORDER BY id DESC LIMIT 1"))
                myth_id = result.fetchone()[0]
                myth_ids.append(myth_id)
                
                # Insert into agent memory - let trigger handle positioning
                conn.execute(text("""
                    INSERT INTO agent_myths (myth_id, agent_id, position, retention) 
                    VALUES (:myth_id, :agent_id, 1, :retention)
                """), {"myth_id": myth_id, "agent_id": agent_id, "retention": retentions[i]})
                
                conn.commit()
                
                # Check memory state after each insertion
                retrieved_myth_ids, retrieved_retentions = get_myth_ids_and_retention_from_agents_memory(agent_id)
                
                if i < 3:  # First 3 insertions - memory not full yet
                    expected_count = i + 1
                    assert len(retrieved_myth_ids) == expected_count, f"After inserting myth {i+1}, should have {expected_count} myths"
                    
                    # Should be in insertion order (bottom to top)
                    for j in range(expected_count):
                        assert retrieved_myth_ids[j] == myth_ids[j], f"Myth at position {j+1} should be myth {j+1}"
                        assert retrieved_retentions[j] == retentions[j], f"Retention at position {j+1} should be {retentions[j]}"
                
                else:  # 4th and 5th insertions - eviction should occur
                    assert len(retrieved_myth_ids) == 3, f"After inserting myth {i+1}, should still have only 3 myths (capacity limit)"
                    
                    # The bottom myth should have been evicted, so we should have the last 3 myths
                    expected_start_idx = i - 2  # For i=3: myths 1,2,3. For i=4: myths 2,3,4
                    for j in range(3):
                        expected_myth_idx = expected_start_idx + j
                        assert retrieved_myth_ids[j] == myth_ids[expected_myth_idx], f"After eviction, position {j+1} should have myth {expected_myth_idx+1}"
                        assert retrieved_retentions[j] == retentions[expected_myth_idx], f"After eviction, position {j+1} should have retention {retentions[expected_myth_idx]}"
        
        # Final verification: should have myths 3, 4, 5 (the last 3 inserted)
        final_myth_ids, final_retentions = get_myth_ids_and_retention_from_agents_memory(agent_id)
        
        assert len(final_myth_ids) == 3, "Final memory should have exactly 3 myths"
        assert final_myth_ids == myth_ids[2:5], "Final memory should contain the last 3 inserted myths"
        assert final_retentions == retentions[2:5], "Final retentions should match the last 3 inserted myths"
        
        # Verify the positions are correct (1, 2, 3)
        with engine.connect() as conn:
            result = conn.execute(text("""
                SELECT myth_id, position, retention 
                FROM agent_myths 
                WHERE agent_id = :agent_id 
                ORDER BY position ASC
            """), {"agent_id": agent_id})
            
            db_data = result.fetchall()
            assert len(db_data) == 3, "Database should have exactly 3 myths"
            
            for i, (myth_id, position, retention) in enumerate(db_data):
                assert position == i + 1, f"Position should be {i + 1}, got {position}"
                assert myth_id == myth_ids[i + 2], f"Myth at position {i + 1} should be myth {i + 3}"
                assert retention == retentions[i + 2], f"Retention at position {i + 1} should be {retentions[i + 2]}"

    @pytest.mark.integration 
    def test_get_myth_ids_and_retention_stack_ordering_verification(self):
        """Test that the function returns myths in correct stack order (bottom to top)."""
        # Insert an agent
        engine = get_engine()
        with engine.connect() as conn:
            conn.execute(text("INSERT INTO agents (name, memory_size) VALUES ('Order Agent', 4)"))
            conn.commit()
            
            # Get agent ID
            result = conn.execute(text("SELECT id FROM agents"))
            agent_id = result.fetchone()[0]
            
            # Insert 4 myths with distinct retentions to track order
            embedding_dim = 4
            myth_ids = []
            retentions = [0.10, 0.20, 0.30, 0.40]  # Bottom to top
            
            for i in range(4):
                embedding = np.random.rand(embedding_dim).tolist()
                conn.execute(text("""
                    INSERT INTO myths (embedding, embedding_ids, offsets, weights) 
                    VALUES (:embedding, ARRAY[:id], ARRAY[]::vector[], ARRAY[]::double precision[])
                """), {"embedding": embedding, "id": i + 100})  # Use distinct IDs
                
                conn.commit()
                
                # Get myth ID
                result = conn.execute(text("SELECT id FROM myths ORDER BY id DESC LIMIT 1"))
                myth_id = result.fetchone()[0]
                myth_ids.append(myth_id)
                
                # Insert into agent memory
                conn.execute(text("""
                    INSERT INTO agent_myths (myth_id, agent_id, position, retention) 
                    VALUES (:myth_id, :agent_id, 1, :retention)
                """), {"myth_id": myth_id, "agent_id": agent_id, "retention": retentions[i]})
                
                conn.commit()
        
        # Get memory and verify order
        retrieved_myth_ids, retrieved_retentions = get_myth_ids_and_retention_from_agents_memory(agent_id)
        
        assert len(retrieved_myth_ids) == 4, "Should have 4 myths"
        assert len(retrieved_retentions) == 4, "Should have 4 retentions"
        
        # Verify the order is bottom to top (position 1, 2, 3, 4)
        # This means first inserted myth should be at position 1 (bottom)
        # and last inserted myth should be at position 4 (top)
        assert retrieved_myth_ids == myth_ids, "Myths should be in insertion order (bottom to top)"
        assert retrieved_retentions == retentions, "Retentions should be in insertion order (bottom to top)"
        
        # Double-check by querying database directly
        with engine.connect() as conn:
            result = conn.execute(text("""
                SELECT myth_id, position, retention 
                FROM agent_myths 
                WHERE agent_id = :agent_id 
                ORDER BY position ASC
            """), {"agent_id": agent_id})
            
            db_data = result.fetchall()
            
            for i, (db_myth_id, db_position, db_retention) in enumerate(db_data):
                assert db_position == i + 1, f"Database position should be {i + 1}"
                assert db_myth_id == myth_ids[i], f"Database myth at position {i + 1} should match"
                assert db_retention == retentions[i], f"Database retention at position {i + 1} should match"
                assert db_myth_id == retrieved_myth_ids[i], "Function result should match database"
                assert db_retention == retrieved_retentions[i], "Function retention should match database"
    
    @pytest.mark.integration
    def test_stack_behavior_with_trigger_insertion_order(self):
        """Test that the function returns memory in the correct stack order using trigger-based insertion."""
        # Insert an agent with memory size 5
        engine = get_engine()
        with engine.connect() as conn:
            conn.execute(text("INSERT INTO agents (name, memory_size) VALUES ('Stack Test Agent', 5)"))
            conn.commit()
            
            # Get agent ID
            result = conn.execute(text("SELECT id FROM agents"))
            agent_id = result.fetchone()[0]
            
            # Insert 3 myths
            embedding_dim = 4
            myth_ids = []
            for i in range(3):
                embedding = np.random.rand(embedding_dim).tolist()
                conn.execute(text("""
                    INSERT INTO myths (embedding, embedding_ids, offsets, weights) 
                    VALUES (:embedding, ARRAY[:id], ARRAY[]::vector[], ARRAY[]::double precision[])
                """), {"embedding": embedding, "id": i + 1})
            
            conn.commit()
            
            # Get myth IDs
            result = conn.execute(text("SELECT id FROM myths ORDER BY id"))
            myth_ids = [row[0] for row in result.fetchall()]
            
            # Insert myths using trigger (position will be auto-assigned)
            # The trigger assigns: first myth -> position 1, second -> position 2, third -> position 3
            retentions = [0.8, 0.9, 0.7]  # Different retentions for each myth
            
            for myth_id, retention in zip(myth_ids, retentions):
                # Insert with position=1, but trigger will override this
                conn.execute(text("""
                    INSERT INTO agent_myths (myth_id, agent_id, position, retention) 
                    VALUES (:myth_id, :agent_id, 1, :retention)
                """), {"myth_id": myth_id, "agent_id": agent_id, "retention": retention})
            
            conn.commit()
            
            # Verify the actual positions assigned by the trigger
            result = conn.execute(text("""
                SELECT myth_id, position, retention 
                FROM agent_myths 
                WHERE agent_id = :agent_id 
                ORDER BY position ASC
            """), {"agent_id": agent_id})
            
            db_data = result.fetchall()
            print(f"Database positions: {[(row[0], row[1], row[2]) for row in db_data]}")
            
            # Expected: myth_ids[0] at position 1, myth_ids[1] at position 2, myth_ids[2] at position 3
            assert db_data[0][0] == myth_ids[0], "First inserted myth should be at position 1 (bottom)"
            assert db_data[0][1] == 1, "First myth should have position 1"
            assert db_data[1][0] == myth_ids[1], "Second inserted myth should be at position 2"
            assert db_data[1][1] == 2, "Second myth should have position 2"
            assert db_data[2][0] == myth_ids[2], "Third inserted myth should be at position 3 (top)"
            assert db_data[2][1] == 3, "Third myth should have position 3"
        
        # Get memory using our function
        retrieved_myth_ids, retrieved_retentions = get_myth_ids_and_retention_from_agents_memory(agent_id)
        
        # Verify the function returns in stack order (bottom to top)
        assert len(retrieved_myth_ids) == 3, "Should return 3 myth IDs"
        assert len(retrieved_retentions) == 3, "Should return 3 retention values"
        
        # Stack order: position 1 (bottom/oldest) to position 3 (top/newest)
        assert retrieved_myth_ids[0] == myth_ids[0], "First in result should be myth at position 1 (bottom)"
        assert retrieved_myth_ids[1] == myth_ids[1], "Second in result should be myth at position 2 (middle)"
        assert retrieved_myth_ids[2] == myth_ids[2], "Third in result should be myth at position 3 (top)"
        
        # Verify retentions match
        assert retrieved_retentions[0] == retentions[0], "First retention should match"
        assert retrieved_retentions[1] == retentions[1], "Second retention should match"
        assert retrieved_retentions[2] == retentions[2], "Third retention should match"
        
        print(f"Function returned (bottom to top): {list(zip(retrieved_myth_ids, retrieved_retentions))}")
    
    @pytest.mark.integration
    def test_stack_vs_queue_semantics(self):
        """Test to clarify if the function should return stack order or queue order."""
        # Insert an agent with memory size 3
        engine = get_engine()
        with engine.connect() as conn:
            conn.execute(text("INSERT INTO agents (name, memory_size) VALUES ('Semantics Test Agent', 3)"))
            conn.commit()
            
            # Get agent ID
            result = conn.execute(text("SELECT id FROM agents"))
            agent_id = result.fetchone()[0]
            
            # Insert 3 myths sequentially
            embedding_dim = 4
            myth_ids = []
            for i in range(3):
                embedding = np.random.rand(embedding_dim).tolist()
                conn.execute(text("""
                    INSERT INTO myths (embedding, embedding_ids, offsets, weights) 
                    VALUES (:embedding, ARRAY[:id], ARRAY[]::vector[], ARRAY[]::double precision[])
                """), {"embedding": embedding, "id": i + 100})  # Use unique IDs
            
            conn.commit()
            
            # Get myth IDs
            result = conn.execute(text("SELECT id FROM myths WHERE embedding_ids[1] >= 100 ORDER BY id"))
            myth_ids = [row[0] for row in result.fetchall()]
            
            # Insert myths in sequence (simulating temporal order)
            retentions = [0.5, 0.6, 0.7]  # Increasing retention for clarity
            
            for i, (myth_id, retention) in enumerate(zip(myth_ids, retentions)):
                conn.execute(text("""
                    INSERT INTO agent_myths (myth_id, agent_id, position, retention) 
                    VALUES (:myth_id, :agent_id, 1, :retention)
                """), {"myth_id": myth_id, "agent_id": agent_id, "retention": retention})
                
                # Check position after each insertion
                result = conn.execute(text("""
                    SELECT myth_id, position FROM agent_myths 
                    WHERE agent_id = :agent_id 
                    ORDER BY position ASC
                """), {"agent_id": agent_id})
                positions = result.fetchall()
                print(f"After inserting myth {i+1}: {positions}")
            
            conn.commit()
        
        # Get memory using our function
        retrieved_myth_ids, retrieved_retentions = get_myth_ids_and_retention_from_agents_memory(agent_id)
        
        print(f"Insertion order: {myth_ids} (oldest to newest)")
        print(f"Function result:  {retrieved_myth_ids} (position 1 to 3)")
        print(f"Retentions:       {retrieved_retentions}")
        
        # The key question: Should the function return:
        # A) Stack order (bottom to top): oldest first, newest last - ORDER BY position ASC
        # B) Queue order (front to back): newest first, oldest last - ORDER BY position DESC
        
        # Based on schema comments, this is a STACK with position 1 = bottom (oldest)
        # So stack order (A) should be correct: oldest myths first
        assert retrieved_myth_ids[0] == myth_ids[0], "Oldest myth should be first (position 1, bottom of stack)"
        assert retrieved_myth_ids[2] == myth_ids[2], "Newest myth should be last (position 3, top of stack)"

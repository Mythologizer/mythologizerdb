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
        
        # Should be ordered by insertion order (stack behavior - last inserted first)
        # Retentions: myth_ids[0]=0.8, myth_ids[1]=0.9, myth_ids[2]=0.7
        # Expected order: 0.7 (last inserted), 0.9 (second inserted), 0.8 (first inserted)
        assert retrieved_myth_ids[0] == myth_ids[2], "Last inserted myth (0.7) should be first"
        assert retrieved_myth_ids[1] == myth_ids[1], "Second inserted myth (0.9) should be second"
        assert retrieved_myth_ids[2] == myth_ids[0], "First inserted myth (0.8) should be third"
        
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
            """), {"myth_id": myth_id, "agent_id": agent_id, "position": 0, "retention": 0.85})
            
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

                    # Should be ordered by insertion order (last inserted first)
                    # Retentions: [0.1, 0.2, 0.3, 0.4, 0.5]
                    # After 3 insertions: [0.3, 0.2, 0.1] (last inserted first)
                    if i == 0:  # After first insertion
                        assert retrieved_myth_ids[0] == myth_ids[0], f"After inserting myth 1, should have myth 1 first"
                        assert retrieved_retentions[0] == retentions[0], f"Retention at position 0 should be {retentions[0]}"
                    elif i == 1:  # After second insertion
                        assert retrieved_myth_ids[0] == myth_ids[1], f"After inserting myth 2, myth 2 (last inserted) should be first"
                        assert retrieved_myth_ids[1] == myth_ids[0], f"After inserting myth 2, myth 1 (first inserted) should be second"
                        assert retrieved_retentions[0] == retentions[1], f"Retention at position 0 should be {retentions[1]}"
                        assert retrieved_retentions[1] == retentions[0], f"Retention at position 1 should be {retentions[0]}"
                    elif i == 2:  # After third insertion
                        assert retrieved_myth_ids[0] == myth_ids[2], f"After inserting myth 3, myth 3 (last inserted) should be first"
                        assert retrieved_myth_ids[1] == myth_ids[1], f"After inserting myth 3, myth 2 (second inserted) should be second"
                        assert retrieved_myth_ids[2] == myth_ids[0], f"After inserting myth 3, myth 1 (first inserted) should be third"
                        assert retrieved_retentions[0] == retentions[2], f"Retention at position 0 should be {retentions[2]}"
                        assert retrieved_retentions[1] == retentions[1], f"Retention at position 1 should be {retentions[1]}"
                        assert retrieved_retentions[2] == retentions[0], f"Retention at position 2 should be {retentions[0]}"
                
                else:  # 4th and 5th insertions - eviction should occur
                    assert len(retrieved_myth_ids) == 3, f"After inserting myth {i+1}, should still have only 3 myths (capacity limit)"
                    
                    # With stack behavior, the oldest myths (highest position) should be evicted
                    # After 4 insertions: should have myths 4, 3, 2 (last 3 inserted)
                    # After 5 insertions: should have myths 5, 4, 3 (last 3 inserted)
                    if i == 3:  # After 4th insertion
                        # Should have myths 4, 3, 2 (last 3 inserted)
                        assert retrieved_myth_ids[0] == myth_ids[3], f"After 4th insertion, myth 4 (last inserted) should be first"
                        assert retrieved_myth_ids[1] == myth_ids[2], f"After 4th insertion, myth 3 (third inserted) should be second"
                        assert retrieved_myth_ids[2] == myth_ids[1], f"After 4th insertion, myth 2 (second inserted) should be third"
                    elif i == 4:  # After 5th insertion
                        # Should have myths 5, 4, 3 (last 3 inserted)
                        assert retrieved_myth_ids[0] == myth_ids[4], f"After 5th insertion, myth 5 (last inserted) should be first"
                        assert retrieved_myth_ids[1] == myth_ids[3], f"After 5th insertion, myth 4 (fourth inserted) should be second"
                        assert retrieved_myth_ids[2] == myth_ids[2], f"After 5th insertion, myth 3 (third inserted) should be third"
        
        # Final verification: should have myths 5, 4, 3 (last 3 inserted - stack behavior)
        final_myth_ids, final_retentions = get_myth_ids_and_retention_from_agents_memory(agent_id)
        
        assert len(final_myth_ids) == 3, "Final memory should have exactly 3 myths"
        expected_final_myths = [myth_ids[4], myth_ids[3], myth_ids[2]]  # Last 3 inserted (stack behavior)
        expected_final_retentions = [retentions[4], retentions[3], retentions[2]]  # Last 3 inserted
        assert final_myth_ids == expected_final_myths, "Final memory should contain the last 3 inserted myths (stack behavior)"
        assert final_retentions == expected_final_retentions, "Final retentions should match the last 3 inserted myths"
        
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
                assert position == i, f"Position should be {i}, got {position}"
                # Should be ordered by insertion order: myths 5, 4, 3 (last 3 inserted)
                expected_myth_idx = 4 - i  # 4, 3, 2 (myths 5, 4, 3)
                assert myth_id == myth_ids[expected_myth_idx], f"Myth at position {i} should be myth {expected_myth_idx + 1}"
                assert retention == retentions[expected_myth_idx], f"Retention at position {i + 1} should be {retentions[expected_myth_idx]}"

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
        
        # Verify the order is by insertion order (last inserted first)
        # Retentions: [0.10, 0.20, 0.30, 0.40] - should be ordered as [0.40, 0.30, 0.20, 0.10] (last inserted first)
        expected_order = [myth_ids[3], myth_ids[2], myth_ids[1], myth_ids[0]]  # Last to first inserted
        assert retrieved_myth_ids == expected_order, "Myths should be ordered by insertion order (last inserted first)"
        expected_retentions = [retentions[3], retentions[2], retentions[1], retentions[0]]  # Last to first inserted
        assert retrieved_retentions == expected_retentions, "Retentions should be ordered by insertion order (last inserted first)"
        
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
                assert db_position == i, f"Database position should be {i}"
                # Should be ordered by insertion order: myths 4, 3, 2, 1 (last inserted first)
                expected_myth_idx = 3 - i  # 3, 2, 1, 0 (myths 4, 3, 2, 1)
                assert db_myth_id == myth_ids[expected_myth_idx], f"Database myth at position {i} should be myth {expected_myth_idx + 1}"
                assert db_retention == retentions[expected_myth_idx], f"Database retention at position {i} should be {retentions[expected_myth_idx]}"
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
            
            # Expected: ordered by insertion order (last inserted first)
            # Insertion order: myth_ids[0]=0.8, myth_ids[1]=0.9, myth_ids[2]=0.7
            # Expected order: 0.7, 0.9, 0.8 (last inserted first)
            assert db_data[0][0] == myth_ids[2], "Last inserted myth (0.7) should be at position 0"
            assert db_data[0][1] == 0, "Last inserted myth should have position 0"
            assert db_data[1][0] == myth_ids[1], "Second inserted myth (0.9) should be at position 1"
            assert db_data[1][1] == 1, "Second inserted myth should have position 1"
            assert db_data[2][0] == myth_ids[0], "First inserted myth (0.8) should be at position 2"
            assert db_data[2][1] == 2, "First inserted myth should have position 2"
        
        # Get memory using our function
        retrieved_myth_ids, retrieved_retentions = get_myth_ids_and_retention_from_agents_memory(agent_id)
        
        # Verify the function returns in stack order (bottom to top)
        assert len(retrieved_myth_ids) == 3, "Should return 3 myth IDs"
        assert len(retrieved_retentions) == 3, "Should return 3 retention values"
        
        # Stack order: last inserted first
        assert retrieved_myth_ids[0] == myth_ids[2], "First in result should be last inserted myth (0.7)"
        assert retrieved_myth_ids[1] == myth_ids[1], "Second in result should be second inserted myth (0.9)"
        assert retrieved_myth_ids[2] == myth_ids[0], "Third in result should be first inserted myth (0.8)"
        
        # Verify retentions match (ordered by insertion)
        assert retrieved_retentions[0] == retentions[2], "First retention should be last inserted (0.7)"
        assert retrieved_retentions[1] == retentions[1], "Second retention should be second inserted (0.9)"
        assert retrieved_retentions[2] == retentions[0], "Third retention should be first inserted (0.8)"
        
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
        
        # Based on stack behavior, myths are ordered by insertion order (last inserted first)
        # Insertion order: [0.5, 0.6, 0.7] - should be ordered as [0.7, 0.6, 0.5] (last inserted first)
        assert retrieved_myth_ids[0] == myth_ids[2], "Last inserted myth (0.7) should be first"
        assert retrieved_myth_ids[2] == myth_ids[0], "First inserted myth (0.5) should be last"

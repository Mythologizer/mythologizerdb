import pytest
from sqlalchemy import text

from mythologizer_postgres.db import session_scope, clear_all_rows
from mythologizer_postgres.connectors.status import (
    get_current_epoch,
    increment_epoch,
    get_n_agents,
    get_n_myths,
    get_n_cultures,
    get_simulation_status,
)


class TestStatusConnector:
    """Test the status connector functions."""
    
    @pytest.mark.integration
    def test_get_current_epoch(self):
        """Test getting the current epoch."""
        with session_scope() as session:
            # Ensure epoch table has initial data
            session.execute(text("""
                INSERT INTO epoch (key, current_epoch) 
                VALUES ('only', 0) 
                ON CONFLICT (key) DO NOTHING
            """))
            session.commit()
            
            # Test getting current epoch
            current_epoch = get_current_epoch()
            assert current_epoch == 0, "Initial epoch should be 0"
            
            # Update epoch and test again
            session.execute(text("UPDATE epoch SET current_epoch = 5 WHERE key = 'only'"))
            session.commit()
            
            current_epoch = get_current_epoch()
            assert current_epoch == 5, "Epoch should be updated to 5"
    
    @pytest.mark.integration
    def test_get_current_epoch_no_record(self):
        """Test get_current_epoch when epoch record doesn't exist."""
        with session_scope() as session:
            # Clear epoch table
            session.execute(text("DELETE FROM epoch"))
            session.commit()
            
            # Should raise ValueError when no epoch record exists
            with pytest.raises(ValueError, match="Epoch record not found"):
                get_current_epoch()
    
    @pytest.mark.integration
    def test_increment_epoch(self):
        """Test incrementing the epoch."""
        with session_scope() as session:
            # Ensure epoch table has initial data
            session.execute(text("""
                INSERT INTO epoch (key, current_epoch) 
                VALUES ('only', 0) 
                ON CONFLICT (key) DO NOTHING
            """))
            session.commit()
            
            # Test initial state
            assert get_current_epoch() == 0, "Initial epoch should be 0"
            
            # Test incrementing
            new_epoch = increment_epoch()
            assert new_epoch == 1, "Incremented epoch should be 1"
            assert get_current_epoch() == 1, "Current epoch should be updated to 1"
            
            # Test incrementing again
            new_epoch = increment_epoch()
            assert new_epoch == 2, "Incremented epoch should be 2"
            assert get_current_epoch() == 2, "Current epoch should be updated to 2"
    
    @pytest.mark.integration
    def test_get_n_agents(self):
        """Test getting the number of agents."""
        with session_scope() as session:
            # Clear agents table
            session.execute(text("DELETE FROM agents"))
            session.commit()
            
            # Test empty agents table
            n_agents = get_n_agents()
            assert n_agents == 0, "Should return 0 for empty agents table"
            
            # Insert some test agents
            session.execute(text("""
                INSERT INTO agents (name, memory_size) 
                VALUES 
                    ('Agent1', 10),
                    ('Agent2', 15),
                    ('Agent3', 20)
            """))
            session.commit()
            
            # Test with agents
            n_agents = get_n_agents()
            assert n_agents == 3, "Should return 3 for 3 agents"
    
    @pytest.mark.integration
    def test_get_n_myths(self):
        """Test getting the number of myths."""
        with session_scope() as session:
            # Clear myths table
            session.execute(text("DELETE FROM myths"))
            session.commit()
            
            # Test empty myths table
            n_myths = get_n_myths()
            assert n_myths == 0, "Should return 0 for empty myths table"
            
            # Insert some test myths (simplified)
            session.execute(text("""
                INSERT INTO myths (embedding, embedding_ids, offsets, weights) 
                VALUES 
                    (ARRAY[0.1, 0.2, 0.3, 0.4], ARRAY[1, 2], ARRAY[]::vector[], ARRAY[]::double precision[]),
                    (ARRAY[0.5, 0.6, 0.7, 0.8], ARRAY[3, 4], ARRAY[]::vector[], ARRAY[]::double precision[])
            """))
            session.commit()
            
            # Test with myths
            n_myths = get_n_myths()
            assert n_myths == 2, "Should return 2 for 2 myths"
    
    @pytest.mark.integration
    def test_get_n_cultures(self):
        """Test getting the number of cultures."""
        with session_scope() as session:
            # Clear cultures table
            session.execute(text("DELETE FROM cultures"))
            session.commit()
            
            # Test empty cultures table
            n_cultures = get_n_cultures()
            assert n_cultures == 0, "Should return 0 for empty cultures table"
            
            # Insert some test cultures
            session.execute(text("""
                INSERT INTO cultures (name, description) 
                VALUES 
                    ('Culture1', 'Description 1'),
                    ('Culture2', 'Description 2'),
                    ('Culture3', 'Description 3'),
                    ('Culture4', 'Description 4')
            """))
            session.commit()
            
            # Test with cultures
            n_cultures = get_n_cultures()
            assert n_cultures == 4, "Should return 4 for 4 cultures"
    
    @pytest.mark.integration
    def test_get_simulation_status(self):
        """Test getting comprehensive simulation status."""
        with session_scope() as session:
            # Set up test data
            session.execute(text("""
                INSERT INTO epoch (key, current_epoch) 
                VALUES ('only', 10) 
                ON CONFLICT (key) DO UPDATE SET current_epoch = 10
            """))
            
            session.execute(text("DELETE FROM agents"))
            session.execute(text("""
                INSERT INTO agents (name, memory_size)
                VALUES ('TestAgent', 10)
            """))
            
            session.execute(text("DELETE FROM myths"))
            session.execute(text("""
                INSERT INTO myths (embedding, embedding_ids, offsets, weights) 
                VALUES (ARRAY[0.1, 0.2, 0.3, 0.4], ARRAY[1], ARRAY[]::vector[], ARRAY[]::double precision[])
            """))
            
            session.execute(text("DELETE FROM cultures"))
            session.execute(text("""
                INSERT INTO cultures (name, description)
                VALUES ('TestCulture', 'Test culture description')
            """))
            
            session.commit()
            
            # Test comprehensive status
            status = get_simulation_status()
            
            assert isinstance(status, dict), "Status should be a dictionary"
            assert 'current_epoch' in status, "Status should contain current_epoch"
            assert 'n_agents' in status, "Status should contain n_agents"
            assert 'n_myths' in status, "Status should contain n_myths"
            assert 'n_cultures' in status, "Status should contain n_cultures"
            
            assert status['current_epoch'] == 10, "Current epoch should be 10"
            assert status['n_agents'] == 1, "Should have 1 agent"
            assert status['n_myths'] == 1, "Should have 1 myth"
            assert status['n_cultures'] == 1, "Should have 1 culture"
    
    @pytest.mark.integration
    def test_status_functions_integration(self):
        """Test integration between status functions."""
        with session_scope() as session:
            # Ensure clean state
            session.execute(text("""
                INSERT INTO epoch (key, current_epoch) 
                VALUES ('only', 0) 
                ON CONFLICT (key) DO UPDATE SET current_epoch = 0
            """))
            session.commit()
            
            # Test that increment_epoch affects get_current_epoch
            initial_epoch = get_current_epoch()
            assert initial_epoch == 0, "Initial epoch should be 0"
            
            new_epoch = increment_epoch()
            assert new_epoch == 1, "Incremented epoch should be 1"
            
            current_epoch = get_current_epoch()
            assert current_epoch == 1, "get_current_epoch should reflect the increment"
            assert current_epoch == new_epoch, "Both functions should return the same value"
            
            # Test that get_simulation_status reflects the current state
            status = get_simulation_status()
            assert status['current_epoch'] == 1, "Status should reflect current epoch"
    
    def teardown_method(self):
        """Clean up after each test."""
        clear_all_rows()

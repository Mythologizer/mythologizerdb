import os
import pytest
import numpy as np
from sqlalchemy import text
from datetime import datetime, timezone

from mythologizer_postgres.db import (
    session_scope,
    get_table_row_counts,
    clear_all_rows,
)


class TestEventsSchema:
    """Test the events table schema and operations."""

    def setup_method(self):
        """Clean up before each test method."""
        clear_all_rows()

    def teardown_method(self):
        """Clean up after each test method."""
        clear_all_rows()
    
    @pytest.mark.integration
    def test_events_table_structure(self):
        """Test that events table has the correct structure."""
        with session_scope() as session:
            # Check events table structure
            result = session.execute(text("""
                SELECT column_name, data_type, is_nullable, column_default
                FROM information_schema.columns 
                WHERE table_name = 'events' 
                ORDER BY ordinal_position
            """))
            events_columns = {row[0]: {'type': row[1], 'nullable': row[2], 'default': row[3]} for row in result.fetchall()}
            
            expected_events_columns = {
                'id': {'type': 'integer', 'nullable': 'NO', 'default': "nextval('events_id_seq'::regclass)"},
                'description': {'type': 'text', 'nullable': 'NO', 'default': None},
                'has_been_triggered': {'type': 'boolean', 'nullable': 'NO', 'default': 'false'},
                'created_at': {'type': 'timestamp with time zone', 'nullable': 'YES', 'default': 'CURRENT_TIMESTAMP'}
            }
            
            for col, expected in expected_events_columns.items():
                assert col in events_columns, f"Column {col} should exist in events table"
                assert events_columns[col]['type'] == expected['type'], f"Column {col} should be {expected['type']}"
                assert events_columns[col]['nullable'] == expected['nullable'], f"Column {col} nullable should be {expected['nullable']}"
                if expected['default']:
                    assert events_columns[col]['default'] == expected['default'], f"Column {col} default should be {expected['default']}"
    
    @pytest.mark.integration
    def test_events_basic_operations(self):
        """Test basic CRUD operations on events table."""
        with session_scope() as session:
            # Insert events
            session.execute(text("""
                INSERT INTO events (description, has_been_triggered) 
                VALUES (:description, :has_been_triggered)
            """), {
                'description': 'First event',
                'has_been_triggered': False
            })
            
            session.execute(text("""
                INSERT INTO events (description, has_been_triggered) 
                VALUES (:description, :has_been_triggered)
            """), {
                'description': 'Second event',
                'has_been_triggered': True
            })
            
            # Query events
            result = session.execute(text("""
                SELECT id, description, has_been_triggered, created_at 
                FROM events 
                ORDER BY created_at
            """))
            events = result.fetchall()
            
            assert len(events) == 2, "Should have 2 events"
            assert events[0][1] == 'First event', "First event should be 'First event'"
            assert events[1][1] == 'Second event', "Second event should be 'Second event'"
            assert events[0][2] == False, "First event should not be triggered"
            assert events[1][2] == True, "Second event should be triggered"
            assert events[0][3] is not None, "created_at should not be null"
            assert events[1][3] is not None, "created_at should not be null"
            
            # Update event
            session.execute(text("""
                UPDATE events 
                SET has_been_triggered = :triggered 
                WHERE description = :description
            """), {
                'triggered': True,
                'description': 'First event'
            })
            
            # Verify update
            result = session.execute(text("""
                SELECT has_been_triggered FROM events WHERE description = 'First event'
            """))
            updated_triggered = result.fetchone()[0]
            assert updated_triggered == True, "Event should be marked as triggered"
        
        # Clean up
        clear_all_rows()
    
    @pytest.mark.integration
    def test_events_constraints(self):
        """Test that events table constraints work correctly."""
        with session_scope() as session:
            # Test valid insertions
            session.execute(text("""
                INSERT INTO events (description) 
                VALUES (:description)
            """), {'description': 'Valid event'})
            
            # Test that description cannot be null
            with pytest.raises(Exception):
                session.execute(text("""
                    INSERT INTO events (description) 
                    VALUES (NULL)
                """))
                session.commit()  # This should not be reached
        
        # Start a new session for the remaining tests
        with session_scope() as session:
            # Test that has_been_triggered defaults to false
            session.execute(text("""
                INSERT INTO events (description) 
                VALUES (:description)
            """), {'description': 'Event with default triggered'})
            
            result = session.execute(text("""
                SELECT has_been_triggered FROM events WHERE description = 'Event with default triggered'
            """))
            triggered_status = result.fetchone()[0]
            assert triggered_status == False, "has_been_triggered should default to false"
            
            # Test that created_at is automatically set
            result = session.execute(text("""
                SELECT created_at FROM events WHERE description = 'Event with default triggered'
            """))
            created_at = result.fetchone()[0]
            assert created_at is not None, "created_at should be automatically set"
            assert isinstance(created_at, datetime), "created_at should be a datetime object"
    
    @pytest.mark.integration
    def test_events_ordering(self):
        """Test that events are properly ordered by creation time."""
        with session_scope() as session:
            # Insert events with small delays to ensure different timestamps
            import time
            
            session.execute(text("""
                INSERT INTO events (description) 
                VALUES (:description)
            """), {'description': 'First event'})
            
            time.sleep(0.1)  # Small delay
            
            session.execute(text("""
                INSERT INTO events (description) 
                VALUES (:description)
            """), {'description': 'Second event'})
            
            time.sleep(0.1)  # Small delay
            
            session.execute(text("""
                INSERT INTO events (description) 
                VALUES (:description)
            """), {'description': 'Third event'})
            
            # Query events ordered by created_at ASC (earliest first)
            result = session.execute(text("""
                SELECT description, created_at 
                FROM events 
                ORDER BY created_at ASC
            """))
            events_asc = result.fetchall()
            
            assert len(events_asc) == 3, "Should have 3 events"
            assert events_asc[0][0] == 'First event', "First event should be earliest"
            assert events_asc[1][0] == 'Second event', "Second event should be middle"
            assert events_asc[2][0] == 'Third event', "Third event should be latest"
            
            # Verify timestamps are in ascending order
            assert events_asc[0][1] <= events_asc[1][1], "First timestamp should be <= second"
            assert events_asc[1][1] <= events_asc[2][1], "Second timestamp should be <= third"
    
    @pytest.mark.integration
    def test_events_triggered_filtering(self):
        """Test filtering events by triggered status."""
        with session_scope() as session:
            # Insert events with different triggered statuses
            session.execute(text("""
                INSERT INTO events (description, has_been_triggered) 
                VALUES (:description, :triggered)
            """), {'description': 'Untriggered 1', 'triggered': False})
            
            session.execute(text("""
                INSERT INTO events (description, has_been_triggered) 
                VALUES (:description, :triggered)
            """), {'description': 'Triggered 1', 'triggered': True})
            
            session.execute(text("""
                INSERT INTO events (description, has_been_triggered) 
                VALUES (:description, :triggered)
            """), {'description': 'Untriggered 2', 'triggered': False})
            
            session.execute(text("""
                INSERT INTO events (description, has_been_triggered) 
                VALUES (:description, :triggered)
            """), {'description': 'Triggered 2', 'triggered': True})
            
            # Query untriggered events
            result = session.execute(text("""
                SELECT description 
                FROM events 
                WHERE has_been_triggered = FALSE 
                ORDER BY created_at ASC
            """))
            untriggered = [row[0] for row in result.fetchall()]
            
            assert len(untriggered) == 2, "Should have 2 untriggered events"
            assert 'Untriggered 1' in untriggered, "Should include first untriggered event"
            assert 'Untriggered 2' in untriggered, "Should include second untriggered event"
            
            # Query triggered events
            result = session.execute(text("""
                SELECT description 
                FROM events 
                WHERE has_been_triggered = TRUE 
                ORDER BY created_at ASC
            """))
            triggered = [row[0] for row in result.fetchall()]
            
            assert len(triggered) == 2, "Should have 2 triggered events"
            assert 'Triggered 1' in triggered, "Should include first triggered event"
            assert 'Triggered 2' in triggered, "Should include second triggered event"
    
    @pytest.mark.integration
    def test_events_next_event_query(self):
        """Test the query pattern used by get_next_event function."""
        with session_scope() as session:
            # Insert events with different triggered statuses
            session.execute(text("""
                INSERT INTO events (description, has_been_triggered) 
                VALUES (:description, :triggered)
            """), {'description': 'Old untriggered', 'triggered': False})
            
            session.execute(text("""
                INSERT INTO events (description, has_been_triggered) 
                VALUES (:description, :triggered)
            """), {'description': 'Triggered event', 'triggered': True})
            
            session.execute(text("""
                INSERT INTO events (description, has_been_triggered) 
                VALUES (:description, :triggered)
            """), {'description': 'New untriggered', 'triggered': False})
            
            # Query for next event (earliest untriggered)
            result = session.execute(text("""
                SELECT id, description, has_been_triggered, created_at
                FROM events
                WHERE has_been_triggered = FALSE
                ORDER BY created_at ASC
                LIMIT 1
            """))
            next_event = result.fetchone()
            
            assert next_event is not None, "Should find a next event"
            assert next_event[1] == 'Old untriggered', "Should return the earliest untriggered event"
            assert next_event[2] == False, "Should not be triggered"
            
            # Mark the first event as triggered
            session.execute(text("""
                UPDATE events 
                SET has_been_triggered = TRUE 
                WHERE id = :id
            """), {'id': next_event[0]})
            
            # Query for next event again
            result = session.execute(text("""
                SELECT id, description, has_been_triggered, created_at
                FROM events
                WHERE has_been_triggered = FALSE
                ORDER BY created_at ASC
                LIMIT 1
            """))
            next_event = result.fetchone()
            
            assert next_event is not None, "Should find the next untriggered event"
            assert next_event[1] == 'New untriggered', "Should return the remaining untriggered event"
    
    @pytest.mark.integration
    def test_events_table_exists(self):
        """Test that the events table exists and is accessible."""
        with session_scope() as session:
            # Check if table exists
            result = session.execute(text("""
                SELECT EXISTS (
                    SELECT FROM information_schema.tables 
                    WHERE table_name = 'events'
                )
            """))
            table_exists = result.fetchone()[0]
            assert table_exists, "Events table should exist"
            
            # Check table row count
            row_count = get_table_row_counts().get('events', 0)
            assert isinstance(row_count, int), "Row count should be an integer"
            assert row_count >= 0, "Row count should be non-negative"

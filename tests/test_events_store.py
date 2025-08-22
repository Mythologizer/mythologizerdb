"""
Tests for the events store connector.
"""

import pytest
import time
from typing import List, Dict, Any

from mythologizer_postgres.connectors.events_store import (
    insert_event,
    get_next_event,
    set_event_triggered,
)
from mythologizer_postgres.db import clear_all_rows


class TestEventsStore:
    """Test the events store connector functions."""

    def setup_method(self):
        """Clean up before each test method."""
        clear_all_rows()

    def teardown_method(self):
        """Clean up after each test method."""
        clear_all_rows()

    @pytest.mark.integration
    def test_insert_event(self):
        """Test inserting a single event."""
        # Insert an event
        event_id = insert_event("Test event description")
        
        # Verify it was inserted
        assert event_id > 0, "Should return a positive ID"
        
        # Retrieve and verify
        event = get_next_event()
        assert event is not None, "Should find the inserted event"
        assert event['id'] == event_id, "Should return the correct event ID"
        assert event['description'] == "Test event description", "Should return the correct description"
        assert event['has_been_triggered'] == False, "Should not be triggered by default"
        assert event['created_at'] is not None, "Should have a created_at timestamp"

    @pytest.mark.integration
    def test_insert_multiple_events(self):
        """Test inserting multiple events."""
        # Insert multiple events
        event1_id = insert_event("First event")
        event2_id = insert_event("Second event")
        event3_id = insert_event("Third event")
        
        # Verify all events were inserted
        assert event1_id > 0, "First event should have positive ID"
        assert event2_id > 0, "Second event should have positive ID"
        assert event3_id > 0, "Third event should have positive ID"
        assert event1_id != event2_id, "Events should have different IDs"
        assert event2_id != event3_id, "Events should have different IDs"

    @pytest.mark.integration
    def test_get_next_event_ordering(self):
        """Test that get_next_event returns events in chronological order."""
        # Insert events with small delays to ensure different timestamps
        event1_id = insert_event("First event")
        time.sleep(0.1)  # Small delay
        event2_id = insert_event("Second event")
        time.sleep(0.1)  # Small delay
        event3_id = insert_event("Third event")
        
        # Get next event (should be the first one)
        next_event = get_next_event()
        assert next_event is not None, "Should find a next event"
        assert next_event['id'] == event1_id, "Should return the earliest event"
        assert next_event['description'] == "First event", "Should return the first event description"
        
        # Mark first event as triggered
        success = set_event_triggered(event1_id)
        assert success == True, "Should successfully mark event as triggered"
        
        # Get next event (should be the second one)
        next_event = get_next_event()
        assert next_event is not None, "Should find the next event"
        assert next_event['id'] == event2_id, "Should return the second event"
        assert next_event['description'] == "Second event", "Should return the second event description"
        
        # Mark second event as triggered
        success = set_event_triggered(event2_id)
        assert success == True, "Should successfully mark event as triggered"
        
        # Get next event (should be the third one)
        next_event = get_next_event()
        assert next_event is not None, "Should find the third event"
        assert next_event['id'] == event3_id, "Should return the third event"
        assert next_event['description'] == "Third event", "Should return the third event description"

    @pytest.mark.integration
    def test_get_next_event_no_events(self):
        """Test get_next_event when no events exist."""
        # Get next event when no events exist
        next_event = get_next_event()
        assert next_event is None, "Should return None when no events exist"

    @pytest.mark.integration
    def test_get_next_event_all_triggered(self):
        """Test get_next_event when all events are triggered."""
        # Insert events
        event1_id = insert_event("First event")
        event2_id = insert_event("Second event")
        
        # Mark all events as triggered
        success1 = set_event_triggered(event1_id)
        success2 = set_event_triggered(event2_id)
        assert success1 == True, "Should successfully mark first event as triggered"
        assert success2 == True, "Should successfully mark second event as triggered"
        
        # Try to get next event
        next_event = get_next_event()
        assert next_event is None, "Should return None when all events are triggered"

    @pytest.mark.integration
    def test_set_event_triggered_valid_id(self):
        """Test setting an event as triggered with a valid ID."""
        # Insert an event
        event_id = insert_event("Test event")
        
        # Verify it's not triggered initially
        event = get_next_event()
        assert event is not None, "Should find the event"
        assert event['has_been_triggered'] == False, "Event should not be triggered initially"
        
        # Mark as triggered
        success = set_event_triggered(event_id)
        assert success == True, "Should successfully mark event as triggered"
        
        # Verify it's now triggered
        next_event = get_next_event()
        assert next_event is None, "Should not find any untriggered events"

    @pytest.mark.integration
    def test_set_event_triggered_invalid_id(self):
        """Test setting an event as triggered with an invalid ID."""
        # Try to mark a non-existent event as triggered
        success = set_event_triggered(99999)
        assert success == False, "Should return False for non-existent event ID"

    @pytest.mark.integration
    def test_set_event_triggered_already_triggered(self):
        """Test setting an already triggered event as triggered."""
        # Insert an event
        event_id = insert_event("Test event")
        
        # Mark as triggered
        success1 = set_event_triggered(event_id)
        assert success1 == True, "Should successfully mark event as triggered"
        
        # Try to mark as triggered again
        success2 = set_event_triggered(event_id)
        assert success2 == True, "Should still return True for already triggered event"

    @pytest.mark.integration
    def test_event_lifecycle(self):
        """Test the complete lifecycle of an event."""
        # Insert an event
        event_id = insert_event("Lifecycle test event")
        
        # Verify it exists and is not triggered
        event = get_next_event()
        assert event is not None, "Should find the event"
        assert event['id'] == event_id, "Should have the correct ID"
        assert event['has_been_triggered'] == False, "Should not be triggered"
        
        # Mark as triggered
        success = set_event_triggered(event_id)
        assert success == True, "Should successfully mark as triggered"
        
        # Verify it's no longer available as next event
        next_event = get_next_event()
        assert next_event is None, "Should not find any untriggered events"

    @pytest.mark.integration
    def test_multiple_events_processing(self):
        """Test processing multiple events in sequence."""
        # Insert multiple events
        events = []
        for i in range(5):
            event_id = insert_event(f"Event {i+1}")
            events.append(event_id)
        
        # Process events in order
        processed_events = []
        for i in range(5):
            next_event = get_next_event()
            assert next_event is not None, f"Should find event {i+1}"
            assert next_event['description'] == f"Event {i+1}", f"Should be event {i+1}"
            
            # Mark as triggered
            success = set_event_triggered(next_event['id'])
            assert success == True, f"Should successfully mark event {i+1} as triggered"
            
            processed_events.append(next_event['id'])
        
        # Verify all events were processed
        assert len(processed_events) == 5, "Should have processed 5 events"
        assert set(processed_events) == set(events), "Should have processed all events"
        
        # Verify no more events are available
        next_event = get_next_event()
        assert next_event is None, "Should not find any more events"

    @pytest.mark.integration
    def test_event_data_integrity(self):
        """Test that event data is stored and retrieved correctly."""
        # Insert an event with a specific description
        description = "Complex event description with special characters: !@#$%^&*()"
        event_id = insert_event(description)
        
        # Retrieve the event
        event = get_next_event()
        assert event is not None, "Should find the event"
        assert event['id'] == event_id, "Should have the correct ID"
        assert event['description'] == description, "Should preserve the exact description"
        assert event['has_been_triggered'] == False, "Should not be triggered"
        assert event['created_at'] is not None, "Should have a timestamp"
        assert isinstance(event['created_at'], type(event['created_at'])), "Should be a datetime object"

    @pytest.mark.integration
    def test_empty_description_handling(self):
        """Test handling of empty description."""
        # Insert an event with empty description
        event_id = insert_event("")
        
        # Verify it was inserted
        assert event_id > 0, "Should return a positive ID"
        
        # Retrieve and verify
        event = get_next_event()
        assert event is not None, "Should find the event"
        assert event['description'] == "", "Should preserve empty description"

    @pytest.mark.integration
    def test_long_description_handling(self):
        """Test handling of long descriptions."""
        # Create a long description
        long_description = "A" * 1000  # 1000 character description
        
        # Insert an event with long description
        event_id = insert_event(long_description)
        
        # Verify it was inserted
        assert event_id > 0, "Should return a positive ID"
        
        # Retrieve and verify
        event = get_next_event()
        assert event is not None, "Should find the event"
        assert event['description'] == long_description, "Should preserve long description"
        assert len(event['description']) == 1000, "Should have correct length"

    @pytest.mark.integration
    def test_concurrent_event_processing_simulation(self):
        """Test simulation of concurrent event processing."""
        # Insert multiple events
        event_ids = []
        for i in range(10):
            event_id = insert_event(f"Concurrent event {i+1}")
            event_ids.append(event_id)
        
        # Simulate processing events (not truly concurrent, but tests the logic)
        processed_count = 0
        while True:
            next_event = get_next_event()
            if next_event is None:
                break
            
            # Mark as triggered
            success = set_event_triggered(next_event['id'])
            assert success == True, "Should successfully mark event as triggered"
            processed_count += 1
        
        # Verify all events were processed
        assert processed_count == 10, "Should have processed all 10 events"
        
        # Verify no more events are available
        next_event = get_next_event()
        assert next_event is None, "Should not find any more events"

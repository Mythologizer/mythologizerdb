"""
Tests for the culture store connector.
"""

import pytest
import numpy as np
from typing import List, Tuple

from mythologizer_postgres.connectors.culture_store import (
    get_cultures_bulk,
    get_culture,
    insert_culture,
    insert_cultures_bulk,
    update_culture,
    delete_culture,
    get_cultures_by_name,
)


class TestCultureStore:
    """Test the culture store connector functions."""

    @pytest.mark.integration
    def test_insert_culture(self):
        """Test inserting a single culture."""
        # Insert a culture
        culture_id = insert_culture("Greek Mythology", "Ancient Greek myths and legends")
        
        # Verify it was inserted
        assert culture_id > 0, "Should return a positive ID"
        
        # Retrieve and verify
        retrieved_id, name, description = get_culture(culture_id)
        assert retrieved_id == culture_id
        assert name == "Greek Mythology"
        assert description == "Ancient Greek myths and legends"

    @pytest.mark.integration
    def test_insert_cultures_bulk(self):
        """Test inserting multiple cultures in bulk."""
        # Insert multiple cultures
        cultures = [
            ("Norse Mythology", "Norse and Viking myths and legends"),
            ("Egyptian Mythology", "Ancient Egyptian myths and legends"),
            ("Roman Mythology", "Ancient Roman myths and legends")
        ]
        
        culture_ids = insert_cultures_bulk(cultures)
        
        # Verify results
        assert len(culture_ids) == 3, "Should return 3 IDs"
        assert all(cid > 0 for cid in culture_ids), "All IDs should be positive"
        
        # Retrieve and verify
        retrieved_cultures = get_cultures_bulk(culture_ids)
        # Sort both lists to handle order differences
        retrieved_ids = [c[0] for c in retrieved_cultures]
        assert sorted(retrieved_ids) == sorted(culture_ids)
        # Verify names and descriptions match (order might be different due to sorting)
        retrieved_names = [c[1] for c in retrieved_cultures]
        retrieved_descriptions = [c[2] for c in retrieved_cultures]
        expected_names = [c[0] for c in cultures]
        expected_descriptions = [c[1] for c in cultures]
        assert set(retrieved_names) == set(expected_names)
        assert set(retrieved_descriptions) == set(expected_descriptions)

    @pytest.mark.integration
    def test_get_cultures_bulk_all(self):
        """Test getting all cultures."""
        # Insert some test cultures
        cultures = [
            ("Test Culture 1", "Description 1"),
            ("Test Culture 2", "Description 2"),
            ("Test Culture 3", "Description 3")
        ]
        insert_cultures_bulk(cultures)
        
        # Get all cultures
        retrieved_cultures = get_cultures_bulk()
        
        # Verify we got at least our test cultures
        assert len(retrieved_cultures) >= 3, "Should have at least 3 cultures"
        assert all(isinstance(culture[0], int) for culture in retrieved_cultures), "All IDs should be integers"
        assert all(isinstance(culture[1], str) for culture in retrieved_cultures), "All names should be strings"
        assert all(isinstance(culture[2], str) for culture in retrieved_cultures), "All descriptions should be strings"

    @pytest.mark.integration
    def test_get_cultures_bulk_by_ids(self):
        """Test getting cultures by specific IDs."""
        # Insert test cultures
        cultures = [
            ("Specific Culture 1", "Specific Description 1"),
            ("Specific Culture 2", "Specific Description 2")
        ]
        culture_ids = insert_cultures_bulk(cultures)
        
        # Get cultures by IDs
        retrieved_cultures = get_cultures_bulk(culture_ids)
        
        # Verify results
        retrieved_ids = [c[0] for c in retrieved_cultures]
        retrieved_names = [c[1] for c in retrieved_cultures]
        retrieved_descriptions = [c[2] for c in retrieved_cultures]
        expected_names = [c[0] for c in cultures]
        expected_descriptions = [c[1] for c in cultures]
        
        assert retrieved_ids == culture_ids
        assert retrieved_names == expected_names
        assert retrieved_descriptions == expected_descriptions

    @pytest.mark.integration
    def test_get_culture_single(self):
        """Test getting a single culture by ID."""
        # Insert a culture
        culture_id = insert_culture("Single Test Culture", "Single test description")
        
        # Get the culture
        retrieved_id, name, description = get_culture(culture_id)
        
        # Verify results
        assert retrieved_id == culture_id
        assert name == "Single Test Culture"
        assert description == "Single test description"

    @pytest.mark.integration
    def test_get_culture_not_found(self):
        """Test getting a culture that doesn't exist."""
        with pytest.raises(KeyError, match="culture 99999 not found"):
            get_culture(99999)

    @pytest.mark.integration
    def test_update_culture(self):
        """Test updating a culture."""
        # Insert a culture
        culture_id = insert_culture("Original Name", "Original description")
        
        # Update the culture
        success = update_culture(culture_id, name="Updated Name", description="Updated description")
        assert success, "Update should succeed"
        
        # Verify the update
        retrieved_id, name, description = get_culture(culture_id)
        assert name == "Updated Name"
        assert description == "Updated description"

    @pytest.mark.integration
    def test_update_culture_partial(self):
        """Test updating only some fields of a culture."""
        # Insert a culture
        culture_id = insert_culture("Partial Update Test", "Original description")
        
        # Update only the name
        success = update_culture(culture_id, name="Updated Name Only")
        assert success, "Update should succeed"
        
        # Verify the update
        retrieved_id, name, description = get_culture(culture_id)
        assert name == "Updated Name Only"
        assert description == "Original description"  # Should be unchanged

    @pytest.mark.integration
    def test_update_culture_not_found(self):
        """Test updating a culture that doesn't exist."""
        success = update_culture(99999, name="New Name")
        assert not success, "Update should fail for non-existent culture"

    @pytest.mark.integration
    def test_delete_culture(self):
        """Test deleting a culture."""
        # Insert a culture
        culture_id = insert_culture("To Delete", "Will be deleted")
        
        # Delete the culture
        success = delete_culture(culture_id)
        assert success, "Delete should succeed"
        
        # Verify it's gone
        with pytest.raises(KeyError):
            get_culture(culture_id)

    @pytest.mark.integration
    def test_delete_culture_not_found(self):
        """Test deleting a culture that doesn't exist."""
        success = delete_culture(99999)
        assert not success, "Delete should fail for non-existent culture"

    @pytest.mark.integration
    def test_get_cultures_by_name_exact(self):
        """Test searching for cultures by exact name match."""
        # Use a unique name to avoid conflicts with previous test runs
        unique_name = f"Exact Match Test {np.random.randint(10000)}"
        
        # Insert test cultures
        insert_culture(unique_name, "Description")
        insert_culture("Another Culture", "Another description")
        
        # Search for exact match
        cultures = get_cultures_by_name(unique_name, exact_match=True)
        
        # Verify results
        assert len(cultures) == 1, "Should find exactly one culture"
        assert cultures[0][1] == unique_name
        assert cultures[0][2] == "Description"

    @pytest.mark.integration
    def test_get_cultures_by_name_pattern(self):
        """Test searching for cultures by name pattern."""
        # Insert test cultures
        insert_culture("Greek Mythology", "Greek myths")
        insert_culture("Roman Mythology", "Roman myths")
        insert_culture("Egyptian Culture", "Egyptian culture")
        
        # Search for cultures containing "Mythology"
        cultures = get_cultures_by_name("Mythology", exact_match=False)
        
        # Verify results
        assert len(cultures) >= 2, "Should find at least 2 cultures with 'Mythology'"
        names = [culture[1] for culture in cultures]
        assert all("Mythology" in name for name in names), "All names should contain 'Mythology'"

    @pytest.mark.integration
    def test_get_cultures_by_name_case_insensitive(self):
        """Test that name search is case insensitive."""
        # Insert test culture
        insert_culture("Case Test Culture", "Description")
        
        # Search with different cases
        cultures1 = get_cultures_by_name("case test", exact_match=False)
        cultures2 = get_cultures_by_name("CASE TEST", exact_match=False)
        
        # Both searches should return the same result
        ids1 = [c[0] for c in cultures1]
        ids2 = [c[0] for c in cultures2]
        names1 = [c[1] for c in cultures1]
        names2 = [c[1] for c in cultures2]
        
        assert ids1 == ids2, "Case insensitive search should return same results"
        assert names1 == names2, "Case insensitive search should return same results"

    @pytest.mark.integration
    def test_insert_cultures_bulk_validation(self):
        """Test that bulk insert validates input format."""
        # Test with invalid input (not tuples)
        invalid_input = ["Name 1", "Name 2"]  # Not tuples
        
        with pytest.raises(ValueError):
            insert_cultures_bulk(invalid_input)

    @pytest.mark.integration
    def test_empty_results(self):
        """Test handling of empty results."""
        # Test getting cultures by non-existent IDs
        cultures = get_cultures_bulk([99999, 99998])
        assert cultures == []
        
        # Test empty name search
        cultures = get_cultures_by_name("NonExistentCulture")
        assert cultures == []

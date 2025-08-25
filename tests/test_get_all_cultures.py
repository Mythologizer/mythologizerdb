"""
Tests for the get_all_cultures function.
"""

import pytest
from mythologizer_postgres.connectors import get_all_cultures, insert_culture, delete_culture
from mythologizer_postgres.db import get_engine
from sqlalchemy import text


class TestGetAllCultures:
    """Test the get_all_cultures function."""
    
    def test_get_all_cultures_empty_database(self):
        """Test getting all cultures when database is empty."""
        cultures = get_all_cultures()
        assert isinstance(cultures, list)
        assert len(cultures) == 0
    
    def test_get_all_cultures_with_data(self):
        """Test getting all cultures when there are cultures in the database."""
        # Insert some test cultures
        culture_ids = []
        test_cultures = [
            ("Greek Mythology", "Ancient Greek myths and legends"),
            ("Norse Mythology", "Norse gods and legends"),
            ("Egyptian Mythology", "Ancient Egyptian beliefs and stories"),
        ]
        
        for name, description in test_cultures:
            culture_id = insert_culture(name, description)
            culture_ids.append(culture_id)
        
        try:
            # Get all cultures
            cultures = get_all_cultures()
            
            # Verify results
            assert isinstance(cultures, list)
            assert len(cultures) == 3
            
            # Check that all cultures are returned
            culture_names = [culture[1] for culture in cultures]  # name is at index 1
            expected_names = ["Egyptian Mythology", "Greek Mythology", "Norse Mythology"]  # ordered by name
            
            assert culture_names == expected_names
            
            # Check structure of returned data
            for culture in cultures:
                assert len(culture) == 3  # (id, name, description)
                assert isinstance(culture[0], int)  # id
                assert isinstance(culture[1], str)  # name
                assert isinstance(culture[2], str)  # description
        
        finally:
            # Clean up
            for culture_id in culture_ids:
                delete_culture(culture_id)
    
    def test_get_all_cultures_ordering(self):
        """Test that cultures are returned in alphabetical order by name."""
        # Insert cultures in non-alphabetical order
        culture_ids = []
        test_cultures = [
            ("Zulu Mythology", "Zulu traditions and stories"),
            ("Aztec Mythology", "Aztec gods and legends"),
            ("Chinese Mythology", "Chinese folklore and myths"),
        ]
        
        for name, description in test_cultures:
            culture_id = insert_culture(name, description)
            culture_ids.append(culture_id)
        
        try:
            # Get all cultures
            cultures = get_all_cultures()
            
            # Verify alphabetical ordering
            culture_names = [culture[1] for culture in cultures]
            expected_names = ["Aztec Mythology", "Chinese Mythology", "Zulu Mythology"]
            
            assert culture_names == expected_names
        
        finally:
            # Clean up
            for culture_id in culture_ids:
                delete_culture(culture_id)
    
    def test_get_all_cultures_with_existing_data(self):
        """Test that get_all_cultures works with existing data in the database."""
        # First, check how many cultures exist
        initial_cultures = get_all_cultures()
        initial_count = len(initial_cultures)
        
        # Add a test culture
        test_culture_id = insert_culture("Test Culture", "A test culture for testing")
        
        try:
            # Get all cultures again
            cultures = get_all_cultures()
            
            # Should have one more culture
            assert len(cultures) == initial_count + 1
            
            # Should find our test culture
            test_culture_names = [culture[1] for culture in cultures]
            assert "Test Culture" in test_culture_names
        
        finally:
            # Clean up
            delete_culture(test_culture_id)
    
    def test_get_all_cultures_return_format(self):
        """Test that the return format is correct."""
        # Insert a test culture
        test_culture_id = insert_culture("Format Test", "Testing return format")
        
        try:
            cultures = get_all_cultures()
            
            # Find our test culture
            test_culture = None
            for culture in cultures:
                if culture[1] == "Format Test":
                    test_culture = culture
                    break
            
            assert test_culture is not None
            assert len(test_culture) == 3
            
            # Check types
            culture_id, name, description = test_culture
            assert isinstance(culture_id, int)
            assert isinstance(name, str)
            assert isinstance(description, str)
            
            # Check values
            assert name == "Format Test"
            assert description == "Testing return format"
        
        finally:
            # Clean up
            delete_culture(test_culture_id)

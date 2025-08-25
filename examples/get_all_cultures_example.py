#!/usr/bin/env python3
"""
Example demonstrating the get_all_cultures function.
"""

from mythologizer_postgres.connectors import get_all_cultures, insert_culture, delete_culture


def main():
    """Example usage of get_all_cultures."""
    print("=== Get All Cultures Example ===\n")
    
    # Get all cultures (initially empty)
    print("Initial cultures:")
    cultures = get_all_cultures()
    print(f"Found {len(cultures)} cultures")
    
    # Add some test cultures
    print("\nAdding test cultures...")
    culture_ids = []
    test_cultures = [
        ("Greek Mythology", "Ancient Greek myths and legends"),
        ("Norse Mythology", "Norse gods and legends"),
        ("Egyptian Mythology", "Ancient Egyptian beliefs and stories"),
    ]
    
    for name, description in test_cultures:
        culture_id = insert_culture(name, description)
        culture_ids.append(culture_id)
        print(f"Added: {name}")
    
    # Get all cultures again
    print("\nAll cultures after adding:")
    cultures = get_all_cultures()
    print(f"Found {len(cultures)} cultures:")
    
    for culture_id, name, description in cultures:
        print(f"  ID: {culture_id}, Name: {name}, Description: {description}")
    
    # Note: cultures are ordered alphabetically by name
    print("\nNote: Cultures are automatically ordered alphabetically by name")
    
    # Clean up
    print("\nCleaning up...")
    for culture_id in culture_ids:
        delete_culture(culture_id)
        print(f"Deleted culture {culture_id}")
    
    # Verify cleanup
    cultures = get_all_cultures()
    print(f"\nAfter cleanup: {len(cultures)} cultures remaining")


if __name__ == "__main__":
    main()

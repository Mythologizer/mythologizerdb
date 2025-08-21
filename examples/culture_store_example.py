#!/usr/bin/env python3
"""
Example script demonstrating how to use the culture store connector.

This script shows how to:
- Insert single and bulk cultures
- Retrieve cultures by ID and in bulk
- Update and delete cultures
- Search cultures by name
"""

from mythologizer_postgres.connectors.culture_store import (
    insert_culture,
    insert_cultures_bulk,
    get_culture,
    get_cultures_bulk,
    update_culture,
    delete_culture,
    get_cultures_by_name,
)


def main():
    """Demonstrate culture store operations."""
    print("=== Culture Store Example ===\n")
    
    # 1. Insert single cultures
    print("1. Inserting single cultures...")
    greek_id = insert_culture("Greek Mythology", "Ancient Greek myths and legends")
    norse_id = insert_culture("Norse Mythology", "Norse and Viking myths and legends")
    print(f"   Inserted Greek Mythology with ID: {greek_id}")
    print(f"   Inserted Norse Mythology with ID: {norse_id}\n")
    
    # 2. Insert cultures in bulk
    print("2. Inserting cultures in bulk...")
    cultures = [
        ("Egyptian Mythology", "Ancient Egyptian myths and legends"),
        ("Roman Mythology", "Ancient Roman myths and legends"),
        ("Celtic Mythology", "Celtic myths and legends")
    ]
    bulk_ids = insert_cultures_bulk(cultures)
    print(f"   Inserted {len(bulk_ids)} cultures with IDs: {bulk_ids}\n")
    
    # 3. Get single culture
    print("3. Getting single culture...")
    culture_id, name, description = get_culture(greek_id)
    print(f"   Culture {culture_id}: {name} - {description}\n")
    
    # 4. Get all cultures
    print("4. Getting all cultures...")
    all_cultures = get_cultures_bulk()
    print(f"   Found {len(all_cultures)} cultures:")
    for i, (cid, cname, cdesc) in enumerate(all_cultures):
        print(f"   {i+1}. {cname} (ID: {cid}) - {cdesc}")
    print()
    
    # 5. Get specific cultures by IDs
    print("5. Getting specific cultures by IDs...")
    specific_ids = [greek_id, norse_id]
    specific_cultures = get_cultures_bulk(specific_ids)
    print(f"   Retrieved {len(specific_cultures)} specific cultures:")
    for cid, cname, cdesc in specific_cultures:
        print(f"   - {cname} (ID: {cid}) - {cdesc}")
    print()
    
    # 6. Search cultures by name (exact match)
    print("6. Searching cultures by exact name...")
    exact_cultures = get_cultures_by_name("Greek Mythology", exact_match=True)
    print(f"   Found {len(exact_cultures)} cultures with exact name 'Greek Mythology':")
    for cid, cname, cdesc in exact_cultures:
        print(f"   - {cname} (ID: {cid}) - {cdesc}")
    print()
    
    # 7. Search cultures by name pattern
    print("7. Searching cultures by name pattern...")
    pattern_cultures = get_cultures_by_name("Mythology", exact_match=False)
    print(f"   Found {len(pattern_cultures)} cultures containing 'Mythology':")
    for cid, cname, cdesc in pattern_cultures:
        print(f"   - {cname} (ID: {cid}) - {cdesc}")
    print()
    
    # 8. Update a culture
    print("8. Updating a culture...")
    success = update_culture(greek_id, description="Updated: Ancient Greek myths and legends with gods and heroes")
    if success:
        updated_id, updated_name, updated_description = get_culture(greek_id)
        print(f"   Updated culture {updated_id}: {updated_name} - {updated_description}")
    else:
        print("   Failed to update culture")
    print()
    
    # 9. Update only the name
    print("9. Updating only the name...")
    success = update_culture(norse_id, name="Norse and Viking Mythology")
    if success:
        updated_id, updated_name, updated_description = get_culture(norse_id)
        print(f"   Updated culture {updated_id}: {updated_name} - {updated_description}")
    else:
        print("   Failed to update culture")
    print()
    
    # 10. Delete a culture
    print("10. Deleting a culture...")
    # First, let's get a culture to delete
    all_cultures = get_cultures_bulk()
    if all_cultures:
        culture_to_delete = all_cultures[0][0]
        culture_name = all_cultures[0][1]
        success = delete_culture(culture_to_delete)
        if success:
            print(f"   Successfully deleted culture: {culture_name} (ID: {culture_to_delete})")
        else:
            print(f"   Failed to delete culture: {culture_name} (ID: {culture_to_delete})")
    else:
        print("   No cultures available to delete")
    print()
    
    # 11. Final count
    print("11. Final culture count...")
    final_cultures = get_cultures_bulk()
    print(f"   Total cultures remaining: {len(final_cultures)}")
    
    print("\n=== Example completed successfully! ===")


if __name__ == "__main__":
    main()

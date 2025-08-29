#!/usr/bin/env python3
"""
Script to query mythemes from the database and create 30 random myths.

This script demonstrates:
1. Querying all mythemes from the database
2. Creating 30 random myths using the mythemes as building blocks
3. Using the mythic algebra connector to compose myths from mythemes
"""

import os
import numpy as np
import random
from typing import List, Tuple
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

from mythologizer_postgres.connectors import (
    get_mythemes_bulk,
    insert_myths_bulk,
    get_myth_embeddings,
)
from mythologizer_postgres.connectors.mythicalgebra import (
    recalc_and_update_myths,
    compose_myth_matrix,
)


def get_embedding_dim():
    """Get the embedding dimension - hardcoded to 384 for now."""
    return 384


def create_random_myth_from_mythemes(
    mytheme_ids: List[int],
    mytheme_sentences: List[str],
    mytheme_embeddings: np.ndarray,
    embedding_dim: int
) -> Tuple[np.ndarray, List[int], List[np.ndarray], List[float]]:
    """
    Create a random myth by combining multiple mythemes.
    
    Args:
        mytheme_ids: List of mytheme IDs
        mytheme_sentences: List of mytheme sentences
        mytheme_embeddings: Array of mytheme embeddings
        embedding_dim: Dimension of embeddings
        
    Returns:
        Tuple of (myth_embedding, embedding_ids, offsets, weights)
    """
    # Randomly select 2-4 mythemes to combine
    num_mythemes = random.randint(2, min(4, len(mytheme_ids)))
    selected_indices = random.sample(range(len(mytheme_ids)), num_mythemes)
    
    # Get selected mythemes
    selected_ids = [mytheme_ids[i] for i in selected_indices]
    selected_embeddings = mytheme_embeddings[selected_indices]
    
    # Create random weights for the combination
    weights = [random.uniform(0.1, 1.0) for _ in range(num_mythemes)]
    # Normalize weights
    weight_sum = sum(weights)
    weights = [w / weight_sum for w in weights]
    
    # Create offsets (small random variations)
    offsets = []
    for _ in range(num_mythemes):
        offset = np.random.normal(0, 0.1, embedding_dim)
        offsets.append(offset.tolist())
    
    # Combine embeddings using weighted average
    myth_embedding = np.zeros(embedding_dim)
    for i, (embedding, weight) in enumerate(zip(selected_embeddings, weights)):
        myth_embedding += embedding * weight
    
    # Add some random variation
    myth_embedding += np.random.normal(0, 0.05, embedding_dim)
    
    return myth_embedding.tolist(), selected_ids, offsets, weights


def main():
    """Main function to query mythemes and create random myths."""
    print("=== Creating Random Myths from Mythemes ===\n")
    
    # Get embedding dimension
    embedding_dim = get_embedding_dim()
    print(f"Using embedding dimension: {embedding_dim}")
    
    # Query all mythemes from the database
    print("\n1. Querying mythemes from database...")
    mytheme_ids, mytheme_sentences, mytheme_embeddings = get_mythemes_bulk()
    
    print(f"Found {len(mytheme_ids)} mythemes in the database")
    
    if len(mytheme_ids) == 0:
        print("No mythemes found in the database. Please add some mythemes first.")
        return
    
    # Display some sample mythemes
    print("\nSample mythemes:")
    for i in range(min(5, len(mytheme_ids))):
        print(f"  ID {mytheme_ids[i]}: {mytheme_sentences[i][:80]}...")
    
    # Create 30 random myths
    print(f"\n2. Creating 30 random myths...")
    
    myth_embeddings = []
    myth_embedding_ids = []
    myth_offsets = []
    myth_weights = []
    
    for i in range(30):
        print(f"  Creating myth {i+1}/30...")
        
        embedding, embedding_ids, offsets, weights = create_random_myth_from_mythemes(
            mytheme_ids, mytheme_sentences, mytheme_embeddings, embedding_dim
        )
        
        myth_embeddings.append(embedding)
        myth_embedding_ids.append(embedding_ids)
        myth_offsets.append(offsets)
        myth_weights.append(weights)
    
    # Insert the myths into the database
    print("\n3. Inserting myths into database...")
    
    # Convert to the format expected by insert_myths_bulk
    # insert_myths_bulk expects: main_embeddings, embedding_ids_list, offsets_list, weights_list
    insert_myths_bulk(
        myth_embeddings,
        myth_embedding_ids,
        myth_offsets,
        myth_weights
    )
    
    print("✅ Successfully created and inserted 30 random myths!")
    
    # Display some information about the created myths
    print("\n4. Myth creation summary:")
    print(f"  - Total myths created: 30")
    print(f"  - Embedding dimension: {embedding_dim}")
    print(f"  - Source mythemes: {len(mytheme_ids)}")
    
    # Show some details about the first few myths
    print("\n5. Sample of created myths:")
    for i in range(min(3, 30)):
        print(f"  Myth {i+1}:")
        print(f"    - Uses {len(myth_embedding_ids[i])} mythemes")
        print(f"    - Mytheme IDs: {myth_embedding_ids[i]}")
        print(f"    - Weights: {[f'{w:.2f}' for w in myth_weights[i]]}")
        print()
    
    print("=== Myth Creation Complete ===")


if __name__ == "__main__":
    main()

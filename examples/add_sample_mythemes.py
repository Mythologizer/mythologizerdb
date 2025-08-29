#!/usr/bin/env python3
"""
Script to add sample mythemes to the database.

This script creates sample mythemes that can be used to generate random myths.
"""

import numpy as np
from mythologizer_postgres.connectors import insert_mythemes_bulk


def get_embedding_dim():
    """Get the embedding dimension from environment variable."""
    import os
    return int(os.getenv('EMBEDDING_DIM', '4'))


def main():
    """Add sample mythemes to the database."""
    print("=== Adding Sample Mythemes ===\n")
    
    # Get embedding dimension
    embedding_dim = get_embedding_dim()
    print(f"Using embedding dimension: {embedding_dim}")
    
    # Sample mythemes from various mythologies
    sample_mythemes = [
        "A hero embarks on a dangerous journey to prove their worth",
        "A powerful being creates the world from chaos",
        "A trickster figure outsmarts the gods",
        "A mortal falls in love with an immortal being",
        "A great flood destroys the world and forces rebirth",
        "A hero must retrieve a sacred object from the underworld",
        "A prophecy foretells the downfall of a great kingdom",
        "A magical weapon grants its wielder immense power",
        "A sacrifice is made to appease the gods",
        "A hero battles a monstrous creature to save their people",
        "A divine being takes human form to walk among mortals",
        "A curse transforms a person into a different form",
        "A hero must solve impossible riddles to succeed",
        "A magical tree or plant grants eternal life",
        "A hero must cross a dangerous bridge or river",
        "A divine messenger brings important news to mortals",
        "A hero must resist the temptation of forbidden knowledge",
        "A great battle between good and evil forces",
        "A hero must find and return a stolen treasure",
        "A magical transformation changes the course of destiny",
        "A hero must prove their lineage to claim their birthright",
        "A divine punishment teaches mortals a lesson",
        "A hero must overcome their own inner demons",
        "A magical artifact contains the essence of a god",
        "A hero must navigate a labyrinth to reach their goal",
        "A prophecy reveals the true identity of a hidden hero",
        "A divine marriage unites different realms or peoples",
        "A hero must break a powerful enchantment",
        "A magical animal guides the hero on their quest",
        "A hero must face their greatest fear to succeed",
    ]
    
    print(f"Creating {len(sample_mythemes)} sample mythemes...")
    
    # Generate random embeddings for each mytheme
    embeddings = []
    for i in range(len(sample_mythemes)):
        # Create a random embedding vector
        embedding = np.random.rand(embedding_dim).tolist()
        embeddings.append(embedding)
        print(f"  Created mytheme {i+1}: {sample_mythemes[i][:50]}...")
    
    # Insert mythemes into the database
    print("\nInserting mythemes into database...")
    insert_mythemes_bulk(sample_mythemes, embeddings)
    
    print("✅ Successfully added sample mythemes to the database!")
    print(f"\nSummary:")
    print(f"  - Mythemes added: {len(sample_mythemes)}")
    print(f"  - Embedding dimension: {embedding_dim}")
    
    print("\n=== Sample Mythemes Added ===")


if __name__ == "__main__":
    main()




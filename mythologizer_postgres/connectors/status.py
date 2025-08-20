from sqlalchemy import text
from typing import Optional

from mythologizer_postgres.db import session_scope


def get_current_epoch() -> int:
    """
    Get the current epoch number.
    
    Returns:
        int: The current epoch number (>= 0)
    """
    with session_scope() as session:
        result = session.execute(text("SELECT current_epoch FROM epoch WHERE key = 'only'"))
        row = result.fetchone()
        if row is None:
            raise ValueError("Epoch record not found. Database may not be properly initialized.")
        return row[0]


def increment_epoch() -> int:
    """
    Increment the current epoch by 1.
    
    Returns:
        int: The new epoch number
    """
    with session_scope() as session:
        # Get current epoch
        current = get_current_epoch()
        new_epoch = current + 1
        
        # Update to new epoch
        session.execute(text("UPDATE epoch SET current_epoch = :new_epoch WHERE key = 'only'"), {
            'new_epoch': new_epoch
        })
        session.commit()
        
        return new_epoch


def get_n_agents() -> int:
    """
    Get the number of agents in the simulation.
    
    Returns:
        int: Number of agents
    """
    with session_scope() as session:
        result = session.execute(text("SELECT COUNT(*) FROM agents"))
        return result.fetchone()[0]


def get_n_myths() -> int:
    """
    Get the number of myths in the simulation.
    
    Returns:
        int: Number of myths
    """
    with session_scope() as session:
        result = session.execute(text("SELECT COUNT(*) FROM myths"))
        return result.fetchone()[0]


def get_n_cultures() -> int:
    """
    Get the number of cultures in the simulation.
    
    Returns:
        int: Number of cultures
    """
    with session_scope() as session:
        result = session.execute(text("SELECT COUNT(*) FROM cultures"))
        return result.fetchone()[0]


def get_simulation_status() -> dict:
    """
    Get a comprehensive status of the simulation.
    
    Returns:
        dict: Dictionary containing current epoch and counts of agents, myths, and cultures
    """
    return {
        'current_epoch': get_current_epoch(),
        'n_agents': get_n_agents(),
        'n_myths': get_n_myths(),
        'n_cultures': get_n_cultures()
    }


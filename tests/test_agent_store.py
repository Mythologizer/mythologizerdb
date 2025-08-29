"""
Tests for the agent store connector.
"""

import pytest
from typing import List, Dict, Any, Tuple
from sqlalchemy import text

from mythologizer_postgres.connectors.agent_store import get_agents_bulk, get_agent_cultures, get_agents_cultures_ids_bulk
from mythologizer_postgres.db import clear_all_rows, session_scope


class TestAgentStore:
    """Test the agent store connector functions."""

    def setup_method(self):
        """Clean up before each test method."""
        clear_all_rows()

    def teardown_method(self):
        """Clean up after each test method."""
        clear_all_rows()

    def _create_test_agents(self, agent_data: List[Dict[str, Any]]) -> List[int]:
        """
        Helper method to create test agents.
        
        Args:
            agent_data: List of dictionaries with 'name' and 'memory_size' keys
            
        Returns:
            List of created agent IDs
        """
        agent_ids = []
        with session_scope() as session:
            for data in agent_data:
                result = session.execute(text("""
                    INSERT INTO agents (name, memory_size)
                    VALUES (:name, :memory_size)
                    RETURNING id
                """), data)
                agent_ids.append(result.fetchone()[0])
        return agent_ids

    def _create_test_cultures(self, culture_data: List[Dict[str, Any]]) -> List[int]:
        """
        Helper method to create test cultures.
        
        Args:
            culture_data: List of dictionaries with 'name' and 'description' keys
            
        Returns:
            List of created culture IDs
        """
        culture_ids = []
        with session_scope() as session:
            for data in culture_data:
                result = session.execute(text("""
                    INSERT INTO cultures (name, description)
                    VALUES (:name, :description)
                    RETURNING id
                """), data)
                culture_ids.append(result.fetchone()[0])
        return culture_ids

    def _create_agent_culture_relationships(self, relationships: List[Tuple[int, int]]):
        """
        Helper method to create agent-culture relationships.
        
        Args:
            relationships: List of tuples (agent_id, culture_id)
        """
        with session_scope() as session:
            for agent_id, culture_id in relationships:
                session.execute(text("""
                    INSERT INTO agent_cultures (agent_id, culture_id)
                    VALUES (:agent_id, :culture_id)
                """), {'agent_id': agent_id, 'culture_id': culture_id})

    @pytest.mark.integration
    def test_get_agents_bulk_empty_list(self):
        """Test get_agents_bulk with empty list."""
        result = get_agents_bulk([])
        assert result == [], "Should return empty list for empty input"

    @pytest.mark.integration
    def test_get_agents_bulk_single_agent(self):
        """Test get_agents_bulk with single agent."""
        # Create a test agent
        agent_ids = self._create_test_agents([
            {'name': 'Test Agent', 'memory_size': 5}
        ])
        
        # Get the agent in bulk
        result = get_agents_bulk(agent_ids)
        
        # Verify result
        assert len(result) == 1, "Should return exactly one agent"
        assert result[0]['id'] == agent_ids[0], "Should return correct agent ID"
        assert result[0]['name'] == 'Test Agent', "Should return correct agent name"
        assert result[0]['memory_size'] == 5, "Should return correct memory size"

    @pytest.mark.integration
    def test_get_agents_bulk_multiple_agents(self):
        """Test get_agents_bulk with multiple agents."""
        # Create multiple test agents
        agent_ids = self._create_test_agents([
            {'name': 'Agent 1', 'memory_size': 5},
            {'name': 'Agent 2', 'memory_size': 10},
            {'name': 'Agent 3', 'memory_size': 15}
        ])
        
        # Get agents in bulk
        result = get_agents_bulk(agent_ids)
        
        # Verify result
        assert len(result) == 3, "Should return exactly three agents"
        
        # Verify all agents are returned with correct data
        for i, agent in enumerate(result):
            assert agent['id'] == agent_ids[i], f"Agent {i} should have correct ID"
            assert agent['name'] == f'Agent {i+1}', f"Agent {i} should have correct name"
            assert agent['memory_size'] == 5 + (i * 5), f"Agent {i} should have correct memory size"

    @pytest.mark.integration
    def test_get_agents_bulk_order_preservation(self):
        """Test that get_agents_bulk preserves the order of input IDs."""
        # Create test agents
        agent_ids = self._create_test_agents([
            {'name': 'Agent 1', 'memory_size': 5},
            {'name': 'Agent 2', 'memory_size': 10},
            {'name': 'Agent 3', 'memory_size': 15}
        ])
        
        # Request agents in reverse order
        reversed_ids = list(reversed(agent_ids))
        result = get_agents_bulk(reversed_ids)
        
        # Verify order is preserved
        assert len(result) == 3, "Should return exactly three agents"
        for i, agent in enumerate(result):
            assert agent['id'] == reversed_ids[i], f"Agent {i} should match reversed order"

    @pytest.mark.integration
    def test_get_agents_bulk_nonexistent_agents(self):
        """Test get_agents_bulk with nonexistent agent IDs."""
        # Create one test agent
        agent_ids = self._create_test_agents([
            {'name': 'Test Agent', 'memory_size': 5}
        ])
        
        # Request existing and nonexistent agents
        request_ids = [agent_ids[0], 999, 1000]
        result = get_agents_bulk(request_ids)
        
        # Should only return the existing agent
        assert len(result) == 1, "Should return only existing agent"
        assert result[0]['id'] == agent_ids[0], "Should return the existing agent"

    @pytest.mark.integration
    def test_get_agents_bulk_all_nonexistent(self):
        """Test get_agents_bulk with all nonexistent agent IDs."""
        result = get_agents_bulk([999, 1000, 1001])
        
        # Should return empty list
        assert result == [], "Should return empty list for nonexistent agents"

    @pytest.mark.integration
    def test_get_agents_bulk_duplicate_ids(self):
        """Test get_agents_bulk with duplicate agent IDs."""
        # Create test agents
        agent_ids = self._create_test_agents([
            {'name': 'Agent 1', 'memory_size': 5},
            {'name': 'Agent 2', 'memory_size': 10}
        ])
        
        # Request with duplicates
        request_ids = [agent_ids[0], agent_ids[1], agent_ids[0], agent_ids[1]]
        result = get_agents_bulk(request_ids)
        
        # Should return agents in the requested order (including duplicates)
        assert len(result) == 4, "Should return exactly four agents (including duplicates)"
        assert result[0]['id'] == agent_ids[0], "First agent should match first request"
        assert result[1]['id'] == agent_ids[1], "Second agent should match second request"
        assert result[2]['id'] == agent_ids[0], "Third agent should match third request (duplicate)"
        assert result[3]['id'] == agent_ids[1], "Fourth agent should match fourth request (duplicate)"

    @pytest.mark.integration
    def test_get_agents_bulk_large_number(self):
        """Test get_agents_bulk with a large number of agents."""
        # Create many test agents
        agent_data = [
            {'name': f'Agent {i}', 'memory_size': 5 + i}
            for i in range(50)
        ]
        agent_ids = self._create_test_agents(agent_data)
        
        # Get all agents in bulk
        result = get_agents_bulk(agent_ids)
        
        # Verify all agents are returned
        assert len(result) == 50, "Should return exactly 50 agents"
        
        # Verify data integrity
        for i, agent in enumerate(result):
            assert agent['id'] == agent_ids[i], f"Agent {i} should have correct ID"
            assert agent['name'] == f'Agent {i}', f"Agent {i} should have correct name"
            assert agent['memory_size'] == 5 + i, f"Agent {i} should have correct memory size"

    @pytest.mark.integration
    def test_get_agents_bulk_mixed_existing_nonexistent(self):
        """Test get_agents_bulk with mix of existing and nonexistent agents."""
        # Create test agents
        agent_ids = self._create_test_agents([
            {'name': 'Agent 1', 'memory_size': 5},
            {'name': 'Agent 2', 'memory_size': 10}
        ])
        
        # Request mix of existing and nonexistent
        request_ids = [999, agent_ids[0], 1000, agent_ids[1], 1001]
        result = get_agents_bulk(request_ids)
        
        # Should return only existing agents in order
        assert len(result) == 2, "Should return only existing agents"
        assert result[0]['id'] == agent_ids[0], "First result should be first existing agent"
        assert result[1]['id'] == agent_ids[1], "Second result should be second existing agent"

    @pytest.mark.integration
    def test_get_agents_bulk_data_structure(self):
        """Test that get_agents_bulk returns correct data structure."""
        # Create a test agent
        agent_ids = self._create_test_agents([
            {'name': 'Test Agent', 'memory_size': 5}
        ])
        
        # Get the agent
        result = get_agents_bulk(agent_ids)
        
        # Verify data structure
        assert len(result) == 1, "Should return one agent"
        agent = result[0]
        
        # Check required keys exist
        assert 'id' in agent, "Agent should have 'id' key"
        assert 'name' in agent, "Agent should have 'name' key"
        assert 'memory_size' in agent, "Agent should have 'memory_size' key"
        
        # Check data types
        assert isinstance(agent['id'], int), "Agent ID should be integer"
        assert isinstance(agent['name'], str), "Agent name should be string"
        assert isinstance(agent['memory_size'], int), "Agent memory_size should be integer"
        
        # Check values
        assert agent['id'] == agent_ids[0], "Agent ID should match"
        assert agent['name'] == 'Test Agent', "Agent name should match"
        assert agent['memory_size'] == 5, "Agent memory_size should match"

    # Tests for get_agent_cultures function
    @pytest.mark.integration
    def test_get_agent_cultures_empty(self):
        """Test get_agent_cultures for agent with no cultures."""
        # Create a test agent
        agent_ids = self._create_test_agents([
            {'name': 'Test Agent', 'memory_size': 5}
        ])
        
        # Get cultures for the agent
        result = get_agent_cultures(agent_ids[0])
        
        # Should return empty list
        assert result == [], "Should return empty list for agent with no cultures"

    @pytest.mark.integration
    def test_get_agent_cultures_single_culture(self):
        """Test get_agent_cultures for agent with single culture."""
        # Create test agent and culture
        agent_ids = self._create_test_agents([
            {'name': 'Test Agent', 'memory_size': 5}
        ])
        culture_ids = self._create_test_cultures([
            {'name': 'Test Culture', 'description': 'A test culture'}
        ])
        
        # Create relationship
        self._create_agent_culture_relationships([(agent_ids[0], culture_ids[0])])
        
        # Get cultures for the agent
        result = get_agent_cultures(agent_ids[0])
        
        # Verify result
        assert len(result) == 1, "Should return exactly one culture"
        assert result[0][0] == culture_ids[0], "Should return correct culture ID"
        assert result[0][1] == 'Test Culture', "Should return correct culture name"
        assert result[0][2] == 'A test culture', "Should return correct culture description"

    @pytest.mark.integration
    def test_get_agent_cultures_multiple_cultures(self):
        """Test get_agent_cultures for agent with multiple cultures."""
        # Create test agent and cultures
        agent_ids = self._create_test_agents([
            {'name': 'Test Agent', 'memory_size': 5}
        ])
        culture_ids = self._create_test_cultures([
            {'name': 'Culture 1', 'description': 'First culture'},
            {'name': 'Culture 2', 'description': 'Second culture'},
            {'name': 'Culture 3', 'description': 'Third culture'}
        ])
        
        # Create relationships
        relationships = [(agent_ids[0], culture_ids[0]), 
                        (agent_ids[0], culture_ids[1]), 
                        (agent_ids[0], culture_ids[2])]
        self._create_agent_culture_relationships(relationships)
        
        # Get cultures for the agent
        result = get_agent_cultures(agent_ids[0])
        
        # Verify result
        assert len(result) == 3, "Should return exactly three cultures"
        
        # Verify all cultures are returned with correct data
        for i, culture in enumerate(result):
            assert culture[0] == culture_ids[i], f"Culture {i} should have correct ID"
            assert culture[1] == f'Culture {i+1}', f"Culture {i} should have correct name"
            assert culture[2] == f'{["First", "Second", "Third"][i]} culture', f"Culture {i} should have correct description"

    @pytest.mark.integration
    def test_get_agent_cultures_nonexistent_agent(self):
        """Test get_agent_cultures for nonexistent agent."""
        result = get_agent_cultures(999)
        
        # Should return empty list
        assert result == [], "Should return empty list for nonexistent agent"

    @pytest.mark.integration
    def test_get_agent_cultures_data_structure(self):
        """Test that get_agent_cultures returns correct data structure."""
        # Create test agent and culture
        agent_ids = self._create_test_agents([
            {'name': 'Test Agent', 'memory_size': 5}
        ])
        culture_ids = self._create_test_cultures([
            {'name': 'Test Culture', 'description': 'A test culture'}
        ])
        
        # Create relationship
        self._create_agent_culture_relationships([(agent_ids[0], culture_ids[0])])
        
        # Get cultures for the agent
        result = get_agent_cultures(agent_ids[0])
        
        # Verify data structure
        assert len(result) == 1, "Should return one culture"
        culture = result[0]
        
        # Check data types
        assert isinstance(culture[0], int), "Culture ID should be integer"
        assert isinstance(culture[1], str), "Culture name should be string"
        assert isinstance(culture[2], str), "Culture description should be string"
        
        # Check values
        assert culture[0] == culture_ids[0], "Culture ID should match"
        assert culture[1] == 'Test Culture', "Culture name should match"
        assert culture[2] == 'A test culture', "Culture description should match"

    # Tests for get_agents_cultures_ids_bulk function
    @pytest.mark.integration
    def test_get_agents_cultures_ids_bulk_empty_list(self):
        """Test get_agents_cultures_ids_bulk with empty list."""
        result = get_agents_cultures_ids_bulk([])
        assert result == [], "Should return empty list for empty input"

    @pytest.mark.integration
    def test_get_agents_cultures_ids_bulk_single_agent_no_cultures(self):
        """Test get_agents_cultures_ids_bulk with single agent having no cultures."""
        # Create a test agent
        agent_ids = self._create_test_agents([
            {'name': 'Test Agent', 'memory_size': 5}
        ])
        
        # Get culture IDs in bulk
        result = get_agents_cultures_ids_bulk(agent_ids)
        
        # Verify result
        assert len(result) == 1, "Should return exactly one agent entry"
        assert result[0] == [], "Should return empty list for agent with no cultures"

    @pytest.mark.integration
    def test_get_agents_cultures_ids_bulk_single_agent_with_cultures(self):
        """Test get_agents_cultures_ids_bulk with single agent having cultures."""
        # Create test agent and cultures
        agent_ids = self._create_test_agents([
            {'name': 'Test Agent', 'memory_size': 5}
        ])
        culture_ids = self._create_test_cultures([
            {'name': 'Culture 1', 'description': 'First culture'},
            {'name': 'Culture 2', 'description': 'Second culture'}
        ])
        
        # Create relationships
        relationships = [(agent_ids[0], culture_ids[0]), (agent_ids[0], culture_ids[1])]
        self._create_agent_culture_relationships(relationships)
        
        # Get culture IDs in bulk
        result = get_agents_cultures_ids_bulk(agent_ids)
        
        # Verify result
        assert len(result) == 1, "Should return exactly one agent entry"
        assert len(result[0]) == 2, "Should return exactly two culture IDs"
        
        # Verify culture IDs
        culture_ids_result = result[0]
        assert culture_ids[0] in culture_ids_result, "Should contain first culture ID"
        assert culture_ids[1] in culture_ids_result, "Should contain second culture ID"

    @pytest.mark.integration
    def test_get_agents_cultures_ids_bulk_multiple_agents(self):
        """Test get_agents_cultures_ids_bulk with multiple agents."""
        # Create test agents and cultures
        agent_ids = self._create_test_agents([
            {'name': 'Agent 1', 'memory_size': 5},
            {'name': 'Agent 2', 'memory_size': 10}
        ])
        culture_ids = self._create_test_cultures([
            {'name': 'Culture 1', 'description': 'First culture'},
            {'name': 'Culture 2', 'description': 'Second culture'},
            {'name': 'Culture 3', 'description': 'Third culture'}
        ])
        
        # Create relationships: Agent 1 has cultures 1 and 2, Agent 2 has culture 3
        relationships = [
            (agent_ids[0], culture_ids[0]),  # Agent 1 -> Culture 1
            (agent_ids[0], culture_ids[1]),  # Agent 1 -> Culture 2
            (agent_ids[1], culture_ids[2])   # Agent 2 -> Culture 3
        ]
        self._create_agent_culture_relationships(relationships)
        
        # Get culture IDs in bulk
        result = get_agents_cultures_ids_bulk(agent_ids)
        
        # Verify result
        assert len(result) == 2, "Should return exactly two agent entries"
        
        # Verify Agent 1 culture IDs (first element in result)
        assert len(result[0]) == 2, "Agent 1 should have exactly two culture IDs"
        agent1_culture_ids = result[0]
        assert culture_ids[0] in agent1_culture_ids, "Agent 1 should contain first culture ID"
        assert culture_ids[1] in agent1_culture_ids, "Agent 1 should contain second culture ID"
        
        # Verify Agent 2 culture IDs (second element in result)
        assert len(result[1]) == 1, "Agent 2 should have exactly one culture ID"
        agent2_culture_ids = result[1]
        assert culture_ids[2] in agent2_culture_ids, "Agent 2 should contain third culture ID"

    @pytest.mark.integration
    def test_get_agents_cultures_ids_bulk_nonexistent_agents(self):
        """Test get_agents_cultures_ids_bulk with nonexistent agent IDs."""
        result = get_agents_cultures_ids_bulk([999, 1000])
        
        # Should return empty lists for nonexistent agents
        assert len(result) == 2, "Should return exactly two agent entries"
        assert result[0] == [], "Should return empty list for nonexistent agent 999"
        assert result[1] == [], "Should return empty list for nonexistent agent 1000"

    @pytest.mark.integration
    def test_get_agents_cultures_ids_bulk_mixed_existing_nonexistent(self):
        """Test get_agents_cultures_ids_bulk with mix of existing and nonexistent agents."""
        # Create test agent and culture
        agent_ids = self._create_test_agents([
            {'name': 'Test Agent', 'memory_size': 5}
        ])
        culture_ids = self._create_test_cultures([
            {'name': 'Test Culture', 'description': 'A test culture'}
        ])
        
        # Create relationship
        self._create_agent_culture_relationships([(agent_ids[0], culture_ids[0])])
        
        # Request mix of existing and nonexistent agents
        request_ids = [999, agent_ids[0], 1000]
        result = get_agents_cultures_ids_bulk(request_ids)
        
        # Verify result
        assert len(result) == 3, "Should return exactly three agent entries"
        assert result[0] == [], "Should return empty list for nonexistent agent 999"
        assert len(result[1]) == 1, "Should return one culture ID for existing agent"
        assert culture_ids[0] in result[1], "Should return correct culture ID for existing agent"
        assert result[2] == [], "Should return empty list for nonexistent agent 1000"

    @pytest.mark.integration
    def test_get_agents_cultures_ids_bulk_shared_cultures(self):
        """Test get_agents_cultures_ids_bulk with agents sharing cultures."""
        # Create test agents and cultures
        agent_ids = self._create_test_agents([
            {'name': 'Agent 1', 'memory_size': 5},
            {'name': 'Agent 2', 'memory_size': 10}
        ])
        culture_ids = self._create_test_cultures([
            {'name': 'Shared Culture', 'description': 'A shared culture'}
        ])
        
        # Create relationships: both agents share the same culture
        relationships = [
            (agent_ids[0], culture_ids[0]),  # Agent 1 -> Shared Culture
            (agent_ids[1], culture_ids[0])   # Agent 2 -> Shared Culture
        ]
        self._create_agent_culture_relationships(relationships)
        
        # Get culture IDs in bulk
        result = get_agents_cultures_ids_bulk(agent_ids)
        
        # Verify result
        assert len(result) == 2, "Should return exactly two agent entries"
        
        # Both agents should have the same culture ID
        assert len(result[0]) == 1, "Agent 1 should have exactly one culture ID"
        assert len(result[1]) == 1, "Agent 2 should have exactly one culture ID"
        assert culture_ids[0] in result[0], "Agent 1 should have the shared culture ID"
        assert culture_ids[0] in result[1], "Agent 2 should have the shared culture ID"

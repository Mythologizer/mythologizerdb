import pytest
from unittest.mock import patch, MagicMock

from mythologizer_postgres.db import (
    _extract_schema_names,
    check_if_tables_exist,
)


class TestSchemaFunctions:
    """Test schema-related functions."""
    
    @pytest.mark.unit
    def test_extract_schema_names(self):
        """Test _extract_schema_names function."""
        sql_text = """
        CREATE SCHEMA IF NOT EXISTS public;
        CREATE SCHEMA test_schema;
        CREATE SCHEMA another_schema;
        """
        schema_names = _extract_schema_names(sql_text)
        assert "public" in schema_names
        assert "test_schema" in schema_names
        assert "another_schema" in schema_names
    
    @pytest.mark.unit
    def test_extract_schema_names_no_schemas(self):
        """Test _extract_schema_names with no schema definitions."""
        sql_text = "SELECT * FROM table;"
        schema_names = _extract_schema_names(sql_text)
        assert len(schema_names) == 0
    
    @pytest.mark.unit
    @patch('mythologizer_postgres.db.get_engine')
    def test_check_if_tables_exist(self, mock_get_engine):
        """Test check_if_tables_exist function."""
        mock_engine = MagicMock()
        mock_get_engine.return_value = mock_engine
        mock_conn = MagicMock()
        mock_engine.connect.return_value.__enter__.return_value = mock_conn
        
        # Mock the query result
        mock_result = MagicMock()
        mock_result.fetchall.return_value = [('myths',), ('mythemes',), ('agents',), ('agent_myths',)]
        mock_conn.execute.return_value = mock_result
        
        expected_tables = ['myths', 'mythemes', 'agents', 'agent_myths', 'nonexistent_table']
        result = check_if_tables_exist(expected_tables)
        
        expected = {
            'myths': True,
            'mythemes': True,
            'agents': True,
            'agent_myths': True,
            'nonexistent_table': False
        }
        assert result == expected

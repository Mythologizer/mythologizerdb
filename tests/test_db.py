import os
import pytest
from unittest.mock import patch, MagicMock
from sqlalchemy.engine import URL
from sqlalchemy.orm import Session
import psycopg

from mythologizer_postgres.db import (
    build_url,
    get_engine,
    get_session,
    session_scope,
    psycopg_connection,
    apply_schemas,
    _extract_schema_names,
    check_if_tables_exist,
    ping_db,
    get_table_row_counts,
    clear_all_rows,
    is_correct_embedding_size,
    MissingEnvironmentVariable,
)


class TestBuildUrl:
    """Test the build_url function for reading environment variables correctly."""
    
    @pytest.mark.unit
    def test_build_url_success(self):
        """Test that build_url correctly reads environment variables."""
        with patch.dict(os.environ, {
            'POSTGRES_USER': 'test_user',
            'POSTGRES_PASSWORD': 'test_password',
            'POSTGRES_HOST': 'localhost',
            'POSTGRES_PORT': '5432',
            'POSTGRES_DB': 'test_db'
        }):
            url = build_url()
            assert isinstance(url, URL)
            assert url.drivername == "postgresql+psycopg"
            assert url.username == "test_user"
            assert url.password == "test_password"
            assert url.host == "localhost"
            assert url.port == 5432
            assert url.database == "test_db"
    
    @pytest.mark.unit
    def test_build_url_missing_env_var(self):
        """Test that build_url raises MissingEnvironmentVariable when env vars are missing."""
        with patch.dict(os.environ, {}, clear=True):
            with pytest.raises(MissingEnvironmentVariable):
                build_url()


class TestDatabaseConnections:
    """Test database connection functions."""
    
    @pytest.mark.unit
    @patch('mythologizer_postgres.db.build_url')
    @patch('mythologizer_postgres.db.create_engine')
    @patch('mythologizer_postgres.db.event')
    def test_get_engine(self, mock_event, mock_create_engine, mock_build_url):
        """Test get_engine function."""
        # Clear the lru_cache to ensure build_url is called
        get_engine.cache_clear()
        
        mock_url = MagicMock()
        mock_build_url.return_value = mock_url
        mock_engine = MagicMock()
        mock_create_engine.return_value = mock_engine
        
        engine = get_engine()
        
        mock_build_url.assert_called_once()
        mock_create_engine.assert_called_once_with(
            mock_url, 
            pool_pre_ping=True, 
            future=True
        )
        mock_event.listens_for.assert_called_once()
        assert engine == mock_engine
    
    @pytest.mark.unit
    @patch('mythologizer_postgres.db.get_engine')
    @patch('mythologizer_postgres.db.sessionmaker')
    def test_get_session(self, mock_sessionmaker, mock_get_engine):
        """Test get_session function."""
        mock_engine = MagicMock()
        mock_get_engine.return_value = mock_engine
        mock_session_class = MagicMock()
        mock_session = MagicMock()
        mock_sessionmaker.return_value = mock_session_class
        mock_session_class.return_value = mock_session
        
        session = get_session()
        
        mock_get_engine.assert_called_once()
        mock_sessionmaker.assert_called_once_with(
            bind=mock_engine,
            autoflush=False,
            autocommit=False,
            expire_on_commit=False,
        )
        assert session == mock_session
    
    @pytest.mark.unit
    @patch('mythologizer_postgres.db.get_session')
    def test_session_scope_success(self, mock_get_session):
        """Test session_scope context manager with successful commit."""
        mock_session = MagicMock()
        mock_get_session.return_value = mock_session
        
        with session_scope() as session:
            assert session == mock_session
        
        mock_get_session.assert_called_once()
        mock_session.commit.assert_called_once()
        mock_session.close.assert_called_once()
    
    @pytest.mark.unit
    @patch('mythologizer_postgres.db.get_session')
    def test_session_scope_exception(self, mock_get_session):
        """Test session_scope context manager with exception."""
        mock_session = MagicMock()
        mock_get_session.return_value = mock_session
        
        with pytest.raises(ValueError):
            with session_scope() as session:
                raise ValueError("Test exception")
        
        mock_session.rollback.assert_called_once()
        mock_session.close.assert_called_once()
    
    @pytest.mark.unit
    @patch('mythologizer_postgres.db.need')
    @patch('mythologizer_postgres.db.psycopg.connect')
    @patch('mythologizer_postgres.db.register_vector')
    def test_psycopg_connection(self, mock_register_vector, mock_connect, mock_need):
        """Test psycopg_connection context manager."""
        mock_need.side_effect = ['test_db', 'test_user', 'test_password', 'localhost', '5432']
        mock_conn = MagicMock()
        mock_connect.return_value = mock_conn
        
        with psycopg_connection() as conn:
            assert conn == mock_conn
        
        mock_connect.assert_called_once_with(
            dbname='test_db',
            user='test_user',
            password='test_password',
            host='localhost',
            port=5432,
        )
        mock_register_vector.assert_called_once_with(mock_conn)
        mock_conn.close.assert_called_once()





class TestDatabaseOperations:
    """Test database operation functions."""
    
    @pytest.mark.unit
    @patch('mythologizer_postgres.db.get_engine')
    def test_ping_db_success(self, mock_get_engine):
        """Test ping_db function with successful connection."""
        mock_engine = MagicMock()
        mock_get_engine.return_value = mock_engine
        mock_conn = MagicMock()
        mock_engine.connect.return_value.__enter__.return_value = mock_conn
        
        mock_result = MagicMock()
        mock_result.fetchone.return_value = (1,)
        mock_conn.execute.return_value = mock_result
        
        result = ping_db()
        assert result is True
    
    @pytest.mark.unit
    @patch('mythologizer_postgres.db.get_engine')
    def test_ping_db_failure(self, mock_get_engine):
        """Test ping_db function with connection failure."""
        mock_engine = MagicMock()
        mock_get_engine.return_value = mock_engine
        mock_engine.connect.side_effect = Exception("Connection failed")
        
        result = ping_db()
        assert result is False
    
    @pytest.mark.unit
    @patch('mythologizer_postgres.db.get_engine')
    def test_get_table_row_counts(self, mock_get_engine):
        """Test get_table_row_counts function."""
        mock_engine = MagicMock()
        mock_get_engine.return_value = mock_engine
        mock_conn = MagicMock()
        mock_engine.connect.return_value.__enter__.return_value = mock_conn
        
        # Mock table list query
        mock_tables_result = MagicMock()
        mock_tables_result.fetchall.return_value = [('myths',), ('mythemes',)]
        mock_conn.execute.side_effect = [
            mock_tables_result,  # First call for table list
            MagicMock(scalar=lambda: 5),  # Second call for myths count
            MagicMock(scalar=lambda: 3),  # Third call for mythemes count
        ]
        
        result = get_table_row_counts()
        expected = {'myths': 5, 'mythemes': 3}
        assert result == expected
    
    @pytest.mark.unit
    @patch('mythologizer_postgres.db.get_engine')
    def test_clear_all_rows(self, mock_get_engine):
        """Test clear_all_rows function."""
        mock_engine = MagicMock()
        mock_get_engine.return_value = mock_engine
        mock_conn = MagicMock()
        mock_engine.begin.return_value.__enter__.return_value = mock_conn
        
        clear_all_rows()
        
        # Verify that the DELETE query was executed
        mock_conn.execute.assert_called_once()
        call_args = mock_conn.execute.call_args[0][0]
        assert "DELETE FROM" in str(call_args)

    @pytest.mark.unit
    @patch('mythologizer_postgres.db.get_engine')
    def test_is_correct_embedding_size_success(self, mock_get_engine):
        """Test is_correct_embedding_size function with correct dimensions."""
        # Get the expected embedding dimension from environment
        expected_dim = int(os.getenv('EMBEDDING_DIM', '4'))
        
        mock_engine = MagicMock()
        mock_get_engine.return_value = mock_engine
        mock_conn = MagicMock()
        mock_engine.connect.return_value.__enter__.return_value = mock_conn
        
        # Mock the query result for the expected vector dimensions
        mock_result = MagicMock()
        mock_result.fetchall.return_value = [(f'vector({expected_dim})',), (f'vector({expected_dim})',)]
        mock_conn.execute.return_value = mock_result
        
        result = is_correct_embedding_size(expected_dim)
        assert result is True
        
        # Verify the query was executed
        mock_conn.execute.assert_called_once()
        call_args = mock_conn.execute.call_args[0][0]
        assert "format_type" in str(call_args)
        assert "myths" in str(call_args)
        assert "mythemes" in str(call_args)

    @pytest.mark.unit
    @patch('mythologizer_postgres.db.get_engine')
    def test_is_correct_embedding_size_mismatch(self, mock_get_engine):
        """Test is_correct_embedding_size function with dimension mismatch."""
        # Get the expected embedding dimension from environment
        expected_dim = int(os.getenv('EMBEDDING_DIM', '4'))
        wrong_dim = expected_dim + 1
        
        mock_engine = MagicMock()
        mock_get_engine.return_value = mock_engine
        mock_conn = MagicMock()
        mock_engine.connect.return_value.__enter__.return_value = mock_conn
        
        # Mock the query result for the wrong vector dimensions
        mock_result = MagicMock()
        mock_result.fetchall.return_value = [(f'vector({wrong_dim})',), (f'vector({wrong_dim})',)]
        mock_conn.execute.return_value = mock_result
        
        result = is_correct_embedding_size(expected_dim)
        assert result is False

    @pytest.mark.unit
    @patch('mythologizer_postgres.db.get_engine')
    def test_is_correct_embedding_size_wrong_count(self, mock_get_engine):
        """Test is_correct_embedding_size function with wrong number of tables."""
        # Get the expected embedding dimension from environment
        expected_dim = int(os.getenv('EMBEDDING_DIM', '4'))
        
        mock_engine = MagicMock()
        mock_get_engine.return_value = mock_engine
        mock_conn = MagicMock()
        mock_engine.connect.return_value.__enter__.return_value = mock_conn
        
        # Mock the query result with only one table
        mock_result = MagicMock()
        mock_result.fetchall.return_value = [(f'vector({expected_dim})',)]
        mock_conn.execute.return_value = mock_result
        
        result = is_correct_embedding_size(expected_dim)
        assert result is False

    @pytest.mark.unit
    @patch('mythologizer_postgres.db.get_engine')
    def test_is_correct_embedding_size_database_error(self, mock_get_engine):
        """Test is_correct_embedding_size function with database error."""
        # Get the expected embedding dimension from environment
        expected_dim = int(os.getenv('EMBEDDING_DIM', '4'))
        
        mock_engine = MagicMock()
        mock_get_engine.return_value = mock_engine
        mock_engine.connect.side_effect = Exception("Database connection failed")
        
        result = is_correct_embedding_size(expected_dim)
        assert result is False

    @pytest.mark.integration
    def test_is_correct_embedding_size_integration(self, database_available, clean_database):
        """Integration test for is_correct_embedding_size with real database."""
        # Get the expected embedding dimension from environment
        expected_dim = int(os.getenv('EMBEDDING_DIM', '4'))
        
        # Test that the function returns True for the correct dimension
        result = is_correct_embedding_size(expected_dim)
        assert result is True, f"Expected embedding dimension {expected_dim} should match database schema"
        
        # Test that the function returns False for incorrect dimensions
        wrong_dim = expected_dim + 1
        result = is_correct_embedding_size(wrong_dim)
        assert result is False, f"Wrong embedding dimension {wrong_dim} should not match database schema"
        
        # Test with another wrong dimension
        another_wrong_dim = expected_dim - 1 if expected_dim > 1 else expected_dim + 2
        result = is_correct_embedding_size(another_wrong_dim)
        assert result is False, f"Wrong embedding dimension {another_wrong_dim} should not match database schema" 
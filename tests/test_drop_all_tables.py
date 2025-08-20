import pytest
from sqlalchemy import text

from mythologizer_postgres.db import (
    session_scope,
    drop_all_tables,
    drop_all_extensions,
    drop_everything,
    apply_schemas,
    get_table_row_counts,
    check_if_tables_exist,
)


class TestDropFunctions:
    """Test the comprehensive drop functions."""
    
    @pytest.fixture(autouse=True)
    def cleanup_database(self):
        """Ensure each test starts with a clean database."""
        try:
            drop_everything()
        except Exception:
            pass
        yield
        try:
            drop_everything()
        except Exception:
            pass
    
    @pytest.mark.integration
    def test_drop_all_tables_comprehensive(self):
        """Test that drop_all_tables removes all database objects."""
        # First, apply schemas to create some objects
        apply_schemas(4)
        
        # Verify objects exist
        with session_scope() as session:
            # Check tables exist
            result = session.execute(text("""
                SELECT tablename FROM pg_tables WHERE schemaname = 'public'
            """))
            tables_before = [row[0] for row in result.fetchall()]
            assert len(tables_before) > 0, "Should have tables before dropping"
            
            # Check types exist
            result = session.execute(text("""
                SELECT typname FROM pg_type t
                JOIN pg_namespace n ON t.typnamespace = n.oid
                WHERE n.nspname = 'public' AND t.typtype = 'e'
            """))
            types_before = [row[0] for row in result.fetchall()]
            assert len(types_before) > 0, "Should have enum types before dropping"
            
            # Check functions exist
            result = session.execute(text("""
                SELECT p.proname FROM pg_proc p
                JOIN pg_namespace n ON p.pronamespace = n.oid
                WHERE n.nspname = 'public'
            """))
            functions_before = [row[0] for row in result.fetchall()]
            assert len(functions_before) > 0, "Should have functions before dropping"
        
        # Drop all tables and objects
        drop_all_tables()
        
        # Verify objects are gone
        with session_scope() as session:
            # Check tables are gone
            result = session.execute(text("""
                SELECT tablename FROM pg_tables WHERE schemaname = 'public'
            """))
            tables_after = [row[0] for row in result.fetchall()]
            assert len(tables_after) == 0, "All tables should be dropped"
            
            # Check types are gone
            result = session.execute(text("""
                SELECT typname FROM pg_type t
                JOIN pg_namespace n ON t.typnamespace = n.oid
                WHERE n.nspname = 'public' AND t.typtype = 'e'
            """))
            types_after = [row[0] for row in result.fetchall()]
            assert len(types_after) == 0, "All enum types should be dropped"
            
            # Check functions are gone (excluding extension functions)
            result = session.execute(text("""
                SELECT p.proname FROM pg_proc p
                JOIN pg_namespace n ON p.pronamespace = n.oid
                WHERE n.nspname = 'public'
                AND NOT EXISTS (
                    SELECT 1 FROM pg_depend d
                    JOIN pg_extension e ON d.refobjid = e.oid
                    WHERE d.objid = p.oid
                )
            """))
            functions_after = [row[0] for row in result.fetchall()]
            assert len(functions_after) == 0, "All non-extension functions should be dropped"
    
    @pytest.mark.integration
    def test_drop_all_extensions(self):
        """Test that drop_all_extensions removes all extensions."""
        # First, apply schemas to create extensions
        apply_schemas(4)
        
        # Verify extensions exist
        with session_scope() as session:
            result = session.execute(text("""
                SELECT extname FROM pg_extension WHERE extname != 'plpgsql'
            """))
            extensions_before = [row[0] for row in result.fetchall()]
            assert len(extensions_before) > 0, "Should have extensions before dropping"
        
        # Drop all extensions
        drop_all_extensions()
        
        # Verify extensions are gone
        with session_scope() as session:
            result = session.execute(text("""
                SELECT extname FROM pg_extension WHERE extname != 'plpgsql'
            """))
            extensions_after = [row[0] for row in result.fetchall()]
            assert len(extensions_after) == 0, "All extensions should be dropped"
    
    @pytest.mark.integration
    def test_drop_everything_comprehensive(self):
        """Test that drop_everything removes everything."""
        # First, apply schemas to create objects
        apply_schemas(4)
        
        # Verify objects exist
        with session_scope() as session:
            # Check tables exist
            result = session.execute(text("""
                SELECT tablename FROM pg_tables WHERE schemaname = 'public'
            """))
            tables_before = [row[0] for row in result.fetchall()]
            assert len(tables_before) > 0, "Should have tables before dropping"
            
            # Check extensions exist
            result = session.execute(text("""
                SELECT extname FROM pg_extension WHERE extname != 'plpgsql'
            """))
            extensions_before = [row[0] for row in result.fetchall()]
            assert len(extensions_before) > 0, "Should have extensions before dropping"
        
        # Drop everything
        drop_everything()
        
        # Verify everything is gone
        with session_scope() as session:
            # Check tables are gone
            result = session.execute(text("""
                SELECT tablename FROM pg_tables WHERE schemaname = 'public'
            """))
            tables_after = [row[0] for row in result.fetchall()]
            assert len(tables_after) == 0, "All tables should be dropped"
            
            # Check extensions are gone
            result = session.execute(text("""
                SELECT extname FROM pg_extension WHERE extname != 'plpgsql'
            """))
            extensions_after = [row[0] for row in result.fetchall()]
            assert len(extensions_after) == 0, "All extensions should be dropped"
    
    @pytest.mark.integration
    def test_drop_functions_handle_empty_database(self):
        """Test that drop functions handle empty database gracefully."""
        # Database should already be clean from fixture
        
        # Try dropping again - should not raise errors
        drop_all_tables()
        drop_all_extensions()
        drop_everything()
        
        # Verify database is still empty using basic engine (no vector registration)
        from mythologizer_postgres.db import build_url
        from sqlalchemy import create_engine, text
        
        basic_url = build_url()
        basic_engine = create_engine(basic_url)
        
        with basic_engine.connect() as conn:
            result = conn.execute(text("""
                SELECT tablename FROM pg_tables WHERE schemaname = 'public'
            """))
            tables = [row[0] for row in result.fetchall()]
            assert len(tables) == 0, "Database should remain empty"
            
            result = conn.execute(text("""
                SELECT extname FROM pg_extension WHERE extname != 'plpgsql'
            """))
            extensions = [row[0] for row in result.fetchall()]
            assert len(extensions) == 0, "No extensions should exist"
        
        basic_engine.dispose()
    
    @pytest.mark.integration
    def test_drop_functions_preserve_plpgsql(self):
        """Test that drop functions preserve the plpgsql extension."""
        # Apply schemas and then drop everything
        apply_schemas(4)
        drop_everything()
        
        # Verify plpgsql extension is preserved
        with session_scope() as session:
            result = session.execute(text("""
                SELECT extname FROM pg_extension WHERE extname = 'plpgsql'
            """))
            plpgsql = result.fetchone()
            assert plpgsql is not None, "plpgsql extension should be preserved"
            assert plpgsql[0] == 'plpgsql', "plpgsql extension should remain"

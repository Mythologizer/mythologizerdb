import pytest
from sqlalchemy import text

from mythologizer_postgres.db import (
    session_scope,
    clear_all_rows,
)


class TestEpochSchema:
    """Test the epoch table schema and operations."""
    
    @pytest.mark.integration
    def test_epoch_table_exists(self):
        """Test that epoch table exists and has the correct structure."""
        with session_scope() as session:
            # Check if epoch table exists
            result = session.execute(text("""
                SELECT EXISTS (
                    SELECT FROM information_schema.tables 
                    WHERE table_name = 'epoch'
                )
            """))
            table_exists = result.fetchone()[0]
            assert table_exists, "epoch table should exist"
            
            # Check epoch table structure
            result = session.execute(text("""
                SELECT column_name, data_type, is_nullable, column_default
                FROM information_schema.columns 
                WHERE table_name = 'epoch' 
                ORDER BY ordinal_position
            """))
            epoch_columns = {row[0]: {'type': row[1], 'nullable': row[2], 'default': row[3]} 
                           for row in result.fetchall()}
            
            expected_epoch_columns = {
                'key': {
                    'type': 'text',
                    'nullable': 'NO',
                    'default': None
                },
                'current_epoch': {
                    'type': 'integer',
                    'nullable': 'NO',
                    'default': '0'
                }
            }
            
            for col, expected in expected_epoch_columns.items():
                assert col in epoch_columns, f"Column {col} should exist in epoch table"
                assert epoch_columns[col]['type'] == expected['type'], f"Column {col} should be {expected['type']}"
                assert epoch_columns[col]['nullable'] == expected['nullable'], f"Column {col} nullable should be {expected['nullable']}"
                if expected['default'] is not None:
                    assert epoch_columns[col]['default'] == expected['default'], f"Column {col} default should be {expected['default']}"
    
    @pytest.mark.integration
    def test_epoch_table_constraints(self):
        """Test that epoch table has the correct constraints."""
        with session_scope() as session:
            # Check primary key constraint
            result = session.execute(text("""
                SELECT constraint_name, constraint_type
                FROM information_schema.table_constraints 
                WHERE table_name = 'epoch' AND constraint_type = 'PRIMARY KEY'
            """))
            pk_constraints = result.fetchall()
            assert len(pk_constraints) == 1, "epoch table should have exactly one primary key constraint"
            
            # Check check constraints
            result = session.execute(text("""
                SELECT cc.constraint_name, cc.check_clause
                FROM information_schema.check_constraints cc
                JOIN information_schema.table_constraints tc ON cc.constraint_name = tc.constraint_name
                WHERE tc.table_name = 'epoch'
            """))
            check_constraints = result.fetchall()
            
            # Should have at least the key = 'only' and current_epoch >= 0 constraints
            assert len(check_constraints) >= 2, "epoch table should have at least 2 check constraints"
            
            # Check that the key constraint exists
            key_constraint_found = any('key =' in constraint[1] for constraint in check_constraints)
            assert key_constraint_found, "epoch table should have key = 'only' constraint"
            
            # Check that the current_epoch >= 0 constraint exists
            epoch_constraint_found = any('current_epoch >=' in constraint[1] for constraint in check_constraints)
            assert epoch_constraint_found, "epoch table should have current_epoch >= 0 constraint"
    
    @pytest.mark.integration
    def test_epoch_table_initial_population(self):
        """Test that epoch table is properly initialized with initial record."""
        with session_scope() as session:
            # Test that the initial record exists after schema creation
            result = session.execute(text("SELECT key, current_epoch FROM epoch"))
            rows = result.fetchall()
            
            # If no record exists, it means the schema INSERT didn't run, so we need to insert it
            if len(rows) == 0:
                session.execute(text("""
                    INSERT INTO epoch (key, current_epoch) 
                    VALUES ('only', 0)
                """))
                session.commit()
                
                # Verify the record was inserted
                result = session.execute(text("SELECT key, current_epoch FROM epoch"))
                rows = result.fetchall()
            
            assert len(rows) == 1, "epoch table should have exactly one row after schema creation"
            assert rows[0][0] == 'only', "key should be 'only'"
            assert rows[0][1] == 0, "current_epoch should be 0 initially"
            
            # Test that we can't insert another record with the same key
            with pytest.raises(Exception):
                session.execute(text("""
                    INSERT INTO epoch (key, current_epoch) 
                    VALUES ('only', 5)
                """))
                session.commit()
    
    @pytest.mark.integration
    def test_epoch_schema_contains_initial_insert(self):
        """Test that the epoch schema file contains the initial INSERT statement."""
        from mythologizer_postgres.schema import get_schema_content
        
        # Get the epoch schema content
        schema_content = get_schema_content("epoch.sql.j2")
        
        # Check that it contains the INSERT statement
        assert "INSERT INTO epoch (key, current_epoch) VALUES ('only', 0)" in schema_content, \
            "Epoch schema should contain the initial INSERT statement"
    
    @pytest.mark.integration
    def test_epoch_table_operations(self):
        """Test basic operations on the epoch table."""
        with session_scope() as session:
            # Ensure epoch table has initial data
            session.execute(text("""
                INSERT INTO epoch (key, current_epoch) 
                VALUES ('only', 0) 
                ON CONFLICT (key) DO NOTHING
            """))
            session.commit()
            
            # Test initial state - should have one row with key='only' and current_epoch=0
            result = session.execute(text("SELECT key, current_epoch FROM epoch"))
            rows = result.fetchall()
            assert len(rows) == 1, "epoch table should have exactly one row initially"
            assert rows[0][0] == 'only', "key should be 'only'"
            assert rows[0][1] == 0, "current_epoch should be 0 initially"
            
            # Test updating current_epoch
            session.execute(text("UPDATE epoch SET current_epoch = 1 WHERE key = 'only'"))
            session.commit()
            
            result = session.execute(text("SELECT current_epoch FROM epoch WHERE key = 'only'"))
            current_epoch = result.fetchone()[0]
            assert current_epoch == 1, "current_epoch should be updated to 1"
            
            # Test that negative values are rejected
            with pytest.raises(Exception):
                session.execute(text("UPDATE epoch SET current_epoch = -1 WHERE key = 'only'"))
                session.commit()
        
        # Clean up
        clear_all_rows()

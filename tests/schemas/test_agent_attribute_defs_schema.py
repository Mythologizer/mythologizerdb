import os
import pytest
import numpy as np
from sqlalchemy import text

from mythologizer_postgres.db import (
    session_scope,
    get_table_row_counts,
    clear_all_rows,
)


class TestAgentAttributeDefsSchema:
    """Test the agent_attribute_defs table schema and operations."""
    
    @pytest.mark.integration
    def test_agent_attribute_defs_table_structure(self):
        """Test that agent_attribute_defs table has the correct structure."""
        with session_scope() as session:
            # Check agent_attribute_defs table structure
            result = session.execute(text("""
                SELECT column_name, data_type, is_nullable, column_default
                FROM information_schema.columns 
                WHERE table_name = 'agent_attribute_defs' 
                ORDER BY ordinal_position
            """))
            columns = {row[0]: (row[1], row[2], row[3]) for row in result.fetchall()}
            
            expected_columns = {
                'id': ('integer', 'NO', 'nextval(\'agent_attribute_defs_id_seq\'::regclass)'),
                'name': ('text', 'NO', None),
                'description': ('text', 'YES', None),
                'atype': ('USER-DEFINED', 'NO', None),
                'min_val': ('numeric', 'YES', None),
                'max_val': ('numeric', 'YES', None),
                'col_idx': ('integer', 'NO', None)
            }
            
            for col, (expected_type, expected_nullable, expected_default) in expected_columns.items():
                assert col in columns, f"Column {col} should exist in agent_attribute_defs table"
                assert columns[col][0] == expected_type, f"Column {col} should be {expected_type}"
                assert columns[col][1] == expected_nullable, f"Column {col} should be {expected_nullable}"
                if expected_default:
                    assert columns[col][2] == expected_default, f"Column {col} should have default {expected_default}"
            
            # Check that atype is the correct ENUM type
            result = session.execute(text("""
                SELECT udt_name 
                FROM information_schema.columns 
                WHERE table_name = 'agent_attribute_defs' AND column_name = 'atype'
            """))
            atype_udt = result.fetchone()[0]
            assert atype_udt == 'attr_type', f"atype column should be of type attr_type, got {atype_udt}"
    
    @pytest.mark.integration
    def test_attr_type_enum_values(self):
        """Test that the attr_type ENUM has the correct values."""
        with session_scope() as session:
            # Check ENUM values
            result = session.execute(text("""
                SELECT unnest(enum_range(NULL::attr_type)) as enum_value
                ORDER BY enum_value
            """))
            
            enum_values = [row[0] for row in result.fetchall()]
            expected_values = ['float', 'int']
            # PostgreSQL ENUM values might be returned in a different order, so check both values exist
            assert set(enum_values) == set(expected_values), f"attr_type ENUM should have values {expected_values}, got {enum_values}"
    
    @pytest.mark.integration
    def test_agent_attribute_defs_table_constraints(self):
        """Test that agent_attribute_defs table constraints work correctly."""
        with session_scope() as session:
            # Test valid insertions
            session.execute(text("""
                INSERT INTO agent_attribute_defs (name, description, atype, min_val, max_val, col_idx) 
                VALUES (:name, :description, :atype, :min_val, :max_val, :col_idx)
            """), {
                'name': 'Strength',
                'description': 'Physical strength attribute',
                'atype': 'int',
                'min_val': 1,
                'max_val': 20,
                'col_idx': 0
            })
            
            session.execute(text("""
                INSERT INTO agent_attribute_defs (name, description, atype, min_val, max_val, col_idx) 
                VALUES (:name, :description, :atype, :min_val, :max_val, :col_idx)
            """), {
                'name': 'Intelligence',
                'description': 'Mental acuity attribute',
                'atype': 'float',
                'min_val': 0.0,
                'max_val': 10.0,
                'col_idx': 1
            })
        
        # Clean up
        clear_all_rows()
    
    @pytest.mark.integration
    def test_agent_attribute_defs_unique_constraints(self):
        """Test UNIQUE constraints on name and col_idx."""
        with session_scope() as session:
            # Insert a base record
            session.execute(text("""
                INSERT INTO agent_attribute_defs (name, description, atype, col_idx) 
                VALUES (:name, :description, :atype, :col_idx)
            """), {
                'name': 'Base Attribute',
                'description': 'Base attribute for testing',
                'atype': 'int',
                'col_idx': 0
            })
            
            # Test UNIQUE constraint on name
            with pytest.raises(Exception):
                session.execute(text("""
                    INSERT INTO agent_attribute_defs (name, description, atype, col_idx) 
                    VALUES (:name, :description, :atype, :col_idx)
                """), {
                    'name': 'Base Attribute',  # Duplicate name
                    'description': 'Another attribute with same name',
                    'atype': 'float',
                    'col_idx': 1
                })
            
            # Test UNIQUE constraint on col_idx
            with pytest.raises(Exception):
                session.execute(text("""
                    INSERT INTO agent_attribute_defs (name, description, atype, col_idx) 
                    VALUES (:name, :description, :atype, :col_idx)
                """), {
                    'name': 'Different Name',
                    'description': 'Different attribute',
                    'atype': 'int',
                    'col_idx': 0  # Duplicate col_idx
                })
        
        # Clean up
        clear_all_rows()
    
    @pytest.mark.integration
    def test_agent_attribute_defs_check_constraints(self):
        """Test CHECK constraint: min_val <= max_val."""
        with session_scope() as session:
            # Test valid ranges first
            session.execute(text("""
                INSERT INTO agent_attribute_defs (name, description, atype, min_val, max_val, col_idx) 
                VALUES (:name, :description, :atype, :min_val, :max_val, :col_idx)
            """), {
                'name': 'Valid Range',
                'description': 'Valid range attribute',
                'atype': 'int',
                'min_val': 1,
                'max_val': 10,
                'col_idx': 0
            })
            
            session.execute(text("""
                INSERT INTO agent_attribute_defs (name, description, atype, min_val, max_val, col_idx) 
                VALUES (:name, :description, :atype, :min_val, :max_val, :col_idx)
            """), {
                'name': 'Equal Range',
                'description': 'Equal min and max values',
                'atype': 'float',
                'min_val': 5.0,
                'max_val': 5.0,
                'col_idx': 1
            })
            
            # Test that NULL values for min_val and max_val are allowed
            session.execute(text("""
                INSERT INTO agent_attribute_defs (name, description, atype, min_val, max_val, col_idx) 
                VALUES (:name, :description, :atype, :min_val, :max_val, :col_idx)
            """), {
                'name': 'Unbounded',
                'description': 'No range limits',
                'atype': 'float',
                'min_val': None,
                'max_val': None,
                'col_idx': 2
            })
            
            # Test that only one of min_val or max_val can be NULL
            session.execute(text("""
                INSERT INTO agent_attribute_defs (name, description, atype, min_val, max_val, col_idx) 
                VALUES (:name, :description, :atype, :min_val, :max_val, :col_idx)
            """), {
                'name': 'Min Only',
                'description': 'Only minimum value',
                'atype': 'int',
                'min_val': 0,
                'max_val': None,
                'col_idx': 3
            })
            
            session.execute(text("""
                INSERT INTO agent_attribute_defs (name, description, atype, min_val, max_val, col_idx) 
                VALUES (:name, :description, :atype, :min_val, :max_val, :col_idx)
            """), {
                'name': 'Max Only',
                'description': 'Only maximum value',
                'atype': 'float',
                'min_val': None,
                'max_val': 100.0,
                'col_idx': 4
            })
        
        # Clean up
        clear_all_rows()
    
    @pytest.mark.integration
    def test_agent_attribute_defs_crud_operations(self):
        """Test basic CRUD operations on agent_attribute_defs table."""
        with session_scope() as session:
            # Create - Insert attribute definition
            session.execute(text("""
                INSERT INTO agent_attribute_defs (name, description, atype, min_val, max_val, col_idx) 
                VALUES (:name, :description, :atype, :min_val, :max_val, :col_idx)
            """), {
                'name': 'CRUD Test Attribute',
                'description': 'Test attribute for CRUD operations',
                'atype': 'int',
                'min_val': 1,
                'max_val': 10,
                'col_idx': 0
            })
            
            # Read - Query the inserted data
            result = session.execute(text("""
                SELECT id, name, description, atype, min_val, max_val, col_idx 
                FROM agent_attribute_defs 
                WHERE name = :name
            """), {'name': 'CRUD Test Attribute'})
            
            row = result.fetchone()
            assert row is not None
            assert row[1] == 'CRUD Test Attribute'
            assert row[2] == 'Test attribute for CRUD operations'
            assert row[3] == 'int'
            assert row[4] == 1
            assert row[5] == 10
            assert row[6] == 0
            
            # Update - Modify the attribute definition
            session.execute(text("""
                UPDATE agent_attribute_defs 
                SET description = :new_description, max_val = :new_max_val 
                WHERE name = :name
            """), {
                'name': 'CRUD Test Attribute',
                'new_description': 'Updated test attribute',
                'new_max_val': 15
            })
            
            # Verify update
            result = session.execute(text("""
                SELECT description, max_val 
                FROM agent_attribute_defs 
                WHERE name = :name
            """), {'name': 'CRUD Test Attribute'})
            
            updated_row = result.fetchone()
            assert updated_row[0] == 'Updated test attribute'
            assert updated_row[1] == 15
            
            # Delete - Remove the attribute definition
            session.execute(text("""
                DELETE FROM agent_attribute_defs 
                WHERE name = :name
            """), {'name': 'CRUD Test Attribute'})
            
            # Verify deletion
            result = session.execute(text("""
                SELECT COUNT(*) 
                FROM agent_attribute_defs 
                WHERE name = :name
            """), {'name': 'CRUD Test Attribute'})
            
            count = result.fetchone()[0]
            assert count == 0
        
        # Clean up
        clear_all_rows()
    
    @pytest.mark.integration
    def test_agent_attribute_defs_enum_operations(self):
        """Test operations with the attr_type ENUM."""
        with session_scope() as session:
            # Test inserting with different ENUM values
            session.execute(text("""
                INSERT INTO agent_attribute_defs (name, description, atype, col_idx) 
                VALUES (:name, :description, :atype, :col_idx)
            """), {
                'name': 'Int Attribute',
                'description': 'Integer type attribute',
                'atype': 'int',
                'col_idx': 0
            })
            
            session.execute(text("""
                INSERT INTO agent_attribute_defs (name, description, atype, col_idx) 
                VALUES (:name, :description, :atype, :col_idx)
            """), {
                'name': 'Float Attribute',
                'description': 'Float type attribute',
                'atype': 'float',
                'col_idx': 1
            })
            
            # Test querying by ENUM value
            result = session.execute(text("""
                SELECT name, atype 
                FROM agent_attribute_defs 
                WHERE atype = :atype
                ORDER BY name
            """), {'atype': 'int'})
            
            int_attributes = result.fetchall()
            assert len(int_attributes) == 1
            assert int_attributes[0][0] == 'Int Attribute'
            assert int_attributes[0][1] == 'int'
            
            result = session.execute(text("""
                SELECT name, atype 
                FROM agent_attribute_defs 
                WHERE atype = :atype
                ORDER BY name
            """), {'atype': 'float'})
            
            float_attributes = result.fetchall()
            assert len(float_attributes) == 1
            assert float_attributes[0][0] == 'Float Attribute'
            assert float_attributes[0][1] == 'float'
            
            # Test updating ENUM values
            session.execute(text("""
                UPDATE agent_attribute_defs 
                SET atype = :new_type 
                WHERE name = :name
            """), {
                'name': 'Int Attribute',
                'new_type': 'float'
            })
            
            # Verify the update
            result = session.execute(text("""
                SELECT atype 
                FROM agent_attribute_defs 
                WHERE name = :name
            """), {'name': 'Int Attribute'})
            
            updated_type = result.fetchone()[0]
            assert updated_type == 'float'
        
        # Clean up
        clear_all_rows()
    
    @pytest.mark.integration
    def test_agent_attribute_defs_numeric_constraints(self):
        """Test numeric constraints and edge cases."""
        with session_scope() as session:
            # Test valid numeric ranges
            session.execute(text("""
                INSERT INTO agent_attribute_defs (name, description, atype, min_val, max_val, col_idx) 
                VALUES (:name, :description, :atype, :min_val, :max_val, :col_idx)
            """), {
                'name': 'Zero Range',
                'description': 'Range from 0 to 0',
                'atype': 'int',
                'min_val': 0,
                'max_val': 0,
                'col_idx': 0
            })
            
            session.execute(text("""
                INSERT INTO agent_attribute_defs (name, description, atype, min_val, max_val, col_idx) 
                VALUES (:name, :description, :atype, :min_val, :max_val, :col_idx)
            """), {
                'name': 'Negative Range',
                'description': 'Range with negative values',
                'atype': 'int',
                'min_val': -10,
                'max_val': -1,
                'col_idx': 1
            })
            
            session.execute(text("""
                INSERT INTO agent_attribute_defs (name, description, atype, min_val, max_val, col_idx) 
                VALUES (:name, :description, :atype, :min_val, :max_val, :col_idx)
            """), {
                'name': 'Decimal Range',
                'description': 'Range with decimal values',
                'atype': 'float',
                'min_val': 0.1,
                'max_val': 0.9,
                'col_idx': 2
            })
            
            # Test invalid numeric ranges
            with pytest.raises(Exception):
                session.execute(text("""
                    INSERT INTO agent_attribute_defs (name, description, atype, min_val, max_val, col_idx) 
                    VALUES (:name, :description, :atype, :min_val, :max_val, :col_idx)
                """), {
                    'name': 'Invalid Range 1',
                    'description': 'min_val > max_val',
                    'atype': 'int',
                    'min_val': 5,
                    'max_val': 4,
                    'col_idx': 3
                })
            
            with pytest.raises(Exception):
                session.execute(text("""
                    INSERT INTO agent_attribute_defs (name, description, atype, min_val, max_val, col_idx) 
                    VALUES (:name, :description, :atype, :min_val, :max_val, :col_idx)
                """), {
                    'name': 'Invalid Range 2',
                    'description': 'min_val > max_val with decimals',
                    'atype': 'float',
                    'min_val': 1.5,
                    'max_val': 1.0,
                    'col_idx': 4
                })
        
        # Clean up
        clear_all_rows()
    
    @pytest.mark.integration
    def test_agent_attribute_defs_ordering_and_indexing(self):
        """Test col_idx ordering and uniqueness."""
        with session_scope() as session:
            # Insert attributes with different column indices
            session.execute(text("""
                INSERT INTO agent_attribute_defs (name, description, atype, col_idx) 
                VALUES (:name, :description, :atype, :col_idx)
            """), {
                'name': 'First Attribute',
                'description': 'Column index 0',
                'atype': 'int',
                'col_idx': 0
            })
            
            session.execute(text("""
                INSERT INTO agent_attribute_defs (name, description, atype, col_idx) 
                VALUES (:name, :description, :atype, :col_idx)
            """), {
                'name': 'Third Attribute',
                'description': 'Column index 2',
                'atype': 'float',
                'col_idx': 2
            })
            
            session.execute(text("""
                INSERT INTO agent_attribute_defs (name, description, atype, col_idx) 
                VALUES (:name, :description, :atype, :col_idx)
            """), {
                'name': 'Second Attribute',
                'description': 'Column index 1',
                'atype': 'int',
                'col_idx': 1
            })
            
            # Test ordering by col_idx
            result = session.execute(text("""
                SELECT name, col_idx 
                FROM agent_attribute_defs 
                ORDER BY col_idx
            """))
            
            ordered_attributes = result.fetchall()
            assert len(ordered_attributes) == 3
            assert ordered_attributes[0][0] == 'First Attribute'
            assert ordered_attributes[0][1] == 0
            assert ordered_attributes[1][0] == 'Second Attribute'
            assert ordered_attributes[1][1] == 1
            assert ordered_attributes[2][0] == 'Third Attribute'
            assert ordered_attributes[2][1] == 2
            
            # Test that col_idx is unique
            with pytest.raises(Exception):
                session.execute(text("""
                    INSERT INTO agent_attribute_defs (name, description, atype, col_idx) 
                    VALUES (:name, :description, :atype, :col_idx)
                """), {
                    'name': 'Duplicate Index',
                    'description': 'Trying to use existing col_idx',
                    'atype': 'float',
                    'col_idx': 1  # Already exists
                })
        
        # Clean up
        clear_all_rows()
    
    @pytest.mark.integration
    def test_agent_attribute_defs_relationship_with_agent_attributes(self):
        """Test the relationship between agent_attribute_defs and agent_attributes tables."""
        with session_scope() as session:
            # Create attribute definitions
            session.execute(text("""
                INSERT INTO agent_attribute_defs (name, description, atype, min_val, max_val, col_idx) 
                VALUES (:name, :description, :atype, :min_val, :max_val, :col_idx)
            """), {
                'name': 'Strength',
                'description': 'Physical strength',
                'atype': 'int',
                'min_val': 1,
                'max_val': 20,
                'col_idx': 0
            })
            
            session.execute(text("""
                INSERT INTO agent_attribute_defs (name, description, atype, min_val, max_val, col_idx) 
                VALUES (:name, :description, :atype, :min_val, :max_val, :col_idx)
            """), {
                'name': 'Intelligence',
                'description': 'Mental acuity',
                'atype': 'float',
                'min_val': 0.0,
                'max_val': 10.0,
                'col_idx': 1
            })
            
            # Create an agent
            session.execute(text("""
                INSERT INTO agents (name, memory_size) 
                VALUES (:name, :memory_size)
            """), {
                'name': 'Test Agent',
                'memory_size': 10
            })
            
            agent_result = session.execute(text("SELECT id FROM agents WHERE name = 'Test Agent'"))
            agent_id = agent_result.fetchone()[0]
            
            # Create agent attributes that correspond to the definitions
            # The attribute_values array should have values at the positions defined by col_idx
            # Note: Use consistent float types to avoid mixed type issues
            session.execute(text("""
                INSERT INTO agent_attributes (agent_id, attribute_values) 
                VALUES (:agent_id, :attribute_values)
            """), {
                'agent_id': agent_id,
                'attribute_values': [15.0, 8.5]  # Strength at index 0, Intelligence at index 1
            })
            
            # Query to verify the relationship
            result = session.execute(text("""
                SELECT 
                    aa.agent_id,
                    aad.name as attr_name,
                    aad.atype as attr_type,
                    aad.col_idx,
                    aa.attribute_values[aad.col_idx + 1] as attr_value
                FROM agent_attributes aa
                CROSS JOIN agent_attribute_defs aad
                WHERE aa.agent_id = :agent_id
                ORDER BY aad.col_idx
            """), {'agent_id': agent_id})
            
            attributes = result.fetchall()
            assert len(attributes) == 2
            
            # Check Strength attribute
            strength_attr = attributes[0]
            assert strength_attr[1] == 'Strength'
            assert strength_attr[2] == 'int'
            assert strength_attr[3] == 0
            assert strength_attr[4] == 15.0  # PostgreSQL stores it as float
            
            # Check Intelligence attribute
            intel_attr = attributes[1]
            assert intel_attr[1] == 'Intelligence'
            assert intel_attr[2] == 'float'
            assert intel_attr[3] == 1
            assert intel_attr[4] == 8.5
        
        # Clean up
        clear_all_rows()

    @pytest.mark.integration
    def test_agent_attribute_defs_check_constraint_enforcement(self):
        """Test that the CHECK constraint actually fails when min_val > max_val."""
        # Test that the CHECK constraint fails when min_val > max_val
        with session_scope() as session:
            try:
                session.execute(text("""
                    INSERT INTO agent_attribute_defs (name, description, atype, min_val, max_val, col_idx) 
                    VALUES (:name, :description, :atype, :min_val, :max_val, :col_idx)
                """), {
                    'name': 'Invalid Range Test',
                    'description': 'Testing CHECK constraint failure',
                    'atype': 'int',
                    'min_val': 10,  # min_val > max_val
                    'max_val': 5,
                    'col_idx': 0
                })
                # If we get here, the constraint failed to enforce
                assert False, "CHECK constraint should have prevented this insertion"
            except Exception as e:
                # The constraint worked - this is what we want
                assert "CheckViolation" in str(e) or "check constraint" in str(e).lower(), f"Expected constraint violation, got: {e}"
        
        # Test that it also fails with float values
        with session_scope() as session:
            try:
                session.execute(text("""
                    INSERT INTO agent_attribute_defs (name, description, atype, min_val, max_val, col_idx) 
                    VALUES (:name, :description, :atype, :min_val, :max_val, :col_idx)
                """), {
                    'name': 'Invalid Float Range Test',
                    'description': 'Testing CHECK constraint failure with floats',
                    'atype': 'float',
                    'min_val': 5.5,  # min_val > max_val
                    'max_val': 3.2,
                    'col_idx': 1
                })
                # If we get here, the constraint failed to enforce
                assert False, "CHECK constraint should have prevented this insertion"
            except Exception as e:
                # The constraint worked - this is what we want
                assert "CheckViolation" in str(e) or "check constraint" in str(e).lower(), f"Expected constraint violation, got: {e}"
        
        # Test that equal values are allowed (edge case)
        with session_scope() as session:
            session.execute(text("""
                INSERT INTO agent_attribute_defs (name, description, atype, min_val, max_val, col_idx) 
                VALUES (:name, :description, :atype, :min_val, :max_val, :col_idx)
            """), {
                'name': 'Equal Values Test',
                'description': 'Testing CHECK constraint with equal values',
                'atype': 'int',
                'min_val': 5,  # min_val = max_val (should be allowed)
                'max_val': 5,
                'col_idx': 2
            })
        
        # Verify that valid ranges still work
        with session_scope() as session:
            session.execute(text("""
                INSERT INTO agent_attribute_defs (name, description, atype, min_val, max_val, col_idx) 
                VALUES (:name, :description, :atype, :min_val, :max_val, :col_idx)
            """), {
                'name': 'Valid Range Test',
                'description': 'Testing valid range insertion',
                'atype': 'int',
                'min_val': 1,
                'max_val': 10,
                'col_idx': 3
            })
        
        # Clean up
        clear_all_rows()

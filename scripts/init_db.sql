-- SwimVision Pro - Database Initialization Script
-- This script runs when the PostgreSQL container is first created

-- Create extensions if needed
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";

-- Create a simple test to verify database is working
DO $$
BEGIN
    RAISE NOTICE 'SwimVision database initialized successfully!';
END $$;

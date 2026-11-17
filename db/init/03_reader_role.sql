-- A login the assistant uses to query the warehouse. It can read every table and change nothing,
-- so even a bad generated query cannot modify data. Change the password outside local development.
DO $$
BEGIN
    IF NOT EXISTS (SELECT FROM pg_roles WHERE rolname = 'rag_reader') THEN
        CREATE ROLE rag_reader LOGIN PASSWORD 'rag_reader';
    END IF;
    EXECUTE format('GRANT CONNECT ON DATABASE %I TO rag_reader', current_database());
END
$$;
GRANT USAGE ON SCHEMA public TO rag_reader;
GRANT SELECT ON ALL TABLES IN SCHEMA public TO rag_reader;
ALTER ROLE rag_reader SET default_transaction_read_only = on;
ALTER ROLE rag_reader SET statement_timeout = '5s';

import os
import psycopg


def get_table_schema_as_dict():
    conn_params = {
        "host": os.getenv("DB_HOST"),
        "port": os.getenv("DB_PORT"),
        "dbname": os.getenv("DB_NAME"),
        "user": os.getenv("DB_USER"),
        "password": os.getenv("DB_PASSWORD"),
    }

    schema_data = {}

    try:
        with psycopg.connect(**conn_params) as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    SELECT table_schema, table_name
                    FROM information_schema.tables
                    WHERE table_type = 'BASE TABLE' AND table_schema NOT IN ('pg_catalog', 'information_schema')
                    ORDER BY table_schema, table_name;
                """
                )
                tables = cur.fetchall()

                for schema, table in tables:
                    schema_data[f"{schema}.{table}"] = []
                    cur.execute(
                        """
                        SELECT column_name, data_type, is_nullable, column_default
                        FROM information_schema.columns
                        WHERE table_schema = %s AND table_name = %s
                        ORDER BY ordinal_position;
                    """,
                        (schema, table),
                    )
                    columns = cur.fetchall()

                    for col in columns:
                        schema_data[f"{schema}.{table}"].append(
                            {
                                "column_name": col[0],
                                "data_type": col[1],
                                "is_nullable": col[2],
                                "column_default": col[3],
                            }
                        )

                    return schema_data

    except Exception:
        return False

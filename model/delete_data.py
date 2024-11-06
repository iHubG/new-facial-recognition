import sqlite3

def reset_database(db_name, table_names):
    # Connect to the SQLite database
    conn = sqlite3.connect(db_name)
    cursor = conn.cursor()

    try:
        # Disable foreign key constraints
        cursor.execute("PRAGMA foreign_keys = OFF;")
        
        # Delete all records from each specified table
        for table in table_names:
            cursor.execute(f"DELETE FROM {table};")
            print(f"Deleted all records from table: {table}")

            # Reset the primary key counter for AUTOINCREMENT columns
            cursor.execute(f"DELETE FROM sqlite_sequence WHERE name='{table}';")
            print(f"Reset primary key for table: {table}")

        # Optional: Reclaim storage space
        cursor.execute("VACUUM;")
        print("Database vacuumed to reclaim space.")

    except sqlite3.Error as e:
        print(f"An error occurred: {e}")
    finally:
        # Enable foreign key constraints again
        cursor.execute("PRAGMA foreign_keys = ON;")
        
        # Commit changes and close the connection
        conn.commit()
        conn.close()
        print("Database connection closed.")

if __name__ == "__main__":
    # Specify your database name and table names here
    database_name = 'your_database.db'  # Change this to your database name
    tables_to_clear = ['table1', 'table2', 'table3']  # Add your table names here

    reset_database(database_name, tables_to_clear)

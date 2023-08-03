import sqlite3

# Function to create a SQLite database connection
def create_connection(database_name):
    try:
        connection = sqlite3.connect(database_name)
        return connection
    except sqlite3.Error as e:
        print(e)
        return None

# Create the resumes table in the database
def create_resumes_table(connection):
    if connection is not None:
        create_table_query = """
            CREATE TABLE IF NOT EXISTS resumes (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                name TEXT,
                gpa REAL,
                email TEXT,
                resume_text TEXT
            );
        """
        try:
            cursor = connection.cursor()
            cursor.execute(create_table_query)
            connection.commit()
            cursor.close()
        except sqlite3.Error as e:
            print(e)

# Function to store resume and information in the database
def insert_resume(connection, candidate_info):
    insert_query = """
        INSERT INTO resumes (name, gpa, email, resume_text)
        VALUES (?, ?, ?, ?);
    """
    cursor = connection.cursor()
    cursor.execute(insert_query, (
        candidate_info["name"],
        candidate_info["gpa"],
        candidate_info["email"],
        candidate_info["resume_text"]
    ))
    connection.commit()
    cursor.close()

# Function to retrieve all resumes from the database
def get_all_resumes(connection):
    select_query = "SELECT * FROM resumes;"
    cursor = connection.cursor()
    cursor.execute(select_query)
    resumes = cursor.fetchall()
    cursor.close()
    return resumes

import sqlite3

# Function to retrieve candidate information from the database based on the candidate's name
def get_candidate_info_from_database(connection, candidate_name):
    select_query = "SELECT name, gpa, email, resume_text FROM resumes WHERE name = ?;"
    cursor = connection.cursor()
    cursor.execute(select_query, (candidate_name,))
    candidate_info = cursor.fetchone()
    cursor.close()

    if candidate_info:
        # Convert the fetched data to a dictionary
        candidate_info_dict = {
            'name': candidate_info[0],
            'gpa': candidate_info[1],
            'email': candidate_info[2],
            'resume_text': candidate_info[3]
        }
        return candidate_info_dict
    else:
        return None

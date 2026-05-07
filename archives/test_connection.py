import psycopg2

from modules.config import get_database_url

conn = psycopg2.connect(get_database_url())
print("Connected!")
conn.close()

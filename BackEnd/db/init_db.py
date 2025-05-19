# db/init_db.py
from BackEnd.db.models import db

def initialize_database():
    db.create_all()
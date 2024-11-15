from sqlalchemy import create_engine
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
import os

from dotenv import load_dotenv

# Load enviroment variable from .env
load_dotenv()

# Get the database URL from enviroment variables
SQLALCHEMY_DATABASE_URL = os.getenv("POSTGRESQL_CONNECTION_URL")

# Creating the Engine
engine = create_engine(SQLALCHEMY_DATABASE_URL)

# Creating the Session Local
SessionLocal = sessionmaker(bind=engine, autocommit=False, autoflush=False)

# Declarative Base
Base = declarative_base()

def get_db():
    db = SessionLocal() # Create an instance of SessionLocal

    try:
        yield db
    finally:
        db.close()
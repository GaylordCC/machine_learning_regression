from sqlalchemy import Column, String, Integer
from .database import Base

class User(Base):
    __tablename__ = "users"
    id = Column(Integer, primary_key=True, index=True)
    name = Column(String, nullable=False)
    email = Column(String, nullable=False)
    # Never store plaintext. Populate via core.security.hash_password().
    hashed_password = Column(String, nullable=False)
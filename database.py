# database.py
from sqlalchemy import create_engine, Column, String, Integer, DateTime
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
from config import config
from datetime import datetime

DATABASE_URL = config["database_url"]

# Determine if SQLite is used to set connect_args appropriately
if DATABASE_URL.startswith("sqlite"):
    engine = create_engine(DATABASE_URL, connect_args={"check_same_thread": False})
else:
    engine = create_engine(DATABASE_URL)

SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()

class ProcessedVideo(Base):
    __tablename__ = "processed_videos"
    id = Column(Integer, primary_key=True, index=True)
    blob_name = Column(String, unique=True, index=True, nullable=False)
    processed_at = Column(DateTime, default=datetime.utcnow)

# Create tables
Base.metadata.create_all(bind=engine)

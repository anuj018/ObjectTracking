# database.py
import os
import sqlalchemy as sa
from sqlalchemy import create_engine
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker

# Get database URL from environment variable
DATABASE_URL = os.getenv("DATABASE_URL")
if not DATABASE_URL:
    # Fallback for development or if env var isn't set
    DATABASE_URL = "postgresql://tracking_user:your_secure_password@localhost:5432/tracking_system"

engine = create_engine(DATABASE_URL)
Session = sessionmaker(bind=engine)
Base = declarative_base()

# Define your models
class Store(Base):
    __tablename__ = "stores"
    
    store_id = sa.Column(sa.Integer, primary_key=True)
    store_name = sa.Column(sa.String, nullable=False)
    
class Camera(Base):
    __tablename__ = "cameras"
    
    camera_id = sa.Column(sa.Integer, primary_key=True)
    store_id = sa.Column(sa.Integer, sa.ForeignKey("stores.store_id"))
    camera_name = sa.Column(sa.String)

class Track(Base):
    __tablename__ = "tracks"
    
    track_id = sa.Column(sa.Integer, primary_key=True)
    camera_id = sa.Column(sa.Integer, sa.ForeignKey("cameras.camera_id"))
    first_seen = sa.Column(sa.DateTime)
    last_seen = sa.Column(sa.DateTime)
    is_active = sa.Column(sa.Boolean, default=True)

class Feature(Base):
    __tablename__ = "features"
    
    feature_id = sa.Column(sa.Integer, primary_key=True)
    track_id = sa.Column(sa.Integer, sa.ForeignKey("tracks.track_id"))
    feature_data = sa.Column(sa.LargeBinary)  # For numpy arrays
    timestamp = sa.Column(sa.DateTime)

def init_db():
    """Initialize the database tables if they don't exist."""
    Base.metadata.create_all(engine)

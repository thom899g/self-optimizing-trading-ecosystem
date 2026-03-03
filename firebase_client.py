"""
Firebase Firestore client for state management and real-time data streaming.
Implements robust error handling and connection management.
"""
import logging
import json
from typing import Dict, Any, Optional, List, Union
from datetime import datetime
from dataclasses import dataclass, asdict
from pathlib import Path

import firebase_admin
from firebase_admin import credentials, firestore
from firebase_admin.exceptions import FirebaseError
from google.cloud.firestore_v1 import Client as FirestoreClient
from google.cloud.firestore_v1.base_query import FieldFilter

from config import config

logger = logging.getLogger(__name__)

@dataclass
class FirestoreDocument:
    """Base document structure with metadata"""
    id: Optional[str] = None
    created_at: datetime = field(default_factory=datetime.utcnow)
    updated_at: datetime = field(default_factory=datetime.utcnow)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert dataclass to Firestore-compatible dictionary"""
        data = asdict(self)
        # Convert datetime to ISO format strings
        for key, value in data.items():
            if isinstance(value, datetime):
                data[key] = value.isoformat()
        return data
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'FirestoreDocument':
        """Create instance from Firestore dictionary"""
        # Convert ISO strings back to datetime
        for key in ['created_at', 'updated_at']:
            if key in data and isinstance(data[key], str):
                data[key] = datetime.fromisoformat(data[key])
        return cls(**data)

class FirebaseClient:
    """Singleton Firebase client with connection management"""
    
    _instance: Optional['FirebaseClient'] = None
    _client: Optional[FirestoreClient] = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance =
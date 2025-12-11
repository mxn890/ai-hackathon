import os
import asyncpg
from dotenv import load_dotenv
import hashlib

load_dotenv()

class DatabaseService:
    def __init__(self):
        self.connection_string = os.getenv("DATABASE_URL")
        self.pool = None
        
    async def get_pool(self):
        if not self.pool:
            self.pool = await asyncpg.create_pool(self.connection_string)
        return self.pool
    
    async def initialize_tables(self):
        """Create necessary tables if they don't exist"""
        pool = await self.get_pool()
        async with pool.acquire() as conn:
            # Users table
            await conn.execute("""
                CREATE TABLE IF NOT EXISTS users (
                    id SERIAL PRIMARY KEY,
                    name TEXT NOT NULL,
                    email TEXT UNIQUE NOT NULL,
                    password_hash TEXT NOT NULL,
                    experience_level TEXT,
                    software_background TEXT,
                    hardware_background TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)
            
            await conn.execute("""
                CREATE TABLE IF NOT EXISTS chat_history (
                    id SERIAL PRIMARY KEY,
                    user_id INTEGER,
                    user_message TEXT NOT NULL,
                    bot_response TEXT NOT NULL,
                    selected_text TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (user_id) REFERENCES users (id)
                )
            """)
            
            await conn.execute("""
                CREATE TABLE IF NOT EXISTS documents (
                    id SERIAL PRIMARY KEY,
                    title TEXT NOT NULL,
                    content TEXT NOT NULL,
                    qdrant_id TEXT UNIQUE,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)
            
        print("✅ Database tables initialized in Neon Postgres")
    
    def hash_password(self, password: str) -> str:
        """Hash password using SHA256"""
        return hashlib.sha256(password.encode()).hexdigest()
    
    async def create_user(self, name: str, email: str, password: str, 
                         experience_level: str, software_background: str, 
                         hardware_background: str):
        """Create a new user"""
        pool = await self.get_pool()
        async with pool.acquire() as conn:
            password_hash = self.hash_password(password)
            
            try:
                await conn.execute("""
                    INSERT INTO users (name, email, password_hash, experience_level, 
                                     software_background, hardware_background)
                    VALUES ($1, $2, $3, $4, $5, $6)
                """, name, email, password_hash, experience_level, 
                     software_background, hardware_background)
                return {"success": True, "message": "User created successfully"}
            except Exception as e:
                if "duplicate key" in str(e).lower():
                    return {"success": False, "message": "Email already exists"}
                return {"success": False, "message": str(e)}
    
    async def authenticate_user(self, email: str, password: str):
        """Authenticate user and return user data"""
        pool = await self.get_pool()
        async with pool.acquire() as conn:
            password_hash = self.hash_password(password)
            
            row = await conn.fetchrow("""
                SELECT id, name, email, experience_level, 
                       software_background, hardware_background
                FROM users 
                WHERE email = $1 AND password_hash = $2
            """, email, password_hash)
            
            if row:
                return {
                    "success": True,
                    "user": {
                        "id": row['id'],
                        "name": row['name'],
                        "email": row['email'],
                        "experienceLevel": row['experience_level'],
                        "softwareBackground": row['software_background'],
                        "hardwareBackground": row['hardware_background']
                    }
                }
            else:
                return {"success": False, "message": "Invalid email or password"}
    
    async def save_chat(self, user_message: str, bot_response: str, 
                       selected_text: str = None, user_id: int = None):
        """Save chat interaction to database"""
        pool = await self.get_pool()
        async with pool.acquire() as conn:
            await conn.execute("""
                INSERT INTO chat_history (user_id, user_message, bot_response, selected_text)
                VALUES ($1, $2, $3, $4)
            """, user_id, user_message, bot_response, selected_text)
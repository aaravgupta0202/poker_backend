from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import redis
import os
import uuid
import time
import threading
import random
import asyncio
from typing import Dict, List, Optional
from enum import Enum
from pydantic import BaseModel
import json
from dotenv import load_dotenv
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

# Redis configuration from environment
REDIS_HOST = os.getenv("REDIS_HOST", "localhost")
REDIS_PORT = int(os.getenv("REDIS_PORT", 6379))
REDIS_PASSWORD = os.getenv("REDIS_PASSWORD", "")
REDIS_DB = int(os.getenv("REDIS_DB", 0))

# Game configuration from environment
MAX_PLAYERS = int(os.getenv("MAX_PLAYERS", 8))
STARTING_CHIPS = int(os.getenv("STARTING_CHIPS", 1000))
TURN_TIMER_SECONDS = int(os.getenv("TURN_TIMER_SECONDS", 30))
ROOM_EXPIRY_SECONDS = int(os.getenv("ROOM_EXPIRY_HOURS", 2)) * 3600
SMALL_BLIND = int(os.getenv("SMALL_BLIND", 10))
BIG_BLIND = int(os.getenv("BIG_BLIND", 20))

logger.info(f"Redis Config: host={REDIS_HOST}, port={REDIS_PORT}")

# Redis connection with error handling
try:
    r = redis.Redis(
        host=REDIS_HOST,
        port=REDIS_PORT,
        password=REDIS_PASSWORD,
        db=REDIS_DB,
        decode_responses=True,
        socket_connect_timeout=5,
        socket_timeout=5,
        retry_on_timeout=True
    )
    
    # Test connection
    r.ping()
    logger.info("✅ Redis connection successful")
    
except Exception as e:
    logger.error(f"❌ Redis connection failed: {e}")
    logger.warning("⚠️ Using in-memory storage only")
    # Create a dummy Redis client that stores in memory
    class DummyRedis:
        def __init__(self):
            self.data = {}
            self.expirations = {}
        
        def setex(self, key, ttl, value):
            self.data[key] = value
            self.expirations[key] = time.time() + ttl
        
        def get(self, key):
            if key in self.expirations and time.time() > self.expirations[key]:
                del self.data[key]
                del self.expirations[key]
                return None
            return self.data.get(key)
        
        def exists(self, key):
            self.get(key)  # Clean expired keys
            return key in self.data
        
        def delete(self, key):
            if key in self.data:
                del self.data[key]
            if key in self.expirations:
                del self.expirations[key]
        
        def ttl(self, key):
            if key in self.expirations:
                remaining = self.expirations[key] - time.time()
                return max(0, int(remaining))
            return -2
        
        def ping(self):
            return True
    
    r = DummyRedis()

# FastAPI setup
app = FastAPI(title="Emoji Poker API", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, specify your frontend URLs
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Game models
class GamePhase(str, Enum):
    WAITING = "waiting"
    PREFLOP = "preflop"
    FLOP = "flop"
    TURN = "turn"
    RIVER = "river"
    SHOWDOWN = "showdown"
    ENDED = "ended"

class PlayerAction(str, Enum):
    FOLD = "fold"
    CHECK = "check"
    CALL = "call"
    RAISE = "raise"
    BET = "bet"
    TIMEOUT = "timeout"

class Player(BaseModel):
    id: str
    name: str
    chips: int = STARTING_CHIPS
    hand: List[str] = []
    is_active: bool = True
    is_in_hand: bool = True
    current_bet: int = 0
    last_action: Optional[str] = None
    is_host: bool = False
    joined_at: float = time.time()

class PokerGame(BaseModel):
    room_code: str
    players: Dict[str, Player] = {}
    player_order: List[str] = []
    dealer_position: int = 0
    current_turn_index: int = 0
    phase: GamePhase = GamePhase.WAITING
    community_cards: List[str] = []
    deck: List[str] = []
    pot: int = 0
    current_bet: int = 0
    small_blind: int = SMALL_BLIND
    big_blind: int = BIG_BLIND
    created_at: float = time.time()
    last_action_time: float = time.time()
    game_started: bool = False
    round_started: bool = False
    turn_timer: int = TURN_TIMER_SECONDS
    max_players: int = MAX_PLAYERS

# Card deck setup
SUITS = ['♠️', '♥️', '♦️', '♣️']
RANKS = ['2', '3', '4', '5', '6', '7', '8', '9', '10', 'J', 'Q', 'K', 'A']

def create_deck() -> List[str]:
    """Create a deck of cards with emojis"""
    deck = []
    for suit in SUITS:
        for rank in RANKS:
            deck.append(f"{rank}{suit}")
    random.shuffle(deck)
    return deck

def create_player_order(game: PokerGame) -> List[str]:
    """Create player order starting from dealer"""
    active_players = [pid for pid, p in game.players.items() if p.is_active]
    if not active_players:
        return []
    
    dealer_index = game.dealer_position % len(active_players)
    order = active_players[dealer_index:] + active_players[:dealer_index]
    return order

def deal_cards(game: PokerGame):
    """Deal cards to all players"""
    logger.info(f"Dealing cards for room {game.room_code}")
    
    # Reset deck for new hand
    game.deck = create_deck()
    game.community_cards = []
    game.pot = 0
    game.current_bet = 0
    
    # Reset player states
    for player in game.players.values():
        player.hand = []
        player.current_bet = 0
        player.last_action = None
        player.is_in_hand = True
    
    # Deal 2 cards to each active player
    for player in game.players.values():
        if player.is_active:
            player.hand = [game.deck.pop(), game.deck.pop()]
    
    # Set blinds if we have enough players
    active_players = [p for p in game.players.values() if p.is_active]
    if len(active_players) >= 2:
        # Small blind
        sb_player = active_players[game.dealer_position % len(active_players)]
        sb_player.chips -= game.small_blind
        sb_player.current_bet = game.small_blind
        game.pot += game.small_blind
        
        # Big blind
        bb_player = active_players[(game.dealer_position + 1) % len(active_players)]
        bb_player.chips -= game.big_blind
        bb_player.current_bet = game.big_blind
        game.current_bet = game.big_blind
        game.pot += game.big_blind
    
    # Set player order for betting
    game.player_order = create_player_order(game)
    game.current_turn_index = 2 % len(game.player_order) if len(game.player_order) > 2 else 0
    game.phase = GamePhase.PREFLOP
    game.round_started = True
    game.last_action_time = time.time()
    
    logger.info(f"Cards dealt. Player order: {game.player_order}")

# In-memory storage for active games
active_games: Dict[str, PokerGame] = {}

@app.get("/")
async def root():
    """Health check endpoint"""
    return {
        "message": "🎰 Emoji Poker API",
        "status": "running",
        "version": "1.0.0",
        "active_games": len(active_games),
        "redis_connected": hasattr(r, 'ping') and r.ping()
    }

@app.get("/health")
async def health_check():
    """Health check with Redis status"""
    redis_status = "connected" if hasattr(r, 'ping') and r.ping() else "disconnected"
    return {
        "status": "healthy",
        "timestamp": time.time(),
        "redis": redis_status,
        "active_games": len(active_games)
    }

@app.post("/poker/create")
async def create_poker_room(player_name: str):
    """Create a new poker room"""
    logger.info(f"Creating room for player: {player_name}")
    
    try:
        # Generate room code and player ID
        room_code = str(uuid.uuid4())[:6].upper()
        player_id = str(uuid.uuid4())[:8]
        
        logger.info(f"Generated room code: {room_code}, player_id: {player_id}")
        
        # Store room in Redis (expires in 2 hours)
        r.setex(f"poker:room:{room_code}", ROOM_EXPIRY_SECONDS, "active")
        
        # Create game and add host player
        game = PokerGame(room_code=room_code)
        player = Player(
            id=player_id,
            name=player_name,
            chips=STARTING_CHIPS,
            is_host=True
        )
        game.players[player_id] = player
        game.player_order.append(player_id)
        
        # Store in active games
        active_games[room_code] = game
        
        logger.info(f"Room created successfully: {room_code} with player {player_name}")
        
        return {
            "success": True,
            "room_code": room_code,
            "player_id": player_id,
            "is_host": True,
            "message": f"Room {room_code} created successfully"
        }
        
    except Exception as e:
        logger.error(f"Error creating room: {e}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"Failed to create room: {str(e)}"
        )

@app.get("/poker/room/{room_code}/exists")
async def check_room_exists(room_code: str):
    """Check if room exists"""
    logger.info(f"Checking if room exists: {room_code}")
    
    try:
        exists = r.exists(f"poker:room:{room_code}")
        
        if not exists:
            logger.info(f"Room {room_code} not found in Redis")
            # Also check in-memory games (in case Redis is down)
            exists = room_code in active_games
        
        if not exists:
            return {
                "exists": False,
                "message": "Room not found"
            }
        
        game = active_games.get(room_code)
        
        response = {
            "exists": True,
            "player_count": len(game.players) if game else 0,
            "game_started": game.game_started if game else False,
            "max_players": MAX_PLAYERS,
            "message": "Room exists"
        }
        
        if game:
            response["players"] = [
                {"id": pid, "name": p.name, "is_host": p.is_host}
                for pid, p in game.players.items()
            ]
        
        logger.info(f"Room {room_code} exists: {response}")
        return response
        
    except Exception as e:
        logger.error(f"Error checking room: {e}", exc_info=True)
        return {
            "exists": False,
            "error": str(e)
        }

@app.post("/poker/room/{room_code}/join")
async def join_room(room_code: str, player_name: str):
    """Join a poker room"""
    logger.info(f"Player {player_name} joining room {room_code}")
    
    try:
        # Check if room exists
        exists = r.exists(f"poker:room:{room_code}")
        if not exists and room_code not in active_games:
            logger.warning(f"Room {room_code} not found")
            raise HTTPException(
                status_code=404,
                detail="Room not found. Check the room code."
            )
        
        # Get or create game
        if room_code not in active_games:
            # Recreate game from Redis
            game = PokerGame(room_code=room_code)
            active_games[room_code] = game
            logger.info(f"Recreated game for room {room_code}")
        else:
            game = active_games[room_code]
        
        # Check if game has started
        if game.game_started:
            logger.warning(f"Game already started in room {room_code}")
            raise HTTPException(
                status_code=400,
                detail="Game has already started. Please wait for the next game."
            )
        
        # Check if room is full
        if len(game.players) >= game.max_players:
            logger.warning(f"Room {room_code} is full")
            raise HTTPException(
                status_code=400,
                detail=f"Room is full (max {game.max_players} players)"
            )
        
        # Generate player ID
        player_id = str(uuid.uuid4())[:8]
        
        # Add player
        player = Player(
            id=player_id,
            name=player_name,
            chips=STARTING_CHIPS
        )
        game.players[player_id] = player
        game.player_order.append(player_id)
        
        logger.info(f"Player {player_name} joined room {room_code}. Total players: {len(game.players)}")
        
        return {
            "success": True,
            "player_id": player_id,
            "room_code": room_code,
            "is_host": False,
            "player_count": len(game.players),
            "message": f"Joined room {room_code} successfully"
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error joining room: {e}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"Failed to join room: {str(e)}"
        )

@app.get("/poker/room/{room_code}/players")
async def get_room_players(room_code: str):
    """Get list of players in room"""
    logger.info(f"Getting players for room {room_code}")
    
    try:
        if room_code not in active_games:
            raise HTTPException(status_code=404, detail="Game not found")
        
        game = active_games[room_code]
        
        players = []
        for player in game.players.values():
            players.append({
                "id": player.id,
                "name": player.name,
                "chips": player.chips,
                "is_host": player.is_host,
                "is_active": player.is_active,
                "joined_at": player.joined_at
            })
        
        return {
            "success": True,
            "players": players,
            "count": len(players),
            "game_started": game.game_started
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting players: {e}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"Failed to get players: {str(e)}"
        )

# Add other endpoints (start, state, action, leave) here...

# Add a cleanup task for inactive games
async def cleanup_inactive_games():
    """Periodically clean up inactive games"""
    while True:
        await asyncio.sleep(300)  # Every 5 minutes
        try:
            current_time = time.time()
            games_to_remove = []
            
            for room_code, game in list(active_games.items()):
                # Remove games older than ROOM_EXPIRY_SECONDS
                if current_time - game.created_at > ROOM_EXPIRY_SECONDS:
                    games_to_remove.append(room_code)
                # Remove empty games
                elif len(game.players) == 0:
                    games_to_remove.append(room_code)
            
            for room_code in games_to_remove:
                logger.info(f"Cleaning up inactive game: {room_code}")
                del active_games[room_code]
                r.delete(f"poker:room:{room_code}")
                
        except Exception as e:
            logger.error(f"Error in cleanup task: {e}")

# Start cleanup task on startup
@app.on_event("startup")
async def startup_event():
    asyncio.create_task(cleanup_inactive_games())
    logger.info("🚀 Poker Backend started successfully")

if __name__ == "__main__":
    import uvicorn
    
    # Get port from environment or use default
    port = int(os.getenv("PORT", 8000))
    
    logger.info(f"Starting server on port {port}")
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=port,
        reload=True if os.getenv("DEBUG") == "True" else False
    )
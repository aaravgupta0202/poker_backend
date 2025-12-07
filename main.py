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

# Redis connection
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

# Test Redis connection
try:
    r.ping()
    print("✅ Redis connection successful")
except redis.ConnectionError as e:
    print(f"❌ Redis connection failed: {e}")
    print("⚠️  Starting without Redis persistence")

# FastAPI setup
app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Game models
class GamePhase(Enum):
    WAITING = "waiting"
    PREFLOP = "preflop"
    FLOP = "flop"
    TURN = "turn"
    RIVER = "river"
    SHOWDOWN = "showdown"
    ENDED = "ended"

class PlayerAction(Enum):
    FOLD = "fold"
    CHECK = "check"
    CALL = "call"
    RAISE = "raise"
    BET = "bet"
    TIMEOUT = "timeout"

class Player(BaseModel):
    id: str
    name: str
    chips: int = 1000
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
    player_order: List[str] = []  # Order of players for turns
    dealer_position: int = 0
    current_turn_index: int = 0
    phase: GamePhase = GamePhase.WAITING
    community_cards: List[str] = []
    deck: List[str] = []
    pot: int = 0
    current_bet: int = 0
    small_blind: int = 10
    big_blind: int = 20
    created_at: float = time.time()
    last_action_time: float = time.time()
    game_started: bool = False
    round_started: bool = False
    turn_timer: int = 30  # 30 seconds per turn
    max_players: int = 8

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
    
    # Start order from player after dealer
    dealer_index = game.dealer_position % len(active_players)
    order = active_players[dealer_index:] + active_players[:dealer_index]
    return order

def deal_cards(game: PokerGame):
    """Deal cards to all players"""
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
        sb_player = active_players[(game.dealer_position) % len(active_players)]
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
    game.current_turn_index = 2 % len(game.player_order)  # Start after blinds
    
    game.phase = GamePhase.PREFLOP
    game.round_started = True
    game.last_action_time = time.time()

def deal_community(game: PokerGame):
    """Deal community cards based on game phase"""
    if game.phase == GamePhase.PREFLOP:
        # Burn card
        game.deck.pop()
        # Deal flop
        game.community_cards = [game.deck.pop() for _ in range(3)]
        game.phase = GamePhase.FLOP
        
    elif game.phase == GamePhase.FLOP:
        # Burn card
        game.deck.pop()
        # Deal turn
        game.community_cards.append(game.deck.pop())
        game.phase = GamePhase.TURN
        
    elif game.phase == GamePhase.TURN:
        # Burn card
        game.deck.pop()
        # Deal river
        game.community_cards.append(game.deck.pop())
        game.phase = GamePhase.RIVER
    
    # Reset bets for new round
    for player in game.players.values():
        player.current_bet = 0
    game.current_bet = 0
    
    # Reset turn order
    game.player_order = [pid for pid in game.player_order if game.players[pid].is_in_hand]
    game.current_turn_index = 0
    game.last_action_time = time.time()

# In-memory storage for active games
active_games: Dict[str, PokerGame] = {}

# Turn timer management
async def check_turn_timeout():
    """Check for players who timed out on their turn"""
    while True:
        await asyncio.sleep(5)  # Check every 5 seconds
        
        for room_code, game in list(active_games.items()):
            if game.phase not in [GamePhase.WAITING, GamePhase.ENDED] and game.round_started:
                time_since_last_action = time.time() - game.last_action_time
                
                if time_since_last_action > game.turn_timer:
                    # Player timed out - auto check/fold
                    if game.player_order:
                        current_player_id = game.player_order[game.current_turn_index]
                        current_player = game.players[current_player_id]
                        
                        # Auto-check if possible, otherwise fold
                        if current_player.current_bet >= game.current_bet:
                            current_player.last_action = "timeout_check"
                            # Move to next player
                            move_to_next_player(game)
                        else:
                            current_player.is_in_hand = False
                            current_player.last_action = "timeout_fold"
                            move_to_next_player(game)
                        
                        game.last_action_time = time.time()
                        
                        # Check if round is over
                        if check_round_complete(game):
                            advance_game_phase(game)

@app.get("/")
def root():
    return {"message": "Poker Backend Running", "status": "healthy"}

@app.post("/poker/create")
def create_poker_room(player_name: str):
    """Create a new poker room"""
    room_code = str(uuid.uuid4())[:6].upper()
    player_id = str(uuid.uuid4())[:8]
    
    # Store room in Redis (expires in 2 hours)
    r.setex(f"poker:room:{room_code}", 7200, "active")
    
    # Create game and add host player
    game = PokerGame(room_code=room_code)
    player = Player(
        id=player_id,
        name=player_name,
        chips=1000,
        is_host=True
    )
    game.players[player_id] = player
    game.player_order.append(player_id)
    
    active_games[room_code] = game
    
    return {
        "room_code": room_code,
        "player_id": player_id,
        "is_host": True
    }

@app.get("/poker/room/{room_code}/exists")
def check_room_exists(room_code: str):
    """Check if room exists"""
    exists = r.exists(f"poker:room:{room_code}")
    if not exists:
        return {"exists": False}
    
    game = active_games.get(room_code)
    if not game:
        return {"exists": True, "status": "room_exists"}
    
    return {
        "exists": True,
        "player_count": len(game.players),
        "game_started": game.game_started,
        "max_players": game.max_players
    }

@app.post("/poker/room/{room_code}/join")
def join_room(room_code: str, player_name: str):
    """Join a poker room"""
    if not r.exists(f"poker:room:{room_code}"):
        raise HTTPException(status_code=404, detail="Room not found")
    
    if room_code not in active_games:
        # Recreate game from Redis
        game = PokerGame(room_code=room_code)
        active_games[room_code] = game
    else:
        game = active_games[room_code]
    
    # Check if game has started
    if game.game_started:
        raise HTTPException(status_code=400, detail="Game already started")
    
    # Check if room is full
    if len(game.players) >= game.max_players:
        raise HTTPException(status_code=400, detail="Room is full")
    
    # Generate player ID
    player_id = str(uuid.uuid4())[:8]
    
    # Add player
    player = Player(id=player_id, name=player_name, chips=1000)
    game.players[player_id] = player
    game.player_order.append(player_id)
    
    return {
        "player_id": player_id,
        "room_code": room_code,
        "is_host": False,
        "player_count": len(game.players)
    }

@app.post("/poker/room/{room_code}/start")
def start_game(room_code: str, player_id: str):
    """Start the poker game (host only)"""
    if room_code not in active_games:
        raise HTTPException(status_code=404, detail="Game not found")
    
    game = active_games[room_code]
    
    # Verify host
    if player_id not in game.players or not game.players[player_id].is_host:
        raise HTTPException(status_code=403, detail="Only host can start the game")
    
    # Check minimum players
    if len(game.players) < 2:
        raise HTTPException(status_code=400, detail="Need at least 2 players")
    
    # Start the game
    game.game_started = True
    deal_cards(game)
    
    return {
        "status": "started",
        "message": "Game started successfully"
    }

@app.get("/poker/room/{room_code}/state")
def get_game_state(room_code: str, player_id: str):
    """Get current game state for a player"""
    if room_code not in active_games:
        raise HTTPException(status_code=404, detail="Game not found")
    
    game = active_games[room_code]
    
    if player_id not in game.players:
        raise HTTPException(status_code=403, detail="Player not in game")
    
    player = game.players[player_id]
    
    # Calculate time remaining for current turn
    time_remaining = max(0, game.turn_timer - (time.time() - game.last_action_time))
    
    # Prepare player data
    players_data = []
    for pid, p in game.players.items():
        players_data.append({
            "id": pid,
            "name": p.name,
            "chips": p.chips,
            "current_bet": p.current_bet,
            "is_active": p.is_active,
            "is_in_hand": p.is_in_hand,
            "is_host": p.is_host,
            "last_action": p.last_action,
            "hand": p.hand if (game.phase == GamePhase.SHOWDOWN and p.is_in_hand) or pid == player_id else ["?", "?"]
        })
    
    # Check whose turn it is
    current_player_id = None
    if game.player_order and game.current_turn_index < len(game.player_order):
        current_player_id = game.player_order[game.current_turn_index]
    
    return {
        "room_code": room_code,
        "phase": game.phase.value,
        "community_cards": game.community_cards,
        "pot": game.pot,
        "current_bet": game.current_bet,
        "players": players_data,
        "current_player_id": current_player_id,
        "is_your_turn": current_player_id == player_id,
        "time_remaining": int(time_remaining),
        "game_started": game.game_started,
        "player_order": game.player_order,
        "current_turn_index": game.current_turn_index,
        "small_blind": game.small_blind,
        "big_blind": game.big_blind
    }

@app.post("/poker/room/{room_code}/action")
def player_action(room_code: str, player_id: str, action: str, amount: int = 0):
    """Process player action"""
    if room_code not in active_games:
        raise HTTPException(status_code=404, detail="Game not found")
    
    game = active_games[room_code]
    
    if player_id not in game.players:
        raise HTTPException(status_code=403, detail="Player not in game")
    
    # Check if it's player's turn
    if not game.player_order or game.current_turn_index >= len(game.player_order):
        raise HTTPException(status_code=400, detail="Not in betting round")
    
    current_player_id = game.player_order[game.current_turn_index]
    if current_player_id != player_id:
        raise HTTPException(status_code=400, detail="Not your turn")
    
    player = game.players[player_id]
    
    # Process action
    if action == "fold":
        player.is_in_hand = False
        player.last_action = "folded"
        
    elif action == "check":
        if player.current_bet < game.current_bet:
            raise HTTPException(status_code=400, detail="Cannot check, must call or fold")
        player.last_action = "checked"
        
    elif action == "call":
        call_amount = game.current_bet - player.current_bet
        if player.chips < call_amount:
            # Player goes all-in
            call_amount = player.chips
            player.is_in_hand = True  # But marked for side pot logic
        
        player.chips -= call_amount
        player.current_bet += call_amount
        game.pot += call_amount
        player.last_action = f"called {call_amount}"
        
    elif action == "raise":
        total_amount = amount
        if total_amount <= game.current_bet:
            raise HTTPException(status_code=400, detail="Raise must be higher than current bet")
        if player.chips < total_amount:
            raise HTTPException(status_code=400, detail="Not enough chips")
        
        raise_amount = total_amount - player.current_bet
        player.chips -= raise_amount
        player.current_bet = total_amount
        game.current_bet = total_amount
        game.pot += raise_amount
        player.last_action = f"raised to {total_amount}"
    
    # Update last action time
    game.last_action_time = time.time()
    
    # Move to next player
    move_to_next_player(game)
    
    # Check if round is complete
    if check_round_complete(game):
        advance_game_phase(game)
    
    return {"status": "success"}

def move_to_next_player(game: PokerGame):
    """Move to next active player in hand"""
    # Find next player who is still in the hand
    start_index = game.current_turn_index
    players_in_hand = len([pid for pid in game.player_order if game.players[pid].is_in_hand])
    
    if players_in_hand <= 1:
        # Only one player left, hand ends
        advance_game_phase(game)
        return
    
    # Find next player
    for i in range(1, len(game.player_order) + 1):
        next_index = (start_index + i) % len(game.player_order)
        next_player_id = game.player_order[next_index]
        next_player = game.players[next_player_id]
        
        if next_player.is_in_hand:
            game.current_turn_index = next_index
            break

def check_round_complete(game: PokerGame) -> bool:
    """Check if betting round is complete"""
    players_in_hand = [pid for pid in game.player_order if game.players[pid].is_in_hand]
    
    if len(players_in_hand) <= 1:
        return True
    
    # Check if all players have matched the current bet or are all-in
    for player_id in players_in_hand:
        player = game.players[player_id]
        if player.current_bet < game.current_bet and player.chips > 0:
            return False
    
    # Check if we've completed a full round of betting
    # (everyone has had a chance to act since last raise)
    return True

def advance_game_phase(game: PokerGame):
    """Advance to next game phase"""
    if game.phase == GamePhase.PREFLOP:
        deal_community(game)  # Deal flop
    elif game.phase == GamePhase.FLOP:
        deal_community(game)  # Deal turn
    elif game.phase == GamePhase.TURN:
        deal_community(game)  # Deal river
    elif game.phase == GamePhase.RIVER:
        game.phase = GamePhase.SHOWDOWN
        determine_winner(game)
    elif game.phase == GamePhase.SHOWDOWN:
        start_new_hand(game)

def determine_winner(game: PokerGame):
    """Determine winner of the hand"""
    players_in_hand = [pid for pid in game.player_order if game.players[pid].is_in_hand]
    
    if len(players_in_hand) == 1:
        # Single player wins
        winner_id = players_in_hand[0]
        winner = game.players[winner_id]
        winner.chips += game.pot
        winner.last_action = f"won {game.pot} chips"
    else:
        # For simplicity, award pot to random player (implement proper hand evaluation)
        winner_id = random.choice(players_in_hand)
        winner = game.players[winner_id]
        winner.chips += game.pot
        winner.last_action = f"won {game.pot} chips"
    
    game.pot = 0

def start_new_hand(game: PokerGame):
    """Start a new hand"""
    # Move dealer button
    game.dealer_position = (game.dealer_position + 1) % len(game.players)
    
    # Reset for new hand
    deal_cards(game)

@app.post("/poker/room/{room_code}/leave")
def leave_room(room_code: str, player_id: str):
    """Player leaves the room"""
    if room_code not in active_games:
        raise HTTPException(status_code=404, detail="Game not found")
    
    game = active_games[room_code]
    
    if player_id in game.players:
        player = game.players[player_id]
        
        # If host leaves and game hasn't started, assign new host
        if player.is_host and not game.game_started and len(game.players) > 1:
            # Find another player to be host
            for pid, p in game.players.items():
                if pid != player_id:
                    p.is_host = True
                    break
        
        # Remove player
        game.player_order = [pid for pid in game.player_order if pid != player_id]
        del game.players[player_id]
        
        # Update current turn index
        if game.current_turn_index >= len(game.player_order):
            game.current_turn_index = max(0, len(game.player_order) - 1)
        
        # If no players left, remove game
        if not game.players:
            del active_games[room_code]
            r.delete(f"poker:room:{room_code}")
    
    return {"status": "left"}

@app.get("/poker/room/{room_code}/players")
def get_room_players(room_code: str):
    """Get list of players in room"""
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
            "is_active": player.is_active
        })
    
    return {"players": players, "count": len(players)}

# Start timeout checker
@app.on_event("startup")
async def startup_event():
    asyncio.create_task(check_turn_timeout())

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
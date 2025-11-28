# mutemplate/web_api.py

import os
import sys
import uuid
import logging
import random
from pathlib import Path
from typing import Dict, List, Optional, Any

from fastapi import FastAPI, HTTPException, Query
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

logging.basicConfig(level=logging.INFO)

# ---------- path setup ----------
CURRENT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = CURRENT_DIR.parent
ENGINE_DIR = PROJECT_ROOT / "muforge"

for p in (PROJECT_ROOT, ENGINE_DIR):
    p_str = str(p)
    if p_str not in sys.path:
        sys.path.insert(0, p_str)

# ---------- app ----------
app = FastAPI(title="Muforge Web API")
STATIC_DIR = CURRENT_DIR / "static"
app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------- try to import real engine ----------
loader_module = None
commands_module = None
models_module = None

try:
    loader_module = __import__("muforge.loader", fromlist=["*"])
    logging.info("✅ imported muforge.loader")
except ImportError:
    logging.warning("⚠ could not import muforge.loader")

try:
    commands_module = __import__("muforge.commands", fromlist=["*"])
    logging.info("✅ imported muforge.commands")
except ImportError:
    logging.warning("⚠ could not import muforge.commands")

try:
    models_module = __import__("muforge.models", fromlist=["*"])
    logging.info("✅ imported muforge.models")
except ImportError:
    logging.warning("⚠ could not import muforge.models")

# ---------- fallback world (used if your loader doesn't give nodes) ----------
WORLD: Dict[str, Dict[str, Any]] = {
    "node.planet.terra": {
        "id": "node.planet.terra",
        "name": "Terra",
        "desc": "A well-mapped starter world. Entry point for all adventurers.",
        "lore": "Terra is the hub of expansion; guilds recruit here.",
        "grid": {"region": "Core", "x": 0, "y": 0},
        "exits": {
            "nova_city": "node.city.nova",
            "raider_outskirts": "node.field.raiders"
        },
        "kind": "planet",
    },
    "node.city.nova": {
        "id": "node.city.nova",
        "name": "Nova City",
        "desc": "Bustling neon market, lots of rumors and contracts.",
        "lore": "Nova grew around the first jump gate.",
        "grid": {"region": "Core", "x": 1, "y": 0},
        "exits": {
            "back_to_terra": "node.planet.terra",
            "lower_district": "node.city.lower"
        },
        "kind": "city",
    },
    "node.field.raiders": {
        "id": "node.field.raiders",
        "name": "Raider Outskirts",
        "desc": "Sparse dunes. Raider signals detected.",
        "lore": "Patrolled by mercs... and raiders.",
        "grid": {"region": "Outer", "x": -1, "y": 1},
        "exits": {
            "back_to_terra": "node.planet.terra"
        },
        "kind": "field",
    },
    "node.city.lower": {
        "id": "node.city.lower",
        "name": "Nova: Lower District",
        "desc": "Shady alleys where info is cheap.",
        "lore": "Fixers operate here.",
        "grid": {"region": "Core", "x": 1, "y": -1},
        "exits": {
            "nova_plaza": "node.city.nova"
        },
        "kind": "district",
    },
}

# ---------- detect loader functions ----------
if loader_module:
    loader_fn = (
        getattr(loader_module, "load_all_nodes", None)
        or getattr(loader_module, "load_nodes", None)
        or getattr(loader_module, "load_world", None)
        or getattr(loader_module, "load", None)
    )
else:
    loader_fn = None

if loader_fn is None:
    def loader_fn():
        logging.info("using built-in WORLD")
else:
    logging.info("using muforge loader")

# try to get node accessor from muforge
if loader_module:
    get_node_real = (
        getattr(loader_module, "get_node", None)
        or getattr(loader_module, "get", None)
        or getattr(loader_module, "get_room", None)
    )
else:
    get_node_real = None

def get_node(node_id: str):
    if get_node_real:
        return get_node_real(node_id)
    # fallback to our WORLD
    return WORLD.get(node_id)

# command execution
if commands_module and hasattr(commands_module, "execute_command"):
    execute_command_real = commands_module.execute_command
else:
    execute_command_real = None

def execute_command_simple(command: str, args, player: dict, session: dict) -> dict:
    """
    Handle simple commands from /api/game/command.

    We use this for inventory item usage (use <item name>),
    working directly on the player dict stored in the session.
    """

    text = (command or "").strip()
    if not text:
        return {"ok": False, "msg": "Empty command."}

    parts = text.split()
    cmd = parts[0].lower()
    extra = parts[1:]

    # args may come from JSON body; merge both sources
    arg_list = list(extra) + list(args or [])

    # For now, only support 'use'
    if cmd != "use":
        return {
            "ok": False,
            "msg": f"Command '{cmd}' is not supported via this endpoint."
        }

    if not arg_list:
        return {"ok": False, "msg": "Use what?"}

    item_name = " ".join(arg_list).strip()
    if not item_name:
        return {"ok": False, "msg": "Use what?"}

    # --- find item in inventory -------------------------------------------
    inventory = player.get("inventory") or []
    idx = None
    item = None

    for i, it in enumerate(inventory):
        name = (it.get("name") or it.get("item") or "").strip()
        if name.lower() == item_name.lower():
            idx = i
            item = it
            break

    if item is None:
        return {"ok": False, "msg": f"You don't have a {item_name}."}

    qty = item.get("qty") or item.get("count") or item.get("amount") or 1

    # --- load stats -------------------------------------------------------
    attrs = player.get("attributes") or {}

    health = attrs.get("health", player.get("health", 100))
    max_health = attrs.get("max_health", player.get("max_health", 100))
    credits = attrs.get("credits", player.get("credits", 0))
    attack = attrs.get("attack", player.get("attack", 10))
    armor = attrs.get("armor", player.get("armor", 0))

    name = item.get("name") or item_name

    heal = 0
    credits_gain = 0
    attack_gain = 0
    max_hp_gain = 0

    # --- item effects -----------------------------------------------------
    if name == "Medpack":
        heal = 25 * qty
    elif name == "Nano Repair Kit":
        heal = 75 * qty
    elif name == "Energy Cell":
        heal = 10 * qty
    elif name == "Charge Cell":
        heal = 5 * qty

    elif name == "Credits":
        credits_gain = 1 * qty
    elif name == "Iron Scrap":
        credits_gain = 2 * qty
    elif name == "Scrap":
        credits_gain = 3 * qty
    elif name == "Scrap Alloy":
        credits_gain = 5 * qty

    elif name in ("Weapon", "Plasma Blaster", "Blaster"):
        # basic weapon vs blaster
        attack_gain = 10 * qty if name in ("Blaster", "Plasma Blaster") else 5 * qty

    elif name in ("Armor", "Energy Shield"):
        max_hp_gain = 20 * qty

    else:
        return {"ok": False, "msg": f"{name} cannot be used directly."}

    msg_parts = [f"You use {qty}× {name}."]

    # healing, capped by max HP (including any boost)
    if heal > 0:
        new_hp = min(max_health + max_hp_gain, health + heal)
        actual_heal = new_hp - health
        health = new_hp
        if actual_heal > 0:
            msg_parts.append(f"Recovered {actual_heal} HP.")

    # max HP boost (e.g. Armor, Energy Shield) → also full heal
    if max_hp_gain > 0:
        max_health += max_hp_gain
        health = max_health
        msg_parts.append(f"Max health increased by {max_hp_gain}.")

    if credits_gain > 0:
        credits += credits_gain
        msg_parts.append(f"Gained {credits_gain} credits.")

    if attack_gain > 0:
        attack += attack_gain
        msg_parts.append(f"Attack increased by {attack_gain}.")

    # --- write stats back into player/attributes --------------------------
    attrs["health"] = health
    attrs["max_health"] = max_health
    attrs["credits"] = credits
    attrs["attack"] = attack
    attrs["armor"] = armor

    player["attributes"] = attrs
    player["health"] = health
    player["max_health"] = max_health
    player["credits"] = credits

    # consume the stack
    inventory.pop(idx)
    player["inventory"] = inventory

    # persist into session
    session["player"] = player

    return {
        "ok": True,
        "msg": " ".join(msg_parts),
        "player": player,
    }

# ---------- sessions ----------
sessions: Dict[str, Dict[str, Any]] = {}

# ---------- models ----------
class CommandRequest(BaseModel):
    session_id: str
    command: str
    args: Optional[List[str]] = []

class ShopBuyRequest(BaseModel):
    session_id: str
    item_name: str

class SearchRequest(BaseModel):
    session_id: str

@app.on_event("startup")
async def startup():
    loader_fn()
    logging.info("✅ game data ready")

# ---------- helpers ----------
def create_player() -> Dict[str, Any]:
    return {
        "id": "player_1",
        "name": "Traveler",
        "health": 100,
        "max_health": 100,
        "credits": 0,
        "level": 1,
        "xp": 0,
        "xp_to_next": 50,
        "current_node_id": "node.planet.terra",
        "inventory": [],
    }

def generate_raider_group(player: Dict[str, Any]) -> Dict[str, Any]:
    # enemy count capped by level
    max_enemies = min(4, player["level"] + 1)
    import random
    count = random.randint(1, max_enemies)
    enemies = []
    for i in range(count):
        raider_level = max(1, player["level"] + (1 if random.random() > 0.7 else 0))
        base_hp = 30 + raider_level * 10
        enemies.append({
            "id": i,
            "name": f"Raider L{raider_level}",
            "level": raider_level,
            "health": base_hp,
            "max_health": base_hp,
            "attack_min": 4 + raider_level,
            "attack_max": 9 + raider_level * 2,
            "xp_reward": 15 + raider_level * 5,
            "credit_reward": 3 + raider_level,
        })
    return {
        "enemies": enemies,
        "story": "A patrol of raiders spots you in the dunes.",
        "loot": [
            {"name": "Scrap Alloy", "qty": 1},
            {"name": "Charge Cell", "qty": 1},
        ]
    }

def grant_xp_and_level(player: Dict[str, Any], amount: int) -> List[str]:
    msgs = []
    player["xp"] += amount
    msgs.append(f"You gained {amount} XP.")
    # level up loop
    while player["xp"] >= player["xp_to_next"]:
        player["xp"] -= player["xp_to_next"]
        player["level"] += 1
        player["max_health"] += 10
        player["health"] = player["max_health"]
        player["xp_to_next"] = int(player["xp_to_next"] * 1.35)
        msgs.append(f"Level up! You are now level {player['level']}.")
    return msgs

# ---------- routes ----------
@app.get("/api/ping")
async def ping():
    return {"status": "ok"}

@app.post("/api/game/start")
async def start_game():
    session_id = str(uuid.uuid4())

    player = {
        "name": "Traveler",
        "health": 100,
        "max_health": 100,
        "xp": 0,
        "xp_to_next": 50,
        "level": 1,
        "credits": 0,
        "inventory": [],
    }

    node = {
        "id": "terra",
        "name": "Terra",
        "desc": "The home planet. Your journey begins here.",
    }

    sessions[session_id] = {
        "player": player,
        "node": node,
        "combat": None,
        "unclaimed_loot": [],
    }

    return {"session_id": session_id}

@app.get("/api/game/state")
async def get_game_state(session_id: str = Query(...)):
    if session_id not in sessions:
        raise HTTPException(status_code=404, detail="Session not found")

    session = sessions[session_id]
    return {
        "player": session["player"],
        "node": session["node"],
        "combat": session["combat"],
        "loot": session.get("unclaimed_loot", []),
    }

@app.post("/api/game/command")
async def run_command(req: CommandRequest):
    if req.session_id not in sessions:
        raise HTTPException(404, "Session not found")
    session = sessions[req.session_id]
    player = session["player"]

    if execute_command_real:
        result = execute_command_real(
            command=req.command,
            args=req.args or [],
            player=player,
            session=session,
        )
        # Legacy format conversion
        return {
            "success": result.get("success", True),
            "message": result.get("message", ""),
            "data": result.get("data", {}),
        }
    else:
        result = execute_command_simple(req.command, req.args or [], player, session)
        # New format: {ok, msg, player}
        return {
            "ok": result.get("ok", False),
            "msg": result.get("msg", ""),
            "error": result.get("msg", "") if not result.get("ok") else None,
            "player": result.get("player", player),
        }
    
@app.post("/api/game/heal")
async def heal_player(session_id: str = Query(...)):
    if session_id not in sessions:
        raise HTTPException(404, "Session not found")
    session = sessions[session_id]
    player = session["player"]

    amount = 15
    player["health"] = min(player["max_health"], player["health"] + amount)

    return {
        "message": f"Healed for {amount}",
        "player": player
    }

@app.post("/api/game/shop/buy")
async def shop_buy(req: ShopBuyRequest):
    if req.session_id not in sessions:
        raise HTTPException(404, "Session not found")
    session = sessions[req.session_id]
    player = session["player"]

    # Basic prices – keep in sync with your front-end ITEM_DATABASE
    PRICES = {
        "Medpack": 25,
        "Nano Repair Kit": 50,
        "Energy Cell": 10,
        "Charge Cell": 5,
        "Armor": 80,
        "Energy Shield": 100,
        "Weapon": 60,
        "Blaster": 90,
        "Plasma Blaster": 90,
    }

    name = req.item_name
    cost = PRICES.get(name)
    if cost is None:
        return {"ok": False, "msg": f"{name} cannot be bought here."}

    credits = player.get("credits", 0)
    if credits < cost:
        return {"ok": False, "msg": "Not enough credits."}

    credits -= cost
    player["credits"] = credits

    inv = player.get("inventory") or []
    # normalize to the same shape we use in /state
    inv.append({"name": name, "qty": 1})
    player["inventory"] = inv

    session["player"] = player

    return {
        "ok": True,
        "msg": f"Purchased {name} for {cost} credits!",
        "player": player,
    }

MAX_INVENTORY_SLOTS = 6   # keep in sync with frontend

@app.post("/api/game/search")
async def search(req: SearchRequest):
    if req.session_id not in sessions:
        raise HTTPException(404, "Session not found")
    session = sessions[req.session_id]
    player = session["player"]

    inventory = player.get("inventory") or []

    # inventory full → nothing added
    if len(inventory) >= MAX_INVENTORY_SLOTS:
        return {
            "ok": False,
            "msg": "Your inventory is full. You leave any scraps you find behind.",
            "player": player,
            "items": [],
            "credits_gained": 0,
        }

    # simple loot table – adjust however you like
    loot_table = [
        ("Scrap Alloy", 1),
        ("Scrap", 1),
        ("Iron Scrap", 1),
        ("Energy Cell", 1),
        ("Nano Repair Kit", 1),
        ("Medpack", 1),
    ]

    # 1–2 rolls of random junk
    rolls = random.randint(1, 2)
    found_items = []
    for _ in range(rolls):
        if len(inventory) >= MAX_INVENTORY_SLOTS:
            break
        name, qty = random.choice(loot_table)
        inventory.append({"name": name, "qty": qty})
        found_items.append({"name": name, "qty": qty})

    # small credit bonus
    credits_gain = random.randint(5, 25)
    credits = player.get("credits", 0) + credits_gain
    player["credits"] = credits

    player["inventory"] = inventory
    session["player"] = player

    return {
        "ok": True,
        "msg": "You scour the area and find some useful scraps.",
        "player": player,
        "items": found_items,
        "credits_gained": credits_gain,
    }

@app.post("/api/game/adventure")
async def start_adventure(session_id: str = Query(...)):
    if session_id not in sessions:
        raise HTTPException(status_code=404, detail="Session not found")

    session = sessions[session_id]
    player = session["player"]

    # simple 1–2 raiders
    import random
    enemy_count = random.randint(1, 2)
    enemies = []
    for i in range(enemy_count):
        level = random.randint(player["level"], player["level"] + 1)
        max_hp = 40 + level * 10
        enemies.append({
            "id": i,
            "name": f"Raider L{level}",
            "health": max_hp,
            "max_health": max_hp,
            "attack": 8 + level * 2,
            "level": level,
            "credit_reward": 10 + level * 5,  # Base reward scales with level
        })

    combat = {"enemies": enemies}
    session["combat"] = combat

    # Pre-generate loot for this fight
    loot = [
        {"name": "Credits", "count": random.randint(20, 60)},
        {"name": "Scrap Alloy", "count": random.randint(1, 4)},
    ]
    session["unclaimed_loot"] = loot

    return {
        "enemies": enemies,
        "description": "You encounter hostile raiders in the outskirts.",
        "loot": loot,
    }

@app.post("/api/game/attack")
async def attack_enemy(
    session_id: str = Query(...),
    enemy_id: int = Query(...),
):
    if session_id not in sessions:
        raise HTTPException(status_code=404, detail="Session not found")

    session = sessions[session_id]
    player = session["player"]
    combat = session.get("combat")
    if not combat:
        raise HTTPException(status_code=400, detail="No active combat")

    enemies = combat["enemies"]

    # enemy_id is the enemy's stable "id" field, not its index
    target_index = None
    for i, enemy in enumerate(enemies):
        if enemy.get("id") == enemy_id:
            target_index = i
            break

    if target_index is None:
        raise HTTPException(status_code=400, detail="Invalid enemy id")

    import random
    events: list[str] = []

    # Player attack
    enemy = enemies[target_index]
    dmg = random.randint(12, 20)
    enemy["health"] = max(0, enemy["health"] - dmg)
    events.append(f"You hit {enemy['name']} for {dmg} damage.")

    # Remove dead enemy and grant rewards
    if enemy["health"] <= 0:
        events.append(f"{enemy['name']} is defeated!")
        # Grant credit reward
        credit_reward = enemy.get("credit_reward", 0)
        if credit_reward > 0:
            player["credits"] += credit_reward
            events.append(f"You loot {credit_reward} credits from {enemy['name']}.")
        enemies.pop(target_index)

    combat_won = False
    loot_to_send = []

    # Enemy turn (only if any left)
    if enemies:
        enemy = random.choice(enemies)
        edmg = random.randint(5, enemy["attack"])
        player["health"] = max(0, player["health"] - edmg)
        events.append(f"{enemy['name']} hits you for {edmg} damage.")
        
        # Check for player death
        if player["health"] <= 0:
            return {
                "events": events + ["You were defeated!"],
                "combat_won": False,
                "player_dead": True,
                "enemies": enemies,
                "player": player,
            }
    else:
        combat_won = True
        session["combat"] = None
        loot_to_send = session.get("unclaimed_loot", [])

    return {
        "events": events,
        "combat_won": combat_won,
        "enemies": enemies,
        "loot": loot_to_send,
        "player": player,
    }

@app.post("/api/game/loot/claim")
async def claim_loot(session_id: str = Query(...)):
    if session_id not in sessions:
        raise HTTPException(status_code=404, detail="Session not found")

    session = sessions[session_id]
    player = session["player"]
    loot = session.get("unclaimed_loot", [])

    if "inventory" not in player or player["inventory"] is None:
        player["inventory"] = []

    # Move loot into inventory
    for item in loot:
        if item["name"] == "Credits":
            player["credits"] += item.get("count", 0)
        else:
            player["inventory"].append({
                "name": item["name"],
                "count": item.get("count", 1),
            })

    session["unclaimed_loot"] = []

    return {
        "claimed": loot,
        "player": player,
    }

@app.post("/api/game/unlock")
async def unlock_location(session_id: str = Query(...), location_id: str = Query(...)):
    if session_id not in sessions:
        raise HTTPException(404, "Session not found")

    session = sessions[session_id]
    player = session["player"]

    # Example costs (Delta Base = 50 credits)
    unlock_costs = {
        "delta_base": 50,
        "node.delta.base": 50
    }

    if location_id not in unlock_costs:
        return {"success": False, "message": "Unknown locked location."}

    cost = unlock_costs[location_id]

    if player["credits"] < cost:
        return {"success": False, "message": "Not enough credits."}

    # Subtract credits
    player["credits"] -= cost

    # Mark as unlocked
    if "unlocked_locations" not in player:
        player["unlocked_locations"] = []

    player["unlocked_locations"].append(location_id)

    return {
        "success": True,
        "message": f"Unlocked {location_id}!",
        "player": player
    }


def render_index() -> HTMLResponse:
    index_path = STATIC_DIR / "index.html"
    if not index_path.exists():
        return HTMLResponse(
            "<h1>Muforge Web UI</h1><p>Put index.html in mutemplate/static/</p>",
            status_code=404,
        )
    return HTMLResponse(index_path.read_text(encoding="utf-8"))


@app.get("/", response_class=HTMLResponse)
async def root():
    return render_index()


@app.get("/index.html", response_class=HTMLResponse)
async def index_html():
    return render_index()


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("web_api:app", host="0.0.0.0", port=8000, reload=False)

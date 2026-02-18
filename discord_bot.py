import os
import requests
import discord
from discord.ext import commands

DISCORD_BOT_TOKEN = os.getenv("DISCORD_BOT_TOKEN", "")
PAID_ROLE_NAME = os.getenv("DISCORD_PAID_ROLE_NAME", "Paid")
BACKEND_URL = os.getenv("BACKEND_EVALUATE_URL", "http://localhost:8000/evaluate")
DISCORD_CHANNEL_ACCESS_URL = os.getenv("DISCORD_CHANNEL_ACCESS_URL", "")

intents = discord.Intents.default()
intents.message_content = True
intents.guilds = True
intents.members = True

bot = commands.Bot(command_prefix="!", intents=intents)


def has_paid_role(member: discord.Member):
    return any(r.name == PAID_ROLE_NAME for r in member.roles)


def access_link_message() -> str:
    if DISCORD_CHANNEL_ACCESS_URL:
        return f"Get channel access here: {DISCORD_CHANNEL_ACCESS_URL}"
    return "Access link is not configured yet. Ask an admin for the channel access link."


@bot.event
async def on_ready():
    print(f"Bot connected as {bot.user}")


@bot.command(name="access")
async def access(ctx):
    await ctx.send(access_link_message())


@bot.command(name="prop")
async def prop(ctx, player: str, prop_type: str, line: float, opponent: str = ""):
    if not has_paid_role(ctx.author):
        await ctx.send(f"This command is for paid members only. {access_link_message()}")
        return

    params = {
        "player": player,
        "prop": prop_type,
        "line": line,
        "opponent": opponent,
    }
    try:
        res = requests.get(BACKEND_URL, params=params, timeout=10).json()
    except Exception:
        await ctx.send("Backend error. Try again later.")
        return

    if "error" in res:
        await ctx.send(res["error"])
        return

    msg = (
        f"{res['player']} {res['prop']} line {res['line']}\n"
        f"Confidence: {res['confidence']}% ({res['recommendation']})\n"
        f"L5: {res['last_5_hit_rate']}% | L10: {res['last_10_hit_rate']}% | H2H: {res['h2h_hit_rate']}%"
    )
    await ctx.send(msg)


if __name__ == "__main__":
    if not DISCORD_BOT_TOKEN:
        raise SystemExit("DISCORD_BOT_TOKEN is missing.")
    bot.run(DISCORD_BOT_TOKEN)

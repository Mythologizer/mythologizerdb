import os
import subprocess
from pathlib import Path
from typing import Optional

import typer

from dotenv import load_dotenv, find_dotenv
load_dotenv(find_dotenv())

from .db import ping_db, apply_schemas, build_url, clear_all_rows, get_table_row_counts, check_if_tables_exist

from sqlalchemy.engine.url import URL

app = typer.Typer(help="MythologizerDB CLI")

def url_as_string(show_password: bool = False) -> str:
    url: URL = build_url()
    return url.render_as_string(hide_password=not show_password)

@app.command("show-url")
def show_url(reveal_password: bool = typer.Option(False, "--reveal-password")):
    typer.echo(url_as_string(show_password=reveal_password))

@app.command()
def ping():
    ping_db(get_connection())
    typer.secho("OK", fg=typer.colors.GREEN)

@app.command("start")
def start_db(
    compose_file: Path = typer.Option("docker-compose.yaml", "--file", "-f"),
    env_file: Path = typer.Option(".env", "--env-file"),
    services: Optional[str] = typer.Option(None, "--services", help="space separated list"),
    detach: bool = typer.Option(True, "--detach/--no-detach", "-d"),
):
    """
    Start database using docker compose with a specific env file.
    """
    args = [
        "docker", "compose",
        "--env-file", str(env_file),
        "-f", str(compose_file),
        "up"
    ]
    if detach:
        args.append("-d")
    if services:
        args += services.split()

    subprocess.run(args, check=True)
    typer.secho("Database started", fg=typer.colors.GREEN)

@app.command("stop")
def stop_db(
    compose_file: Path = typer.Option("docker-compose.yaml", "--file", "-f"),
    services: Optional[str] = typer.Option(None, "--services", help="space separated list"),
):
    """
    Stop the database containers.
    """
    args = ["docker", "compose", "-f", str(compose_file), "down"]
    if services:
        args = ["docker", "compose", "-f", str(compose_file), "stop"] + services.split()
    subprocess.run(args, check=True)
    typer.secho("Database stopped", fg=typer.colors.GREEN)

@app.command("clear")
def clear(
    yes: bool = typer.Option(False, "--yes", "-y", help="Skip confirmation")
):
    """
    Delete all rows from all tables in the database.
    """
    if not yes and not typer.confirm(
        "This will remove all rows from every table. Continue?"
    ):
        raise typer.Abort()

    clear_all_rows()
    typer.secho("All rows deleted", fg=typer.colors.RED)


@app.command("setup")
def setup_db(
    dim: Optional[int] = typer.Option(None, "--dim", help="Embedding dimensionality (e.g. 384, 768)")
):
    """
    Apply database schema for the given dimensionality.
    
    If --dim is not provided, uses EMBEDDING_DIM from .env file.
    If EMBEDDING_DIM is not set, defaults to 384.
    """
    # Get dimension from parameter, environment, or default
    if dim is not None:
        embedding_dim = dim
    else:
        env_dim = os.getenv('EMBEDDING_DIM')
        if env_dim:
            try:
                embedding_dim = int(env_dim)
            except ValueError:
                typer.secho(f"Invalid EMBEDDING_DIM value in .env: {env_dim}. Using default 384.", fg=typer.colors.YELLOW)
                embedding_dim = 384
        else:
            embedding_dim = 384
    
    typer.echo(f"Using embedding dimension: {embedding_dim}")
    apply_schemas(embedding_dim)
    typer.secho("Schemas created", fg=typer.colors.GREEN)

@app.command("destroy")
def destroy(
    compose_file: Path = typer.Option("docker-compose.yaml", "--file", "-f"),
    env_file: Path = typer.Option(".env", "--env-file"),
    yes: bool = typer.Option(False, "--yes", "-y", help="Skip confirmation")
):
    """
    Stop containers and remove all volumes (deletes persistent data).
    """
    if not yes and not typer.confirm(
        "This will stop containers and delete every volume. Continue?"
    ):
        raise typer.Abort()

    args = [
        "docker", "compose",
        "--env-file", str(env_file),
        "-f", str(compose_file),
        "down", "-v"
    ]
    subprocess.run(args, check=True)
    typer.secho("Containers stopped and all volumes removed", fg=typer.colors.GREEN)

@app.command("status")
def status(dim: int = typer.Option(384, help="Embedding dimensionality used to generate schema")):
    typer.echo("Checking database connectivity...")
    if ping_db():
        typer.secho("Database connection OK", fg=typer.colors.GREEN)
    else:
        typer.secho("Database not reachable", fg=typer.colors.RED)
        raise typer.Exit(code=1)

    typer.echo("Checking required tables...")
    expected_tables = ["mythemes", "myths"]
    table_status = check_if_tables_exist(expected_tables)
    for table, exists in table_status.items():
        color = typer.colors.GREEN if exists else typer.colors.RED
        typer.secho(f"Table '{table}': {'OK' if exists else 'Missing'}", fg=color)

    typer.echo("Getting row counts for all public tables...")
    counts = get_table_row_counts()
    if counts:
        for table, count in counts.items():
            typer.echo(f"{table}: {count} rows")
    else:
        typer.secho("No user tables found or error occurred while counting", fg=typer.colors.YELLOW)

if __name__ == "__main__":
    app()

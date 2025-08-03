from dotenv import load_dotenv, find_dotenv
load_dotenv(find_dotenv())

from .cli import app
app()
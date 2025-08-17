import asyncio, json
import psycopg
from mythologizer_postgres.db import build_url
from dotenv import load_dotenv, find_dotenv
load_dotenv(find_dotenv())

async def listen_and_fetch():
    url = build_url()

    # 1) Listener connection
    listen_conn = await psycopg.AsyncConnection.connect(
        dbname=url.database, user=url.username, password=url.password,
        host=url.host, port=url.port,
    )
    await listen_conn.set_autocommit(True)
    await listen_conn.execute("LISTEN myth_writings_marked_read")
    print("Listening on myth_writings_marked_read")

    # 2) Worker connection for queries
    work_conn = await psycopg.AsyncConnection.connect(
        dbname=url.database, user=url.username, password=url.password,
        host=url.host, port=url.port,
    )
    # optional nice printing
    work_conn.row_factory = psycopg.rows.dict_row

    async for notify in listen_conn.notifies():
        payload = json.loads(notify.payload)
        print("notify:", payload)

        async with work_conn.cursor() as cur:
            await cur.execute(
                "SELECT * FROM public.myth_writings WHERE id = %s",
                (payload["id"],),
            )
            row = await cur.fetchone()
            print("row:", row)

asyncio.run(listen_and_fetch())

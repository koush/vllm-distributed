# VLLM Distributed Launcher

Run VLLM easily distributing model weights across multiple servers.

# Server 

1. Copy the `.env.server` to `.env`:
`cp .env.server .env`
2. Modify the `.env` with server's host address and vllm server parameters.
3. `docker compose up -d`


# Client(s)

1. Copy the `.env.client` to `.env`:
`cp .env.client .env`
2. Modify the `.env` with client's host address AND also the server's host addresses. These will be different.
3. `docker compose up -d`

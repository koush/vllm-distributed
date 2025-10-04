# VLLM Distributed Launcher

Run VLLM easily distributing model weights across multiple servers.

# Server 

1. Copy the `.env.server` to `.env`:
`cp .env.server .env`
2. Modify the `.env` with your address and vllm server parameters.
3. `docker compose up -d`


# Client(s)

1. Copy the `.env.client` to `.env`:
`cp .env.client .env`
2. Modify the `.env` with your address and vllm server AND client addresses.
3. `docker compose up -d`

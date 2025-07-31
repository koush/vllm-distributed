import asyncio
import concurrent.futures
import gc
import importlib
import multiprocessing
import multiprocessing.connection
import os
import socket
import sys
import threading
import weakref
from argparse import Namespace
from asyncio.events import AbstractEventLoop
from collections.abc import Awaitable, Callable
from contextlib import asynccontextmanager
from typing import Any, AsyncIterator, List, Optional, Union
import traceback
import rpc

import uvloop
import vllm.envs as envs
from vllm.config import VllmConfig
from vllm.distributed.kv_transfer.kv_connector.utils import KVOutputAggregator
from vllm.engine.arg_utils import AsyncEngineArgs
from vllm.engine.protocol import EngineClient
from vllm.entrypoints.launcher import serve_http
from vllm.entrypoints.openai.api_server import (
    build_app,
    build_async_engine_client_from_engine_args,
    init_app_state,
    load_log_config,
    maybe_register_tokenizer_info_endpoint,
    setup_server,
)

# yapf conflicts with isort for this block
# yapf: disable
# yapf: enable
from vllm.entrypoints.openai.tool_parsers import ToolParserManager
from vllm.logger import init_logger
from vllm.utils import (
    get_distributed_init_method,
    get_ip,
    get_open_port,
    run_method,
    cuda_device_count_stateless,
)
from vllm.v1.executor.abstract import Executor, FailureCallback
from vllm.v1.outputs import ModelRunnerOutput
from vllm.worker.worker_base import WorkerWrapperBase
from vllm.platforms import current_platform

import rpc_reader

VLLM_SERVER_PORT = int(os.environ.get("VLLM_SERVER_PORT", 30044))

logger = init_logger("vllm.entrypoints.openai.api_server")

RunWorkerType = Callable[[Union[str, bytes], Optional[int], Any, Any], Awaitable[Any]]
CreateWorkerType = Callable[[VllmConfig, int], Awaitable[RunWorkerType]]

class CustomExecutor(Executor):
    uses_ray: bool = False

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def _init_executor(self):
        # Call self.shutdown at exit to clean up
        # and ensure workers will be terminated.
        self._finalizer = weakref.finalize(self, self.shutdown)
        self.is_failed = False
        self.shutdown_event = threading.Event()
        self.failure_callback: Optional[FailureCallback] = None

        self.world_size = self.parallel_config.world_size
        tensor_parallel_size = self.parallel_config.tensor_parallel_size
        pp_parallel_size = self.parallel_config.pipeline_parallel_size
        assert self.world_size == tensor_parallel_size * pp_parallel_size, (
            f"world_size ({self.world_size}) must be equal to the "
            f"tensor_parallel_size ({tensor_parallel_size}) x pipeline"
            f"_parallel_size ({pp_parallel_size}). "
        )

        distributed_init_method = get_distributed_init_method(get_ip(), get_open_port())

        remote_nodes = asyncio.queues.Queue[list[CreateWorkerType]]()
        pending_remote_nodes: dict[str, list[RunWorkerType]] = {}

        async def handle_client(
            reader: asyncio.StreamReader, writer: asyncio.StreamWriter
        ):
            addr = writer.get_extra_info("peername")
            print(f"Connection from {addr}")

            remote_ip, remote_port = addr
            remote_node = pending_remote_nodes.get(remote_ip, [])
            pending_remote_nodes[remote_ip] = remote_node

            rpc_transport = rpc_reader.RpcStreamTransport(reader, writer)
            loop = asyncio.get_event_loop()
            peer, readLoop = await rpc_reader.prepare_peer_readloop(loop, rpc_transport)

            loop_task = loop.create_task(readLoop())

            try:
                available_devices: int = await peer.getParam("available_devices")
                create_worker = await peer.getParam("create_worker")
                remote_node.append(create_worker)
                # after the remote node has all available devices
                if len(remote_node) == available_devices:
                    remote_nodes.put_nowait(remote_node)
                await loop_task
            except:
                print(f"Error handling client {addr}")
                traceback.print_exc()
            finally:
                print(f"Closing connection {addr}")
                # remove from remote_worker list
                remote_node.remove(create_worker)
                writer.close()
                await writer.wait_closed()

        loop_ready = concurrent.futures.Future[AbstractEventLoop]()
        workers_ready = concurrent.futures.Future[tuple[int, list[RunWorkerType]]]()

        async def spawn_and_listen_for_workers():
            loop = asyncio.get_event_loop()
            loop_ready.set_result(loop)

            async def get_run_worker(local_rank: int, pipeline_rank: int) -> RunWorkerType:
                rank = local_rank + pipeline_rank * tensor_parallel_size
                parent_conn, child_conn = multiprocessing.Pipe()
                process_kwargs = {
                    "vllm_config": self.vllm_config,
                    "rank": rank,
                }
                worker = multiprocessing.Process(
                    target=worker_main,
                    args=(child_conn,),
                    kwargs=process_kwargs,
                    daemon=True,
                )
                worker.start()

                rpcTransport = rpc_reader.RpcConnectionTransport(parent_conn)
                forkPeer, readLoop = await rpc_reader.prepare_peer_readloop(
                    loop, rpcTransport
                )
                forkPeer.peerName = "vllm-worker"

                async def workerReadLoop():
                    try:
                        await readLoop()
                    except:
                        import traceback

                        traceback.print_exc()
                        print("worker read loop exited")
                    finally:
                        parent_conn.close()
                        rpcTransport.executor.shutdown()
                        worker.terminate()

                asyncio.run_coroutine_threadsafe(workerReadLoop(), loop=loop)
                return await forkPeer.getParam("run_worker")

            need_devices = self.world_size
            available_devices = 1
            if current_platform.is_cuda():
                available_devices = cuda_device_count_stateless()

            run_worker_futures: list[asyncio.Future[RunWorkerType]] = []
            for pipeline_rank in range(self.parallel_config.pipeline_parallel_size):
                # try to fulfill as many workers as possible locally
                if available_devices >= tensor_parallel_size:
                    for local_rank in range(tensor_parallel_size):
                        run_worker_futures.append(
                            asyncio.create_task(
                                get_run_worker(local_rank, pipeline_rank)
                            )
                        )
                    available_devices -= tensor_parallel_size
                else:
                    create_workers = await remote_nodes.get()
                    # node is not usable
                    if len(create_workers) < tensor_parallel_size:
                        print(
                            f"WARNING: Not enough workers on remote node {create_workers}, skipping... this may be a bug."
                        )
                        continue

                    for local_rank in range(tensor_parallel_size):
                        rank = local_rank + pipeline_rank * tensor_parallel_size
                        create_worker = create_workers.pop()
                        run_worker_futures.append(loop.create_task(create_worker(self.vllm_config, rank)))

                    # the node may still be usable, if so readd it
                    # note: does this work? possibly different number of workers per node?
                    if len(create_workers) >= tensor_parallel_size:
                        remote_nodes.put_nowait(create_workers)

                need_devices -= tensor_parallel_size

            run_workers: list[RunWorkerType] = []
            for future in run_worker_futures:
                run_workers.append(await future)
            return run_workers

        async def worker_loop():
            server = await asyncio.start_server(
                handle_client, "0.0.0.0", VLLM_SERVER_PORT
            )
            print(f"Server listening on 0.0.0.0:{VLLM_SERVER_PORT}")
            run_workers = await spawn_and_listen_for_workers()
            workers_ready.set_result((VLLM_SERVER_PORT, run_workers))

            async with server:
                await server.serve_forever()

        def worker_thread():
            uvloop.run(worker_loop())

        threading.Thread(target=worker_thread, daemon=True).start()
        port, run_workers = workers_ready.result()
        self.loop = loop_ready.result()

        self.workers: List[RunWorkerType] = run_workers

        all_kwargs = []

        for pipeline_rank in range(self.parallel_config.pipeline_parallel_size):
            for local_rank in range(self.parallel_config.tensor_parallel_size):
                rank = local_rank + pipeline_rank * tensor_parallel_size
                kwargs = dict(
                    vllm_config=self.vllm_config,
                    local_rank=local_rank,
                    rank=rank,
                    distributed_init_method=distributed_init_method,
                    is_driver_worker=(not self.parallel_config)
                    or (rank % self.parallel_config.tensor_parallel_size == 0),
                )
                all_kwargs.append(kwargs)

        self.collective_rpc("init_worker", args=[all_kwargs])
        self.collective_rpc("init_device")
        self.collective_rpc("load_model")

        self.output_rank = self._get_output_rank()
        self.has_connector = self.vllm_config.kv_transfer_config is not None
        self.kv_output_aggregator = KVOutputAggregator(self.parallel_config.world_size)

    @property
    def max_concurrent_batches(self) -> int:
        if self.scheduler_config.async_scheduling:
            return 2
        return self.parallel_config.pipeline_parallel_size

    def _get_output_rank(self) -> int:
        # Only returns ModelRunnerOutput from TP rank=0 and PP rank=-1
        # (the first TP worker of the last PP stage).
        # Example:
        # Assuming TP=8, PP=4, then the world_size=32
        # 0-7, PP rank 0
        # 8-15, PP rank 1
        # 16-23, PP rank 2
        # 24-31, PP rank 3
        # so world_size - tp_size = 32 - 8 = 24 should be PP rank = -1 (i.e. 3)
        return self.world_size - self.parallel_config.tensor_parallel_size

    def register_failure_callback(self, callback: FailureCallback):
        if self.is_failed:
            callback()
        else:
            self.failure_callback = callback

    def execute_model(
        self,
        scheduler_output,
    ) -> Union[ModelRunnerOutput, concurrent.futures.Future[ModelRunnerOutput]]:
        non_block = self.max_concurrent_batches > 1

        if not self.has_connector:
            # get output only from a single worker (output_rank)
            (output,) = self.collective_rpc(
                "execute_model",
                args=(scheduler_output,),
                unique_reply_rank=self.output_rank,
                non_block=non_block,
                timeout=envs.VLLM_EXECUTE_MODEL_TIMEOUT_SECONDS,
            )
            return output

        # get output from all workers
        outputs = self.collective_rpc(
            "execute_model",
            args=(scheduler_output,),
            non_block=non_block,
            timeout=envs.VLLM_EXECUTE_MODEL_TIMEOUT_SECONDS,
        )

        # aggregate all workers output to a single output
        if non_block:
            return self.kv_output_aggregator.async_aggregate(outputs, self.output_rank)
        return self.kv_output_aggregator.aggregate(outputs, self.output_rank)

    def collective_rpc(
        self,
        method: Union[str, Callable],
        timeout: Optional[float] = None,
        args: tuple = (),
        kwargs: Optional[dict] = None,
        non_block: bool = False,
        unique_reply_rank: Optional[int] = None,
    ) -> list[Any]:
        if self.is_failed:
            raise RuntimeError("Executor failed.")

        async def run_workers():
            run_worker_result_coros: list[asyncio.Future[Any]] = []
            for run_worker in self.workers:
                task = asyncio.create_task(
                    run_worker(method, unique_reply_rank, list(args), kwargs)
                )
                run_worker_result_coros.append(task)

            result = await asyncio.gather(*run_worker_result_coros)
            return result

        run_worker_results_future = asyncio.run_coroutine_threadsafe(
            run_workers(), self.loop
        )

        worker_outputs = run_worker_results_future.result()
        worker_outputs = (
            [worker_outputs[unique_reply_rank]]
            if unique_reply_rank is not None
            else worker_outputs
        )
        return worker_outputs

    def check_health(self):
        self.collective_rpc("check_health", timeout=10)
        return


@asynccontextmanager
async def build_async_engine_client(
    args: Namespace,
    client_config: Optional[dict[str, Any]] = None,
) -> AsyncIterator[EngineClient]:

    # Context manager to handle engine_client lifecycle
    # Ensures everything is shutdown and cleaned up on error/exit
    engine_args = AsyncEngineArgs.from_cli_args(args)
    engine_args.distributed_executor_backend = CustomExecutor

    async with build_async_engine_client_from_engine_args(
        engine_args, args.disable_frontend_multiprocessing, client_config
    ) as engine:
        yield engine


async def run_server(args, client_config=None, **uvicorn_kwargs) -> None:
    """Run a single-worker API server."""
    listen_address, sock = setup_server(args)

    if args.tool_parser_plugin and len(args.tool_parser_plugin) > 3:
        ToolParserManager.import_tool_parser(args.tool_parser_plugin)

    server_index = client_config.get("client_index", 0) if client_config else 0

    # Load logging config for uvicorn if specified
    log_config = load_log_config(args.log_config_file)
    if log_config is not None:
        uvicorn_kwargs["log_config"] = log_config

    async with build_async_engine_client(args, client_config) as engine_client:
        maybe_register_tokenizer_info_endpoint(args)
        app = build_app(args)

        vllm_config = await engine_client.get_vllm_config()
        await init_app_state(engine_client, vllm_config, app.state, args)

        logger.info("Starting vLLM API server %d on %s", server_index, listen_address)
        shutdown_task = await serve_http(
            app,
            sock=sock,
            enable_ssl_refresh=args.enable_ssl_refresh,
            host=args.host,
            port=args.port,
            log_level=args.uvicorn_log_level,
            # NOTE: When the 'disable_uvicorn_access_log' value is True,
            # no access log will be output.
            access_log=not args.disable_uvicorn_access_log,
            timeout_keep_alive=envs.VLLM_HTTP_TIMEOUT_KEEP_ALIVE,
            ssl_keyfile=args.ssl_keyfile,
            ssl_certfile=args.ssl_certfile,
            ssl_ca_certs=args.ssl_ca_certs,
            ssl_cert_reqs=args.ssl_cert_reqs,
            **uvicorn_kwargs,
        )

    # NB: Await server shutdown only after the backend context is exited
    try:
        await shutdown_task
    finally:
        sock.close()


def vllm_main():
    import vllm.entrypoints.cli.benchmark.main
    import vllm.entrypoints.cli.collect_env
    import vllm.entrypoints.cli.openai
    import vllm.entrypoints.cli.run_batch
    import vllm.entrypoints.cli.serve
    from vllm.entrypoints.utils import VLLM_SUBCMD_PARSER_EPILOG, cli_env_setup
    from vllm.utils import FlexibleArgumentParser

    CMD_MODULES = [
        vllm.entrypoints.cli.openai,
        vllm.entrypoints.cli.serve,
        vllm.entrypoints.cli.benchmark.main,
        vllm.entrypoints.cli.collect_env,
        vllm.entrypoints.cli.run_batch,
    ]

    cli_env_setup()

    parser = FlexibleArgumentParser(
        description="vLLM CLI",
        epilog=VLLM_SUBCMD_PARSER_EPILOG,
    )
    parser.add_argument(
        "-v",
        "--version",
        action="version",
        version=importlib.metadata.version("vllm"),
    )
    subparsers = parser.add_subparsers(required=False, dest="subparser")
    cmds = {}
    for cmd_module in CMD_MODULES:
        new_cmds = cmd_module.cmd_init()
        for cmd in new_cmds:
            cmd.subparser_init(subparsers).set_defaults(dispatch_function=cmd.cmd)
            cmds[cmd.name] = cmd

    args = parser.parse_args()

    # force external launcher and dummy up ranks that are checked by vllm
    args.distributed_executor_backend = "external_launcher"
    os.environ["RANK"] = "-1"
    os.environ["LOCAL_RANK"] = "-1"

    if hasattr(args, "model_tag") and args.model_tag is not None:
        args.model = args.model_tag

    if args.subparser in cmds:
        cmds[args.subparser].validate(args)

    if hasattr(args, "dispatch_function"):
        uvloop.run(run_server(args))
    else:
        parser.print_help()


def set_peer_run_worker(peer: rpc.RpcPeer, wrapper: WorkerWrapperBase, rank: int):
    async def run_worker(
        method: Union[str, bytes], unique_reply_rank: Optional[int], args, kwargs
    ):
        args = args or []
        kwargs = kwargs or {}
        results = run_method(wrapper, method, args, kwargs)
        if unique_reply_rank is not None and unique_reply_rank != rank:
            return None
        return results

    peer.params["run_worker"] = run_worker
    return run_worker


async def remote_worker_async_main(server_ip: str, available_devices: int):
    loop = asyncio.get_event_loop()
    wrapper: WorkerWrapperBase = None
    while True:
        try:
            # Attempt to connect to the server
            reader, writer = await asyncio.open_connection(server_ip, VLLM_SERVER_PORT)

            rpc_transport = rpc_reader.RpcStreamTransport(reader, writer)
            peer, readLoop = await rpc_reader.prepare_peer_readloop(loop, rpc_transport)

            peer.params["print"] = print
            peer.params["available_devices"] = available_devices

            async def create_worker(vllm_config: VllmConfig, rank: int):
                nonlocal wrapper
                if wrapper:
                    raise Exception("Worker already created.")
                wrapper = WorkerWrapperBase(vllm_config=vllm_config, rpc_rank=rank)
                run_worker = set_peer_run_worker(peer, wrapper, rank)
                return run_worker

            peer.params["create_worker"] = create_worker

            await readLoop()
        except:
            # allow retrying as long as a worker has never been created.
            # once a worker has been created the entire node needs to tear down and restart.
            traceback.print_exc()
            if wrapper:
                raise
            print(
                f"Connection to {server_ip}:{VLLM_SERVER_PORT} failed. Retrying momentarily..."
            )
            await asyncio.sleep(10)


def gc_runner_loop(loop: AbstractEventLoop):
    def gc_runner():
        gc.collect()
        loop.call_later(10, gc_runner)

    gc_runner()


def remote_worker_main(server_ip: str, **kwargs):
    loop = asyncio.new_event_loop()

    loop.run_until_complete(remote_worker_async_main(loop, server_ip, **kwargs))
    loop.close()

    gc_runner_loop(loop)


def remote_main(server_ip: str):
    available_devices = 1
    if current_platform.is_cuda():
        available_devices = cuda_device_count_stateless()

    workers: list[multiprocessing.Process] = []
    for _ in range(available_devices):
        worker = multiprocessing.Process(
            target=remote_worker_main,
            args=(server_ip, available_devices),
            daemon=True,
        )
        workers.append(worker)
        worker.start()

    alive_sentinels = [p.sentinel for p in workers]

    multiprocessing.connection.wait([p.sentinel for p in alive_sentinels])
    print("A worker exited, shutting down node.")

    for worker in workers:
        worker.terminate()

    for worker in workers:
        worker.join()

    sys.exit(1)


async def worker_async_main(
    rpc_transport: rpc_reader.RpcTransport,
    vllm_config: VllmConfig,
    rank: int,
):
    loop = asyncio.get_event_loop()
    peer, readLoop = await rpc_reader.prepare_peer_readloop(loop, rpc_transport)
    peer.params["print"] = print

    wrapper = WorkerWrapperBase(vllm_config=vllm_config, rpc_rank=rank)
    set_peer_run_worker(peer, wrapper, rank)

    try:
        await readLoop()
    finally:
        os._exit(0)


def worker_main(
    conn: multiprocessing.connection.Connection,
    **kwargs,
):
    rpc_transport = rpc_reader.RpcConnectionTransport(conn)
    loop = asyncio.new_event_loop()

    gc_runner_loop(loop)

    loop.run_until_complete(worker_async_main(rpc_transport, **kwargs))
    loop.close()


def main():
    # if first arg is worker, launch the worker with the ip
    if sys.argv[1] != "remote":
        vllm_main()
        return

    server_ip = sys.argv[2]
    remote_main(server_ip)


if __name__ == "__main__":
    main()

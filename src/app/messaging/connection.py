import asyncio
import json
import uuid
import zmq
import zmq.asyncio
from typing import Dict, Any, Optional
from concurrent.futures import ThreadPoolExecutor
from src.app.domain.exceptions import MessagingError
from src.app.infrastructure.logging import get_logger

logger = get_logger(__name__)


class AsynchronousZMQClient:
    """
    Asynchronous ZeroMQ Client using DEALER socket.
    Supports high-throughput, multiplexed concurrent request-reply.
    """
    def __init__(self, broker_url: str, timeout_ms: int = 15000):
        self.broker_url = broker_url
        self.timeout_ms = timeout_ms
        self.context: Optional[zmq.asyncio.Context] = None
        self.socket: Optional[zmq.asyncio.Socket] = None
        self.pending_requests: Dict[str, asyncio.Future] = {}
        self._listener_task: Optional[asyncio.Task] = None
        self._lock = asyncio.Lock()

    async def start(self):
        async with self._lock:
            if self.context is not None:
                return

            logger.info("Initializing ZeroMQ Client...", extra={"broker_url": self.broker_url})
            self.context = zmq.asyncio.Context()
            # Use DEALER socket for asynchronous concurrent requests
            self.socket = self.context.socket(zmq.DEALER)
            # Set identity to track client routing
            self.socket.setsockopt_string(zmq.IDENTITY, f"client-{uuid.uuid4()}")
            self.socket.connect(self.broker_url)

            # Start background listener task to receive worker replies
            self._listener_task = asyncio.create_task(self._listen_replies())
            logger.info("ZeroMQ Client started successfully.")

    async def _listen_replies(self):
        """Continuously listens for replies from the ZeroMQ socket."""
        try:
            while self.socket is not None:
                # DEALER socket receives parts: [empty_delimiter, request_id, message_body]
                parts = await self.socket.recv_multipart()
                if len(parts) < 3:
                    logger.warning("Received invalid ZMQ message format", extra={"parts_count": len(parts)})
                    continue

                empty_delim, req_id_bytes, body_bytes = parts[0], parts[1], parts[2]
                req_id = req_id_bytes.decode("utf-8")

                future = self.pending_requests.pop(req_id, None)
                if future and not future.done():
                    try:
                        response = json.loads(body_bytes.decode("utf-8"))
                        future.set_result(response)
                    except Exception as e:
                        future.set_exception(e)
        except zmq.ZMQError as e:
            # Clean exit on socket close
            logger.debug(f"ZeroMQ listener socket closed: {e}")
        except Exception as e:
            logger.error("Error in ZeroMQ Client listener loop", extra={"error": str(e)})

    async def send_request(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        """
        Sends a request asynchronously and awaits the response.
        Multiplexes requests using unique request IDs.
        """
        if self.socket is None:
            await self.start()

        req_id = str(uuid.uuid4())
        future = asyncio.get_running_loop().create_future()
        self.pending_requests[req_id] = future

        # DEALER message: [empty_delimiter, request_id, body]
        message = [b"", req_id.encode("utf-8"), json.dumps(payload).encode("utf-8")]

        try:
            await self.socket.send_multipart(message) # type: ignore
            
            # Wait for response with timeout
            try:
                response = await asyncio.wait_for(future, timeout=self.timeout_ms / 1000.0)
                return response
            except asyncio.TimeoutError:
                self.pending_requests.pop(req_id, None)
                logger.error("ZeroMQ Request timed out", extra={"request_id": req_id})
                raise MessagingError(f"Request timed out after {self.timeout_ms} ms")
        except zmq.ZMQError as e:
            self.pending_requests.pop(req_id, None)
            logger.error("ZeroMQ connection error during send", extra={"error": str(e)})
            raise MessagingError(f"ZeroMQ connection error: {e}")

    async def stop(self):
        async with self._lock:
            if self._listener_task:
                self._listener_task.cancel()
                try:
                    await self._listener_task
                except asyncio.CancelledError:
                    pass
                self._listener_task = None

            if self.socket:
                self.socket.close(linger=0)
                self.socket = None

            if self.context:
                self.context.term()
                self.context = None

            logger.info("ZeroMQ Client stopped.")


class ZMQWorker:
    """
    ZeroMQ Worker using ROUTER socket (or REP).
    Listens for requests, executes CPU-bound preprocessing in a ThreadPoolExecutor,
    and returns responses asynchronously.
    """
    def __init__(self, broker_url: str, request_handler):
        self.broker_url = broker_url
        self.request_handler = request_handler
        self.context: Optional[zmq.asyncio.Context] = None
        self.socket: Optional[zmq.asyncio.Socket] = None
        self.executor = ThreadPoolExecutor(max_workers=4)
        self.is_running = False

    async def start(self):
        logger.info("Initializing ZeroMQ Worker...", extra={"broker_url": self.broker_url})
        self.context = zmq.asyncio.Context()
        # ROUTER socket allows worker to track client identity and send multiplexed replies
        self.socket = self.context.socket(zmq.ROUTER)
        self.socket.bind(self.broker_url)
        self.is_running = True

        logger.info("ZeroMQ Worker listening for requests...")
        try:
            while self.is_running:
                # ROUTER receives parts: [client_identity, empty_delim, request_id, request_body]
                parts = await self.socket.recv_multipart()
                if len(parts) < 4:
                    logger.warning("Worker received invalid message parts count", extra={"count": len(parts)})
                    continue

                client_id, empty_delim, req_id, body_bytes = parts[0], parts[1], parts[2], parts[3]
                
                # Handle request in the thread pool to avoid blocking the network loop
                asyncio.create_task(
                    self._handle_request_async(client_id, empty_delim, req_id, body_bytes)
                )
        except zmq.ZMQError as e:
            logger.debug(f"Worker socket closed: {e}")
        except Exception as e:
            logger.error("Error in ZeroMQ Worker loop", extra={"error": str(e)})

    async def _handle_request_async(self, client_id: bytes, empty_delim: bytes, req_id: bytes, body_bytes: bytes):
        try:
            payload = json.loads(body_bytes.decode("utf-8"))
            
            # Delegate CPU-bound ML/CV processing to thread pool
            loop = asyncio.get_running_loop()
            response = await loop.run_in_executor(self.executor, self.request_handler, payload)
        except Exception as e:
            logger.error("Error processing request in worker", extra={"error": str(e)})
            response = {"success": False, "error": str(e)}

        if self.socket is not None:
            # ROUTER reply: [client_identity, empty_delim, request_id, response_body]
            reply = [client_id, empty_delim, req_id, json.dumps(response).encode("utf-8")]
            try:
                await self.socket.send_multipart(reply)
            except Exception as e:
                logger.error("Failed to send reply to client", extra={"error": str(e)})

    def stop(self):
        self.is_running = False
        if self.socket:
            self.socket.close(linger=0)
            self.socket = None
        if self.context:
            self.context.term()
            self.context = None
        self.executor.shutdown(wait=False)
        logger.info("ZeroMQ Worker stopped.")

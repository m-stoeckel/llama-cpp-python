from __future__ import annotations

import asyncio
import json
import logging
import os
import queue
from functools import partial
from logging.handlers import QueueHandler, QueueListener
from threading import Lock
from typing import Dict, Final, Iterator, List, Optional, Union

import anyio
import llama_cpp
from anyio.streams.memory import MemoryObjectSendStream
from fastapi import (
    APIRouter,
    BackgroundTasks,
    Depends,
    FastAPI,
    HTTPException,
    Request,
    status,
)
from fastapi.middleware import Middleware
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import HTTPBearer
from llama_cpp.server.errors import RouteErrorHandler
from llama_cpp.server.model import LlamaProxy
from llama_cpp.server.settings import (
    ConfigFileSettings,
    ModelSettings,
    ServerSettings,
    Settings,
)
from llama_cpp.server.types import (
    CreateChatCompletionRequest,
    CreateCompletionRequest,
    CreateEmbeddingRequest,
    ModelList,
)
from sse_starlette.sse import EventSourceResponse
from starlette.concurrency import iterate_in_threadpool, run_in_threadpool
from starlette_context.middleware import RawContextMiddleware
from starlette_context.plugins import RequestIdPlugin  # type: ignore

logging.basicConfig(
    format="[%(asctime)s] [%(levelname)s] - %(message)s", level=logging.INFO
)
logger = logging.getLogger(__name__)

# log_queue = queue.Queue()
# queue_handler = QueueHandler(log_queue)
# logger.addHandler(queue_handler)

# queue_listener = QueueListener(logging.StreamHandler())
# queue_listener.start()


class AlreadyLockedError(Exception):
    pass


class UnloadManager:
    def __init__(
        self,
        timeout: float,
        timeout_step: float = 0.1,
        event_loop: asyncio.BaseEventLoop = None,
    ) -> None:
        self.__unload_timer = _UnloadTimer(timeout, timeout_step)
        self.__event_loop = event_loop or asyncio.new_event_loop()

        self.__lock = asyncio.Lock()
        self.__task: asyncio.Task = None

    async def __aenter__(self):
        logger.debug("Aquiring Unload Lock")
        if not self.__lock.locked():
            await self.__lock.acquire()
            logger.debug("Aquired Unload Lock")
            return self
        else:
            raise AlreadyLockedError()

    async def __aexit__(self, exc_type, exc_value, traceback):
        match (exc_type, exc_value, traceback):
            case (None, None, None):
                self.__task = None
                self.__lock.release()
            case _:
                raise RuntimeError(
                    "Error while exiting UnloadManager context"
                ) from exc_type(exc_value).with_traceback(traceback)

    def create_task(self, event_loop=None) -> asyncio.Task:
        logger.debug("Creating Unload Task")
        event_loop = event_loop or self.__event_loop
        self.__task = event_loop.create_task(self.__unload_timer.schedule())
        return self.__task

    async def reset(self) -> None:
        return await self.__unload_timer.reset()

    async def suspend(self) -> bool:
        """Cancel the unload task.

        Returns:
            bool: True if the task was canceled, False otherwise
        """
        if self.__task is None:
            return False

        logger.debug("Suspending Unload Task")
        return await self.__unload_timer.suspend()

    async def unsuspend(self) -> bool:
        """Unancel the unload task.

        Returns:
            bool: True if the task was uncanceled, False otherwise
        """
        if self.__task is None:
            return False

        logger.debug("Unsuspending Unload Task")
        return await self.__unload_timer.unsuspend()


class _UnloadTimer:
    def __init__(self, timeout: float, timeout_step: float = 0.1) -> None:
        self.__timeout: float = timeout
        self.__timeout_remainder: float = self.__timeout
        self.__timeout_step: float = timeout_step
        self.__suspend_counter = 0

        self.__lock = asyncio.Lock()

    async def schedule(self):
        await self.reset()
        while await self.timeout > 0:
            await asyncio.sleep(self.timeout_step)
            await self.decrement(self.timeout_step)
        return _Unloader()

    @property
    async def timeout(self) -> float:
        """Get the remaining timeout (async).

        Returns:
            float: The remaining timeout
        """
        async with self.__lock:
            if bool(self.__suspend_counter):
                return 1.0
            return self.__timeout_remainder

    @property
    def timeout_step(self) -> float:
        """Return the timeout polling step.

        Returns:
            float: The timeout polling step length
        """
        return self.__timeout_step

    @property
    async def suspended(self) -> bool:
        async with self.__lock:
            return bool(self.__suspend_counter)

    async def suspend(self) -> bool:
        async with self.__lock:
            self.__suspend_counter += 1
            return bool(self.__suspend_counter)

    async def unsuspend(self) -> bool:
        async with self.__lock:
            self.__suspend_counter = max(0, self.__suspend_counter - 1)
            return bool(self.__suspend_counter)

    async def decrement(self, step: float) -> float:
        """Decrement the timeout remainder by the given step.

        Args:
            step (float): The step to decrement by

        Returns:
            float: The remaining timeout after decrementing
        """
        async with self.__lock:
            if not bool(self.__suspend_counter):
                self.__timeout_remainder -= step
            return self.__timeout_remainder

    async def reset(self) -> float:
        """Reset the timeout remainder to the original timeout.

        Returns:
            float: The remaining timeout after resetting
        """
        async with self.__lock:
            self.__timeout_remainder = self.__timeout
            return self.__timeout_remainder

    async def reschedule(self, timeout: float) -> float:
        """Reschedule the timeout remainder to the given timeout.

        Args:
            timeout (float): The new timeout

        Returns:
            float: The remaining timeout after rescheduling
        """
        self.__timeout = timeout
        return await self.reset()


class _Unloader:
    @staticmethod
    async def unload():
        try:
            logger.debug("_Unloader: get_llama_proxy()")
            llama_proxy = next(get_llama_proxy())
            # llama_proxy = await asyncio.wait_for(get_llama_proxy(), timeout=1)

            if llama_proxy is not None:
                logger.debug("_Unloader: Unloading Model")
                # del llama_proxy._current_model
                llama_proxy._current_model = None

            return True
        except asyncio.TimeoutError:
            # Model still in use?
            logger.warning(
                "_Unloader: get_llama_proxy() timed out. Is the model still in use?"
            )
            return False


UNLOAD_TIMEOUT: Final[float] = os.getenv("UNLOAD_TIMEOUT", 3600.)
UNLOAD_POLLING: Final[float] = os.getenv("UNLOAD_POLLING", 10.)
UNLOAD_MANAGER: Final[UnloadManager] = UnloadManager(UNLOAD_TIMEOUT, timeout_step=UNLOAD_POLLING)


async def schedule_unload(event_loop, did_suspend: bool = False):
    logger.debug("schedule_unload() called")

    try:
        if did_suspend is True:
            logger.debug("Unsuspending Unload")
            await UNLOAD_MANAGER.unsuspend()

        async with UNLOAD_MANAGER as ctx:
            logger.info("Scheduling Unload")

            unload_task = ctx.create_task(event_loop)
            logger.debug("Aquired Unload Task")

            await unload_task
            logger.debug("Awaited Unload Task")

            if unload_task.cancelled():
                logger.warning("Unload Task was canceled")
                return None

            unloader = unload_task.result()
            if unloader is not None:
                logger.info("Unloading Model")
                await unloader.unload()
            else:
                logger.warning("Unload Task returned None")
    except AlreadyLockedError as e:
        logger.warning("Unload Lock already aquired, resetting timer.")
        await UNLOAD_MANAGER.reset()
    except Exception as e:
        raise RuntimeError("Error while scheduling unload") from e


async def suspend_unload():
    logger.info("Suspending Unload")
    return await UNLOAD_MANAGER.suspend()


router = APIRouter(route_class=RouteErrorHandler)

_server_settings: Optional[ServerSettings] = None


def set_server_settings(server_settings: ServerSettings):
    global _server_settings
    _server_settings = server_settings


def get_server_settings():
    yield _server_settings


_llama_proxy: Optional[LlamaProxy] = None

llama_outer_lock = Lock()
llama_inner_lock = Lock()


def set_llama_proxy(model_settings: List[ModelSettings]):
    global _llama_proxy
    _llama_proxy = LlamaProxy(models=model_settings)


def get_llama_proxy():
    # NOTE: This double lock allows the currently streaming llama model to
    # check if any other requests are pending in the same thread and cancel
    # the stream if so.
    llama_outer_lock.acquire()
    release_outer_lock = True
    try:
        llama_inner_lock.acquire()
        try:
            llama_outer_lock.release()
            release_outer_lock = False
            yield _llama_proxy
        finally:
            llama_inner_lock.release()
    finally:
        if release_outer_lock:
            llama_outer_lock.release()


def create_app(
    settings: Settings | None = None,
    server_settings: ServerSettings | None = None,
    model_settings: List[ModelSettings] | None = None,
):
    config_file = os.environ.get("CONFIG_FILE", None)
    if config_file is not None:
        if not os.path.exists(config_file):
            raise ValueError(f"Config file {config_file} not found!")
        with open(config_file, "rb") as f:
            config_file_settings = ConfigFileSettings.model_validate_json(f.read())
            server_settings = ServerSettings.model_validate(config_file_settings)
            model_settings = config_file_settings.models

    if server_settings is None and model_settings is None:
        if settings is None:
            settings = Settings()
        server_settings = ServerSettings.model_validate(settings)
        model_settings = [ModelSettings.model_validate(settings)]

    assert (
        server_settings is not None and model_settings is not None
    ), "server_settings and model_settings must be provided together"

    set_server_settings(server_settings)
    middleware = [Middleware(RawContextMiddleware, plugins=(RequestIdPlugin(),))]
    app = FastAPI(
        middleware=middleware,
        title="🦙 llama.cpp Python API",
        version=llama_cpp.__version__,
    )
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    app.include_router(router)

    assert model_settings is not None
    set_llama_proxy(model_settings=model_settings)

    return app


async def get_event_publisher(
    request: Request,
    inner_send_chan: MemoryObjectSendStream,
    iterator: Iterator,
):
    async with inner_send_chan:
        try:
            async for chunk in iterate_in_threadpool(iterator):
                await inner_send_chan.send(dict(data=json.dumps(chunk)))
                if await request.is_disconnected():
                    raise anyio.get_cancelled_exc_class()()
                if (
                    next(get_server_settings()).interrupt_requests
                    and llama_outer_lock.locked()
                ):
                    await inner_send_chan.send(dict(data="[DONE]"))
                    raise anyio.get_cancelled_exc_class()()
            await inner_send_chan.send(dict(data="[DONE]"))
        except anyio.get_cancelled_exc_class() as e:
            print("disconnected")
            with anyio.move_on_after(1, shield=True):
                print(f"Disconnected from client (via refresh/close) {request.client}")
                raise e


def _logit_bias_tokens_to_input_ids(
    llama: llama_cpp.Llama,
    logit_bias: Dict[str, float],
) -> Dict[str, float]:
    to_bias: Dict[str, float] = {}
    for token, score in logit_bias.items():
        token = token.encode("utf-8")
        for input_id in llama.tokenize(token, add_bos=False, special=True):
            to_bias[str(input_id)] = score
    return to_bias


# Setup Bearer authentication scheme
bearer_scheme = HTTPBearer(auto_error=False)


async def authenticate(
    settings: Settings = Depends(get_server_settings),
    authorization: Optional[str] = Depends(bearer_scheme),
):
    # Skip API key check if it's not set in settings
    if settings.api_key is None:
        return True

    # check bearer credentials against the api_key
    if authorization and authorization.credentials == settings.api_key:
        # api key is valid
        return authorization.credentials

    # raise http error 401
    raise HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Invalid API key",
    )


@router.post(
    "/v1/completions",
    summary="Completion",
    dependencies=[Depends(authenticate)],
    response_model=Union[
        llama_cpp.CreateCompletionResponse,
        str,
    ],
    responses={
        "200": {
            "description": "Successful Response",
            "content": {
                "application/json": {
                    "schema": {
                        "anyOf": [
                            {"$ref": "#/components/schemas/CreateCompletionResponse"}
                        ],
                        "title": "Completion response, when stream=False",
                    }
                },
                "text/event-stream": {
                    "schema": {
                        "type": "string",
                        "title": "Server Side Streaming response, when stream=True. "
                        + "See SSE format: https://developer.mozilla.org/en-US/docs/Web/API/Server-sent_events/Using_server-sent_events#Event_stream_format",  # noqa: E501
                        "example": """data: {... see CreateCompletionResponse ...} \\n\\n data: ... \\n\\n ... data: [DONE]""",
                    }
                },
            },
        }
    },
)
@router.post(
    "/v1/engines/copilot-codex/completions",
    include_in_schema=False,
    dependencies=[Depends(authenticate)],
)
async def create_completion(
    request: Request,
    body: CreateCompletionRequest,
    background_tasks: BackgroundTasks,
    llama_proxy: LlamaProxy = Depends(get_llama_proxy),
) -> llama_cpp.Completion:
    suspended = await suspend_unload()
    background_tasks.add_task(
        schedule_unload, asyncio.get_event_loop(), did_suspend=suspended
    )
    if isinstance(body.prompt, list):
        assert len(body.prompt) <= 1
        body.prompt = body.prompt[0] if len(body.prompt) > 0 else ""

    llama = llama_proxy(
        body.model
        if request.url.path != "/v1/engines/copilot-codex/completions"
        else "copilot-codex"
    )

    exclude = {
        "n",
        "best_of",
        "logit_bias_type",
        "user",
    }
    kwargs = body.model_dump(exclude=exclude)

    if body.logit_bias is not None:
        kwargs["logit_bias"] = (
            _logit_bias_tokens_to_input_ids(llama, body.logit_bias)
            if body.logit_bias_type == "tokens"
            else body.logit_bias
        )

    if body.grammar is not None:
        kwargs["grammar"] = llama_cpp.LlamaGrammar.from_string(body.grammar)

    iterator_or_completion: Union[
        llama_cpp.CreateCompletionResponse,
        Iterator[llama_cpp.CreateCompletionStreamResponse],
    ] = await run_in_threadpool(llama, **kwargs)

    if isinstance(iterator_or_completion, Iterator):
        # EAFP: It's easier to ask for forgiveness than permission
        first_response = await run_in_threadpool(next, iterator_or_completion)

        # If no exception was raised from first_response, we can assume that
        # the iterator is valid and we can use it to stream the response.
        def iterator() -> Iterator[llama_cpp.CreateCompletionStreamResponse]:
            yield first_response
            yield from iterator_or_completion

        send_chan, recv_chan = anyio.create_memory_object_stream(10)
        return EventSourceResponse(
            recv_chan,
            data_sender_callable=partial(  # type: ignore
                get_event_publisher,
                request=request,
                inner_send_chan=send_chan,
                iterator=iterator(),
            ),
            sep='\n',
        )
    else:
        return iterator_or_completion


@router.post(
    "/v1/embeddings", summary="Embedding", dependencies=[Depends(authenticate)]
)
async def create_embedding(
    request: CreateEmbeddingRequest,
    background_tasks: BackgroundTasks,
    llama_proxy: LlamaProxy = Depends(get_llama_proxy),
):
    suspended = await suspend_unload()
    background_tasks.add_task(
        schedule_unload, asyncio.get_event_loop(), did_suspend=suspended
    )
    return await run_in_threadpool(
        llama_proxy(request.model).create_embedding,
        **request.model_dump(exclude={"user"}),
    )


@router.post(
    "/v1/chat/completions",
    summary="Chat",
    dependencies=[Depends(authenticate)],
    response_model=Union[llama_cpp.ChatCompletion, str],
    responses={
        "200": {
            "description": "Successful Response",
            "content": {
                "application/json": {
                    "schema": {
                        "anyOf": [
                            {
                                "$ref": "#/components/schemas/CreateChatCompletionResponse"
                            }
                        ],
                        "title": "Completion response, when stream=False",
                    }
                },
                "text/event-stream": {
                    "schema": {
                        "type": "string",
                        "title": "Server Side Streaming response, when stream=True"
                        + "See SSE format: https://developer.mozilla.org/en-US/docs/Web/API/Server-sent_events/Using_server-sent_events#Event_stream_format",  # noqa: E501
                        "example": """data: {... see CreateChatCompletionResponse ...} \\n\\n data: ... \\n\\n ... data: [DONE]""",
                    }
                },
            },
        }
    },
)
async def create_chat_completion(
    request: Request,
    body: CreateChatCompletionRequest,
    background_tasks: BackgroundTasks,
    llama_proxy: LlamaProxy = Depends(get_llama_proxy),
) -> llama_cpp.ChatCompletion:
    suspended = await suspend_unload()
    background_tasks.add_task(
        schedule_unload, asyncio.get_event_loop(), did_suspend=suspended
    )
    exclude = {
        "n",
        "logit_bias_type",
        "user",
    }
    kwargs = body.model_dump(exclude=exclude)
    llama = llama_proxy(body.model)
    if body.logit_bias is not None:
        kwargs["logit_bias"] = (
            _logit_bias_tokens_to_input_ids(llama, body.logit_bias)
            if body.logit_bias_type == "tokens"
            else body.logit_bias
        )

    if body.grammar is not None:
        kwargs["grammar"] = llama_cpp.LlamaGrammar.from_string(body.grammar)

    iterator_or_completion: Union[
        llama_cpp.ChatCompletion, Iterator[llama_cpp.ChatCompletionChunk]
    ] = await run_in_threadpool(llama.create_chat_completion, **kwargs)

    if isinstance(iterator_or_completion, Iterator):
        # EAFP: It's easier to ask for forgiveness than permission
        first_response = await run_in_threadpool(next, iterator_or_completion)

        # If no exception was raised from first_response, we can assume that
        # the iterator is valid and we can use it to stream the response.
        def iterator() -> Iterator[llama_cpp.ChatCompletionChunk]:
            yield first_response
            yield from iterator_or_completion

        send_chan, recv_chan = anyio.create_memory_object_stream(10)
        return EventSourceResponse(
            recv_chan,
            data_sender_callable=partial(  # type: ignore
                get_event_publisher,
                request=request,
                inner_send_chan=send_chan,
                iterator=iterator(),
            ),
            sep='\n',
        )
    else:
        return iterator_or_completion


@router.get("/v1/models", summary="Models", dependencies=[Depends(authenticate)])
async def get_models(
    llama_proxy: LlamaProxy = Depends(get_llama_proxy),
) -> ModelList:
    return {
        "object": "list",
        "data": [
            {
                "id": model_alias,
                "object": "model",
                "owned_by": "me",
                "permissions": [],
            }
            for model_alias in llama_proxy
        ],
    }

from groq import APIError
from qdrant_client.http.exceptions import ResponseHandlingException, UnexpectedResponse
from redis.exceptions import RedisError
from sqlalchemy.exc import SQLAlchemyError

# Failures of our own infrastructure (not the user's fault). Routes turn these into HTTP 503.
BACKEND_ERRORS = (RedisError, SQLAlchemyError, ResponseHandlingException, UnexpectedResponse)

# Failures of the language model provider. Routes turn these into HTTP 502.
LLM_ERRORS = (APIError,)


class SessionNotFoundError(Exception):
    """The client sent a session_id that was never created by this server."""
from typing import List, Optional, Dict
from pydantic import BaseModel
import asyncio
import uvicorn
import os
from dotenv import load_dotenv
import sys
import logging
import itertools
import json
import argparse
import time
from httpx import AsyncClient
from fastapi import FastAPI, HTTPException, Depends, APIRouter, Request
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from fastapi.responses import StreamingResponse, JSONResponse
from fastapi_poe.types import ProtocolMessage
from fastapi_poe.client import get_bot_response, get_final_response, QueryRequest, BotError

# 加载环境变量
load_dotenv()

app = FastAPI()
security = HTTPBearer()
router = APIRouter()

# Request logging middleware for debug mode
@app.middleware("http")
async def log_requests(request: Request, call_next):
    if DEBUG_MODE:
        start_time = time.time()
        
        # Log request details
        logger.debug(f"=== Incoming Request ===")
        logger.debug(f"Method: {request.method}")
        logger.debug(f"URL: {request.url}")
        logger.debug(f"Headers: {dict(request.headers)}")
        logger.debug(f"Path Parameters: {request.path_params}")
        logger.debug(f"Query Parameters: {dict(request.query_params)}")
        
        # Read and log body for POST requests
        if request.method in ["POST", "PUT", "PATCH"]:
            try:
                body = await request.body()
                if body:
                    # Try to parse as JSON for pretty logging
                    try:
                        body_json = json.loads(body.decode())
                        # Mask sensitive data in logs
                        if isinstance(body_json, dict):
                            masked_body = body_json.copy()
                            # Mask authorization headers and tokens
                            for key in ['authorization', 'token', 'api_key', 'password']:
                                if key in masked_body:
                                    masked_body[key] = "*" * 8
                            logger.debug(f"Request Body: {json.dumps(masked_body, indent=2, ensure_ascii=False)}")
                        else:
                            logger.debug(f"Request Body: {body_json}")
                    except json.JSONDecodeError:
                        # If not JSON, log as string (truncated if too long)
                        body_str = body.decode()[:500]
                        logger.debug(f"Request Body (first 500 chars): {body_str}")

            except Exception as e:
                logger.debug(f"Could not read request body: {str(e)}")
    
    # Process the request
    response = await call_next(request)
    
    if DEBUG_MODE:
        process_time = time.time() - start_time
        logger.debug(f"=== Response ===")
        logger.debug(f"Status Code: {response.status_code}")
        logger.debug(f"Response Headers: {dict(response.headers)}")
        logger.debug(f"Processing Time: {process_time:.4f} seconds")
        logger.debug(f"Callnext: {call_next}")
        logger.debug(f"=== End Request ===\n")
    
    return response

# 从环境变量获取配置
PORT = int(os.getenv("PORT", 3700))
TIMEOUT = int(os.getenv("TIMEOUT", 120))
PROXY = os.getenv("PROXY", "")

# Global debug flag
DEBUG_MODE = False

# 设置日志
def setup_logging(debug: bool = False):
    global DEBUG_MODE
    DEBUG_MODE = debug
    
    if debug:
        logging.basicConfig(
            level=logging.DEBUG, 
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        logger = logging.getLogger(__name__)
        logger.setLevel(logging.DEBUG)
        logger.info("Debug mode enabled - detailed request logging activated")
    else:
        logging.basicConfig(
            level=logging.INFO, 
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        logger = logging.getLogger(__name__)
    
    return logger

logger = setup_logging()

# 解析JSON数组格式的环境变量
def parse_json_env(env_name, default=None):
    value = os.getenv(env_name)
    if value:
        try:
            value = value.strip()
            if not value.startswith('['):
                if value.startswith('"') or value.startswith("'"):
                    value = value[1:]
                if value.endswith('"') or value.endswith("'"):
                    value = value[:-1]
            return json.loads(value)
        except json.JSONDecodeError as e:
            logger.warning(f"Failed to parse {env_name} as JSON: {str(e)}, using default value")
            logger.debug(f"Attempted to parse value: {value}")
    return default or []

ACCESS_TOKENS = set(parse_json_env("ACCESS_TOKENS"))
BOT_NAMES = parse_json_env("BOT_NAMES")
POE_API_KEYS = parse_json_env("POE_API_KEYS")

# 初始化代理
proxy = None
if not PROXY:
    proxy = AsyncClient(timeout=TIMEOUT)
else:
    proxy = AsyncClient(proxy=PROXY, timeout=TIMEOUT)

# 初始化客户端字典和API密钥循环
client_dict = {}
api_key_cycle = None

bot_names_map = {name.lower(): name for name in BOT_NAMES}


class Message(BaseModel):
    role: str
    content: str

class ToolFunction(BaseModel):
    name: str
    description: str
    parameters: Dict

class Tool(BaseModel):
    type: str
    function: ToolFunction

class CompletionRequest(BaseModel):
    model: str
    messages: List[Message]
    stream: Optional[bool] = False
    temperature: Optional[float] = 0.7
    skip_system_prompt: Optional[bool] = None
    frequency_penalty: Optional[float] = 0.0
    presence_penalty: Optional[float] = 0.0
    logit_bias: Optional[Dict[str, int]] = None
    stop_sequences: Optional[List[str]] = None
    tools: Optional[List[Tool]] = []

    class Config:
        json_schema_extra = {
            "example": {
                "model": "GPT-3.5-Turbo",
                "messages": [
                    {"role": "system", "content": "You are a helpful assistant."},
                    {"role": "user", "content": "Hello!"}
                ],
                "stream": True, 
                  "tools": [
                        {
                        "type": "function",
                        "function": {
                            "name": "builtin_read_file",
                            "description": "Use this tool if you need to view the contents of an existing file.",
                            "parameters": {
                                "type": "object",
                                "required": [
                                    "filepath"
                                ],
                                "properties": {
                                    "filepath": {
                                        "type": "string",
                                        "description": "The path of the file to read, relative to the root of the workspace (NOT uri or absolute path)"
                                    }
                                }
                            }
                        }
                        }
                  ]
            }
        }


class TextCompletionRequest(BaseModel):
    model: str
    prompt: str
    stream: Optional[bool] = False
    temperature: Optional[float] = 0.7
    max_tokens: Optional[int] = 1000
    top_p: Optional[float] = 1.0
    frequency_penalty: Optional[float] = 0.0
    presence_penalty: Optional[float] = 0.0
    stop: Optional[List[str]] = None
    logit_bias: Optional[Dict[str, int]] = None
    tools: Optional[List[Tool]] = []
    
    class Config:
        json_schema_extra = {
            "example": {
                "model": "GPT-3.5-Turbo",
                "prompt": "The weather today is",
                "stream": False,
                "max_tokens": 100, 
            }
        }


async def add_token(token: str):
    """Add a new API token to the client dictionary.

    This function attempts to validate and add a new API token to the system.
    It first checks if the token is empty or already exists, then tries to
    validate it by making a test request to the POE API.

    Args:
        token (str): The API token to be added.

    Returns:
        str: A status message indicating the result:
            - "ok" if token was successfully added
            - "exist" if token already exists
            - "failed: [error message]" if token validation failed
    """
    global api_key_cycle
    if not token:
        logger.error("Empty token provided")
        return "failed: empty token"

    if token not in client_dict:
        try:
            logger.info(f"Attempting to add apikey: {token[:6]}...")  # 只记录前6位
            request = CompletionRequest(
                model="GPT-4o-mini",
                messages=[Message(role="user", content="Please return 'OK'")],
                temperature=0.7
            )
            ret = await get_responses(request, token)
            if ret == "OK":
                client_dict[token] = token
                api_key_cycle = itertools.cycle(client_dict.values())
                logger.info(f"apikey added successfully: {token[:6]}...")
                return "ok"
            else:
                logger.error(f"Failed to add apikey: {token[:6]}..., response: {ret}")
                return "failed"
        except Exception as exception:
            logger.error(f"Failed to connect to poe due to {str(exception)}")
            if isinstance(exception, BotError):
                try:
                    error_json = json.loads(exception.text)
                    return f"failed: {json.dumps(error_json)}"
                except json.JSONDecodeError:
                    return f"failed: {str(exception)}"
            return f"failed: {str(exception)}"
    else:
        logger.info(f"apikey already exists: {token[:6]}...")
        return "exist"


async def get_responses(request: CompletionRequest, token: str):
    """Get responses from the POE API for a completion request.

    This function processes a completion request and returns the response from
    the POE API. It handles model name mapping and converts the request format
    to match POE's API requirements.

    Args:
        request (CompletionRequest): The completion request object containing
            model, messages, and other parameters.
        token (str): The API token to use for authentication.

    Returns:
        str: The response text from the POE API.

    Raises:
        HTTPException: If the token is missing or invalid, or if the model
            is not supported.
    """
    if not token:
        raise HTTPException(status_code=400, detail="Token is required")

    model_lower = request.model.lower()
    if model_lower in bot_names_map:
        request.model = bot_names_map[model_lower]
        message = [
            ProtocolMessage(role=msg.role if msg.role in ["user", "system"] else "bot", content=msg.content)
            for msg in request.messages
        ]
        additional_params = {
            "temperature": request.temperature,
            "skip_system_prompt": request.skip_system_prompt if request.skip_system_prompt is not None else False,
            "logit_bias": request.logit_bias if request.logit_bias is not None else {},
            "stop_sequences": request.stop_sequences if request.stop_sequences is not None else []
        }
        query = QueryRequest(
            query=message,
            user_id="",
            conversation_id="",
            message_id="",
            version="1.0",
            type="query",
            **additional_params
        )
        try:
            return await get_final_response(query, bot_name=request.model, api_key=token, session=proxy)
        except Exception as e:
            logger.error(f"Error in get_final_response: {str(e)}")
            raise HTTPException(status_code=500, detail=str(e))
    else:
        raise HTTPException(status_code=400, detail=f"Model {request.model} is not supported")


async def get_text_completion_response(request: TextCompletionRequest, token: str):
    """Convert text completion request to POE API format and get response.

    This function converts a text completion request into the format expected
    by the POE API and retrieves the response.

    Args:
        request (TextCompletionRequest): The text completion request object
            containing model, prompt, and other parameters.
        token (str): The API token to use for authentication.

    Returns:
        str: The response text from the POE API.

    Raises:
        HTTPException: If the token is missing or invalid, or if the model
            is not supported.
    """
    if not token:
        raise HTTPException(status_code=400, detail="Token is required")

    model_lower = request.model.lower()
    if model_lower not in bot_names_map:
        raise HTTPException(status_code=400, detail=f"Model {request.model} is not supported")

    # Convert prompt to messages format for POE API
    protocol_messages = [
        ProtocolMessage(role="user", content=request.prompt)
    ]
    
    additional_params = {
        "temperature": request.temperature,
        "skip_system_prompt": False,
        "logit_bias": request.logit_bias if request.logit_bias is not None else {},
        "stop_sequences": request.stop if request.stop is not None else []
    }
    
    query = QueryRequest(
        query=protocol_messages,
        user_id="",
        conversation_id="",
        message_id="",
        version="1.0",
        type="query",
        **additional_params
    )
    
    try:
        return await get_final_response(query, bot_name=bot_names_map[model_lower], api_key=token, session=proxy)
    except Exception as e:
        logger.error(f"Error in get_final_response: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


async def verify_token(credentials: HTTPAuthorizationCredentials = Depends(security)):
    """Verify the authentication token.

    This function validates the provided authentication token against the
    list of allowed access tokens.

    Args:
        credentials (HTTPAuthorizationCredentials): The credentials object
            containing the token to verify.

    Returns:
        str: The verified token.

    Raises:
        HTTPException: If the credentials are missing, invalid, or not
            in the allowed access tokens list.
    """
    if not credentials:
        raise HTTPException(
            status_code=401,
            detail="Missing authentication credentials",
            headers={"WWW-Authenticate": "Bearer"},
        )
    if not credentials.credentials:
        raise HTTPException(
            status_code=401,
            detail="Invalid authentication credentials",
            headers={"WWW-Authenticate": "Bearer"},
        )
    if credentials.credentials not in ACCESS_TOKENS:
        raise HTTPException(
            status_code=401,
            detail="Invalid API key",
            headers={"WWW-Authenticate": "Bearer"},
        )
    return credentials.credentials

@router.options("/{full_path:path}")
async def options_handler(full_path: str, request: Request):
    response = JSONResponse(content={"message": "OK"})
    response.headers["Access-Control-Allow-Origin"] = "*"
    response.headers["Access-Control-Allow-Methods"] = "GET, POST, OPTIONS"
    response.headers["Access-Control-Allow-Headers"] = "Authorization, Content-Type"
    response.headers["Access-Control-Max-Age"] = "86400"
    return response

# Helper function to process tools
def process_tools(tools: List[Dict]) -> str:
    """Reformat tools into a human-readable string for system message.

    This function takes a list of tool definitions and converts them into
    a formatted string that can be appended to the system message.

    Args:
        tools (List[Dict]): List of tool definitions, each containing
            function name, description, and parameters.

    Returns:
        str: A formatted string describing the available tools and
            instructions for using them.
    """
    logger.warning("Processing tools")
    tool_descriptions = []
    for tool in tools:
        try:
            tool_name = tool.function.name
            tool_description = tool.function.description
            tool_properties = ", ".join(
                [f"{key}: {value['description']}" for key, value in tool.function.parameters["properties"].items()]
            )

            tool_descriptions.append(
                f"- **{tool_name}**: {tool_description}\n  Parameters: {tool_properties}"
            )
        except KeyError as e:
            # Log and skip invalid tool definitions
            logger.warning(f"Invalid tool format: {tool}. Missing key: {str(e)}")
            continue

    tool_call_instruction = """
To call a tool, your response should end with a json like this:
{
    'tool_call': [Tool name], 
    'parameters': [Tool parameter dict]
}
Do not use a code block for tool call. Do not ask for confirmation before calling a tool, you are granted such permission to call the tool in agent mode. Additionally, you must keep going until your have completed all the assigned tasks. Do not stop until all tasks are completed.
    """

    if tool_descriptions:
        return (
            "The following tools are available to assist with your request:\n"
            + "\n".join(tool_descriptions) + "\n" + tool_call_instruction
        )
    return ""


async def process_completion_request(
    request: CompletionRequest,
    token: str,
    request_id: str,
    is_text_completion: bool = False,
):
    """Process completion requests for both chat and text completion.

    This function handles the core logic for processing both chat and text
    completion requests, including streaming and non-streaming responses.
    It also handles tool calls in the response.

    Args:
        request (CompletionRequest): The completion request object.
        token (str): The API token for authentication.
        request_id (str): A unique identifier for the request.
        is_text_completion (bool, optional): Whether this is a text completion
            request. Defaults to False.

    Returns:
        Union[Dict, StreamingResponse]: Either a JSON response for non-streaming
            requests or a StreamingResponse for streaming requests.

    Raises:
        HTTPException: If there are any errors during request processing.
    """
    try:
        # Log the incoming request safely
        safe_request = request.model_dump()
        logger.info(f"Full request [{request_id}]: {json.dumps(request.dict(), ensure_ascii=False)}")
        if "messages" in safe_request:
            safe_request["messages"] = [
                {**msg, "content": msg["content"][:100] + "..." if len(msg["content"]) > 100 else msg["content"]}
                for msg in safe_request["messages"]
            ]
        elif "prompt" in safe_request:
            safe_request["prompt"] = safe_request["prompt"][:100] + "..." if len(safe_request["prompt"]) > 100 else safe_request["prompt"]
        logger.info(f"Request [{request_id}]: {json.dumps(safe_request, ensure_ascii=False)}")

        # Extract tools from the request, if present
        tools_section = ""
        if hasattr(request, "tools"):
            logger.debug(f"Full request [{request_id}]: {json.dumps(request.dict(), ensure_ascii=False)}")
            tools_section = process_tools(request.tools)

        # Check for valid API tokens
        if not api_key_cycle:
            raise HTTPException(status_code=500, detail="No valid API tokens available")

        # Validate the model
        model_lower = request.model.lower()
        if model_lower not in bot_names_map:
            raise HTTPException(status_code=400, detail=f"Model {request.model} not found")

        request.model = bot_names_map[model_lower]
        poe_token = next(api_key_cycle)

        # Build protocol messages for chat or text completion
        if is_text_completion:
            protocol_messages = [ProtocolMessage(role="user", content=request.prompt)]
        else:
            protocol_messages = [
                ProtocolMessage(role=msg.role if msg.role in ["user", "system"] else "bot", content=msg.content)
                for msg in request.messages
            ]

        # Append tools information to the system prompt if tools exist
        if tools_section:
            tools_prompt = (
                f"{tools_section}\n\n"
                "Use the tools as needed to fulfill the user's request. If unsure, ask for clarification."
            )
            protocol_messages.insert(0, ProtocolMessage(role="system", content=tools_prompt))

        # Handle streaming or non-streaming responses
        if request.stream:
            import re
            async def response_generator():
                total_response = ""
                last_sent_base_content = None
                elapsed_time_pattern = re.compile(r" \(\d+s elapsed\)$")
                tool_call_pattern = re.compile(r'\{\s*[\'"]tool_call[\'"]\s*:\s*[\'"]([^\'"]+)[\'"]\s*,\s*[\'"]parameters[\'"]\s*:\s*({[^}]+})\s*\}')
                
                # Buffer for accumulating tool call fragments
                tool_call_buffer = ""
                in_tool_call = False
                brace_count = 0

                try:
                    async for partial in get_bot_response(
                        protocol_messages, bot_name=request.model, api_key=poe_token, session=proxy
                    ):
                        if partial and partial.text:
                            if partial.text.strip() in ["Thinking...", "Generating image..."]:
                                continue

                            base_content = elapsed_time_pattern.sub("", partial.text)

                            if last_sent_base_content == base_content:
                                continue

                            total_response += base_content

                            # Check if we're in a tool call or if this might be the start of one
                            if '{' in base_content and not in_tool_call:
                                in_tool_call = True
                                tool_call_buffer = base_content
                                brace_count = base_content.count('{') - base_content.count('}')
                            elif in_tool_call:
                                tool_call_buffer += base_content
                                brace_count += base_content.count('{') - base_content.count('}')
                                
                                # If we have a complete JSON object
                                if brace_count == 0:
                                    in_tool_call = False
                                    tool_call_match = tool_call_pattern.search(tool_call_buffer)
                                    if tool_call_match:
                                        logger.info(f"Complete tool call detected: {tool_call_match.group(0)}")
                                        tool_name = tool_call_match.group(1).strip().strip("'\"")
                                        try:
                                            tool_params = json.loads(tool_call_match.group(2))
                                            chunk = {
                                                "id": request_id,
                                                "object": "chat.completion.chunk" if not is_text_completion else "text_completion",
                                                "created": int(asyncio.get_event_loop().time()),
                                                "model": request.model,
                                                "choices": [{
                                                    "delta": {
                                                        "content": None,
                                                        "tool_calls": [{
                                                            "index": 0,
                                                            "id": f"call_{request_id}",
                                                            "type": "function",
                                                            "function": {
                                                                "name": tool_name,
                                                                "arguments": json.dumps(tool_params)
                                                            }
                                                        }]
                                                    },
                                                    "index": 0,
                                                    "finish_reason": None
                                                }]
                                            }
                                            yield f"data: {json.dumps(chunk)}\n\n"
                                            tool_call_buffer = ""
                                            continue
                                        except json.JSONDecodeError:
                                            logger.warning(f"Invalid tool call JSON: {tool_call_buffer}")
                                            # Fall back to sending as regular content
                                            in_tool_call = False
                                            tool_call_buffer = ""
                                    else:
                                        logger.info(f"No tool call pattern in complete JSON: {tool_call_buffer}")
                                        # Send the accumulated content as regular text
                                        chunk = {
                                            "id": request_id,
                                            "object": "chat.completion.chunk" if not is_text_completion else "text_completion",
                                            "created": int(asyncio.get_event_loop().time()),
                                            "model": request.model,
                                            "choices": [{
                                                "delta" if not is_text_completion else "text": {
                                                    "content": tool_call_buffer
                                                },
                                                "index": 0,
                                                "finish_reason": None
                                            }]
                                        }
                                        yield f"data: {json.dumps(chunk)}\n\n"
                                        tool_call_buffer = ""
                                        continue

                            # If we're not in a tool call or haven't accumulated enough yet
                            if not in_tool_call:
                                chunk = {
                                    "id": request_id,
                                    "object": "chat.completion.chunk" if not is_text_completion else "text_completion",
                                    "created": int(asyncio.get_event_loop().time()),
                                    "model": request.model,
                                    "choices": [{
                                        "delta" if not is_text_completion else "text": {
                                            "content": base_content
                                        },
                                        "index": 0,
                                        "finish_reason": None
                                    }]
                                }
                                yield f"data: {json.dumps(chunk)}\n\n"
                                last_sent_base_content = base_content

                    # If we have any remaining buffered content, send it
                    if tool_call_buffer:
                        chunk = {
                            "id": request_id,
                            "object": "chat.completion.chunk" if not is_text_completion else "text_completion",
                            "created": int(asyncio.get_event_loop().time()),
                            "model": request.model,
                            "choices": [{
                                "delta" if not is_text_completion else "text": {
                                    "content": tool_call_buffer
                                },
                                "index": 0,
                                "finish_reason": None
                            }]
                        }
                        yield f"data: {json.dumps(chunk)}\n\n"

                    # Send end marker
                    end_chunk = {
                        "id": request_id,
                        "object": "chat.completion.chunk" if not is_text_completion else "text_completion",
                        "created": int(asyncio.get_event_loop().time()),
                        "model": request.model,
                        "choices": [{
                            "delta" if not is_text_completion else "text": {},
                            "index": 0,
                            "finish_reason": "stop"
                        }]
                    }
                    yield f"data: {json.dumps(end_chunk)}\n\n"
                    yield "data: [DONE]\n\n"

                    # Log the complete streaming response (limit length)
                    logger.info(f"Stream Response [{request_id}]: {total_response[:200]}..." if len(
                        total_response) > 200 else total_response)
                except BotError as be:
                    error_chunk = {
                        "id": request_id,
                        "object": "chat.completion.chunk" if not is_text_completion else "text_completion",
                        "created": int(asyncio.get_event_loop().time()),
                        "model": request.model,
                        "choices": [{
                            "delta" if not is_text_completion else "text": {
                                "content": json.loads(be.args[0])["text"]
                            },
                            "index": 0,
                            "finish_reason": "error"
                        }]
                    }
                    yield f"data: {json.dumps(error_chunk)}\n\n"
                    yield "data: [DONE]\n\n"
                    logger.error(f"BotError in stream generation for [{request_id}]: {str(be)}")
                except Exception as e:
                    logger.error(f"Error in stream generation for [{request_id}]: {str(e)}")
                    raise

            return StreamingResponse(response_generator(), media_type="text/event-stream")
        else:
            # Non-streaming response
            response = await get_responses(request, poe_token)
            
            # Check for tool calls in the response
            tool_call_match = re.search(r'{\s*[\'"]tool_call[\'"]\s*:\s*[\'"]([^\'"]+)[\'"]\s*,\s*[\'"]parameters[\'"]\s*:\s*({[^}]+})\s*}', response)
            if tool_call_match:
                tool_name = tool_call_match.group(1).strip().strip("'\"")
                try:
                    tool_params = json.loads(tool_call_match.group(2))
                    response_data = {
                        "id": request_id,
                        "object": "chat.completion" if not is_text_completion else "text_completion",
                        "created": int(asyncio.get_event_loop().time()),
                        "model": request.model,
                        "choices": [{
                            "index": 0,
                            "message": {
                                "role": "assistant",
                                "content": None,
                                "tool_calls": [{
                                    "id": f"call_{request_id}",
                                    "type": "function",
                                    "function": {
                                        "name": tool_name,
                                        "arguments": json.dumps(tool_params)
                                    }
                                }]
                            },
                            "finish_reason": "tool_calls"
                        }],
                        "usage": {
                            "prompt_tokens": -1,
                            "completion_tokens": -1,
                            "total_tokens": -1
                        }
                    }
                except json.JSONDecodeError:
                    # If tool parameters are not valid JSON, send as regular content
                    response_data = {
                        "id": request_id,
                        "object": "chat.completion" if not is_text_completion else "text_completion",
                        "created": int(asyncio.get_event_loop().time()),
                        "model": request.model,
                        "choices": [{
                            "index": 0,
                            "message" if not is_text_completion else "text": {
                                "role": "assistant",
                                "content": response
                            },
                            "finish_reason": "stop"
                        }],
                        "usage": {
                            "prompt_tokens": -1,
                            "completion_tokens": -1,
                            "total_tokens": -1
                        }
                    }
            else:
                response_data = {
                    "id": request_id,
                    "object": "chat.completion" if not is_text_completion else "text_completion",
                    "created": int(asyncio.get_event_loop().time()),
                    "model": request.model,
                    "choices": [{
                        "index": 0,
                        "message" if not is_text_completion else "text": {
                            "role": "assistant",
                            "content": response
                        },
                        "finish_reason": "stop"
                    }],
                    "usage": {
                        "prompt_tokens": -1,
                        "completion_tokens": -1,
                        "total_tokens": -1
                    }
                }
            # Log the complete response (limit length)
            safe_response = {**response_data}
            if len(response) > 200:
                logger.info(f"Response [{request_id}]: {json.dumps(safe_response, ensure_ascii=False)[:200]}...")
            else:
                logger.info(f"Response [{request_id}]: {json.dumps(safe_response, ensure_ascii=False)}")
            return response_data

    except Exception as e:
        error_msg = f"Error during response for request [{request_id}]: {str(e)}"
        logger.error(error_msg)
        raise HTTPException(status_code=500, detail=str(e))


# Chat completion endpoint
@router.post("/v1/chat/completions")
@router.post("/chat/completions")
async def create_completion(request: CompletionRequest, token: str = Depends(verify_token)):
    request_id = f"chat$poe-to-gpt$-{token[:6]}"
    return await process_completion_request(request, token, request_id)


# Text completion endpoint
@router.post("/v1/completion")
@router.post("/completion")
async def create_text_completion(request: TextCompletionRequest, token: str = Depends(verify_token)):
    request_id = f"completion$poe-to-gpt$-{token[:6]}"
    return await process_completion_request(request, token, request_id, is_text_completion=True)


@router.get("/models")
@router.get("/v1/models")
async def get_models():
    model_list = [{"id": name, "object": "model", "type": "llm"} for name in BOT_NAMES]
    return {"data": model_list, "object": "list"}


async def initialize_tokens(tokens: List[str]):
    """Initialize API tokens for the system.

    This function validates and initializes the provided API tokens,
    setting up the token rotation system.

    Args:
        tokens (List[str]): List of API tokens to initialize.

    Raises:
        SystemExit: If no valid tokens are found or if initialization fails.
    """
    if not tokens or all(not token for token in tokens):
        logger.error("No API keys found in the configuration.")
        sys.exit(1)
    else:
        for token in tokens:
            await add_token(token)
        if not client_dict:
            logger.error("No valid tokens were added.")
            sys.exit(1)
        else:
            global api_key_cycle
            api_key_cycle = itertools.cycle(client_dict.values())
            logger.info(f"Successfully initialized {len(client_dict)} API tokens")


app.include_router(router)


def parse_arguments():
    """Parse command line arguments for the server.

    Returns:
        argparse.Namespace: Parsed command line arguments including:
            - debug: Enable debug mode
            - port: Server port number
            - host: Server host address
    """
    parser = argparse.ArgumentParser(description="POE to GPT API Bridge Server")
    parser.add_argument(
        "--debug", 
        action="store_true", 
        help="Enable debug mode with detailed request logging"
    )
    parser.add_argument(
        "--port", 
        type=int, 
        default=PORT, 
        help=f"Port to run the server on (default: {PORT})"
    )
    parser.add_argument(
        "--host", 
        type=str, 
        default="0.0.0.0", 
        help="Host to bind the server to (default: 0.0.0.0)"
    )
    return parser.parse_args()


async def main(tokens: List[str] = None, debug: bool = False, host: str = "0.0.0.0", port: int = None):
    """Main entry point for the POE to GPT API Bridge Server.

    This function initializes and starts both HTTP and HTTPS servers
    concurrently.

    Args:
        tokens (List[str], optional): List of API tokens to initialize.
            Defaults to None.
        debug (bool, optional): Enable debug mode. Defaults to False.
        host (str, optional): Host address to bind to. Defaults to "0.0.0.0".
        port (int, optional): Port number to use. Defaults to None.

    Raises:
        SystemExit: If server startup fails.
    """
    try:
        # Setup logging based on debug mode
        global logger
        logger = setup_logging(debug)
        
        if debug:
            logger.info("Starting server in DEBUG mode")
        
        await initialize_tokens(tokens)
        
        # Use provided port or default
        server_port = port if port is not None else PORT
        
        ssl_keyfile = './server.key'
        ssl_certfile = './server.crt'
        
        # Set log level based on debug mode
        log_level = "debug" if debug else "info"
        
        # HTTP server config
        http_conf = uvicorn.Config(
            app,
            host=host,
            port=server_port,
            log_level=log_level
        )
        
        # HTTPS server config
        https_conf = uvicorn.Config(
            app,
            host=host,
            port=server_port + 1,  # Use next port for HTTPS
            log_level=log_level,
            ssl_keyfile=ssl_keyfile,
            ssl_certfile=ssl_certfile
        )
        
        logger.info(f"Starting HTTP server on {host}:{server_port}")
        logger.info(f"Starting HTTPS server on {host}:{server_port + 1}")
        
        # Create and start both servers
        http_server = uvicorn.Server(http_conf)
        https_server = uvicorn.Server(https_conf)
        
        # Run both servers concurrently
        await asyncio.gather(
            http_server.serve(),
            https_server.serve()
        )
    except Exception as e:
        logger.error(f"Failed to start server: {str(e)}")
        sys.exit(1)


if __name__ == "__main__":
    args = parse_arguments()
    asyncio.run(main(
        tokens=POE_API_KEYS, 
        debug=args.debug, 
        host=args.host, 
        port=args.port
    ))

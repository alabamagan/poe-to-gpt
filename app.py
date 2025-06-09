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
                
                # Create a new request object with the body
                async def receive():
                    return {"type": "http.request", "body": body}
                
                request._receive = receive
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

    class Config:
        json_schema_extra = {
            "example": {
                "model": "GPT-3.5-Turbo",
                "messages": [
                    {"role": "system", "content": "You are a helpful assistant."},
                    {"role": "user", "content": "Hello!"}
                ],
                "stream": True
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
    
    class Config:
        json_schema_extra = {
            "example": {
                "model": "GPT-3.5-Turbo",
                "prompt": "The weather today is",
                "stream": False,
                "max_tokens": 100
            }
        }


async def add_token(token: str):
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
    """Convert text completion request to POE API format and get response"""
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

@router.post("/v1/chat/completions")
@router.post("/chat/completions")
async def create_completion(request: CompletionRequest, token: str = Depends(verify_token)):
    request_id = "chat$poe-to-gpt$-" + token[:6]

    try:
        # 打印请求参数（隐藏敏感信息）
        safe_request = request.model_dump()
        if "messages" in safe_request:
            safe_request["messages"] = [
                {**msg, "content": msg["content"][:100] + "..." if len(msg["content"]) > 100 else msg["content"]}
                for msg in safe_request["messages"]
            ]
        logger.info(f"Request [{request_id}]: {json.dumps(safe_request, ensure_ascii=False)}")

        if not api_key_cycle:
            raise HTTPException(status_code=500, detail="No valid API tokens available")

        model_lower = request.model.lower()
        if model_lower not in bot_names_map:
            raise HTTPException(status_code=400, detail=f"Model {request.model} not found")

        request.model = bot_names_map[model_lower]
        
        protocol_messages = [
            ProtocolMessage(role=msg.role if msg.role in ["user", "system"] else "bot", content=msg.content)
            for msg in request.messages
        ]
        poe_token = next(api_key_cycle)

        if request.stream:
            import re
            async def response_generator():
                total_response = ""
                last_sent_base_content = None
                elapsed_time_pattern = re.compile(r" \(\d+s elapsed\)$")

                try:
                    async for partial in get_bot_response(protocol_messages, bot_name=request.model, api_key=poe_token,
                                                          session=proxy):
                        if partial and partial.text:
                            if partial.text.strip() in ["Thinking...", "Generating image..."]:
                                continue
                                
                            base_content = elapsed_time_pattern.sub("", partial.text)

                            if last_sent_base_content == base_content:
                                continue

                            total_response += base_content
                            chunk = {
                                "id": request_id,
                                "object": "chat.completion.chunk",
                                "created": int(asyncio.get_event_loop().time()),
                                "model": request.model,
                                "choices": [{
                                    "delta": {
                                        "content": base_content
                                    },
                                    "index": 0,
                                    "finish_reason": None
                                }],
                                "usage": {
                                    "prompt_tokens": -1,
                                    "completion_tokens": -1,
                                    "total_tokens": -1
                                }
                            }
                            yield f"data: {json.dumps(chunk)}\n\n"
                            last_sent_base_content = base_content

                    # 发送结束标记
                    end_chunk = {
                        "id": request_id,
                        "object": "chat.completion.chunk",
                        "created": int(asyncio.get_event_loop().time()),
                        "model": request.model,
                        "choices": [{
                            "delta": {},
                            "index": 0,
                            "finish_reason": "stop"
                        }]
                    }
                    yield f"data: {json.dumps(end_chunk)}\n\n"
                    yield "data: [DONE]\n\n"

                    # 打印完整的流式响应（限制长度）
                    logger.info(f"Stream Response [{request_id}]: {total_response[:200]}..." if len(
                        total_response) > 200 else total_response)
                except BotError as be:
                    
                    error_chunk = {
                        "id": request_id,
                        "object": "chat.completion.chunk",
                        "created": int(asyncio.get_event_loop().time()),
                        "model": request.model,
                        "choices": [{
                            "delta": {
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
            response = await get_responses(request, poe_token)
            response_data = {
                "id": request_id,
                "object": "chat.completion",
                "created": int(asyncio.get_event_loop().time()),
                "model": request.model,
                "choices": [{
                    "index": 0,
                    "message": {
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
            # 打印完整响应（限制长度）
            safe_response = {**response_data}
            if len(response) > 200:
                logger.info(f"Response [{request_id}]: {json.dumps(safe_response, ensure_ascii=False)[:200]}...")
            else:
                logger.info(f"Response [{request_id}]: {json.dumps(safe_response, ensure_ascii=False)}")
            return response_data
    except GeneratorExit:
        logger.info(f"GeneratorExit exception caught for request [{request_id}]")
    except Exception as e:
        error_msg = f"Error during response for request [{request_id}]: {str(e)}"
        logger.error(error_msg)
        raise HTTPException(status_code=500, detail=str(e))




@router.post("/v1/completion")
@router.post("/completion")
async def create_text_completion(request: TextCompletionRequest, token: str = Depends(verify_token)):
    request_id = "completion$poe-to-gpt$-" + token[:6]

    try:
        # Print request parameters (hiding sensitive information)
        safe_request = request.model_dump()
        if "prompt" in safe_request:
            safe_request["prompt"] = safe_request["prompt"][:100] + "..." if len(safe_request["prompt"]) > 100 else safe_request["prompt"]
        logger.info(f"Text Completion Request [{request_id}]: {json.dumps(safe_request, ensure_ascii=False)}")

        if not api_key_cycle:
            raise HTTPException(status_code=500, detail="No valid API tokens available")

        model_lower = request.model.lower()
        if model_lower not in bot_names_map:
            raise HTTPException(status_code=400, detail=f"Model {request.model} not found")

        actual_model = bot_names_map[model_lower]
        protocol_messages = [
            ProtocolMessage(role="user", content=request.prompt)
        ]
        poe_token = next(api_key_cycle)

        if request.stream:
            import re
            async def response_generator():
                total_response = ""
                last_sent_base_content = None
                elapsed_time_pattern = re.compile(r" \(\d+s elapsed\)$")

                try:
                    async for partial in get_bot_response(protocol_messages, bot_name=actual_model, api_key=poe_token,
                                                          session=proxy):
                        if partial and partial.text:
                            if partial.text.strip() in ["Thinking...", "Generating image..."]:
                                continue
                                
                            base_content = elapsed_time_pattern.sub("", partial.text)

                            if last_sent_base_content == base_content:
                                continue

                            total_response += base_content
                            chunk = {
                                "id": request_id,
                                "object": "text_completion",
                                "created": int(asyncio.get_event_loop().time()),
                                "model": actual_model,
                                "choices": [{
                                    "text": base_content,
                                    "index": 0,
                                    "finish_reason": None
                                }]
                            }
                            yield f"data: {json.dumps(chunk)}\n\n"
                            last_sent_base_content = base_content

                    # Send end marker
                    end_chunk = {
                        "id": request_id,
                        "object": "text_completion",
                        "created": int(asyncio.get_event_loop().time()),
                        "model": actual_model,
                        "choices": [{
                            "text": "",
                            "index": 0,
                            "finish_reason": "stop"
                        }]
                    }
                    yield f"data: {json.dumps(end_chunk)}\n\n"
                    yield "data: [DONE]\n\n"

                    # Print complete streaming response (limit length)
                    logger.info(f"Stream Text Completion Response [{request_id}]: {total_response[:200]}..." if len(
                        total_response) > 200 else total_response)
                except BotError as be:
                    error_chunk = {
                        "id": request_id,
                        "object": "text_completion",
                        "created": int(asyncio.get_event_loop().time()),
                        "model": actual_model,
                        "choices": [{
                            "text": json.loads(be.args[0])["text"],
                            "index": 0,
                            "finish_reason": "error"
                        }]
                    }
                    yield f"data: {json.dumps(error_chunk)}\n\n"
                    yield "data: [DONE]\n\n"
                    logger.error(f"BotError in text completion stream generation for [{request_id}]: {str(be)}")
                except Exception as e:
                    logger.error(f"Error in text completion stream generation for [{request_id}]: {str(e)}")
                    raise

            return StreamingResponse(response_generator(), media_type="text/event-stream")
        else:
            response = await get_text_completion_response(request, poe_token)
            response_data = {
                "id": request_id,
                "object": "text_completion",
                "created": int(asyncio.get_event_loop().time()),
                "model": actual_model,
                "choices": [{
                    "index": 0,
                    "text": response,
                    "finish_reason": "stop"
                }],
                "usage": {
                    "prompt_tokens": -1,
                    "completion_tokens": -1,
                    "total_tokens": -1
                }
            }
            # Print complete response (limit length)
            safe_response = {**response_data}
            if len(response) > 200:
                logger.info(f"Text Completion Response [{request_id}]: {json.dumps(safe_response, ensure_ascii=False)[:200]}...")
            else:
                logger.info(f"Text Completion Response [{request_id}]: {json.dumps(safe_response, ensure_ascii=False)}")
            return response_data
    except GeneratorExit:
        logger.info(f"GeneratorExit exception caught for text completion request [{request_id}]")
    except Exception as e:
        error_msg = f"Error during text completion response for request [{request_id}]: {str(e)}"
        logger.error(error_msg)
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/models")
@router.get("/v1/models")
async def get_models():
    model_list = [{"id": name, "object": "model", "type": "llm"} for name in BOT_NAMES]
    return {"data": model_list, "object": "list"}


async def initialize_tokens(tokens: List[str]):
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
    """Parse command line arguments"""
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

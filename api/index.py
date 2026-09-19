import json
import time
import uuid

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, StreamingResponse

from gemini_web2api.config import CONFIG
from gemini_web2api.models import MODELS, resolve_model
from gemini_web2api.gemini import generate, generate_stream
from gemini_web2api.tools import messages_to_prompt

app = FastAPI()


def authorized(request: Request) -> bool:
    keys = CONFIG.get("api_keys", [])

    # If no keys are configured, allow access.
    if not keys:
        return True

    auth = request.headers.get("authorization", "")

    if auth.startswith("Bearer ") and auth[7:] in keys:
        return True

    if request.headers.get("x-api-key", "") in keys:
        return True

    if request.headers.get("x-goog-api-key", "") in keys:
        return True

    return False


@app.get("/")
async def root():
    return {
        "status": "ok",
        "version": "1.1.0",
        "models": list(MODELS.keys()),
    }


@app.get("/api/v1/models")
@app.get("/v1/models")
async def models(request: Request):

    if not authorized(request):
        return JSONResponse(
            {"error": {"message": "invalid api key"}},
            status_code=401,
        )

    return {
        "object": "list",
        "data": [
            {
                "id": name,
                "object": "model",
                "created": 1700000000,
                "owned_by": "google",
                "description": cfg["desc"],
            }
            for name, cfg in MODELS.items()
        ],
    }


@app.post("/api/v1/chat/completions")
@app.post("/v1/chat/completions")
async def chat_completions(request: Request):

    if not authorized(request):
        return JSONResponse(
            {"error": {"message": "invalid api key"}},
            status_code=401,
        )

    body = await request.json()

    model_name = body.get(
        "model",
        CONFIG.get("default_model", "gemini-3.6-flash"),
    )

    model_name, model_id, think_mode, error = resolve_model(model_name)

    if error:
        return JSONResponse(
            {"error": {"message": error}},
            status_code=400,
        )

    messages = body.get("messages", [])

    prompt, images = messages_to_prompt(
        messages,
        body.get("tools"),
    )

    if not prompt.strip():
        return JSONResponse(
            {"error": {"message": "empty prompt"}},
            status_code=400,
        )

    stream = body.get("stream", False)

    if stream:

        def generate_sse():
            completion_id = f"chatcmpl-{uuid.uuid4().hex[:12]}"

            first = {
                "id": completion_id,
                "object": "chat.completion.chunk",
                "created": int(time.time()),
                "model": model_name,
                "choices": [
                    {
                        "index": 0,
                        "delta": {"role": "assistant"},
                        "finish_reason": None,
                    }
                ],
            }

            yield f"data: {json.dumps(first)}\n\n"

            for delta in generate_stream(
                prompt,
                model_id,
                think_mode,
            ):
                chunk = {
                    "id": completion_id,
                    "object": "chat.completion.chunk",
                    "created": int(time.time()),
                    "model": model_name,
                    "choices": [
                        {
                            "index": 0,
                            "delta": {"content": delta},
                            "finish_reason": None,
                        }
                    ],
                }

                yield f"data: {json.dumps(chunk, ensure_ascii=False)}\n\n"

            final = {
                "id": completion_id,
                "object": "chat.completion.chunk",
                "created": int(time.time()),
                "model": model_name,
                "choices": [
                    {
                        "index": 0,
                        "delta": {},
                        "finish_reason": "stop",
                    }
                ],
            }

            yield f"data: {json.dumps(final)}\n\n"
            yield "data: [DONE]\n\n"

        return StreamingResponse(
            generate_sse(),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
            },
        )

    try:
        text = generate(
            prompt,
            model_id,
            think_mode,
        )

    except Exception as e:
        return JSONResponse(
            {
                "error": {
                    "message": f"upstream error: {str(e)}"
                }
            },
            status_code=502,
        )

    completion_id = f"chatcmpl-{uuid.uuid4().hex[:12]}"

    return {
        "id": completion_id,
        "object": "chat.completion",
        "created": int(time.time()),
        "model": model_name,
        "choices": [
            {
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": text,
                },
                "finish_reason": "stop",
            }
        ],
        "usage": {
            "prompt_tokens": len(prompt) // 4,
            "completion_tokens": len(text or "") // 4,
            "total_tokens": (
                len(prompt) + len(text or "")
            ) // 4,
        },
    }

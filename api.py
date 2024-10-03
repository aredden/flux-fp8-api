from typing import Literal, Optional, TYPE_CHECKING

import numpy as np
from fastapi import FastAPI
from fastapi.responses import StreamingResponse, JSONResponse
from pydantic import BaseModel, Field
from platform import system

if TYPE_CHECKING:
    from flux_pipeline import FluxPipeline

if system() == "Windows":
    MAX_RAND = 2**16 - 1
else:
    MAX_RAND = 2**32 - 1


class AppState:
    model: "FluxPipeline"


class FastAPIApp(FastAPI):
    state: AppState


class LoraArgs(BaseModel):
    scale: Optional[float] = 1.0
    path: Optional[str] = None
    name: Optional[str] = None
    action: Optional[Literal["load", "unload"]] = "load"


class LoraLoadResponse(BaseModel):
    status: Literal["success", "error"]
    message: Optional[str] = None


class GenerateArgs(BaseModel):
    prompt: str
    width: Optional[int] = Field(default=720)
    height: Optional[int] = Field(default=1024)
    num_steps: Optional[int] = Field(default=24)
    guidance: Optional[float] = Field(default=3.5)
    seed: Optional[int] = Field(
        default_factory=lambda: np.random.randint(0, MAX_RAND), gt=0, lt=MAX_RAND
    )
    strength: Optional[float] = 1.0
    init_image: Optional[str] = None


app = FastAPIApp()


@app.post("/generate")
def generate(args: GenerateArgs):
    """
    Generates an image from the Flux flow transformer.

    Args:
        args (GenerateArgs): Arguments for image generation:

            - `prompt`: The prompt used for image generation.

            - `width`: The width of the image.

            - `height`: The height of the image.

            - `num_steps`: The number of steps for the image generation.

            - `guidance`: The guidance for image generation, represents the
                influence of the prompt on the image generation.

            - `seed`: The seed for the image generation.

            - `strength`: strength for image generation, 0.0 - 1.0.
                Represents the percent of diffusion steps to run,
                setting the init_image as the noised latent at the
                given number of steps.

            - `init_image`: Base64 encoded image or path to image to use as the init image.

    Returns:
        StreamingResponse: The generated image as streaming jpeg bytes.
    """
    result = app.state.model.generate(**args.model_dump())
    return StreamingResponse(result, media_type="image/jpeg")


@app.post("/lora", response_model=LoraLoadResponse)
def lora_action(args: LoraArgs):
    """
    Loads or unloads a LoRA checkpoint into / from the Flux flow transformer.

    Args:
        args (LoraArgs): Arguments for the LoRA action:

            - `scale`: The scaling factor for the LoRA weights.
            - `path`: The path to the LoRA checkpoint.
            - `name`: The name of the LoRA checkpoint.
            - `action`: The action to perform, either "load" or "unload".

    Returns:
        LoraLoadResponse: The status of the LoRA action.
    """
    try:
        if args.action == "load":
            app.state.model.load_lora(args.path, args.scale, args.name)
        elif args.action == "unload":
            app.state.model.unload_lora(args.name if args.name else args.path)
        else:
            return JSONResponse(
                content={
                    "status": "error",
                    "message": f"Invalid action, expected 'load' or 'unload', got {args.action}",
                },
                status_code=400,
            )
    except Exception as e:
        return JSONResponse(
            status_code=500, content={"status": "error", "message": str(e)}
        )
    return JSONResponse(status_code=200, content={"status": "success"})

from typing import Optional

import numpy as np
from fastapi import FastAPI
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field
from platform import system

if system() == "Windows":
    MAX_RAND = 2**16 - 1
else:
    MAX_RAND = 2**32 - 1

app = FastAPI()


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


@app.post("/generate")
def generate(args: GenerateArgs):
    result = app.state.model.generate(**args.model_dump())
    return StreamingResponse(result, media_type="image/jpeg")

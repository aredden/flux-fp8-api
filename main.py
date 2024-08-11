import argparse
import uvicorn
from api import app
from flux_pipeline import FluxPipeline
from util import load_config, ModelVersion


def parse_args():
    parser = argparse.ArgumentParser(description="Launch Flux API server")
    parser.add_argument(
        "-c",
        "--config-path",
        type=str,
        help="Path to the configuration file, if not provided, the model will be loaded from the command line arguments",
    )
    parser.add_argument(
        "-p",
        "--port",
        type=int,
        default=8088,
        help="Port to run the server on",
    )
    parser.add_argument(
        "-H",
        "--host",
        type=str,
        default="0.0.0.0",
        help="Host to run the server on",
    )
    parser.add_argument(
        "-f", "--flow-model-path", type=str, help="Path to the flow model"
    )
    parser.add_argument(
        "-t", "--text-enc-path", type=str, help="Path to the text encoder"
    )
    parser.add_argument(
        "-a", "--autoencoder-path", type=str, help="Path to the autoencoder"
    )
    parser.add_argument(
        "-m",
        "--model-version",
        type=str,
        choices=["flux-dev", "flux-schnell"],
        default="flux-dev",
        help="Choose model version",
    )
    parser.add_argument(
        "-F",
        "--flux-device",
        type=str,
        default="cuda:0",
        help="Device to run the flow model on",
    )
    parser.add_argument(
        "-T",
        "--text-enc-device",
        type=str,
        default="cuda:0",
        help="Device to run the text encoder on",
    )
    parser.add_argument(
        "-A",
        "--autoencoder-device",
        type=str,
        default="cuda:0",
        help="Device to run the autoencoder on",
    )
    parser.add_argument(
        "-q",
        "--num-to-quant",
        type=int,
        default=20,
        help="Number of linear layers in flow transformer (the 'unet') to quantize",
    )
    parser.add_argument(
        "-C",
        "--compile",
        action="store_true",
        default=False,
        help="Compile the flow model with extra optimizations",
    )

    return parser.parse_args()


def main():
    args = parse_args()

    if args.config_path:
        app.state.model = FluxPipeline.load_pipeline_from_config_path(args.config_path)
    else:
        model_version = (
            ModelVersion.flux_dev
            if args.model_version == "flux-dev"
            else ModelVersion.flux_schnell
        )
        config = load_config(
            model_version,
            flux_path=args.flow_model_path,
            flux_device=args.flux_device,
            ae_path=args.autoencoder_path,
            ae_device=args.autoencoder_device,
            text_enc_path=args.text_enc_path,
            text_enc_device=args.text_enc_device,
            flow_dtype="float16",
            text_enc_dtype="bfloat16",
            ae_dtype="bfloat16",
            num_to_quant=args.num_to_quant,
            compile_extras=args.compile,
            compile_blocks=args.compile,
        )
        app.state.model = FluxPipeline.load_pipeline_from_config(config)

    uvicorn.run(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
